// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Copyright contributors to the AFDEP project
// Adapted from DeepEP via the pinned FastAFD snapshot. See THIRD_PARTY_NOTICES.md.
#include <cstdlib>
#include <cstring>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "checks.hpp"
#include "layout.hpp"
#include "nccl_runtime.hpp"

namespace afdep {

pybind11::bytearray get_nccl_unique_id() {
    ncclUniqueId id;
    AFDEP_NCCL_CHECK(ncclGetUniqueId(&id));
    return {reinterpret_cast<const char *>(&id), sizeof(id)};
}

int64_t create_nccl_comm(const pybind11::bytearray &unique_id, int world_size, int rank) {
    const auto bytes = unique_id.cast<std::string>();
    TORCH_CHECK(bytes.size() == sizeof(ncclUniqueId), "Invalid NCCL unique ID size");
    TORCH_CHECK(world_size == 8 && rank >= 0 && rank < world_size,
                "The current directional backend requires eight ranks");
    ncclUniqueId id;
    std::memcpy(&id, bytes.data(), sizeof(id));
    ncclComm_t comm = nullptr;
    const auto status = ncclCommInitRank(&comm, world_size, id, rank);
    if (status != ncclSuccess && comm != nullptr)
        ncclCommAbort(comm);
    AFDEP_NCCL_CHECK(status);
    return reinterpret_cast<int64_t>(comm);
}

void destroy_nccl_comm(int64_t handle) {
    if (handle != 0)
        AFDEP_NCCL_CHECK(ncclCommAbort(reinterpret_cast<ncclComm_t>(handle)));
}

int get_nccl_version() {
    int version;
    AFDEP_NCCL_CHECK(ncclGetVersion(&version));
    return version;
}

std::tuple<int, int> get_physical_domain_size(int64_t handle) {
    TORCH_CHECK(handle != 0, "NCCL communicator is not initialized");
    const auto comm = reinterpret_cast<ncclComm_t>(handle);
    const int world = ncclTeamWorld(comm).nRanks;
    const int local = ncclTeamLsa(comm).nRanks;
    TORCH_CHECK(local > 0 && world % local == 0, "Invalid NCCL NVLink domain size");
    return {world / local, local};
}

NcclContext::NcclContext(int64_t handle, int world_size, int rank, size_t size, size_t alignment,
                         int traffic_class, int num_qps, bool allow_hybrid_mode)
    : rank_idx(rank), num_ranks(world_size), comm(reinterpret_cast<ncclComm_t>(handle)) {
    TORCH_CHECK(handle != 0 && num_qps > 0, "NCCL communicator and positive GIN contexts are required");
    const char *disable_gin = std::getenv("EP_DISABLE_GIN");
    TORCH_CHECK(disable_gin == nullptr || std::string(disable_gin) == "0",
                "The current directional backend requires NCCL GIN; unset EP_DISABLE_GIN or set it to 0");
    TORCH_CHECK(size > 0 && alignment > 0 && size % alignment == 0,
                "Registered window size must satisfy its alignment");
    int actual_rank, actual_world, communicator_device, current_device;
    AFDEP_NCCL_CHECK(ncclCommUserRank(comm, &actual_rank));
    AFDEP_NCCL_CHECK(ncclCommCount(comm, &actual_world));
    AFDEP_NCCL_CHECK(ncclCommCuDevice(comm, &communicator_device));
    AFDEP_CUDA_CHECK(cudaGetDevice(&current_device));
    TORCH_CHECK(actual_rank == rank && actual_world == world_size && communicator_device == current_device,
                "AFDEP NCCL communicator rank, size, or CUDA device does not match its buffer");

    // AFDEP-specific extraction of DeepEP's host runtime: rail/full GIN,
    // physical domain validation, explicit initialization-failure cleanup.
    ncclCommProperties properties = NCCL_COMM_PROPERTIES_INITIALIZER;
    AFDEP_NCCL_CHECK(ncclCommQueryProperties(comm, &properties));
    TORCH_CHECK(
        (allow_hybrid_mode ? properties.railedGinType : properties.ginType) != NCCL_GIN_TYPE_NONE,
        "The current directional backend requires NCCL GPU-initiated RDMA between its NVLink domains");
    ncclDevCommRequirements_t requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    requirements.ginContextCount = num_qps;
    requirements.ginExclusiveContexts = true;
    requirements.ginQueueDepth = 1024;
    requirements.ginTrafficClass = traffic_class;
    // Hybrid retains the measured allocation; direct uses one signal per peer
    // for each request/reply readiness and credit phase.
    requirements.ginSignalCount = allow_hybrid_mode ? world_size + 2 * 2 : kDirectSignalCount;
    requirements.ginConnectionType = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_FULL;
    try {
        AFDEP_NCCL_CHECK(ncclDevCommCreate(comm, &requirements, &dev_comm));
        device_comm_created = true;
        num_nvl_ranks = dev_comm.lsaSize;
        nvl_rank_idx = dev_comm.lsaRank;
        TORCH_CHECK(
            num_nvl_ranks == 4 && world_size == 8 && nvl_rank_idx == rank % 4,
            "The current directional backend requires contiguous four-GPU NVLink domains [F0..F3,A0..A3]");
        num_rdma_ranks = world_size / num_nvl_ranks;
        rdma_rank_idx = rank / num_nvl_ranks;
        AFDEP_NCCL_CHECK(ncclMemAlloc(&raw_window_ptr, size));
        AFDEP_NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, size, &window, NCCL_WIN_DEFAULT));
        window_registered = true;
        AFDEP_NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr));
        nvl_window_ptrs.resize(num_nvl_ranks);
        for (int i = 0; i < num_nvl_ranks; ++i)
            AFDEP_NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]));
        TORCH_CHECK(reinterpret_cast<uintptr_t>(raw_window_ptr) % alignment == 0 &&
                        reinterpret_cast<uintptr_t>(mapped_window_ptr) % alignment == 0,
                    "NCCL window pointer does not satisfy the required alignment");
    } catch (...) {
        finalize_noexcept();
        throw;
    }
}

void NcclContext::finalize() {
    if (window_registered) {
        AFDEP_NCCL_CHECK(ncclCommWindowDeregister(comm, window));
        window_registered = false;
    }
    if (raw_window_ptr != nullptr) {
        AFDEP_NCCL_CHECK(ncclMemFree(raw_window_ptr));
        raw_window_ptr = nullptr;
    }
    if (device_comm_created) {
        AFDEP_NCCL_CHECK(ncclDevCommDestroy(comm, &dev_comm));
        device_comm_created = false;
    }
    mapped_window_ptr = nullptr;
    nvl_window_ptrs.clear();
}

void NcclContext::finalize_noexcept() noexcept {
    // Initialization failure only. During normal use, finalize() is explicit
    // and preserves failures for the caller after the all-rank drain/barrier.
    if (window_registered && ncclCommWindowDeregister(comm, window) == ncclSuccess)
        window_registered = false;
    if (!window_registered && raw_window_ptr != nullptr && ncclMemFree(raw_window_ptr) == ncclSuccess)
        raw_window_ptr = nullptr;
    if (device_comm_created && ncclDevCommDestroy(comm, &dev_comm) == ncclSuccess)
        device_comm_created = false;
}

} // namespace afdep
