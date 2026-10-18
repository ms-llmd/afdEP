// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Copyright contributors to the AFDEP project
// Adapted from DeepEP via the pinned FastAFD snapshot. See THIRD_PARTY_NOTICES.md.
#pragma once

#include <cstdint>
#include <nccl.h>
#include <nccl_device.h>
#include <pybind11/pytypes.h>
#include <tuple>
#include <vector>

namespace afdep {

pybind11::bytearray get_nccl_unique_id();
int64_t create_nccl_comm(const pybind11::bytearray &unique_id, int world_size, int rank);
void destroy_nccl_comm(int64_t handle);
std::tuple<int, int> get_physical_domain_size(int64_t handle);
int get_nccl_version();

// Only registered memory and the GPU-initiated communicator are retained
// from DeepEP's runtime. No dispatch/combine layout or expert protocol is used.
struct NcclContext {
  private:
    void *raw_window_ptr = nullptr;
    bool device_comm_created = false;
    bool window_registered = false;

  public:
    int rank_idx, num_ranks;
    int num_rdma_ranks = 0, num_nvl_ranks = 0;
    int rdma_rank_idx = 0, nvl_rank_idx = 0;
    ncclComm_t comm;
    ncclDevComm_t dev_comm{};
    ncclWindow_t window{};
    void *mapped_window_ptr = nullptr;
    std::vector<void *> nvl_window_ptrs;

    NcclContext(int64_t handle, int world_size, int rank, size_t size, size_t alignment, int traffic_class,
                int num_qps, bool allow_hybrid_mode = true);
    NcclContext(const NcclContext &) = delete;
    NcclContext &operator=(const NcclContext &) = delete;

    // The owner must drain every GPU and synchronize ranks before finalizing.
    // A destructor must not silently deregister windows still read by peers.
    void finalize();
    void finalize_noexcept() noexcept;
};

} // namespace afdep
