// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <torch/python.h>
#include <vector>

#include "checks.hpp"
#include "kernels.hpp"
#include "layout.hpp"
#include "nccl_runtime.hpp"

namespace afdep {

// Directional request/reply protocol. All payload kernels belong to AFDEP;
// the minimal NCCL host runtime is adapted from DeepEP under its MIT license.
class DirectionalM2NBuffer {
    Layout layout;
    int rank, device, stripes, num_sms;
    uint64_t timeout_cycles;
    bool destroyed = false;
    int source_rows = 0;
    int worker_bucket = 0;
    int worker_rows = 0;
    std::shared_ptr<NcclContext> context;
    torch::Tensor counts, starts, cursors;

    static Layout checked_layout(int capacity, int hidden, int topk, int experts, int alignment, bool fp8,
                                 bool allow_hybrid_mode) {
        TORCH_CHECK(capacity > 0 && hidden > 0 && hidden % 128 == 0,
                    "Directional capacity must be positive and hidden a positive multiple of 128");
        TORCH_CHECK(topk > 0 && topk <= 32 && topk <= experts && experts > 0 && experts % 4 == 0,
                    "Directional experts must divide over four workers; top-k must be in [1,32]");
        TORCH_CHECK(alignment > 0, "Directional expert alignment must be positive");
        constexpr int64_t kMaxIndex = std::numeric_limits<int>::max();
        const int64_t rows =
            int64_t(4) * capacity * std::min(topk, experts / 4) + int64_t(experts / 4) * (alignment - 1);
        TORCH_CHECK(int64_t(4) * capacity * topk <= kMaxIndex && rows <= kMaxIndex - 3 &&
                        int64_t(hidden) * 2 <= kMaxIndex &&
                        int64_t(capacity) * hidden * 2 / 16 <= kMaxIndex &&
                        rows * (int64_t(hidden) * 2 / 16) <= kMaxIndex,
                    "Directional dimensions exceed the supported 32-bit device index range");
        return Layout::make(capacity, hidden, topk, experts, alignment, fp8, allow_hybrid_mode);
    }

    void check_live(bool worker) const {
        TORCH_CHECK(!destroyed, "DirectionalM2NBuffer was destroyed");
        TORCH_CHECK((rank < 4) == worker,
                    worker ? "worker API requires an FFN rank" : "source API requires an Attention rank");
        int current;
        AFDEP_CUDA_CHECK(cudaGetDevice(&current));
        TORCH_CHECK(current == device, "Directional buffer used on a different CUDA device");
    }

    void check_tensor(const torch::Tensor &tensor, torch::ScalarType dtype, int dimensions,
                      bool contiguous = true) const {
        TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == device,
                    "Directional tensors must be on the buffer's CUDA device");
        TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == dimensions,
                    "Unexpected directional tensor dtype/dimensions");
        TORCH_CHECK(!contiguous || tensor.is_contiguous(), "Directional payload tensors must be contiguous");
    }

    static void check_vector_alignment(const torch::Tensor &tensor) {
        constexpr uintptr_t kVectorAlignment = 16;
        TORCH_CHECK(tensor.numel() == 0 ||
                        reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kVectorAlignment == 0,
                    "Directional vector-copy tensors require a 16-byte aligned data pointer");
    }

    DeviceArgs args(int bucket) const {
        DeviceArgs result{};
        result.layout = layout;
        result.comm = context->dev_comm;
        result.window = context->window;
        result.base = static_cast<uint8_t *>(context->mapped_window_ptr);
        if (rank < 4) {
            for (int i = 0; i < 4; ++i)
                result.peers[i] = static_cast<uint8_t *>(context->nvl_window_ptrs[i]);
        }
        result.rank = rank;
        result.bucket = bucket;
        result.stripes = stripes;
        result.timeout_cycles = timeout_cycles;
        return result;
    }

    void launch(int operation, DeviceArgs arguments, int blocks = 1) {
        const auto stream = at::cuda::getCurrentCUDAStream();
        AFDEP_CUDA_CHECK(launch_kernel(operation, arguments, blocks, stream.stream()));
    }

  public:
    const int64_t num_buffer_bytes;

    DirectionalM2NBuffer(int64_t nccl_comm, int rank, int world, int ffn_size, int attn_size, int capacity,
                         int hidden, int topk, int real_experts, int expert_alignment, bool use_fp8,
                         int num_qps, int gpu_timeout_s, int traffic_class, int transport_stripes = 8,
                         int num_sms = 132, bool allow_hybrid_mode = true)
        : layout(checked_layout(capacity, hidden, topk, real_experts, expert_alignment, use_fp8,
                                allow_hybrid_mode)), rank(rank),
          stripes(transport_stripes), num_sms(num_sms),
          num_buffer_bytes(rank < 4 ? layout.worker_bytes : layout.source_bytes) {
        TORCH_CHECK(world == 8 && ffn_size == 4 && attn_size == 4 && rank >= 0 && rank < 8,
                    "The current directional backend requires world [F0..F3,A0..A3]");
        TORCH_CHECK(expert_alignment > 0 && gpu_timeout_s > 0,
                    "Directional alignment and timeout settings must be positive");
        TORCH_CHECK(num_qps > 0 && stripes > 0 && stripes <= num_qps,
                    "Directional stripes must be in [1,num_qps]");
        AFDEP_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        AFDEP_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        TORCH_CHECK(num_sms > 0 && num_sms <= properties.multiProcessorCount,
                    "Directional num_sms exceeds the device SM count");
        const auto physical = get_physical_domain_size(nccl_comm);
        TORCH_CHECK(std::get<0>(physical) == 2 && std::get<1>(physical) == 4,
                    "The current directional backend requires exactly two four-GPU NVLink domains");
        TORCH_CHECK(properties.major == 9, "The current directional CUDA backend requires SM90");
        int clock_rate_khz;
        AFDEP_CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device));
        timeout_cycles = uint64_t(gpu_timeout_s) * uint64_t(clock_rate_khz) * 1000;

        context = std::make_shared<NcclContext>(nccl_comm, world, rank, size_t(num_buffer_bytes), 128,
                                                traffic_class, num_qps, allow_hybrid_mode);
        try {
            TORCH_CHECK(stripes <= context->dev_comm.ginContextCount,
                        "NCCL provided fewer GIN contexts than requested transport stripes");
            TORCH_CHECK(context->num_nvl_ranks == 4 && context->num_rdma_ranks == 2 &&
                            context->rdma_rank_idx == (rank < 4 ? 0 : 1) && context->nvl_rank_idx == rank % 4,
                        "Unsupported directional placement: workers must fill node0, sources node1");
            AFDEP_CUDA_CHECK(cudaMemsetAsync(context->mapped_window_ptr, 0, sizeof(Control),
                                             at::cuda::getCurrentCUDAStream().stream()));
            // The Python owner performs a Gloo barrier after this local init. Do
            // not create ordinary NCCL data-plane QPs just for synchronization.
            AFDEP_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream()));
            const auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device);
            counts = torch::empty({layout.local_experts()}, options);
            starts = torch::empty_like(counts);
            cursors = torch::empty_like(counts);
        } catch (...) {
            context->finalize_noexcept();
            throw;
        }
    }

    int64_t registered_bytes() const { return num_buffer_bytes; }

    void source_dispatch(const torch::Tensor &x, const std::optional<torch::Tensor> &scale,
                         const torch::Tensor &ids, const torch::Tensor &weights, int bucket) {
        check_live(false);
        check_tensor(x, layout.fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16, 2);
        check_vector_alignment(x);
        check_tensor(ids, torch::kInt64, 2);
        check_tensor(weights, torch::kFloat32, 2);
        TORCH_CHECK(bucket > 0 && bucket <= layout.capacity && x.size(0) <= bucket &&
                        x.size(1) == layout.hidden && ids.size(0) == x.size(0) &&
                        ids.size(1) == layout.topk && weights.sizes() == ids.sizes(),
                    "Directional source payload shape/bucket mismatch");
        TORCH_CHECK(scale.has_value() == bool(layout.fp8), "FP8 payload requires block128 FP32 scales");
        auto a = args(bucket);
        a.tokens = source_rows = x.size(0);
        a.x = static_cast<const uint8_t *>(x.data_ptr());
        a.ids = ids.data_ptr<int64_t>();
        a.weights = weights.data_ptr<float>();
        if (scale.has_value()) {
            check_tensor(*scale, torch::kFloat32, 2, false);
            TORCH_CHECK(scale->size(0) == x.size(0) && scale->size(1) == layout.hidden / 128,
                        "Expected FP8 scale shape [tokens,hidden/128]");
            a.scales = scale->data_ptr<float>();
            a.scale_stride0 = scale->stride(0);
            a.scale_stride1 = scale->stride(1);
        }
        launch(BeginSource, a);
        if (!layout.hybrid)
            launch(PlanDirectRequests, a, num_sms);
        launch(PackSource, a, num_sms);
        launch(SendRequest, a, stripes * (layout.hybrid ? 1 : 4));
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::vector<int>,
               torch::Tensor>
    worker_receive(int bucket, bool read_host_counts) {
        check_live(true);
        TORCH_CHECK(bucket > 0 && bucket <= layout.capacity, "Directional worker bucket exceeds capacity");
        worker_bucket = bucket;
        worker_rows = 4 * bucket * std::min(layout.topk, layout.local_experts()) +
                      layout.local_experts() * (layout.alignment - 1);
        const auto options = torch::TensorOptions().device(torch::kCUDA, device);
        auto recv_x = torch::empty({worker_rows, layout.hidden},
                                   options.dtype(layout.fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
        auto recv_weights = torch::zeros({worker_rows}, options.dtype(torch::kFloat32));
        auto psum = torch::empty({layout.local_experts()}, options.dtype(torch::kInt32));
        auto receipt = torch::full({4, bucket, layout.topk}, -1, options.dtype(torch::kInt32));
        std::optional<torch::Tensor> recv_scale;
        auto a = args(bucket);
        if (layout.fp8) {
            const int scale_stride = (worker_rows + 3) / 4 * 4;
            recv_scale = torch::empty_strided({worker_rows, layout.scale_groups()}, {1, scale_stride},
                                              options.dtype(torch::kFloat32));
            a.recv_scales = recv_scale->data_ptr<float>();
            a.recv_scale_stride1 = scale_stride;
        }
        a.recv_x = static_cast<uint8_t *>(recv_x.data_ptr());
        a.recv_weights = recv_weights.data_ptr<float>();
        a.counts = counts.data_ptr<int>();
        a.starts = starts.data_ptr<int>();
        a.cursors = cursors.data_ptr<int>();
        a.psum = psum.data_ptr<int>();
        a.receipt = receipt.data_ptr<int>();
        counts.zero_();
        launch(BeginWorker, a);
        launch(Histogram, a, num_sms);
        launch(Prefix, a);
        launch(PackRows, a, num_sms);
        launch(PublishPackAndCredit, a);
        std::vector<int> host_counts;
        if (read_host_counts) {
            const auto cpu_counts = counts.to(torch::kCPU);
            const auto *data = cpu_counts.data_ptr<int>();
            for (int e = 0; e < layout.local_experts(); ++e)
                host_counts.push_back((data[e] + layout.alignment - 1) / layout.alignment * layout.alignment);
        }
        return {recv_x, recv_scale, recv_weights, psum, host_counts, receipt};
    }

    void worker_reply(const torch::Tensor &rows, const torch::Tensor &receipt) {
        check_live(true);
        check_tensor(rows, torch::kBFloat16, 2);
        check_tensor(receipt, torch::kInt32, 3);
        TORCH_CHECK(worker_bucket > 0 && rows.size(0) == worker_rows && rows.size(1) == layout.hidden &&
                        receipt.size(0) == 4 && receipt.size(1) == worker_bucket &&
                        receipt.size(2) == layout.topk,
                    "Directional reply must use its matching receive receipt/row shape");
        auto a = args(worker_bucket);
        a.rows = static_cast<const uint16_t *>(rows.data_ptr());
        a.receipt = receipt.data_ptr<int>();
        launch(MakePartials, a, num_sms);
        if (layout.hybrid) {
            launch(PublishPartialAndWait, a);
            launch(ReduceReply, a, num_sms);
        }
        launch(PublishConsumedAndWaitCredit, a);
        launch(SendReply, a, stripes * (layout.hybrid ? 1 : 4));
    }

    void source_collect(const torch::Tensor &out) {
        check_live(false);
        check_tensor(out, torch::kBFloat16, 2);
        check_vector_alignment(out);
        TORCH_CHECK(out.size(0) == source_rows && out.size(1) == layout.hidden,
                    "Directional output must match the source dispatch shape");
        auto a = args(std::max(source_rows, 1));
        a.out = static_cast<uint16_t *>(out.data_ptr());
        launch(CollectReply, a, num_sms);
        launch(ReplyConsumedCredit, a);
    }

    void destroy() {
        if (destroyed)
            return;
        int current;
        AFDEP_CUDA_CHECK(cudaGetDevice(&current));
        TORCH_CHECK(current == device, "Directional buffer must be destroyed on its owning CUDA device");
        // The Python owner synchronizes every rank's GPU, then performs a
        // Gloo barrier, before any rank may deregister its worker windows.
        AFDEP_CUDA_CHECK(cudaDeviceSynchronize());
        context->finalize();
        destroyed = true;
    }
};

inline void register_apis(pybind11::module_ &m) {
    namespace py = pybind11;
    py::class_<DirectionalM2NBuffer>(m, "DirectionalM2NBuffer", py::module_local())
        .def(py::init<int64_t, int, int, int, int, int, int, int, int, int, bool, int, int, int, int, int, bool>(),
             py::arg("nccl_comm"), py::arg("rank"), py::arg("world"), py::arg("ffn_size"),
             py::arg("attn_size"), py::arg("capacity"), py::arg("hidden"), py::arg("top_k"),
             py::arg("real_experts"), py::arg("expert_alignment"), py::arg("use_fp8"), py::arg("num_qps"),
             py::arg("gpu_timeout_s"), py::arg("traffic_class"), py::arg("transport_stripes") = 8,
             py::arg("num_sms") = 132, py::arg("allow_hybrid_mode") = true)
        .def_readonly("num_buffer_bytes", &DirectionalM2NBuffer::num_buffer_bytes)
        .def("registered_bytes", &DirectionalM2NBuffer::registered_bytes)
        .def("source_dispatch", &DirectionalM2NBuffer::source_dispatch)
        .def("worker_receive", &DirectionalM2NBuffer::worker_receive)
        .def("worker_reply", &DirectionalM2NBuffer::worker_reply)
        .def("source_collect", &DirectionalM2NBuffer::source_collect)
        .def("destroy", &DirectionalM2NBuffer::destroy);
}

} // namespace afdep
