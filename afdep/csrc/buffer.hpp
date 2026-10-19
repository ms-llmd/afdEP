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

// Directional request/reply protocol with adapted DeepEP copy/reduction
// primitives and minimal NCCL host runtime. See ORIGIN.json and MIT notices.
class DirectionalM2NBuffer {
    Layout layout;
    int rank, device, stripes, num_sms;
    int wide_tma_threads = kNVLinkTmaWideThreads;
    uint64_t timeout_cycles;
    bool destroyed = false;
    bool nvlink_pair_pending = false;
    int source_rows = 0;
    int worker_bucket = 0;
    int worker_rows = 0;
    std::shared_ptr<NcclContext> context;
    torch::Tensor counts, starts, cursors;

    static Layout checked_layout(int capacity, int hidden, int topk, int experts, int alignment, bool fp8,
                                 bool allow_hybrid_mode, int workers, int sources, bool nvlink) {
        TORCH_CHECK(capacity > 0 && hidden > 0 && hidden % 128 == 0,
                    "Directional capacity must be positive and hidden a positive multiple of 128");
        TORCH_CHECK(workers == (nvlink ? 2 : 4) && sources == workers,
                    "Directional transport supports 2F2A NVLink or 4F4A GIN");
        TORCH_CHECK(topk > 0 && topk <= 32 && topk <= experts && experts > 0 && experts % workers == 0,
                    "Directional experts must divide over workers; top-k must be in [1,32]");
        TORCH_CHECK(alignment > 0, "Directional expert alignment must be positive");
        constexpr int64_t kMaxIndex = std::numeric_limits<int>::max();
        const int64_t rows =
            int64_t(sources) * capacity * std::min(topk, experts / workers) + int64_t(experts / workers) * (alignment - 1);
        TORCH_CHECK(int64_t(sources) * capacity * topk <= kMaxIndex && rows <= kMaxIndex - 3 &&
                        int64_t(hidden) * 2 <= kMaxIndex &&
                        int64_t(capacity) * hidden * 2 / 16 <= kMaxIndex &&
                        rows * (int64_t(hidden) * 2 / 16) <= kMaxIndex,
                    "Directional dimensions exceed the supported 32-bit device index range");
        return Layout::make(capacity, hidden, topk, experts, alignment, fp8, allow_hybrid_mode, workers, sources, nvlink);
    }

    void check_live(bool worker) const {
        TORCH_CHECK(!destroyed, "DirectionalM2NBuffer was destroyed");
        TORCH_CHECK((rank < layout.workers) == worker,
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
        if (layout.nvlink) {
            const bool worker = rank < layout.workers;
            const int peers = worker ? layout.sources : layout.workers;
            for (int i = 0; i < peers; ++i)
                result.peers[i] = static_cast<uint8_t *>(
                    context->nvl_window_ptrs[worker ? layout.workers + i : i]);
        } else if (rank < 4) {
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
        if (layout.nvlink && blocks > 1) {
            // Bulk row kernels need more resident warps to hide memory latency.
            // Bound every grid by its fixed shape, including empty exchanges;
            // live counts remain entirely device-side during graph replay.
            size_t work = blocks;
            if (operation == TmaPackRows || operation == TmaPackRowsWide) {
                const int warps_per_block =
                    (operation == TmaPackRowsWide ? wide_tma_threads : kNVLinkTmaThreads) / kWarpSize;
                work = (size_t(layout.sources) * arguments.bucket + warps_per_block - 1) / warps_per_block;
            } else if (operation == WarpPartials || operation == WarpCollect ||
                       operation == WarpPartialsWide || operation == WarpCollectWide) {
                constexpr int warps_per_block = kThreadsPerBlock / kWarpSize;
                const bool partials = operation == WarpPartials || operation == WarpPartialsWide;
                const size_t tokens = size_t(arguments.bucket) * (partials ? layout.sources : 1);
                work = (tokens + warps_per_block - 1) / warps_per_block;
                blocks *= kNVLinkWarpGridMultiplier;
            } else if (operation == PackRows || operation == MakePartials) {
                work = size_t(layout.sources) * arguments.bucket;
                blocks *= kNVLinkBulkGridMultiplier;
            } else if (operation == CollectReply) {
                work = arguments.bucket;
            } else if (operation == PackSource || operation == PackSourceCounts) {
                constexpr int bytes_per_block = kThreadsPerBlock * kPayloadVectorBytes;
                work = (layout.packet_bytes(arguments.tokens) + bytes_per_block - 1) / bytes_per_block;
            } else if (operation == Histogram) {
                work = (size_t(layout.sources) * arguments.bucket * layout.topk + kThreadsPerBlock - 1) / kThreadsPerBlock;
            }
            blocks = std::min<size_t>(blocks, std::max<size_t>(work, 1));
        }
        const auto stream = at::cuda::getCurrentCUDAStream();
        AFDEP_CUDA_CHECK(launch_kernel(operation, arguments, blocks, stream.stream(), wide_tma_threads));
    }

    int receive_row_capacity(int bucket) const {
        return layout.sources * bucket * std::min(layout.topk, layout.local_experts()) +
               layout.local_experts() * (layout.alignment - 1);
    }

    bool nvlink_bulk_shape(int tokens) const {
        return layout.nvlink && (layout.hidden == kNVLinkWarpHidden || layout.hidden == kNVLinkWideHidden) &&
               layout.topk == kNVLinkWarpTopK &&
               tokens >= kNVLinkWarpMinTokens;
    }

    DeviceArgs discover_receive(int bucket, const torch::Tensor &psum,
                               float* weights_to_clear = nullptr) {
        TORCH_CHECK(!layout.nvlink || !nvlink_pair_pending, "NVLink receive requires the previous reply");
        nvlink_pair_pending = layout.nvlink;
        auto a = args(bucket);
        a.counts = counts.data_ptr<int>();
        a.starts = starts.data_ptr<int>();
        a.cursors = cursors.data_ptr<int>();
        a.psum = psum.data_ptr<int>();
        a.recv_weights = weights_to_clear;
        if (!layout.nvlink) counts.zero_();
        launch(BeginWorker, a);
        if (!layout.nvlink) {
            launch(Histogram, a, num_sms);
            launch(Prefix, a);
        }
        return a;
    }

    std::vector<int> host_receive_counts(int bucket) const {
        const auto cpu_counts = counts.to(torch::kCPU);
        const auto *data = cpu_counts.data_ptr<int>();
        std::vector<int> result;
        int64_t assignments = 0, rows = 0;
        for (int e = 0; e < layout.local_experts(); ++e) {
            TORCH_CHECK(data[e] >= 0 && int64_t(data[e]) <= int64_t(layout.sources) * bucket,
                        "Received expert count exceeds the source-token bound");
            const int64_t aligned = (int64_t(data[e]) + layout.alignment - 1) /
                                    layout.alignment * layout.alignment;
            assignments += data[e];
            rows += aligned;
            TORCH_CHECK(rows <= receive_row_capacity(bucket),
                        "Received aligned counts exceed the receive capacity");
            result.push_back(static_cast<int>(aligned));
        }
        TORCH_CHECK(assignments <= int64_t(layout.sources) * bucket *
                                       std::min(layout.topk, layout.local_experts()),
                    "Received assignments exceed the routing bound");
        return result;
    }

    void pack_receive(DeviceArgs a, int rows, const torch::Tensor &recv_x,
                      const std::optional<torch::Tensor> &recv_scale,
                      const torch::Tensor &recv_weights, const torch::Tensor &receipt,
                      bool weights_cleared = false) {
        worker_bucket = a.bucket;
        worker_rows = rows;
        a.recv_x = static_cast<uint8_t *>(recv_x.data_ptr());
        a.recv_weights = recv_weights.data_ptr<float>();
        a.receipt = receipt.data_ptr<int>();
        if (recv_scale.has_value()) {
            a.recv_scales = recv_scale->data_ptr<float>();
            a.recv_scale_stride1 = recv_scale->stride(1);
        }
        if (!weights_cleared) recv_weights.zero_();
        if (!layout.nvlink) receipt.fill_(-1);
        const int bulk_pack = layout.hidden == kNVLinkWideHidden ? TmaPackRowsWide : TmaPackRows;
        launch(nvlink_bulk_shape(a.bucket) ? bulk_pack : PackRows, a, num_sms);
        if (!layout.nvlink) launch(PublishPackAndCredit, a);
    }

  public:
    const int64_t num_buffer_bytes;

    DirectionalM2NBuffer(int64_t nccl_comm, int rank, int world, int ffn_size, int attn_size, int capacity,
                         int hidden, int topk, int real_experts, int expert_alignment, bool use_fp8,
                         int num_qps, int gpu_timeout_s, int traffic_class, int transport_stripes = 8,
                         int num_sms = 132, bool allow_hybrid_mode = true, bool nvlink_only = false)
        : layout(checked_layout(capacity, hidden, topk, real_experts, expert_alignment, use_fp8,
                                allow_hybrid_mode, ffn_size, attn_size, nvlink_only)), rank(rank),
          stripes(transport_stripes), num_sms(num_sms),
          num_buffer_bytes(rank < ffn_size ? layout.worker_bytes : layout.source_bytes) {
        TORCH_CHECK(world == ffn_size + attn_size && rank >= 0 && rank < world,
                    "Directional world must contain FFN ranks followed by Attention ranks");
        TORCH_CHECK(expert_alignment > 0 && gpu_timeout_s > 0,
                    "Directional alignment and timeout settings must be positive");
        TORCH_CHECK(num_qps > 0 && stripes > 0 && stripes <= num_qps,
                    "Directional stripes must be in [1,num_qps]");
        AFDEP_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        AFDEP_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        TORCH_CHECK(num_sms > 0 && num_sms <= properties.multiProcessorCount,
                    "Directional num_sms exceeds the device SM count");
        if (nvlink_bulk_shape(layout.capacity)) {
            int wide_max_shared = 0;
            if (layout.hidden == kNVLinkWideHidden) {
                cudaFuncAttributes attributes{};
                AFDEP_CUDA_CHECK(get_wide_tma_attributes(&attributes));
                TORCH_CHECK(properties.sharedMemPerBlockOptin > attributes.sharedSizeBytes,
                            "Wide TMA static storage exceeds device shared memory");
                const size_t shared_budget = properties.sharedMemPerBlockOptin - attributes.sharedSizeBytes;
                const int max_threads = std::min(properties.maxThreadsPerBlock, attributes.maxThreadsPerBlock);
                const int max_warps = std::min(max_threads / kWarpSize, kNVLinkTmaMaxWarps);
                const auto staging_bytes = [&](int element_bytes) {
                    return layout.hidden * element_bytes + int(sizeof(uint64_t));
                };
                const int bf16_warps = std::min<int>(shared_budget / staging_bytes(2), max_warps);
                const int fp8_warps = std::min<int>(shared_budget / staging_bytes(1), max_warps);
                TORCH_CHECK(bf16_warps > 0 && fp8_warps > 0, "Wide TMA staging exceeds device shared memory");
                wide_tma_threads = (layout.fp8 ? fp8_warps : bf16_warps) * kWarpSize;
                // Both payload constructors retain the same per-function limit.
                wide_max_shared = std::max(bf16_warps * staging_bytes(2), fp8_warps * staging_bytes(1));
            }
            AFDEP_CUDA_CHECK(configure_tma_kernel(layout.hidden, wide_max_shared));
        }
        const auto physical = get_physical_domain_size(nccl_comm);
        TORCH_CHECK(nvlink_only ? (std::get<0>(physical) == 1 && std::get<1>(physical) == world) :
                        (std::get<0>(physical) == 2 && std::get<1>(physical) == 4),
                    "Directional buffer topology does not match its transport");
        TORCH_CHECK(properties.major == 9, "The current directional CUDA backend requires SM90");
        int clock_rate_khz;
        AFDEP_CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device));
        timeout_cycles = uint64_t(gpu_timeout_s) * uint64_t(clock_rate_khz) * 1000;

        context = std::make_shared<NcclContext>(nccl_comm, world, rank, size_t(num_buffer_bytes), 128,
                                                traffic_class, num_qps, allow_hybrid_mode, nvlink_only);
        try {
            TORCH_CHECK(nvlink_only || stripes <= context->dev_comm.ginContextCount,
                        "NCCL provided fewer GIN contexts than requested transport stripes");
            TORCH_CHECK(nvlink_only || (context->num_nvl_ranks == 4 && context->num_rdma_ranks == 2 &&
                            context->rdma_rank_idx == (rank < 4 ? 0 : 1) && context->nvl_rank_idx == rank % 4),
                        "Unsupported directional placement: workers must fill node0, sources node1");
            AFDEP_CUDA_CHECK(cudaMemsetAsync(context->mapped_window_ptr, 0,
                                             nvlink_only ? sizeof(NVLinkControl) : sizeof(Control),
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
                         const torch::Tensor &ids, const torch::Tensor &weights, int bucket,
                         const std::optional<torch::Tensor> &live_tokens = std::nullopt) {
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
        if (live_tokens.has_value()) {
            check_tensor(*live_tokens, torch::kInt32, 1);
            TORCH_CHECK(live_tokens->numel() == 1,
                        "Live source token count must have shape [1]");
            a.live_tokens = live_tokens->data_ptr<int>();
        }
        if (scale.has_value()) {
            check_tensor(*scale, torch::kFloat32, 2, false);
            TORCH_CHECK(scale->size(0) == x.size(0) && scale->size(1) == layout.hidden / 128,
                        "Expected FP8 scale shape [tokens,hidden/128]");
            a.scales = scale->data_ptr<float>();
            a.scale_stride0 = scale->stride(0);
            a.scale_stride1 = scale->stride(1);
        }
        TORCH_CHECK(!layout.nvlink || !nvlink_pair_pending, "NVLink dispatch requires the previous collect");
        nvlink_pair_pending = layout.nvlink;
        launch(BeginSource, a);
        if (!layout.nvlink && !layout.hybrid)
            launch(PlanDirectRequests, a, num_sms);
        launch(layout.nvlink && bucket > kNVLinkFusedMetadataTokens ? PackSourceCounts : PackSource, a, num_sms);
        launch(SendRequest, a, layout.nvlink ? 1 : stripes * (layout.hybrid ? 1 : 4));
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::vector<int>,
               torch::Tensor>
    worker_receive(int bucket, bool read_host_counts, bool exact_rows = false) {
        check_live(true);
        TORCH_CHECK(bucket > 0 && bucket <= layout.capacity, "Directional worker bucket exceeds capacity");
        TORCH_CHECK(!exact_rows || read_host_counts,
                    "Exact receive allocation requires host receive counts");
        if (exact_rows) {
            cudaStreamCaptureStatus capture;
            AFDEP_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(), &capture));
            TORCH_CHECK(capture == cudaStreamCaptureStatusNone,
                        "Exact receive allocation is unavailable during CUDA capture");
        }
        const auto options = torch::TensorOptions().device(torch::kCUDA, device);
        auto psum = torch::empty({layout.local_experts()}, options.dtype(torch::kInt32));
        auto rows = receive_row_capacity(bucket);
        DeviceArgs discovered{};
        std::vector<int> host_counts;
        if (exact_rows) {
            // Sources retain request storage until packing publishes credits.
            // Only count metadata is allocated before this host decision.
            discovered = discover_receive(bucket, psum);
            host_counts = host_receive_counts(bucket);
            rows = 0;
            for (const int count : host_counts)
                rows += count;
        }
        auto recv_x = torch::empty({rows, layout.hidden},
                                   options.dtype(layout.fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
        auto recv_weights = torch::empty({rows}, options.dtype(torch::kFloat32));
        auto receipt = torch::empty({layout.sources, bucket, layout.topk}, options.dtype(torch::kInt32));
        std::optional<torch::Tensor> recv_scale;
        if (layout.fp8) {
            const int scale_stride = std::max(4, (rows + 3) / 4 * 4);
            recv_scale = torch::empty_strided({rows, layout.scale_groups()}, {1, scale_stride},
                                              options.dtype(torch::kFloat32));
        }
        if (!exact_rows)
            return worker_receive_into(bucket, read_host_counts, recv_x, recv_scale, recv_weights, psum, receipt);
        pack_receive(discovered, rows, recv_x, recv_scale, recv_weights, receipt);
        return {recv_x, recv_scale, recv_weights, psum, host_counts, receipt};
    }

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::vector<int>,
               torch::Tensor>
    worker_receive_into(int bucket, bool read_host_counts, const torch::Tensor &recv_x,
                        const std::optional<torch::Tensor> &recv_scale,
                        const torch::Tensor &recv_weights, const torch::Tensor &psum,
                        const torch::Tensor &receipt) {
        check_live(true);
        TORCH_CHECK(bucket > 0 && bucket <= layout.capacity, "Directional worker bucket exceeds capacity");
        const int rows = receive_row_capacity(bucket);
        check_tensor(recv_x, layout.fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16, 2);
        check_vector_alignment(recv_x);
        check_tensor(recv_weights, torch::kFloat32, 1);
        check_tensor(psum, torch::kInt32, 1);
        check_tensor(receipt, torch::kInt32, 3);
        TORCH_CHECK(recv_x.size(0) == rows && recv_x.size(1) == layout.hidden &&
                        recv_weights.size(0) == rows && psum.size(0) == layout.local_experts() &&
                        receipt.size(0) == layout.sources && receipt.size(1) == bucket && receipt.size(2) == layout.topk,
                    "Reusable receive tensors must match their bucket and buffer configuration");
        TORCH_CHECK(recv_scale.has_value() == bool(layout.fp8), "Reusable receive scales mismatch payload");
        if (recv_scale.has_value()) {
            check_tensor(*recv_scale, torch::kFloat32, 2, false);
            TORCH_CHECK(recv_scale->size(0) == rows && recv_scale->size(1) == layout.scale_groups() &&
                            recv_scale->stride(0) == 1 && recv_scale->stride(1) >= rows &&
                            recv_scale->stride(1) % 4 == 0,
                        "Reusable FP8 scales require padded column-major storage");
        }
        const bool fuse_weight_clear = layout.nvlink && bucket <= kNVLinkFusedMetadataTokens;
        auto a = discover_receive(bucket, psum, fuse_weight_clear ? recv_weights.data_ptr<float>() : nullptr);
        pack_receive(a, rows, recv_x, recv_scale, recv_weights, receipt, fuse_weight_clear);
        std::vector<int> host_counts;
        if (read_host_counts)
            host_counts = host_receive_counts(bucket);
        return {recv_x, recv_scale, recv_weights, psum, host_counts, receipt};
    }

    void worker_reply(const torch::Tensor &rows, const torch::Tensor &receipt) {
        check_live(true);
        check_tensor(rows, torch::kBFloat16, 2);
        check_tensor(receipt, torch::kInt32, 3);
        TORCH_CHECK(worker_bucket > 0 && rows.size(0) == worker_rows && rows.size(1) == layout.hidden &&
                        receipt.size(0) == layout.sources && receipt.size(1) == worker_bucket &&
                        receipt.size(2) == layout.topk,
                    "Directional reply must use its matching receive receipt/row shape");
        TORCH_CHECK(!layout.nvlink || nvlink_pair_pending, "NVLink reply requires a receive");
        auto a = args(worker_bucket);
        a.rows = static_cast<const uint16_t *>(rows.data_ptr());
        a.receipt = receipt.data_ptr<int>();
        const int bulk_partials = layout.hidden == kNVLinkWideHidden ? WarpPartialsWide : WarpPartials;
        launch(nvlink_bulk_shape(worker_bucket) ? bulk_partials : MakePartials, a, num_sms);
        if (layout.hybrid && !layout.nvlink) {
            launch(PublishPartialAndWait, a);
            launch(ReduceReply, a, num_sms);
        }
        if (!layout.nvlink) launch(PublishConsumedAndWaitCredit, a);
        launch(SendReply, a, layout.nvlink ? 1 : stripes * (layout.hybrid ? 1 : 4));
        nvlink_pair_pending = false;
    }

    void source_collect(const torch::Tensor &out) {
        check_live(false);
        check_tensor(out, torch::kBFloat16, 2);
        check_vector_alignment(out);
        TORCH_CHECK(out.size(0) == source_rows && out.size(1) == layout.hidden,
                    "Directional output must match the source dispatch shape");
        TORCH_CHECK(!layout.nvlink || nvlink_pair_pending, "NVLink collect requires a dispatch");
        auto a = args(std::max(source_rows, 1));
        a.out = static_cast<uint16_t *>(out.data_ptr());
        if (nvlink_bulk_shape(source_rows)) launch(WaitReply, a);
        const int bulk_collect = layout.hidden == kNVLinkWideHidden ? WarpCollectWide : WarpCollect;
        launch(nvlink_bulk_shape(source_rows) ? bulk_collect : CollectReply, a, num_sms);
        if (!layout.nvlink) launch(ReplyConsumedCredit, a);
        nvlink_pair_pending = false;
    }

    void drain() {
        check_live(rank < layout.workers);
        if (!layout.nvlink) launch(Drain, args(1), stripes);
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
        .def(py::init<int64_t, int, int, int, int, int, int, int, int, int, bool, int, int, int, int, int, bool, bool>(),
             py::arg("nccl_comm"), py::arg("rank"), py::arg("world"), py::arg("ffn_size"),
             py::arg("attn_size"), py::arg("capacity"), py::arg("hidden"), py::arg("top_k"),
             py::arg("real_experts"), py::arg("expert_alignment"), py::arg("use_fp8"), py::arg("num_qps"),
             py::arg("gpu_timeout_s"), py::arg("traffic_class"), py::arg("transport_stripes") = 8,
             py::arg("num_sms") = 132, py::arg("allow_hybrid_mode") = true,
             py::arg("nvlink_only") = false)
        .def_readonly("num_buffer_bytes", &DirectionalM2NBuffer::num_buffer_bytes)
        .def("registered_bytes", &DirectionalM2NBuffer::registered_bytes)
        .def("source_dispatch", &DirectionalM2NBuffer::source_dispatch, py::arg("x"),
             py::arg("scale"), py::arg("ids"), py::arg("weights"), py::arg("bucket"),
             py::arg("live_tokens") = py::none())
        .def("worker_receive", &DirectionalM2NBuffer::worker_receive, py::arg("bucket"),
             py::arg("read_host_counts"), py::arg("exact_rows") = false)
        .def("worker_receive_into", &DirectionalM2NBuffer::worker_receive_into)
        .def("worker_reply", &DirectionalM2NBuffer::worker_reply)
        .def("source_collect", &DirectionalM2NBuffer::source_collect)
        .def("drain", &DirectionalM2NBuffer::drain)
        .def("destroy", &DirectionalM2NBuffer::destroy);
}

} // namespace afdep
