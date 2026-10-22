#pragma once

#include <nccl.h>
#include <string>
#include <utility>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"
#include "afdep_suites.hpp"

namespace deep_ep::elastic {

class CombineRuntime final : public jit::LaunchRuntime<CombineRuntime> {
public:
    struct Args {
        // Templated arguments
        bool is_scaleup_nvlink;
        bool use_expanded_layout, allow_multiple_reduction;
        int num_scaleup_warps, num_forward_warps;
        int num_scaleout_ranks, num_scaleup_ranks;
        int hidden;
        int num_max_tokens_per_rank;
        int num_experts;
        int num_topk;
        int num_qps;
        int64_t num_timeout_cycles;
        // afdEP: false/zero keep the upstream instantiation
        bool afd_skip_entry_barrier;
        int afd_directed_ffn_ranks;
        bool afd_per_cta_exit;
        bool afd_single_ffn_reduce;
        int afd_ffn_scaleout_ranks;
        // afdEP mixed-role hybrid only (afdep_mixed_hybrid_combine.cuh): F, or 0
        int afd_mixed_ffn_ranks;

        // Parameters
        nv_bfloat16* x;
        float* topk_weights;
        int* src_metadata;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* token_metadata_at_forward;
        int* channel_linked_list;
        ncclDevComm_t nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace;
        int scaleout_rank_idx, scaleup_rank_idx;
        int num_reduced_tokens;
        void* afd_scaleout_send_base;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        std::string header_name, func_name;
        if (args.num_scaleout_ranks == 1) {
            header_name = "combine";
            const auto afdep_args = args.afd_skip_entry_barrier or args.afd_directed_ffn_ranks > 0 ?
                fmt::format(", {}, {}, {}{}", args.afd_skip_entry_barrier, args.afd_directed_ffn_ranks,
                            args.afd_per_cta_exit, args.afd_single_ffn_reduce ? ", true" : "") : std::string();
            func_name = fmt::format("combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}{}>",
                                    args.is_scaleup_nvlink,
                                    args.use_expanded_layout, args.allow_multiple_reduction,
                                    args.launch_args.grid_dim.first,
                                    args.launch_args.num_threads / 32,
                                    args.num_scaleup_ranks * args.num_scaleout_ranks,
                                    args.hidden,
                                    args.num_max_tokens_per_rank,
                                    args.num_experts,
                                    args.num_topk,
                                    args.num_qps, args.num_timeout_cycles,
                                    afdep_args);
        } else {
            header_name = "hybrid_combine";
            func_name = fmt::format("hybrid_combine_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}{}>",
                                    args.use_expanded_layout, args.allow_multiple_reduction,
                                    args.launch_args.grid_dim.first,
                                    args.num_scaleup_warps, args.num_forward_warps,
                                    args.num_scaleout_ranks, args.num_scaleup_ranks,
                                    args.hidden,
                                    args.num_max_tokens_per_rank,
                                    args.num_experts,
                                    args.num_topk,
                                    args.num_qps,
                                    args.num_timeout_cycles,
                                    args.afd_ffn_scaleout_ranks > 0 ?
                                        fmt::format(", {}", args.afd_ffn_scaleout_ranks) : std::string());
        }
        return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        if (args.num_scaleout_ranks == 1) {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,
                                                     args.src_metadata, args.psum_num_recv_tokens_per_scaleup_rank,
                                                     args.nccl_dev_comm, args.nccl_window,
                                                     args.buffer, args.workspace,
                                                     args.scaleup_rank_idx,
                                                     args.num_reduced_tokens));
        } else {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                     args.x, args.topk_weights,
                                                     args.src_metadata,
                                                     args.psum_num_recv_tokens_per_scaleup_rank,
                                                     args.token_metadata_at_forward,
                                                     args.channel_linked_list,
                                                     args.nccl_dev_comm, args.nccl_window,
                                                     args.buffer, args.workspace,
                                                     args.scaleout_rank_idx, args.scaleup_rank_idx,
                                                     args.num_reduced_tokens,
                                                     args.afd_scaleout_send_base));
        }
    }
};

// afdEP Attention directional hybrid combine receive (afdep_suites.hpp).
class AfdepHybridCollectRuntime final : public jit::LaunchRuntime<AfdepHybridCollectRuntime> {
public:
    struct Args {
        int num_scaleout_ranks, num_scaleup_ranks;
        int num_experts, num_channels;
        int afd_ffn_scaleout_ranks;
        int64_t num_timeout_cycles;

        void* workspace;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_hybrid_collect.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_hybrid_collect_impl<{}, {}, {}, {}, {}, {}, {}>);
}}
)",
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.num_experts, args.num_channels,
                           args.afd_ffn_scaleout_ranks, args.num_timeout_cycles,
                           args.launch_args.num_threads);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.workspace, args.scaleout_rank_idx, args.scaleup_rank_idx));
    }
};

// afdEP: a runtime's generated code re-pointed at a generated copy of its kernel (header and
// entry point renamed, template arguments unchanged).
static std::string afdep_repoint_kernel(std::string code, const std::string& from, const std::string& to) {
    for (const auto& [old_text, new_text] : {
             std::pair<std::string, std::string>{"<deep_ep/impls/" + from + ".cuh>", "<deep_ep/impls/" + to + ".cuh>"},
             std::pair<std::string, std::string>{"&" + from + "_impl<", "&" + to + "_impl<"}}) {
        const auto pos = code.find(old_text);
        EP_HOST_ASSERT(pos != std::string::npos and code.find(old_text, pos + 1) == std::string::npos);
        code.replace(pos, old_text.size(), new_text);
    }
    return code;
}

// afdEP sparse combine, FFN (afdep_suites.hpp): the shared hybrid combine's arguments and shape
// through afdep_ffn_sparse_hybrid_combine.cuh, generated by tests/role_kernels.py.
class AfdepFfnSparseHybridCombineRuntime final : public jit::LaunchRuntime<AfdepFfnSparseHybridCombineRuntime> {
public:
    using Args = CombineRuntime::Args;

    static std::string generate_impl(const Args& args) {
        EP_HOST_ASSERT(args.num_scaleout_ranks > 1 and args.afd_ffn_scaleout_ranks > 0);
        return afdep_repoint_kernel(CombineRuntime::generate_impl(args), "hybrid_combine",
                                    "afdep_ffn_sparse_hybrid_combine");
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        CombineRuntime::launch_impl(kernel, config, args);
    }
};

// afdEP sparse combine, Attention collect (afdep_suites.hpp): the collect's arguments and shape
// through afdep_sparse_hybrid_collect.cuh, generated by tests/role_kernels.py.
class AfdepSparseHybridCollectRuntime final : public jit::LaunchRuntime<AfdepSparseHybridCollectRuntime> {
public:
    using Args = AfdepHybridCollectRuntime::Args;

    static std::string generate_impl(const Args& args) {
        return afdep_repoint_kernel(AfdepHybridCollectRuntime::generate_impl(args), "afdep_hybrid_collect",
                                    "afdep_sparse_hybrid_collect");
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        AfdepHybridCollectRuntime::launch_impl(kernel, config, args);
    }
};

// afdEP mixed-role hybrid (afdep_roles.hpp): the shared hybrid combine's arguments and shape, with F
// appended, through afdep_mixed_hybrid_combine.cuh, generated by tests/role_kernels.py.
class AfdepMixedHybridCombineRuntime final : public jit::LaunchRuntime<AfdepMixedHybridCombineRuntime> {
public:
    using Args = CombineRuntime::Args;

    static std::string generate_impl(const Args& args) {
        EP_HOST_ASSERT(args.num_scaleout_ranks > 1 and args.afd_ffn_scaleout_ranks == 1 and args.afd_mixed_ffn_ranks > 0);
        auto code = afdep_repoint_kernel(CombineRuntime::generate_impl(args), "hybrid_combine",
                                         "afdep_mixed_hybrid_combine");
        const std::string end = ">);";
        const auto pos = code.find(end);
        EP_HOST_ASSERT(pos != std::string::npos and code.find(end, pos + 1) == std::string::npos);
        code.insert(pos, fmt::format(", {}", args.afd_mixed_ffn_ranks));
        return code;
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        CombineRuntime::launch_impl(kernel, config, args);
    }
};

// afdEP mixed-role hybrid: every rank's reply inbox, just before the staging slice (calculate_buffer_size).
static void* afdep_mixed_reply_buffer(void* afd_scaleout_send_base, const layout::TokenLayout& token_layout,
                                      const int& num_scaleout_ranks, const int& num_max_tokens_per_rank,
                                      const int& num_topk, const bool& allow_multiple_reduction) {
    EP_HOST_ASSERT(afd_scaleout_send_base != nullptr);
    const auto reply = layout::BufferLayout<false>(
        token_layout, allow_multiple_reduction and num_scaleout_ranks <= num_topk ? num_scaleout_ranks : num_topk,
        num_max_tokens_per_rank);
    return math::advance_ptr(afd_scaleout_send_base, -reply.get_num_bytes());
}

static layout::TokenLayout get_combine_token_layout(
    const int& hidden, const int& elem_size, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, 0, num_topk, false);
}

static void* get_hybrid_reduce_buffer(void* buffer, const layout::TokenLayout& token_layout,
                                      const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                      const int& num_max_tokens_per_rank, const int& num_topk,
                                      const bool& allow_multiple_reduction) {
    // For hybrid mode, we have to skip the scale-up buffer
    const bool is_scaleup_buffer_rank_layout =
        allow_multiple_reduction ? (num_scaleup_ranks <= num_topk) : false;
    const auto scaleup_buffer = layout::BufferLayout<false>(
        token_layout, 
        is_scaleup_buffer_rank_layout ? num_scaleup_ranks : num_topk,
        num_scaleout_ranks * num_max_tokens_per_rank,
        buffer);
    return scaleup_buffer.get_buffer_end_ptr();
}

static void* launch_combine(void* x,
                            void* topk_weights,
                            int* src_metadata,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* token_metadata_at_forward,
                            int* channel_linked_list,
                            const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
                            void* buffer, void* workspace,
                            const int& num_reduced_tokens, const int& num_max_tokens_per_rank,
                            const int& hidden,
                            const int& num_experts, const int& num_topk,
                            const int& num_qps, const int64_t& num_timeout_cycles,
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                            const bool& is_scaleup_nvlink,
                            const int& num_sms, const int& num_smem_bytes,
                            const int& num_channels,
                            const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                            const at::cuda::CUDAStream& stream,
                            const bool& afd_skip_entry_barrier = false,
                            const int& afd_directed_ffn_ranks = 0,
                            const bool& afd_per_cta_exit = false,
                            const int& afd_ffn_scaleout_ranks = 0,
                            const bool& afd_single_ffn_reduce = false,
                            void* afd_scaleout_send_base = nullptr,
                            const AfdLaunchShape afd_shape = AfdLaunchShape::kShared,
                            const int& afd_mixed_ffn_ranks = 0) {
    // Maximize shared memory utilization
    const auto token_layout = get_combine_token_layout(hidden, sizeof(nv_bfloat16), num_topk);
    auto num_warps = std::min(num_smem_bytes / token_layout.get_num_bytes<true>(), 32);

    // Decide warps
    int num_scaleup_warps = 0, num_forward_warps = 0;
    if (num_scaleout_ranks > 1) {
        EP_HOST_ASSERT(num_channels % num_sms == 0 and
                       "Invalid number of channels or SMs, you may use a different SM count than dispatch");
        EP_HOST_ASSERT(num_channels / num_sms <= 16);

        num_scaleup_warps = num_forward_warps = num_channels / num_sms;
        num_warps = num_scaleup_warps + num_forward_warps;
        EP_HOST_ASSERT(num_warps * token_layout.get_num_bytes<true>() <= num_smem_bytes and
                       "Invalid combine SM count, please try to match your dispatch config");
    }

    // Generate, build and launch
    const auto num_threads = num_warps * 32;
    if (afd_shape == AfdLaunchShape::kRoleKernel) {
        // afdEP Attention directional hybrid combine (afdep_suites.hpp): one CTA, a thread per
        // (channel, FFN domain) signal while they fit.
        EP_HOST_ASSERT(num_scaleout_ranks > 1 and afd_ffn_scaleout_ranks > 0 and
                       scaleout_rank_idx >= afd_ffn_scaleout_ranks);
        const AfdepHybridCollectRuntime::Args collect_args = {
            .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
            .num_experts = num_experts, .num_channels = num_channels,
            .afd_ffn_scaleout_ranks = afd_ffn_scaleout_ranks,
            .num_timeout_cycles = num_timeout_cycles,
            .workspace = workspace,
            .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
            .launch_args = jit::LaunchArgs(
                1, std::min(1024, math::align(num_channels * afd_ffn_scaleout_ranks, 32)), num_smem_bytes, 1, false)};
        if (afd_mixed_ffn_ranks == 0 and afdep_hybrid_sparse_combine()) {
            const auto sparse_code = AfdepSparseHybridCollectRuntime::generate(collect_args);
            const auto sparse_runtime = jit::compiler->build("afdep_sparse_hybrid_collect", sparse_code);
            AfdepSparseHybridCollectRuntime::launch(sparse_runtime, collect_args, stream);
        } else {
            const auto code = AfdepHybridCollectRuntime::generate(collect_args);
            const auto runtime = jit::compiler->build("afdep_hybrid_collect", code);
            AfdepHybridCollectRuntime::launch(runtime, collect_args, stream);
        }
        if (afd_mixed_ffn_ranks > 0)
            return afdep_mixed_reply_buffer(afd_scaleout_send_base, token_layout, num_scaleout_ranks,
                                            num_max_tokens_per_rank, num_topk, allow_multiple_reduction);
        return get_hybrid_reduce_buffer(buffer, token_layout, num_scaleout_ranks, num_scaleup_ranks,
                                        num_max_tokens_per_rank, num_topk, allow_multiple_reduction);
    }
    const CombineRuntime::Args args = {
        .is_scaleup_nvlink = is_scaleup_nvlink,
        .use_expanded_layout = use_expanded_layout,
        .allow_multiple_reduction = allow_multiple_reduction,
        .num_scaleup_warps = num_scaleup_warps, .num_forward_warps = num_forward_warps,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts,
        .num_topk = num_topk,
        .num_qps = num_qps, .num_timeout_cycles = num_timeout_cycles,
        .afd_skip_entry_barrier = num_scaleout_ranks == 1 and afd_skip_entry_barrier,
        .afd_directed_ffn_ranks = num_scaleout_ranks == 1 and is_scaleup_nvlink ? afd_directed_ffn_ranks : 0,
        .afd_per_cta_exit = num_scaleout_ranks == 1 and is_scaleup_nvlink and afd_directed_ffn_ranks > 0 and
                            afd_per_cta_exit,
        .afd_single_ffn_reduce = num_scaleout_ranks == 1 and (afd_skip_entry_barrier or afd_directed_ffn_ranks > 0) and
                                 afd_single_ffn_reduce,
        .afd_ffn_scaleout_ranks = num_scaleout_ranks > 1 ? afd_ffn_scaleout_ranks : 0,
        .afd_mixed_ffn_ranks = num_scaleout_ranks > 1 and afd_ffn_scaleout_ranks == 1 ? afd_mixed_ffn_ranks : 0,
        .x = static_cast<nv_bfloat16*>(x),
        .topk_weights = static_cast<float*>(topk_weights),
        .src_metadata = src_metadata,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .token_metadata_at_forward = token_metadata_at_forward,
        .channel_linked_list = channel_linked_list,
        .nccl_dev_comm = nccl_dev_comm, .nccl_window = nccl_window,
        .buffer = buffer, .workspace = workspace,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .num_reduced_tokens = num_reduced_tokens,
        .afd_scaleout_send_base = afd_scaleout_send_base,
        // NOTES: make cluster dim 2 to overlap with clustered computation kernels
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 2 - (num_sms % 2), true)
    };
    if (args.afd_mixed_ffn_ranks > 0) {
        // afdEP mixed-role hybrid (afdep_roles.hpp): every rank that runs the main combine (domain 0,
        // and pure-Attention domains under AFDEP_SHARED_HYBRID_COLLECT=1) uses the generated copy.
        const auto mixed_code = AfdepMixedHybridCombineRuntime::generate(args);
        const auto mixed_runtime = jit::compiler->build("afdep_mixed_hybrid_combine", mixed_code);
        AfdepMixedHybridCombineRuntime::launch(mixed_runtime, args, stream);
        return afdep_mixed_reply_buffer(afd_scaleout_send_base, token_layout, num_scaleout_ranks,
                                        num_max_tokens_per_rank, num_topk, allow_multiple_reduction);
    }
    if (args.afd_ffn_scaleout_ranks > 0 and scaleout_rank_idx < args.afd_ffn_scaleout_ranks and
        afdep_hybrid_sparse_combine()) {
        // afdEP sparse combine, FFN (afdep_suites.hpp): the shared shape through the generated copy.
        const auto sparse_code = AfdepFfnSparseHybridCombineRuntime::generate(args);
        const auto sparse_runtime = jit::compiler->build("afdep_ffn_sparse_hybrid_combine", sparse_code);
        AfdepFfnSparseHybridCombineRuntime::launch(sparse_runtime, args, stream);
    } else {
        const auto code = CombineRuntime::generate(args);
        const auto runtime = jit::compiler->build("combine", code);
        CombineRuntime::launch(runtime, args, stream);
    }

    // Return the buffer to be reduced
    if (num_scaleout_ranks == 1)
        return buffer;
    return get_hybrid_reduce_buffer(buffer, token_layout, num_scaleout_ranks, num_scaleup_ranks,
                                    num_max_tokens_per_rank, num_topk, allow_multiple_reduction);
}

class CombineReduceEpilogueRuntime final : public jit::LaunchRuntime<CombineReduceEpilogueRuntime> {
public:
    struct Args {
        // Templated arguments
        bool use_expanded_layout, allow_multiple_reduction;
        int num_scaleout_ranks, num_scaleup_ranks;
        int hidden;
        int num_max_tokens_per_rank;
        int num_experts, num_topk;

        // Parameters
        nv_bfloat16* combined_x;
        float* combined_topk_weights;
        topk_idx_t* combined_topk_idx;
        void* reduce_buffer;
        void* bias_0;
        void* bias_1;
        int num_combined_tokens;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/combine_reduce_epilogue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_reduce_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",                        args.use_expanded_layout, args.allow_multiple_reduction,
                           args.launch_args.grid_dim.first,
                           args.launch_args.num_threads / 32,
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.hidden,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.combined_x,
                                                 args.combined_topk_weights,
                                                 args.combined_topk_idx,
                                                 args.reduce_buffer,
                                                 args.bias_0, args.bias_1,
                                                 args.num_combined_tokens,
                                                 args.scaleout_rank_idx, args.scaleup_rank_idx));
    }
};

static void launch_combine_reduce_epilogue(void* combined_x,
                                           float* combined_topk_weights,
                                           topk_idx_t* combined_topk_idx,
                                           const int& num_combined_tokens, const int& num_max_tokens_per_rank,
                                           const int& hidden,
                                           const int& num_experts, const int& num_topk,
                                           void* reduce_buffer,
                                           void* bias_0, void* bias_1,
                                           const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                           const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                           const int& num_sms, const int& num_smem_bytes,
                                           const bool& use_expanded_layout, const bool& allow_multiple_reduction,
                                           const at::cuda::CUDAStream& stream) {
    // Maximize shared memory utilization
    // Too many warps may cause performance degrade, so we limit into 1024
    const auto token_layout = layout::TokenLayout(hidden * sizeof(nv_bfloat16), 0, 0, false);
    const auto num_warps = std::min<int>(num_smem_bytes / token_layout.get_num_bytes<false>(), 32);
    const auto num_threads = num_warps * 32;

    // Generate, build and launch
    const CombineReduceEpilogueRuntime::Args args = {
        .use_expanded_layout = use_expanded_layout,
        .allow_multiple_reduction = allow_multiple_reduction,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .hidden = hidden,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk,
        .combined_x = static_cast<nv_bfloat16*>(combined_x),
        .combined_topk_weights = combined_topk_weights,
        .combined_topk_idx = combined_topk_idx,
        .reduce_buffer = reduce_buffer,
        .bias_0 = bias_0, .bias_1 = bias_1,
        .num_combined_tokens = num_combined_tokens,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, false, true)
    };
    const auto code = CombineReduceEpilogueRuntime::generate(args);
    const auto runtime = jit::compiler->build("combine_reduce_epilogue", code);
    CombineReduceEpilogueRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
