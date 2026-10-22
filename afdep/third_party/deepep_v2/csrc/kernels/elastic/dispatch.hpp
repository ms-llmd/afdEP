#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"
#include "afdep_suites.hpp"

namespace deep_ep::elastic {

class DispatchPrologueRuntime final : public jit::LaunchRuntime<DispatchPrologueRuntime> {
public:
    struct Args {
        // Templated arguments
        int num_warps;
        int num_ranks;
        int num_max_tokens_per_rank;
        int num_experts, num_topk;

        // Parameters
        topk_idx_t* topk_idx;
        int* rank_count_buffer;
        int* dst_buffer_slot_idx;
        int num_tokens;
        int rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_deterministic_prologue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_deterministic_prologue_impl<{}, {}, {}, {}, {}, {}>);
}}
)",
                           args.launch_args.grid_dim.first,
                           args.num_warps,
                           args.num_ranks,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel,
                                                 config,
                                                 args.topk_idx,
                                                 args.rank_count_buffer,
                                                 args.dst_buffer_slot_idx,
                                                 args.num_tokens,
                                                 args.rank_idx));
    }
};

static void launch_dispatch_deterministic_prologue(topk_idx_t* topk_idx, int* rank_count_buffer,
                                                   int* dst_buffer_slot_idx,
                                                   const int& num_tokens, const int& num_max_tokens_per_rank,
                                                   const int& num_experts, const int& num_topk,
                                                   const int& rank_idx, const int& num_ranks,
                                                   const int& num_sms, const int& num_smem_bytes,
                                                   const at::cuda::CUDAStream& stream) {
    constexpr auto num_warps = 8;
    constexpr auto num_threads = num_warps * 32;
    EP_HOST_ASSERT((2 * num_warps + 1) * num_ranks * sizeof(int) <= num_smem_bytes and
                   "Insufficient shared memory");

    // Generate, build and launch
    const DispatchPrologueRuntime::Args args = {
        .num_warps = num_warps,
        .num_ranks = num_ranks,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk,
        .topk_idx = topk_idx,
        .rank_count_buffer = rank_count_buffer,
        .dst_buffer_slot_idx = dst_buffer_slot_idx,
        .num_tokens = num_tokens,
        .rank_idx = rank_idx,
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 1, true)};
    const auto code = DispatchPrologueRuntime::generate(args);
    const auto runtime = jit::compiler->build("dispatch_deterministic_prologue", code);
    DispatchPrologueRuntime::launch(runtime, args, stream);
}

class DispatchRuntime final : public jit::LaunchRuntime<DispatchRuntime> {
public:
    struct Args {
        // Templated arguments
        bool is_scaleup_nvlink;
        bool do_cpu_sync;
        bool reuse_slot_indices;
        int num_notify_warps;
        int num_dispatch_warps; // For hybrid dispatch
        int num_scaleout_warps, num_forward_warps; // For direct dispatch
        int num_scaleout_ranks, num_scaleup_ranks;
        int num_hidden_bytes, num_sf_packs;
        int num_max_tokens_per_rank;
        int num_experts, num_topk, expert_alignment;
        int num_qps;
        int64_t num_timeout_cycles;
        int afd_ffn_ranks;
        bool afd_skip_entry_barrier;
        bool afd_directional_notify;
        bool afd_directed_exit;
        bool afd_role_buffers;
        bool afd_per_cta_exit;
        int afd_zero_collect_row_bytes;
        int afd_ffn_scaleout_ranks;
        bool afd_directional_routes;
        bool afd_skip_attention_entry_sync;
        // afdEP count-first source only (AfdepSourceDispatchRuntime)
        int afd_ungated_tokens;
        bool afd_thread_count;
        // afdEP sparse directional hybrid only (afdep_sparse_hybrid.cuh); set by the launcher
        int afd_stagger_max_tokens;
        bool afd_sparse_dispatch;
        bool afd_sparse_combine;
        // afdEP mixed-role hybrid only (afdep_mixed_hybrid_dispatch.cuh): F, or 0
        int afd_mixed_ffn_ranks;

        // Parameters
        void* x; sf_pack_t* sf; topk_idx_t* topk_idx; float* topk_weights;
        topk_idx_t* copied_topk_idx;
        int* cumulative_local_expert_recv_stats;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* psum_num_recv_tokens_per_expert;
        int* dst_buffer_slot_idx;
        int* token_metadata_at_forward;
        int num_tokens;
        int sf_token_stride, sf_hidden_stride;
        ncclDevComm_t nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* buffer;
        void* workspace; void* mapped_host_workspace;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        std::string header_name, func_name;
        if (args.num_scaleout_ranks == 1) {
            header_name = "dispatch";
            func_name = "dispatch_impl" + get_direct_template_args(args);
        } else {
            header_name = "hybrid_dispatch";
            func_name = "hybrid_dispatch_impl" + get_hybrid_template_args(args);
        }

        return fmt::format(R"(
#include <deep_ep/impls/{}.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", header_name, func_name);
    }

    // Shared with the afdEP FFN receive of one NVLink domain, a copy of dispatch_impl.
    static std::string get_direct_template_args(const Args& args) {
        return fmt::format("<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}{}>",
            args.is_scaleup_nvlink,
            args.do_cpu_sync,
            args.reuse_slot_indices,
            args.launch_args.grid_dim.first,
            args.num_notify_warps, args.num_dispatch_warps,
            args.num_scaleup_ranks,
            args.num_hidden_bytes, args.num_sf_packs,
            args.num_max_tokens_per_rank,
            args.num_experts, args.num_topk, args.expert_alignment,
            args.num_qps, args.num_timeout_cycles, args.afd_ffn_ranks,
            args.afd_ffn_ranks > 0 ? fmt::format(", {}, {}, {}, {}, {}{}", args.afd_skip_entry_barrier,
                                                 args.afd_directional_notify, args.afd_directed_exit,
                                                 args.afd_role_buffers, args.afd_per_cta_exit,
                                                 args.afd_zero_collect_row_bytes > 0 ?
                                                     fmt::format(", {}", args.afd_zero_collect_row_bytes) : "")
                                   : std::string());
    }

    // Shared with the afdEP FFN hybrid dispatch, a copy of hybrid_dispatch_impl.
    static std::string get_hybrid_template_args(const Args& args) {
        return fmt::format("<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}{}>",
            args.do_cpu_sync,
            args.reuse_slot_indices,
            args.launch_args.grid_dim.first,
            args.num_notify_warps, args.num_scaleout_warps, args.num_forward_warps,
            args.num_scaleout_ranks, args.num_scaleup_ranks,
            args.num_hidden_bytes, args.num_sf_packs,
            args.num_max_tokens_per_rank,
            args.num_experts, args.num_topk, args.expert_alignment,
            args.num_qps, args.num_timeout_cycles,
            args.afd_ffn_scaleout_ranks > 0 ?
                fmt::format(", {}, {}{}", args.afd_ffn_scaleout_ranks, args.afd_directional_routes,
                            args.afd_skip_attention_entry_sync ? ", true" : "") :
                std::string());
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        if (args.num_scaleout_ranks == 1) {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
                kernel, config,
                args.x, args.sf, args.topk_idx, args.topk_weights,
                args.copied_topk_idx,
                args.cumulative_local_expert_recv_stats,
                args.psum_num_recv_tokens_per_scaleup_rank,
                args.psum_num_recv_tokens_per_expert,
                args.dst_buffer_slot_idx,
                args.num_tokens,
                args.sf_token_stride, args.sf_hidden_stride,
                args.nccl_dev_comm, args.nccl_window,
                args.buffer,
                args.workspace, args.mapped_host_workspace,
                args.scaleup_rank_idx));
        } else {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
                kernel, config,
                args.x, args.sf, args.topk_idx, args.topk_weights,
                args.copied_topk_idx,
                args.cumulative_local_expert_recv_stats,
                args.psum_num_recv_tokens_per_scaleup_rank,
                args.psum_num_recv_tokens_per_expert,
                args.dst_buffer_slot_idx,
                args.token_metadata_at_forward,
                args.num_tokens,
                args.sf_token_stride, args.sf_hidden_stride,
                args.nccl_dev_comm, args.nccl_window,
                args.buffer,
                args.workspace, args.mapped_host_workspace,
                args.scaleout_rank_idx, args.scaleup_rank_idx
            ));
        }
    }
};

// afdEP FFN directional hybrid dispatch (afdep_suites.hpp): the shared argument
// list, with no scale-out warps, through a generated copy of hybrid_dispatch.cuh.
class AfdepFfnHybridDispatchRuntime final : public jit::LaunchRuntime<AfdepFfnHybridDispatchRuntime> {
public:
    using Args = DispatchRuntime::Args;

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_ffn_hybrid_dispatch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_ffn_hybrid_dispatch_impl{});
}}
)", DispatchRuntime::get_hybrid_template_args(args));
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        DispatchRuntime::launch_impl(kernel, config, args);
    }
};

// afdEP sparse directional hybrid (afdep_suites.hpp): the shared argument list, with the
// entry-sync flag always explicit, then the stagger bound and sparse-dispatch and -combine flags.
static std::string afdep_sparse_hybrid_template_args(const DispatchRuntime::Args& args) {
    EP_HOST_ASSERT(args.afd_ffn_scaleout_ranks > 0);
    auto shared = DispatchRuntime::get_hybrid_template_args(args);
    shared.pop_back();
    return fmt::format("{}{}, {}, {}, {}>", shared, args.afd_skip_attention_entry_sync ? "" : ", false",
                       args.afd_stagger_max_tokens, args.afd_sparse_dispatch, args.afd_sparse_combine);
}

// afdEP Attention directional hybrid send, staggered and/or sparse (afdep_suites.hpp): the
// send-block shape through a generated copy of hybrid_dispatch.cuh.
class AfdepSourceHybridDispatchRuntime final : public jit::LaunchRuntime<AfdepSourceHybridDispatchRuntime> {
public:
    using Args = DispatchRuntime::Args;

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_source_hybrid_dispatch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_source_hybrid_dispatch_impl{});
}}
)", afdep_sparse_hybrid_template_args(args));
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        DispatchRuntime::launch_impl(kernel, config, args);
    }
};

// afdEP mixed-role hybrid dispatch (afdep_roles.hpp): the shared argument list, with the entry-sync
// flag always explicit, then F, through a generated copy of hybrid_dispatch.cuh. Domain 0 launches
// the full shape and pure-Attention domains the send block.
class AfdepMixedHybridDispatchRuntime final : public jit::LaunchRuntime<AfdepMixedHybridDispatchRuntime> {
public:
    using Args = DispatchRuntime::Args;

    static std::string generate_impl(const Args& args) {
        EP_HOST_ASSERT(args.afd_ffn_scaleout_ranks == 1 and args.afd_mixed_ffn_ranks > 0);
        auto shared = DispatchRuntime::get_hybrid_template_args(args);
        shared.pop_back();
        return fmt::format(R"(
#include <deep_ep/impls/afdep_mixed_hybrid_dispatch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_mixed_hybrid_dispatch_impl{}{}, {}>);
}}
)", shared, args.afd_skip_attention_entry_sync ? "" : ", false", args.afd_mixed_ffn_ranks);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        DispatchRuntime::launch_impl(kernel, config, args);
    }
};

// afdEP FFN directional hybrid dispatch whose forwarders retire unused channels by source
// token count (afdep_suites.hpp): the FFN copy's shape.
class AfdepFfnSparseHybridDispatchRuntime final : public jit::LaunchRuntime<AfdepFfnSparseHybridDispatchRuntime> {
public:
    using Args = DispatchRuntime::Args;

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_ffn_sparse_hybrid_dispatch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_ffn_sparse_hybrid_dispatch_impl{});
}}
)", afdep_sparse_hybrid_template_args(args));
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        DispatchRuntime::launch_impl(kernel, config, args);
    }
};

// afdEP FFN dispatch receive of one NVLink domain (afdep_suites.hpp): the shared argument
// list and launch shape, through a generated copy of dispatch.cuh.
class AfdepFfnDispatchRuntime final : public jit::LaunchRuntime<AfdepFfnDispatchRuntime> {
public:
    using Args = DispatchRuntime::Args;

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_ffn_dispatch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_ffn_dispatch_impl{});
}}
)", DispatchRuntime::get_direct_template_args(args));
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        DispatchRuntime::launch_impl(kernel, config, args);
    }
};

// afdEP count-first Attention source of one NVLink domain (afdep_suites.hpp): the shared argument
// list and launch shape, through a generated copy of dispatch.cuh whose template always takes the
// collect row bytes, the ungated tokens and the counting loop.
class AfdepSourceDispatchRuntime final : public jit::LaunchRuntime<AfdepSourceDispatchRuntime> {
public:
    using Args = DispatchRuntime::Args;

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_source_dispatch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_source_dispatch_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
                           args.is_scaleup_nvlink,
                           args.do_cpu_sync,
                           args.reuse_slot_indices,
                           args.launch_args.grid_dim.first,
                           args.num_notify_warps, args.num_dispatch_warps,
                           args.num_scaleup_ranks,
                           args.num_hidden_bytes, args.num_sf_packs,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk, args.expert_alignment,
                           args.num_qps, args.num_timeout_cycles, args.afd_ffn_ranks,
                           args.afd_skip_entry_barrier, args.afd_directional_notify, args.afd_directed_exit,
                           args.afd_role_buffers, args.afd_per_cta_exit,
                           args.afd_zero_collect_row_bytes, args.afd_ungated_tokens, args.afd_thread_count);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        DispatchRuntime::launch_impl(kernel, config, args);
    }
};

// afdEP count-first FFN receive of one NVLink domain (afdep_suites.hpp): the shared argument list
// and launch shape, through a generated copy of dispatch.cuh.
class AfdepFfnStreamDispatchRuntime final : public jit::LaunchRuntime<AfdepFfnStreamDispatchRuntime> {
public:
    using Args = DispatchRuntime::Args;

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_ffn_stream_dispatch.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_ffn_stream_dispatch_impl{});
}}
)", DispatchRuntime::get_direct_template_args(args));
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        DispatchRuntime::launch_impl(kernel, config, args);
    }
};

constexpr int kNumNotifyWarps = 4;

static int get_num_notify_smem_bytes(const int& num_ranks, const int& num_experts) {
    return math::align(num_ranks + num_experts, kNumNotifyWarps * 32) * sizeof(int);
}

static layout::TokenLayout get_dispatch_token_layout(
    const int& hidden, const int& elem_size, const int& num_sf_packs, const int& num_topk) {
    return layout::TokenLayout(hidden * elem_size, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
}

static void launch_dispatch(void* x, void* sf,
                            topk_idx_t* topk_idx, float* topk_weights,
                            topk_idx_t* copied_topk_idx,
                            int* cumulative_local_expert_recv_stats,
                            int* psum_num_recv_tokens_per_scaleup_rank,
                            int* psum_num_recv_tokens_per_expert,
                            int* dst_buffer_slot_idx,
                            int* token_metadata_at_forward,
                            const int& num_tokens, const int& num_max_tokens_per_rank,
                            const int& hidden, const int& elem_size,
                            const int& num_sf_packs, const int& sf_token_stride, const int& sf_hidden_stride,
                            const int& num_experts, const int& num_topk, const int& expert_alignment,
                            const ncclDevComm_t& nccl_dev_comm, const ncclWindow_t& nccl_window,
                            void* buffer,
                            void* workspace, void* mapped_host_workspace,
                            const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                            const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                            const bool& is_scaleup_nvlink,
                            const int& num_sms, const int& num_channels_per_sm,
                            const int& num_smem_bytes,
                            const int& num_qps, const int64_t& num_timeout_cycles,
                            const bool& cached_mode,
                            const bool& deterministic,
                            const bool& do_cpu_sync,
                            const at::cuda::CUDAStream& stream,
                            const int afd_ffn_ranks = 0,
                            const bool afd_skip_entry_barrier = false,
                            const bool afd_directional_notify = false,
                            const bool afd_directed_exit = false,
                            const bool afd_role_buffers = false,
                            const bool afd_per_cta_exit = false,
                            const int afd_ffn_scaleout_ranks = 0,
                            const bool afd_directional_routes = false,
                            const bool afd_skip_attention_entry_sync = false,
                            const int afd_zero_collect_row_bytes = 0,
                            const AfdLaunchShape afd_shape = AfdLaunchShape::kShared,
                            const int afd_ungated_tokens = 0,
                            const bool afd_thread_count = true,
                            const int afd_mixed_ffn_ranks = 0) {
    // Cached mode does not support expert token counting
    if (cached_mode)
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats == nullptr);

    // Utils
    const auto num_ranks = num_scaleout_ranks * num_scaleup_ranks;

    // Notify warps
    // TODO: why don't we use 4 notify warps?
    const int num_notify_warps = cached_mode ? 0 : kNumNotifyWarps;
    const bool reuse_slot_indices = cached_mode or deterministic;
    const int num_notify_smem_bytes = cached_mode ? 0 : get_num_notify_smem_bytes(num_ranks, num_experts);
    EP_HOST_ASSERT(num_notify_warps % 4 == 0);

    // Other warps
    int num_dispatch_warps = 0;
    int num_scaleout_warps = 0, num_forward_warps = 0;
    int num_threads = 0;

    // Maximize shared memory utilization
    if (num_scaleout_ranks == 1) {
        // Too many warps may cause performance degrade, so we limit the total warps into 512
        const auto token_layout = get_dispatch_token_layout(hidden, elem_size, num_sf_packs, num_topk);
        num_dispatch_warps = std::min<int>(std::min<int>(
            (num_smem_bytes - num_notify_smem_bytes) / token_layout.get_num_bytes<true>(), 32 - num_notify_warps),
            math::ceil_div(512, num_sms));
        num_threads = (num_notify_warps + num_dispatch_warps) * 32;
    } else {
        // Hybrid kernels
        // Some unimplemented assertions
        EP_HOST_ASSERT(not deterministic);

        num_scaleout_warps = num_channels_per_sm;
        num_forward_warps = num_channels_per_sm;
        num_threads = (num_notify_warps + num_scaleout_warps + num_forward_warps) * 32;
    }

    // Generate, build and launch
    const DispatchRuntime::Args args = {
        .is_scaleup_nvlink = is_scaleup_nvlink,
        .do_cpu_sync = do_cpu_sync,
        .reuse_slot_indices = reuse_slot_indices,
        .num_notify_warps = num_notify_warps,
        .num_dispatch_warps = num_dispatch_warps,
        .num_scaleout_warps = num_scaleout_warps, .num_forward_warps = num_forward_warps,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .num_hidden_bytes = hidden * elem_size, .num_sf_packs = num_sf_packs,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk, .expert_alignment = expert_alignment,
        .num_qps = num_qps, .num_timeout_cycles = num_timeout_cycles,
        .afd_ffn_ranks = afd_ffn_ranks,
        .afd_skip_entry_barrier = num_scaleout_ranks == 1 and afd_ffn_ranks > 0 and afd_skip_entry_barrier,
        .afd_directional_notify = num_scaleout_ranks == 1 and afd_ffn_ranks > 0 and is_scaleup_nvlink and
                                  not cached_mode and afd_directional_notify,
        .afd_directed_exit = num_scaleout_ranks == 1 and afd_ffn_ranks > 0 and is_scaleup_nvlink and
                             afd_directed_exit,
        .afd_role_buffers = num_scaleout_ranks == 1 and afd_ffn_ranks > 0 and is_scaleup_nvlink and
                            afd_role_buffers,
        .afd_per_cta_exit = num_scaleout_ranks == 1 and afd_ffn_ranks > 0 and is_scaleup_nvlink and
                            afd_directed_exit and afd_per_cta_exit,
        .afd_zero_collect_row_bytes = num_scaleout_ranks == 1 and afd_ffn_ranks > 0 and is_scaleup_nvlink ?
                                      afd_zero_collect_row_bytes : 0,
        .afd_ffn_scaleout_ranks = num_scaleout_ranks > 1 ? afd_ffn_scaleout_ranks : 0,
        .afd_directional_routes = num_scaleout_ranks > 1 and afd_ffn_scaleout_ranks > 0 and afd_directional_routes,
        .afd_skip_attention_entry_sync = num_scaleout_ranks > 1 and afd_ffn_scaleout_ranks > 0 and
                                         afd_skip_attention_entry_sync,
        .afd_ungated_tokens = afd_ungated_tokens,
        .afd_thread_count = afd_thread_count,
        .afd_mixed_ffn_ranks = num_scaleout_ranks > 1 and afd_ffn_scaleout_ranks == 1 ? afd_mixed_ffn_ranks : 0,
        .x = x, .sf = static_cast<sf_pack_t*>(sf), .topk_idx = topk_idx, .topk_weights = topk_weights,
        .copied_topk_idx = copied_topk_idx,
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,
        .dst_buffer_slot_idx = dst_buffer_slot_idx,
        .token_metadata_at_forward = token_metadata_at_forward,
        .num_tokens = num_tokens,
        .sf_token_stride = sf_token_stride, .sf_hidden_stride = sf_hidden_stride,
        .nccl_dev_comm = nccl_dev_comm, .nccl_window = nccl_window,
        .buffer = buffer,
        .workspace = workspace, .mapped_host_workspace = mapped_host_workspace,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        // NOTES: make cluster dim 2 to overlap with clustered computation kernels
        .launch_args = jit::LaunchArgs(num_sms, num_threads, num_smem_bytes, 2 - (num_sms % 2), true)};
    if (afd_shape == AfdLaunchShape::kCountFirst) {
        // afdEP count-first of one NVLink domain (afdep_suites.hpp): the shared shape and launch
        // arguments; sources and FFN ranks each run their generated copy of dispatch.cuh.
        EP_HOST_ASSERT(num_scaleout_ranks == 1 and is_scaleup_nvlink and args.afd_ffn_ranks > 0 and
                       args.afd_skip_entry_barrier and args.afd_directional_notify and args.afd_directed_exit and
                       args.afd_role_buffers and args.afd_per_cta_exit and not reuse_slot_indices);
        if (scaleup_rank_idx < args.afd_ffn_ranks) {
            EP_HOST_ASSERT(num_tokens == 0);
            const auto role_code = AfdepFfnStreamDispatchRuntime::generate(args);
            const auto role_runtime = jit::compiler->build("afdep_ffn_stream_dispatch", role_code);
            AfdepFfnStreamDispatchRuntime::launch(role_runtime, args, stream);
        } else {
            const auto role_code = AfdepSourceDispatchRuntime::generate(args);
            const auto role_runtime = jit::compiler->build("afdep_source_dispatch", role_code);
            AfdepSourceDispatchRuntime::launch(role_runtime, args, stream);
        }
        return;
    }
    if (args.afd_mixed_ffn_ranks > 0) {
        // afdEP mixed-role hybrid (afdep_roles.hpp): one generated copy for every role; pure-Attention
        // domains launch it without the forward warps (the directional send block).
        EP_HOST_ASSERT(args.afd_ffn_scaleout_ranks == 1 and num_notify_warps > 0 and
                       (afd_shape == AfdLaunchShape::kShared or afd_shape == AfdLaunchShape::kSendBlock));
        auto role_args = args;
        if (afd_shape == AfdLaunchShape::kSendBlock) {
            EP_HOST_ASSERT(scaleout_rank_idx >= args.afd_ffn_scaleout_ranks and
                           num_scaleup_ranks <= (num_notify_warps + num_scaleout_warps) * 32);
            role_args.launch_args = jit::LaunchArgs(num_sms, (num_notify_warps + num_scaleout_warps) * 32,
                                                    num_smem_bytes, 2 - (num_sms % 2), true);
        }
        const auto role_code = AfdepMixedHybridDispatchRuntime::generate(role_args);
        const auto role_runtime = jit::compiler->build("afdep_mixed_hybrid_dispatch", role_code);
        AfdepMixedHybridDispatchRuntime::launch(role_runtime, role_args, stream);
        return;
    }
    if (afd_shape == AfdLaunchShape::kRoleKernel and num_scaleout_ranks == 1) {
        // afdEP FFN receive of one NVLink domain (afdep_suites.hpp): the shared shape; the copy
        // triggers the copy epilogue only after the arrival wait.
        EP_HOST_ASSERT(is_scaleup_nvlink and num_tokens == 0 and args.afd_ffn_ranks > 0 and
                       args.afd_skip_entry_barrier and args.afd_directional_notify and args.afd_directed_exit and
                       scaleup_rank_idx < args.afd_ffn_ranks and not reuse_slot_indices);
        const auto role_code = AfdepFfnDispatchRuntime::generate(args);
        const auto role_runtime = jit::compiler->build("afdep_ffn_dispatch", role_code);
        AfdepFfnDispatchRuntime::launch(role_runtime, args, stream);
        return;
    }
    if (afd_shape == AfdLaunchShape::kRoleKernel) {
        // afdEP FFN directional hybrid dispatch (afdep_suites.hpp): notify and forward warps.
        EP_HOST_ASSERT(num_scaleout_ranks > 1 and args.afd_ffn_scaleout_ranks > 0 and
                       scaleout_rank_idx < args.afd_ffn_scaleout_ranks and num_tokens == 0 and
                       not reuse_slot_indices and num_notify_warps > 0 and
                       num_scaleup_ranks <= (num_notify_warps + num_forward_warps) * 32);
        auto role_args = args;
        role_args.num_scaleout_warps = 0;
        role_args.launch_args = jit::LaunchArgs(num_sms, (num_notify_warps + num_forward_warps) * 32,
                                                num_smem_bytes, 2 - (num_sms % 2), true);
        if (afdep_hybrid_sparse_dispatch()) {
            role_args.afd_stagger_max_tokens = afdep_hybrid_stagger_max_tokens();
            role_args.afd_sparse_dispatch = true;
            role_args.afd_sparse_combine = afdep_hybrid_sparse_combine();
            const auto sparse_code = AfdepFfnSparseHybridDispatchRuntime::generate(role_args);
            const auto sparse_runtime = jit::compiler->build("afdep_ffn_sparse_hybrid_dispatch", sparse_code);
            AfdepFfnSparseHybridDispatchRuntime::launch(sparse_runtime, role_args, stream);
            return;
        }
        const auto role_code = AfdepFfnHybridDispatchRuntime::generate(role_args);
        const auto role_runtime = jit::compiler->build("afdep_ffn_hybrid_dispatch", role_code);
        AfdepFfnHybridDispatchRuntime::launch(role_runtime, role_args, stream);
        return;
    }
    if (afd_shape == AfdLaunchShape::kSendBlock and
        (afdep_hybrid_stagger() or afdep_hybrid_sparse_dispatch() or afdep_hybrid_sparse_combine())) {
        // afdEP staggered or sparse send (afdep_suites.hpp): the send-block shape through the generated copy.
        EP_HOST_ASSERT(num_scaleout_ranks > 1 and args.afd_ffn_scaleout_ranks > 0 and
                       scaleout_rank_idx >= args.afd_ffn_scaleout_ranks and num_notify_warps > 0 and
                       num_scaleup_ranks <= (num_notify_warps + num_scaleout_warps) * 32);
        auto role_args = args;
        role_args.afd_stagger_max_tokens = afdep_hybrid_stagger_max_tokens();
        role_args.afd_sparse_dispatch = afdep_hybrid_sparse_dispatch();
        role_args.afd_sparse_combine = afdep_hybrid_sparse_combine();
        role_args.launch_args = jit::LaunchArgs(num_sms, (num_notify_warps + num_scaleout_warps) * 32,
                                                num_smem_bytes, 2 - (num_sms % 2), true);
        const auto role_code = AfdepSourceHybridDispatchRuntime::generate(role_args);
        const auto role_runtime = jit::compiler->build("afdep_source_hybrid_dispatch", role_code);
        AfdepSourceHybridDispatchRuntime::launch(role_runtime, role_args, stream);
        return;
    }
    const auto code = DispatchRuntime::generate(args);
    const auto runtime = jit::compiler->build("dispatch", code);
    auto shaped_args = args;
    if (afd_shape == AfdLaunchShape::kSendBlock) {
        // afdEP Attention directional hybrid dispatch (afdep_suites.hpp): notify and scale-out
        // warps; the trailing forward warps are not launched.
        EP_HOST_ASSERT(num_scaleout_ranks > 1 and args.afd_ffn_scaleout_ranks > 0 and
                       scaleout_rank_idx >= args.afd_ffn_scaleout_ranks and num_notify_warps > 0 and
                       num_scaleup_ranks <= (num_notify_warps + num_scaleout_warps) * 32);
        shaped_args.launch_args = jit::LaunchArgs(num_sms, (num_notify_warps + num_scaleout_warps) * 32,
                                                  num_smem_bytes, 2 - (num_sms % 2), true);
    }
    DispatchRuntime::launch(runtime, shaped_args, stream);
}

class DispatchCopyEpilogueRuntime final : public jit::LaunchRuntime<DispatchCopyEpilogueRuntime> {
public:
    struct Args {
        // Templated arguments
        bool do_expand, cached_mode, read_device_count;
        int num_channels;
        int num_warps;
        int num_scaleout_ranks, num_scaleup_ranks;
        int num_hidden_bytes, num_sf_packs;
        int num_max_tokens_per_rank;
        int num_experts, num_topk;
        int afd_region_offset;
        // afdEP count-first only (AfdepFfnStreamEpilogueRuntime)
        int64_t afd_timeout_cycles;
        int afd_unit_min_rows;

        // Parameters
        void *buffer, *workspace;
        int* psum_num_recv_tokens_per_scaleup_rank;
        int* psum_num_recv_tokens_per_expert;
        void* recv_x; void* recv_sf;
        topk_idx_t* recv_topk_idx; float* recv_topk_weights;
        int* recv_src_metadata;
        int* channel_linked_list;
        int num_recv_tokens;
        int recv_sf_token_stride, recv_sf_hidden_stride;
        int scaleout_rank_idx, scaleup_rank_idx;

        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/dispatch_copy_epilogue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_copy_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}{}>);
}}
)",
                           args.do_expand, args.cached_mode, args.read_device_count,
                           args.launch_args.grid_dim.first, args.num_channels, args.num_warps,
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.num_hidden_bytes, args.num_sf_packs,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk,
                           args.afd_region_offset > 0 ? fmt::format(", {}", args.afd_region_offset) : std::string());
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config,
                                                 args.buffer, args.workspace,
                                                 args.psum_num_recv_tokens_per_scaleup_rank,
                                                 args.psum_num_recv_tokens_per_expert,
                                                 args.recv_x, args.recv_sf, args.recv_topk_idx, args.recv_topk_weights,
                                                 args.recv_src_metadata,
                                                 args.channel_linked_list,
                                                 args.num_recv_tokens,
                                                 args.recv_sf_token_stride, args.recv_sf_hidden_stride,
                                                 args.scaleout_rank_idx, args.scaleup_rank_idx));
    }
};

// afdEP count-first copy epilogue (afdep_suites.hpp): the shared argument list through a generated
// copy of dispatch_copy_epilogue.cuh, whose template always takes the region offset, the wait
// timeout and the fewest rows per claimed unit.
class AfdepFfnStreamEpilogueRuntime final : public jit::LaunchRuntime<AfdepFfnStreamEpilogueRuntime> {
public:
    using Args = DispatchCopyEpilogueRuntime::Args;

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_ffn_stream_epilogue.cuh>

using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_ffn_stream_epilogue_impl<{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}>);
}}
)",
                           args.do_expand, args.cached_mode, args.read_device_count,
                           args.launch_args.grid_dim.first, args.num_channels, args.num_warps,
                           args.num_scaleout_ranks, args.num_scaleup_ranks,
                           args.num_hidden_bytes, args.num_sf_packs,
                           args.num_max_tokens_per_rank,
                           args.num_experts, args.num_topk,
                           args.afd_region_offset, args.afd_timeout_cycles, args.afd_unit_min_rows);
    }

    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        DispatchCopyEpilogueRuntime::launch_impl(kernel, config, args);
    }
};

static void launch_dispatch_copy_epilogue(void* buffer, void* workspace,
                                          int* psum_num_recv_tokens_per_scaleup_rank,
                                          int* psum_num_recv_tokens_per_expert,
                                          void* recv_x, void* recv_sf,
                                          topk_idx_t* recv_topk_idx, float* recv_topk_weights,
                                          int* recv_src_metadata,
                                          int* channel_linked_list,
                                          const int& num_recv_tokens, const int& num_max_tokens_per_rank,
                                          const int& num_hidden_bytes,
                                          const int& num_sf_packs, const int& recv_sf_token_stride, const int& recv_sf_hidden_stride,
                                          const int& num_experts, const int& num_topk,
                                          const int& scaleout_rank_idx, const int& scaleup_rank_idx,
                                          const int& num_scaleout_ranks, const int& num_scaleup_ranks,
                                          const int& num_sms, const int& num_smem_bytes,
                                          const int& num_channels,
                                          const bool& do_expand, const bool& cached_mode,
                                          const bool& read_device_count,
                                          const at::cuda::CUDAStream& stream,
                                          const int& afd_region_offset = 0,
                                          const bool& afd_count_first = false,
                                          const int64_t& afd_timeout_cycles = 0,
                                          const int& afd_unit_min_rows = 1,
                                          const int& afd_epilogue_sms = 0) {
    // Maximize shared memory utilization
    // afdEP count-first keeps its unit claim after the TMA buffers, on afd_epilogue_sms CTAs if set.
    const auto token_layout = layout::TokenLayout(num_hidden_bytes, num_sf_packs * sizeof(sf_pack_t), num_topk, true);
    const auto num_warps = std::min((num_smem_bytes - (afd_count_first ? kAfdepCountFirstClaimBytes : 0)) /
                                    token_layout.get_num_bytes<true>(), 32);
    const auto num_threads = num_warps * 32;
    const auto num_grid_sms = afd_count_first and afd_epilogue_sms > 0 ? std::min(afd_epilogue_sms, num_sms) : num_sms;

    // Generate, build and launch
    const DispatchCopyEpilogueRuntime::Args args = {
        .do_expand = do_expand, .cached_mode = cached_mode, .read_device_count = read_device_count,
        .num_channels = num_channels, .num_warps = num_warps,
        .num_scaleout_ranks = num_scaleout_ranks, .num_scaleup_ranks = num_scaleup_ranks,
        .num_hidden_bytes = num_hidden_bytes, .num_sf_packs = num_sf_packs,
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .num_experts = num_experts, .num_topk = num_topk,
        .afd_region_offset = afd_region_offset,
        .afd_timeout_cycles = afd_timeout_cycles,
        .afd_unit_min_rows = std::clamp(afd_unit_min_rows, 1, std::max(num_warps, 1)),
        .buffer = buffer, .workspace = workspace,
        .psum_num_recv_tokens_per_scaleup_rank = psum_num_recv_tokens_per_scaleup_rank,
        .psum_num_recv_tokens_per_expert = psum_num_recv_tokens_per_expert,
        .recv_x = recv_x, .recv_sf = recv_sf,
        .recv_topk_idx = recv_topk_idx, .recv_topk_weights = recv_topk_weights,
        .recv_src_metadata = recv_src_metadata,
        .channel_linked_list = channel_linked_list,
        .num_recv_tokens = num_recv_tokens,
        .recv_sf_token_stride = recv_sf_token_stride, .recv_sf_hidden_stride = recv_sf_hidden_stride,
        .scaleout_rank_idx = scaleout_rank_idx, .scaleup_rank_idx = scaleup_rank_idx,
        .launch_args = jit::LaunchArgs(num_grid_sms, num_threads, num_smem_bytes, 1, false, true)};
    if (afd_count_first) {
        // afdEP count-first (afdep_suites.hpp): CTAs claim units of rows that wait on P and on
        // their flags; the last claimant alone waits for the dispatch kernel.
        EP_HOST_ASSERT(do_expand and not cached_mode and num_scaleout_ranks == 1 and afd_region_offset > 0 and
                       num_scaleup_ranks - afd_region_offset <= kAfdepCountFirstMaxSources and
                       afd_timeout_cycles > 0 and num_warps > 0);
        const auto role_code = AfdepFfnStreamEpilogueRuntime::generate(args);
        const auto role_runtime = jit::compiler->build("afdep_ffn_stream_epilogue", role_code);
        AfdepFfnStreamEpilogueRuntime::launch(role_runtime, args, stream);
        return;
    }
    const auto code = DispatchCopyEpilogueRuntime::generate(args);
    const auto runtime = jit::compiler->build("dispatch_copy_epilogue", code);
    DispatchCopyEpilogueRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
