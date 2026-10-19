// Copyright (c) 2026 afdEP contributors. SPDX-License-Identifier: MIT
#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class AfdEPRouteGuardRuntime final : public jit::LaunchRuntime<AfdEPRouteGuardRuntime> {
public:
    struct Args {
        int real_experts, top_k;
        topk_idx_t* ids;
        int num_tokens;
        jit::LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_roles.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_validate_source_routes_impl<{}, {}>);
}}
)", args.real_experts, args.top_k);
    }
    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config, args.ids, args.num_tokens));
    }
};

static void launch_afdep_route_guard(topk_idx_t* ids, const int num_tokens,
                                     const int real_experts, const int top_k,
                                     const int num_sms, const at::cuda::CUDAStream& stream) {
    if (num_tokens == 0)
        return;
    const AfdEPRouteGuardRuntime::Args args = {
        real_experts, top_k, ids, num_tokens,
        jit::LaunchArgs(std::min(num_sms, (num_tokens + 7) / 8), 256)
    };
    const auto runtime = jit::compiler->build("afdep_route_guard", AfdEPRouteGuardRuntime::generate(args));
    AfdEPRouteGuardRuntime::launch(runtime, args, stream);
}

class AfdEPCompletionRuntime final : public jit::LaunchRuntime<AfdEPCompletionRuntime> {
public:
    struct Args {
        bool source;
        int scaleout_ranks, scaleup_ranks, experts, channels;
        void* workspace;
        int* channel_linked_list;
        int* rank_psum;
        jit::LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args) {
        const auto function = args.source ? fmt::format(
            "afdep_source_finalize_impl<{}, {}, {}, {}>",
            args.scaleout_ranks, args.scaleup_ranks, args.experts, args.channels)
            : "afdep_ffn_complete_impl<>";
        return fmt::format(R"(
#include <deep_ep/impls/afdep_roles.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", function);
    }
    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        if (args.source) {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
                kernel, config, args.workspace, args.channel_linked_list, args.rank_psum));
        } else {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config, args.workspace));
        }
    }
};

static void launch_afdep_completion(const bool source, void* workspace,
                                    int* channel_linked_list, int* rank_psum,
                                    const int scaleout_ranks, const int scaleup_ranks,
                                    const int experts, const int channels,
                                    const at::cuda::CUDAStream& stream) {
    const AfdEPCompletionRuntime::Args args = {
        source, scaleout_ranks, scaleup_ranks, experts, channels,
        workspace, channel_linked_list, rank_psum,
        jit::LaunchArgs(1, 128, 0, 1, false, true)
    };
    const auto runtime = jit::compiler->build(
        source ? "afdep_source_finalize" : "afdep_ffn_complete",
        AfdEPCompletionRuntime::generate(args));
    AfdEPCompletionRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
