// Copyright (c) 2026 afdEP contributors. SPDX-License-Identifier: MIT
// Local afdEP role kernel; transport kernels remain upstream-derived.
#pragma once  // afdEP: generated from afdep_hybrid_collect.cuh by tests/role_kernels.py; do not edit

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/impls/afdep_sparse_hybrid.cuh>

namespace deep_ep::elastic {

// Directional hybrid combine receive of an Attention domain (csrc/kernels/elastic/
// afdep_suites.hpp). On an Attention rank the shared hybrid combine's scale-up warps
// return at once, and the forward warp of channel c only clears the channel's
// scale-up tails (which no peer writes in directional mode) and waits for, then
// clears, each FFN domain's per-channel finish signal; it moves no data. This CTA
// runs those statements for every channel: the same words, values and clears, read
// with the same acquire. The reduce epilogue still waits for this grid to complete.
// The shared kernel's entry gpu_barrier is two local grid syncs on an Attention
// domain (neither barrier half runs there); they are dropped. They ordered nothing:
// each tail and signal word belongs to one channel and is touched by one thread, and
// reuse across exchanges is ordered by the kernel boundary and by the FFN dispatch
// N+1, which needs this rank's dispatch N+1 finish tails, sent after this kernel.
template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts, int kNumChannels,
          int kAfdFfnScaleoutRanks, int64_t kNumTimeoutCycles, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
afdep_sparse_hybrid_collect_impl(void* workspace, const int scaleout_rank_idx, const int scaleup_rank_idx) {
    EP_STATIC_ASSERT(kNumScaleoutRanks <= 32, "Invalid ranks");
    EP_STATIC_ASSERT(0 < kAfdFfnScaleoutRanks and kAfdFfnScaleoutRanks < kNumScaleoutRanks, "Need both roles");
    EP_DEVICE_ASSERT(scaleout_rank_idx >= kAfdFfnScaleoutRanks);
    const auto thread_idx = static_cast<int>(threadIdx.x);
    const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);

    // Clean scale-up tails
    for (int i = thread_idx; i < kNumChannels * kNumScaleupRanks; i += kNumThreads)
        *workspace_layout.get_channel_scaleup_tail_ptr(i / kNumScaleupRanks, i % kNumScaleupRanks) = 0;

    // Wait for every FFN domain's signal on every channel, and clean it
    const auto expected_signal = math::pack2<int, int64_t>(1, 0);
    for (int i = thread_idx; i < kNumChannels * kAfdFfnScaleoutRanks; i += kNumThreads) {
        const int channel_idx = i / kAfdFfnScaleoutRanks, lane_idx = i % kAfdFfnScaleoutRanks;
        // afdEP sparse combine: only the channels this rank sent on, and channel 0
        if (channel_idx != 0 and ((*afdep_sparse_sent_ptr(workspace_layout, channel_idx) >> lane_idx) & 1) == 0)
            continue;
        const auto wait_ptr = workspace_layout.get_scaleout_channel_signaled_tail_ptr(channel_idx, lane_idx);
        comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
            const auto signal = ptx::ld_acquire_sys<int64_t>(wait_ptr);
            if (signal == expected_signal) {
                // Clean for next usages
                *wait_ptr = 0;
                return true;
            }

            if (is_last_check) {
                printf("DeepEP combine (scale-out wait all) timeout, scale-out: %d/%d, scale-up: %d/%d, "
                       "channel: %d, lane: %d, signal: %lld, expected: %lld\n",
                       scaleout_rank_idx, kNumScaleoutRanks, scaleup_rank_idx, kNumScaleupRanks,
                       channel_idx, lane_idx,
                       signal, expected_signal);
            }
            return false;
        });
    }
}

}  // namespace deep_ep::elastic
