// Copyright (c) 2026 afdEP contributors. SPDX-License-Identifier: MIT
// Local afdEP role specialization; transport kernels remain upstream-derived.
#pragma once

#include <cooperative_groups.h>

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic {

// Directed exit (role path, one NVLink domain). Senders flush, grid-sync and
// release a per-peer monotonic arrival counter in each receiver's workspace;
// only receivers wait. The direct path never uses the scale-out channel tails,
// which hold [phase][sender] arrival counters followed by per-phase local rounds.
constexpr int kAfdepDispatchPhase = 0;
constexpr int kAfdepCombinePhase = 1;

template <int kNumRanks>
__device__ __forceinline__ int64_t* afdep_arrival_ptr(const layout::WorkspaceLayout& workspace,
                                                      const int phase, const int sender) {
    return workspace.get_scaleout_channel_signaled_tail_ptr(0, 0) + phase * kNumRanks + sender;
}

// Per-CTA exit: source CTAs that have finished this dispatch (after the rounds).
template <int kNumRanks>
__device__ __forceinline__ int64_t* afdep_done_ptr(const layout::WorkspaceLayout& workspace) {
    return workspace.get_scaleout_channel_signaled_tail_ptr(0, 0) + 2 * kNumRanks + 2;
}

template <int kNumRanks>
__device__ __forceinline__ int64_t* afdep_round_ptr(const layout::WorkspaceLayout& workspace, const int phase) {
    return workspace.get_scaleout_channel_signaled_tail_ptr(0, 0) + 2 * kNumRanks + phase;
}

template <int kNumRanks, typename team_t, bool kPerCta = false>
__device__ __forceinline__ void afdep_signal_receivers(const handle::NCCLGin& gin,
                                                       const layout::WorkspaceLayout& workspace,
                                                       const int phase, const int rank_idx,
                                                       const int first_receiver, const int num_receivers,
                                                       const int sm_idx, const int thread_idx) {
    // Every CTA's TMA stores complete before the arrival release.
    ptx::tma_store_commit();
    ptx::tma_store_wait();
    __syncwarp();
    if constexpr (kPerCta) {
        // The sending role receives nothing in this phase, so it needs no grid
        // sync: each CTA releases its own arrival (receivers count every CTA).
        __syncthreads();
    } else {
        cooperative_groups::this_grid().sync();
    }
    if ((kPerCta or sm_idx == 0) and thread_idx < num_receivers) {
        const auto dst = gin.get_sym_ptr<team_t>(afdep_arrival_ptr<kNumRanks>(workspace, phase, rank_idx),
                                                 first_receiver + thread_idx);
        ptx::red_add_rel_sys(dst, static_cast<int64_t>(1));
    }
}

// kNumSenderCtas: arrivals per sender per round (its grid size with per-CTA release, else 1).
template <int kNumRanks, int64_t kNumTimeoutCycles, int kNumSenderCtas = 1>
__device__ __forceinline__ void afdep_wait_senders(const layout::WorkspaceLayout& workspace,
                                                   const int phase, const int rank_idx,
                                                   const int first_sender, const int num_senders,
                                                   const int sm_idx, const int thread_idx) {
    if (sm_idx != 0 or thread_idx != 0)
        return;
    const int64_t target = (++ *afdep_round_ptr<kNumRanks>(workspace, phase)) * kNumSenderCtas;
    for (int sender = first_sender; sender < first_sender + num_senders; ++ sender) {
        comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
            const auto arrived = ptx::ld_acquire_sys(afdep_arrival_ptr<kNumRanks>(workspace, phase, sender));
            if (arrived >= target)
                return true;
            if (is_last_check)
                printf("afdEP directed exit timeout, rank: %d, phase: %d, sender: %d, arrived: %lld, target: %lld\n",
                       rank_idx, phase, sender, static_cast<long long>(arrived), static_cast<long long>(target));
            return false;
        });
    }
}

template <int kNumRealExperts>
__device__ __forceinline__ int afdep_checked_expert(const topk_idx_t expert, const int lane) {
    // Check the native index width before narrowing: a large int64 ID may
    // otherwise wrap into a valid expert/rank and escape the role boundary.
    EP_DEVICE_ASSERT(expert >= -1 and expert < kNumRealExperts);
    const int narrowed = static_cast<int>(expert);
    const bool unique = ptx::deduplicate(narrowed, lane);
    EP_DEVICE_ASSERT(expert == -1 or unique);
    return narrowed;
}

template <int kNumRealExperts, int kNumTopk>
__global__ void afdep_validate_source_routes_impl(const topk_idx_t* ids, const int num_tokens) {
    const auto lane = ptx::get_lane_idx();
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int warps = gridDim.x * blockDim.x / 32;
    for (int token = warp; token < num_tokens; token += warps) {
        const topk_idx_t expert = lane < kNumTopk ? ids[token * kNumTopk + lane] : -1;
        // Checked on every graph replay, before any transport can target a
        // zero-expert source. Duplicate real experts also violate row bounds.
        afdep_checked_expert<kNumRealExperts>(expert, lane);
    }
}

template <int kNumScaleoutRanks, int kNumScaleupRanks, int kNumExperts, int kNumChannels>
__global__ void afdep_source_finalize_impl(void* workspace, int* channel_linked_list,
                                          const int* psum_num_recv_tokens_per_scaleup_rank) {
    // Preserve the original copy epilogue's PDL dependency even with no rows.
    cudaGridDependencySynchronize();
    EP_DEVICE_ASSERT(psum_num_recv_tokens_per_scaleup_rank[kNumScaleupRanks - 1] == 0);
    if constexpr (kNumScaleoutRanks > 1) {
        const auto workspace_layout = layout::WorkspaceLayout(
            workspace, kNumScaleoutRanks, kNumScaleupRanks, kNumExperts);
        for (int index = threadIdx.x; index < kNumChannels * kNumScaleupRanks;
             index += blockDim.x) {
            const int channel = index / kNumScaleupRanks, peer = index % kNumScaleupRanks;
            auto tail = workspace_layout.get_channel_scaleup_tail_ptr(channel, peer);
            // A source may still forward traffic across NIC rails. Its
            // linked-list endings and tail reset are required by combine.
            channel_linked_list[*tail] = -1;
            *tail = 0;
        }
    }
}

template <int = 0>
__global__ void afdep_ffn_complete_impl(void*) {
    // The zero-source FFN has no combined output, but still consumes the main
    // combine kernel's completion dependency before a buffer may be reused.
    cudaGridDependencySynchronize();
}

}  // namespace deep_ep::elastic
