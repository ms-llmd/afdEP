// Copyright (c) 2026 afdEP contributors. SPDX-License-Identifier: MIT
// Local afdEP role specialization; transport kernels remain upstream-derived.
#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic {

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
