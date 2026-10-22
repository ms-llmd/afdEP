// Copyright (c) 2026 afdEP contributors. SPDX-License-Identifier: MIT
// afdEP sparse directional hybrid (opt-in): staggered source channels (AFDEP_HYBRID_STAGGER=1),
// per-source token counts (AFDEP_HYBRID_SPARSE_DISPATCH=1) that let FFN forwarders retire the
// channels a source never uses, so sources send finish tails only on channels they used, and
// combine signals only on channels that carried a reply (AFDEP_HYBRID_SPARSE_COMBINE=1). Only the
// generated copies use it (tests/role_kernels.py). Every rank must use the same settings: the FFN
// recomputes each source's channels from its token count and the stagger.
#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>

namespace deep_ep::elastic {

// Words, as int64 indices from the signaled-tail base. Hybrid tails use [0, C * domains), at most
// 1,024 channels x 32 domains; count-first and directed-exit words exist only in one NVLink domain.
// The region is zeroed at construction.
//   TOKENS(d)  FFN        source domain d's token count + 1 for this exchange (0: not yet arrived);
//                         written once per exchange by d's rail peer, cleared by the FFN after its
//                         exit barrier
//   SENT(c)    Attention  bit f: this rank's dispatch put a token for FFN domain f on channel c;
//                         written for every channel by the source dispatch, read by the collect
//                         (sparse combine: the FFN replies, and signals, on exactly those channels)
constexpr int64_t kAfdSparseRegionWords =
    static_cast<int64_t>(layout::WorkspaceLayout::kNumMaxRanks) * deep_ep::kNumMaxChannels;
constexpr int kAfdSparseMaxDomains = 32;
constexpr int64_t kAfdSparseTokens = kAfdSparseRegionWords - 4096;
constexpr int64_t kAfdSparseSent = kAfdSparseTokens + kAfdSparseMaxDomains;
EP_STATIC_ASSERT(kAfdSparseTokens >= static_cast<int64_t>(deep_ep::kNumMaxChannels) * kAfdSparseMaxDomains,
                 "Sparse hybrid words overlap the hybrid tails");
EP_STATIC_ASSERT(kAfdSparseSent + deep_ep::kNumMaxChannels <= kAfdSparseRegionWords,
                 "Sparse hybrid words exceed the signaled-tail region");

__device__ __forceinline__ int64_t* afdep_sparse_tokens_ptr(const layout::WorkspaceLayout& workspace,
                                                             const int domain) {
    return workspace.get_scaleout_channel_signaled_tail_ptr(0, 0) + kAfdSparseTokens + domain;
}

__device__ __forceinline__ int64_t* afdep_sparse_sent_ptr(const layout::WorkspaceLayout& workspace,
                                                           const int channel_idx) {
    return workspace.get_scaleout_channel_signaled_tail_ptr(0, 0) + kAfdSparseSent + channel_idx;
}

// The first token a source of T tokens places on a channel: (i + s * C / S) mod C staggered, which a
// source is while T <= kStaggerMaxTokens (0: never), else i mod C. Sources and FFN ranks both decide
// from T. The channel carries at least one of the source's tokens iff this is below T.
template <int kNumChannels, int kNumSources, int kStaggerMaxTokens>
__device__ __forceinline__ int afdep_first_token(const int channel_idx, const int source_idx, const int num_tokens) {
    if (num_tokens > kStaggerMaxTokens)
        return channel_idx;
    const int offset = static_cast<int>(static_cast<int64_t>(source_idx) * kNumChannels / kNumSources);
    return (channel_idx - offset + kNumChannels) % kNumChannels;
}

}  // namespace deep_ep::elastic
