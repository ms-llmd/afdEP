// Copyright (c) 2026 afdEP contributors. SPDX-License-Identifier: MIT
// afdEP count-first dispatch (role path, one NVLink domain): epoch-tagged source counts, per-row
// readiness flags and a claimed streaming copy epilogue. Only the generated copies use it:
// afdep_source_dispatch.cuh, afdep_ffn_stream_dispatch.cuh and afdep_ffn_stream_epilogue.cuh
// (tests/role_kernels.py). AFDEP_PAYLOAD_FIRST=1 keeps the payload-first kernels.
#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/handle.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic {

// Words, as int64 indices from the directed-exit base (afdep_roles.cuh), whose words occupy only
// [0, 2 * ranks + 3). The region is zeroed at construction; each scalar word owns a 128-byte line.
//   GATE        source  SM0 thread 0 stores the exchange's epoch once the counts are issued
//   SRC_EPOCH   source  the last CTA (per-CTA exit) commits the epoch every thread read at entry
//   P           FFN     dispatch CTA 0 releases 1 after the prefix sums; the last claimant clears it
//   CLAIM       FFN     epilogue unit claims; the last claimant clears it
//   E64[s]      FFN     the 64-bit epoch of source s, rebuilt from the high half of its count word
//   FLAG(s, q)  FFN     source s stores its epoch once slot q's row is stored; never cleared
constexpr int kAfdCfBase = 4110;
constexpr int kAfdCfGate = kAfdCfBase;
constexpr int kAfdCfSourceEpoch = kAfdCfBase + 16;
constexpr int kAfdCfPrefixReady = kAfdCfBase + 32;
constexpr int kAfdCfClaim = kAfdCfBase + 48;
constexpr int kAfdCfEpochs = kAfdCfBase + 64;
constexpr int kAfdCfFlags = kAfdCfBase + 128;
// Sources fit one warp's lanes; FFN ranks one 64-bit destination mask per token.
constexpr int kAfdCfMaxSources = 32;
constexpr int kAfdCfMaxFfnRanks = 64;
// Wire-relevant (sources write the flags FFN ranks poll), so a header constant: the preflight
// source digest covers it. afdep_suites.hpp mirrors the flag geometry (pinned by a unit test).
constexpr int kAfdepFlagStride = 1;
constexpr int64_t kAfdCfRegionWords =
    static_cast<int64_t>(layout::WorkspaceLayout::kNumMaxRanks) * deep_ep::kNumMaxChannels;
EP_STATIC_ASSERT(kAfdCfBase >= 2 * layout::WorkspaceLayout::kNumMaxRanks + 3, "Count-first words overlap directed exit");
EP_STATIC_ASSERT(kAfdCfEpochs + kAfdCfMaxSources <= kAfdCfFlags, "Count-first epochs overlap the flags");

__host__ __device__ constexpr bool afdep_cf_flags_fit(const int num_sources, const int bucket) {
    return kAfdCfFlags + static_cast<int64_t>(num_sources) * bucket * kAfdepFlagStride <= kAfdCfRegionWords;
}

__device__ __forceinline__ int64_t* afdep_cf_word(const layout::WorkspaceLayout& workspace, const int idx) {
    return workspace.get_scaleout_channel_signaled_tail_ptr(0, 0) + idx;
}

template <int kBucket>
__device__ __forceinline__ int64_t* afdep_cf_flag_ptr(const layout::WorkspaceLayout& workspace,
                                                      const int source, const int slot) {
    return afdep_cf_word(workspace, kAfdCfFlags) + (static_cast<int64_t>(source) * kBucket + slot) * kAfdepFlagStride;
}

// Every access below is asm volatile with a "memory" clobber, so the compiler moves no prefix-sum,
// top-k or flag access across it (ptx::ld_acquire_sys, ld_volatile and named_barrier lack one,
// and common/ stays upstream-pinned).
__device__ __forceinline__ int64_t afdep_cf_ld_relaxed_gpu(const int64_t* ptr) {
    int64_t value;
    asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ int64_t afdep_cf_ld_relaxed_sys(const int64_t* ptr) {
    int64_t value;
    asm volatile("ld.relaxed.sys.global.L1::no_allocate.b64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ int64_t afdep_cf_ld_acquire_gpu(const int64_t* ptr) {
    int64_t value;
    asm volatile("ld.acquire.gpu.global.b64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ int64_t afdep_cf_ld_acquire_sys(const int64_t* ptr) {
    int64_t value;
    asm volatile("ld.acquire.sys.global.L1::no_allocate.b64 %0, [%1];" : "=l"(value) : "l"(ptr) : "memory");
    return value;
}

__device__ __forceinline__ void afdep_cf_st_relaxed_gpu(int64_t* ptr, const int64_t value) {
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" :: "l"(ptr), "l"(value) : "memory");
}

__device__ __forceinline__ void afdep_cf_st_release_gpu(int64_t* ptr, const int64_t value) {
    asm volatile("st.release.gpu.global.b64 [%0], %1;" :: "l"(ptr), "l"(value) : "memory");
}

__device__ __forceinline__ void afdep_cf_st_release_sys(int64_t* ptr, const int64_t value) {
    asm volatile("st.release.sys.global.b64 [%0], %1;" :: "l"(ptr), "l"(value) : "memory");
}

__device__ __forceinline__ int64_t afdep_cf_atom_acq_rel_gpu_add(int64_t* ptr, const int64_t value) {
    int64_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;" : "=l"(old) : "l"(ptr), "l"(value) : "memory");
    return old;
}

__device__ __forceinline__ void afdep_cf_fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;" ::: "memory");
}

template <int kNumThreads>
__device__ __forceinline__ void afdep_cf_named_barrier_mem(const int idx) {
    asm volatile("bar.sync %0, %1;" :: "r"(idx), "r"(kNumThreads) : "memory");
}

// Epochs. A source's epoch grows by one per count-first exchange and travels in the high half of
// its rank count word (one 8-byte put, so count and epoch are single-copy atomic). A low half of
// 0xFFFFFFFF is skipped: that is a payload-first word (encoded count, sign-extended).
__device__ __forceinline__ int64_t afdep_cf_source_epoch(const layout::WorkspaceLayout& workspace) {
    auto epoch = afdep_cf_ld_relaxed_gpu(afdep_cf_word(workspace, kAfdCfSourceEpoch)) + 1;
    if (static_cast<uint32_t>(epoch) == 0xffffffffu)
        epoch += 1;
    return epoch;
}

__device__ __forceinline__ int64_t afdep_cf_tag_count(const int encoded, const int64_t epoch) {
    return static_cast<int64_t>((static_cast<uint64_t>(static_cast<uint32_t>(epoch)) << 32) |
                                static_cast<uint32_t>(encoded));
}

// The FFN rebuilds the exact 64-bit epoch from the 32-bit delta (no wrap hazard): payload-first
// sources trap here instead of leaving rows unflagged.
template <int kAfdFfnRanks>
__device__ __forceinline__ void afdep_cf_record_epoch(const layout::WorkspaceLayout& workspace, const int source,
                                                      const int64_t word, const int rank_idx) {
    const auto high = static_cast<uint32_t>(static_cast<uint64_t>(word) >> 32);
    if (high == 0xffffffffu) {
        printf("afdEP count-first: source %d is not count-first, rank: %d\n", kAfdFfnRanks + source, rank_idx);
        ptx::trap();
    }
    const auto ptr = afdep_cf_word(workspace, kAfdCfEpochs + source);
    const auto previous = afdep_cf_ld_relaxed_gpu(ptr);
    const auto delta = high - static_cast<uint32_t>(previous);
    if (delta == 0 or delta >= 0x80000000u) {
        printf("afdEP count-first: source %d epoch did not advance, rank: %d, previous: %lld, received: %u\n",
               kAfdFfnRanks + source, rank_idx, static_cast<long long>(previous), high);
        ptx::trap();
    }
    afdep_cf_st_relaxed_gpu(ptr, previous + delta);
}

__device__ __forceinline__ void afdep_cf_commit_source_epoch(const layout::WorkspaceLayout& workspace,
                                                             const int64_t epoch) {
    afdep_cf_st_relaxed_gpu(afdep_cf_word(workspace, kAfdCfSourceEpoch), epoch);
}

// Counting: one token per notify thread over the tokens this CTA's payload warps send
// (sm_idx mod kNumSMs). Scalar loads keep any topk_idx_t width; every route is validated at its
// native width before the first route-derived index, as afdep_checked_expert does.
template <int kNumSMs, int kNumNotifyThreads, int kNumTopk, int kNumRealExperts, int kNumExpertsPerRank,
          int kFfnRanks>
__device__ __forceinline__ void afdep_cf_count_tokens(const topk_idx_t* topk_idx, const int num_tokens,
                                                      const int sm_idx, const int thread_idx,
                                                      int* rank_count, int* expert_count) {
    EP_STATIC_ASSERT(kNumTopk <= 32 and kFfnRanks <= kAfdCfMaxFfnRanks, "Count-first counting: top-k or FFN ranks");
    EP_STATIC_ASSERT(kNumRealExperts == kFfnRanks * kNumExpertsPerRank, "Count-first counting: FFN experts");
    for (int i = sm_idx + kNumSMs * thread_idx; i < num_tokens; i += kNumSMs * kNumNotifyThreads) {
        int experts[kNumTopk];
        bool in_range = true, unique = true;
        #pragma unroll
        for (int j = 0; j < kNumTopk; ++ j) {
            const topk_idx_t raw = __ldg(topk_idx + i * kNumTopk + j);
            in_range &= raw >= -1 and raw < kNumRealExperts;
            experts[j] = static_cast<int>(raw);
            #pragma unroll
            for (int k = 0; k < j; ++ k)
                unique &= experts[j] == -1 or experts[k] != experts[j];
        }
        EP_DEVICE_ASSERT(in_range);
        EP_DEVICE_ASSERT(unique);

        uint64_t ranks = 0;
        #pragma unroll
        for (int j = 0; j < kNumTopk; ++ j) {
            if (experts[j] >= 0) {
                atomicAdd_block(expert_count + experts[j], 1);
                ranks |= 1ull << (experts[j] / kNumExpertsPerRank);
            }
        }
        for (; ranks != 0; ranks &= ranks - 1)
            atomicAdd_block(rank_count + __ffsll(static_cast<long long>(ranks)) - 1, 1);
    }
}

// Gate (performance only; the flags carry the ordering): payload holds its NVLink stores until
// SM0 has issued the counts.
__device__ __forceinline__ void afdep_cf_open_gate(const layout::WorkspaceLayout& workspace, const int64_t epoch) {
    afdep_cf_st_relaxed_gpu(afdep_cf_word(workspace, kAfdCfGate), epoch);
}

template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ void afdep_cf_wait_gate(const layout::WorkspaceLayout& workspace, const int64_t epoch,
                                                   const int lane_idx, const int rank_idx) {
    if (lane_idx == 0) {
        const auto gate = afdep_cf_word(workspace, kAfdCfGate);
        comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
            const auto value = afdep_cf_ld_relaxed_gpu(gate);
            EP_DEVICE_ASSERT(value <= epoch);
            if (value == epoch)
                return true;
            if (is_last_check)
                printf("afdEP count-first gate timeout, rank: %d, gate: %lld, epoch: %lld\n",
                       rank_idx, static_cast<long long>(value), static_cast<long long>(epoch));
            return false;
        });
    }
    __syncwarp();
}

// The lane that issued a row's NVLink store releases its flag once its own bulk group completed
// (cp.async.bulk.wait_group waits for the writes, followed by an implicit generic-async proxy fence).
template <int kNumRanks, int kBucket, typename team_t>
__device__ __forceinline__ void afdep_cf_release_row(const handle::NCCLGin& gin, const layout::WorkspaceLayout& workspace,
                                                     int& dst_rank_idx, int& slot_idx,
                                                     const int source, const int64_t epoch) {
    EP_STATIC_ASSERT(kNumRanks <= layout::WorkspaceLayout::kNumMaxRanks, "Invalid number of ranks");
    if (slot_idx >= 0) {
        afdep_cf_st_release_sys(gin.get_sym_ptr<team_t>(afdep_cf_flag_ptr<kBucket>(workspace, source, slot_idx),
                                                        dst_rank_idx), epoch);
        slot_idx = -1;
    }
}

// P: the copy epilogue no longer waits for dispatch completion before reading the prefix sums,
// epochs and per-expert cursors. The release is cumulative through the preceding barrier.
__device__ __forceinline__ void afdep_cf_publish_prefix(const layout::WorkspaceLayout& workspace, const int rank_idx) {
    const auto ptr = afdep_cf_word(workspace, kAfdCfPrefixReady);
    if (afdep_cf_ld_relaxed_gpu(ptr) != 0) {
        printf("afdEP count-first: P not cleared, rank: %d\n", rank_idx);
        ptx::trap();
    }
    afdep_cf_st_release_gpu(ptr, 1);
}

template <int64_t kNumTimeoutCycles>
__device__ __forceinline__ void afdep_cf_wait_prefix(const layout::WorkspaceLayout& workspace,
                                                     const int thread_idx, const int rank_idx) {
    if (thread_idx == 0) {
        const auto ptr = afdep_cf_word(workspace, kAfdCfPrefixReady);
        comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
            const auto value = afdep_cf_ld_relaxed_gpu(ptr);
            if (value == 1)
                return true;
            if (is_last_check)
                printf("afdEP count-first prefix timeout, rank: %d, P: %lld\n", rank_idx, static_cast<long long>(value));
            return false;
        });
        // The acquire stays outside the assert: the ordering needs it even if the assert is compiled out
        const auto ready = afdep_cf_ld_acquire_gpu(ptr);
        EP_DEVICE_ASSERT(ready == 1);
    }
    __syncthreads();
}

// Copy epilogue. Lane s < S of every warp holds source s's extent; a unit is up to `rows` slots of
// one source, ordered chunk-major, source-minor.
struct AfdCfSources {
    int n, base;
    int64_t epoch;
    // Warp-uniform: rows received, rows per unit and units
    int total, rows, units;
};

struct AfdCfUnit {
    int source, first_slot, len, base;
    int64_t epoch;
    bool valid;
};

struct AfdCfCursor {
    // Claims only grow, so each warp's chunk walk only moves forward
    int chunk, chunk_start, buffer;
    bool last;
};

template <int kNumScaleupRanks, int kOffset, int kBucket>
__device__ __forceinline__ AfdCfSources afdep_cf_load_sources(const layout::WorkspaceLayout& workspace,
                                                              const int* rank_psum, const int lane_idx) {
    constexpr int kNumSources = kNumScaleupRanks - kOffset;
    EP_STATIC_ASSERT(kOffset > 0 and kNumSources > 0 and kNumSources <= kAfdCfMaxSources, "Invalid count-first sources");
    AfdCfSources sources = {};
    if (lane_idx < kNumSources) {
        // FFN-origin counts are zero, so the first source's base is rank_psum[kOffset - 1] == 0
        sources.base = rank_psum[kOffset + lane_idx - 1];
        sources.n = rank_psum[kOffset + lane_idx] - sources.base;
        sources.epoch = afdep_cf_ld_relaxed_gpu(afdep_cf_word(workspace, kAfdCfEpochs + lane_idx));
        EP_DEVICE_ASSERT(sources.n >= 0 and sources.n <= kBucket and sources.epoch > 0);
    }
    sources.total = __reduce_add_sync(0xffffffff, sources.n);
    return sources;
}

// Identical in every CTA: kMinRows trades small-batch latency (1 spreads rows over every SM)
// against SMs held during ingress (kNumWarps concentrates them).
template <int kNumWarps, int kMinRows>
__device__ __forceinline__ void afdep_cf_unit_rows(AfdCfSources& sources, const int num_ctas) {
    EP_STATIC_ASSERT(kMinRows >= 1 and kNumWarps <= 32, "Invalid count-first unit rows");
    sources.rows = min(max(math::ceil_div(sources.total, num_ctas), kMinRows), kNumWarps);
    sources.units = __reduce_add_sync(0xffffffff, math::ceil_div(sources.n, sources.rows));
}

__device__ __forceinline__ AfdCfUnit afdep_cf_claim(const layout::WorkspaceLayout& workspace, const AfdCfSources& sources,
                                                    AfdCfCursor& cursor, int* claims, const int thread_idx) {
    // Every thread's rows (reads and writes) precede the CTA's next claim, and claims form one
    // acq_rel chain: a CTA's failed claim follows all of its work, including its P read.
    __syncthreads();
    if (thread_idx == 0) {
        claims[cursor.buffer] =
            static_cast<int>(afdep_cf_atom_acq_rel_gpu_add(afdep_cf_word(workspace, kAfdCfClaim), 1));
    }
    __syncthreads();
    const int claimed = claims[cursor.buffer];
    cursor.buffer ^= 1;

    AfdCfUnit unit = {};
    if (claimed >= sources.units) {
        // Each CTA fails exactly once and never claims again: claim units + grid - 1 is the last
        cursor.last = claimed == sources.units + static_cast<int>(gridDim.x) - 1;
        return unit;
    }

    // Unit (c, s): slots [cK, min(n_s, (c + 1)K)) of source s, over the sources with n_s > cK
    while (true) {
        const auto active = __ballot_sync(0xffffffff, sources.n > cursor.chunk * sources.rows);
        const int num_active = __popc(active);
        EP_DEVICE_ASSERT(num_active > 0);
        if (claimed < cursor.chunk_start + num_active) {
            auto rest = active;
            for (int k = claimed - cursor.chunk_start; k > 0; -- k)
                rest &= rest - 1;
            unit.source = __ffs(static_cast<int>(rest)) - 1;
            break;
        }
        cursor.chunk_start += num_active;
        ++ cursor.chunk;
    }
    unit.first_slot = cursor.chunk * sources.rows;
    unit.len = min(ptx::exchange(sources.n, unit.source) - unit.first_slot, sources.rows);
    unit.base = ptx::exchange(sources.base, unit.source);
    unit.epoch = ptx::exchange(sources.epoch, unit.source);
    unit.valid = true;
    return unit;
}

// One relaxed poll and one acquire per row. Epochs strictly grow per source and the flag is
// never cleared, so only this exchange's write of this row can equal the expected epoch.
// Diagnostics name the source by global rank (kAfdFfnRanks + source), as afdep_cf_record_epoch does.
template <int64_t kNumTimeoutCycles, int kBucket, int kAfdFfnRanks>
__device__ __forceinline__ void afdep_cf_wait_row(const layout::WorkspaceLayout& workspace, const int source,
                                                  const int slot, const int64_t epoch,
                                                  const int lane_idx, const int rank_idx) {
    if (lane_idx == 0) {
        const auto flag = afdep_cf_flag_ptr<kBucket>(workspace, source, slot);
        comm::timeout_while<kNumTimeoutCycles>([=](const bool& is_last_check) {
            const auto value = afdep_cf_ld_relaxed_sys(flag);
            EP_DEVICE_ASSERT(value <= epoch);
            if (value == epoch)
                return true;
            if (is_last_check)
                printf("afdEP count-first row timeout, rank: %d, source: %d, slot: %d, flag: %lld, epoch: %lld\n",
                       rank_idx, kAfdFfnRanks + source, slot, static_cast<long long>(value),
                       static_cast<long long>(epoch));
            return false;
        });
        // The acquire stays outside the assert: the ordering needs it even if the assert is compiled out
        const auto acquired = afdep_cf_ld_acquire_sys(flag);
        EP_DEVICE_ASSERT(acquired == epoch);
    }
    __syncwarp();
    // The elected TMA lane is unspecified: every lane orders its async-proxy reads after the acquire
    afdep_cf_fence_proxy_async_global();
}

__device__ __forceinline__ void afdep_cf_epilogue_end(const layout::WorkspaceLayout& workspace,
                                                      const bool last, const int thread_idx) {
    if (last and thread_idx == 0) {
        // Epilogue completion still implies dispatch completion (the dispatch writes after its
        // trigger: the arrival round and the counter clear); the clears follow every CTA's P read.
        cudaGridDependencySynchronize();
        afdep_cf_st_relaxed_gpu(afdep_cf_word(workspace, kAfdCfPrefixReady), 0);
        afdep_cf_st_relaxed_gpu(afdep_cf_word(workspace, kAfdCfClaim), 0);
        __threadfence();
    }
}

}  // namespace deep_ep::elastic
