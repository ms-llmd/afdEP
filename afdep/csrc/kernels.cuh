// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#pragma once

#include "layout.hpp"
#include "deepep_reduce.cuh"
#include "deepep_tma.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <nccl_device.h>

namespace afdep {

__device__ __forceinline__ Control* control(uint8_t* base) {
    return reinterpret_cast<Control*>(base);
}
__device__ __forceinline__ uint64_t acquire(const uint64_t* p) {
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(p) : "memory");
    return value;
}
__device__ __forceinline__ void release(uint64_t* p, uint64_t value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(value) : "memory");
}
__device__ __forceinline__ void fail(const char* reason, const DeviceArgs& a) {
    printf("AFDEP timeout/bounds: %s rank=%d epoch=%llu\n", reason,
           a.rank, static_cast<unsigned long long>(control(a.base)->epoch));
    asm volatile("trap;");
}
__device__ __forceinline__ void wait_local(const uint64_t* p, uint64_t target,
                                         const DeviceArgs& a) {
    const uint64_t start = clock64();
    while (acquire(p) < target) {
        if (clock64() - start >= a.timeout_cycles) fail("local phase", a);
    }
}
__device__ __forceinline__ void wait_signal(int context, int signal, uint64_t target,
                                          const DeviceArgs& a) {
    ncclGin gin(a.comm, context, NCCL_GIN_RESOURCE_SHARING_CTA);
    const uint64_t start = clock64();
    while (gin.readSignal(static_cast<ncclGinSignal_t>(signal)) < target) {
        if (clock64() - start >= a.timeout_cycles) fail("network phase", a);
    }
}
__device__ __forceinline__ void send_credit(int signal, int peer, const DeviceArgs& a,
                                           bool direct = false) {
    ncclGin gin(a.comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    gin.signal(direct ? ncclTeamWorld(a.comm) : ncclTeamRail(a.comm), peer,
               ncclGin_StrongSignalInc{static_cast<ncclGinSignal_t>(signal)});
}
__device__ __forceinline__ int packet_tokens(uint8_t* base, const Layout& l) {
    return static_cast<int>(reinterpret_cast<volatile Packet*>(base + l.request)->tokens);
}
__device__ __forceinline__ int live_source_tokens(const DeviceArgs& a) {
    const int tokens = a.live_tokens ? *a.live_tokens : a.tokens;
    if (tokens < 0 || tokens > a.tokens || tokens > a.bucket)
        fail("live source tokens exceed input capacity/bucket", a);
    return tokens;
}
__device__ __forceinline__ uint4 load_remote_vector(const void* pointer) {
    uint4 value;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(pointer) : "memory");
    return value;
}

// A token may select several experts on this worker. Read each activation
// vector once (over NVLink in hybrid mode), then fan it out to all local rows.
template <bool kRemote>
__device__ __forceinline__ void copy_expert_rows(const DeviceArgs& a, const uint8_t* packet,
                                                int tokens, int token, const int* slots) {
    const Layout l = a.layout;
    bool selected = false;
    for (int k = 0; k < l.topk; ++k) selected |= slots[k] >= 0;
    if (!selected) return;
    const int vectors = l.hidden * l.element_bytes() / sizeof(uint4);
    for (int h = threadIdx.x; h < vectors; h += blockDim.x) {
        const auto* source = reinterpret_cast<const uint4*>(packet + l.packet_x()) + token * vectors + h;
        uint4 value;
        if constexpr (kRemote) value = load_remote_vector(source);
        else value = *source;
        for (int k = 0; k < l.topk; ++k)
            if (slots[k] >= 0)
                reinterpret_cast<uint4*>(a.recv_x)[slots[k] * vectors + h] = value;
    }
    for (int group = threadIdx.x; group < l.scale_groups(); group += blockDim.x) {
        const auto* source = reinterpret_cast<const float*>(packet + l.packet_scales(tokens)) +
                             token * l.scale_groups() + group;
        float value;
        if constexpr (kRemote) value = *reinterpret_cast<const volatile float*>(source);
        else value = *source;
        for (int k = 0; k < l.topk; ++k)
            if (slots[k] >= 0)
                a.recv_scales[slots[k] + group * a.recv_scale_stride1] = value;
    }
}

// Eight adjacent BF16 values share one vector load/store and one route lookup.
struct BF16VectorSum {
    static constexpr int kValues = sizeof(uint4) / sizeof(nv_bfloat16);
    float values[kValues] = {};
    __device__ __forceinline__ void add(uint4 input) {
        const uint32_t words[] = {input.x, input.y, input.z, input.w};
#pragma unroll
        for (int pair = 0; pair < kValues / 2; ++pair) {
            const __nv_bfloat162_raw bits = {static_cast<uint16_t>(words[pair]),
                                            static_cast<uint16_t>(words[pair] >> 16)};
            const float2 value = __bfloat1622float2(__nv_bfloat162(bits));
            values[2 * pair] += value.x;
            values[2 * pair + 1] += value.y;
        }
    }
    __device__ __forceinline__ uint4 pack() const {
        uint32_t words[kValues / 2];
#pragma unroll
        for (int pair = 0; pair < kValues / 2; ++pair) {
            const __nv_bfloat162_raw bits =
                __floats2bfloat162_rn(values[2 * pair], values[2 * pair + 1]);
            words[pair] = uint32_t(bits.x) | (uint32_t(bits.y) << 16);
        }
        return make_uint4(words[0], words[1], words[2], words[3]);
    }
};

template <bool kDirect>
__device__ __forceinline__ void make_partial_vectors(const DeviceArgs& a) {
    const Layout l = a.layout;
    const Control* state = control(a.base);
    constexpr int kMaxTopK = 32;
    __shared__ int rows[kMaxTopK];
    const int vectors = l.hidden / BF16VectorSum::kValues;
    int compact_token = blockIdx.x;
    for (int source = 0; source < l.sources; ++source) {
        const int n = state->received_tokens[source];
        for (; compact_token < n; compact_token += gridDim.x) {
            const int token = kDirect ?
                *reinterpret_cast<const int*>(a.base + l.token_index(source, compact_token)) : compact_token;
            if (threadIdx.x < l.topk)
                rows[threadIdx.x] = a.receipt[(source * a.bucket + token) * l.topk + threadIdx.x];
            __syncthreads();
            for (int h = threadIdx.x; h < vectors; h += blockDim.x) {
                BF16VectorSum sum;
                for (int k = 0; k < l.topk; ++k)
                    if (rows[k] >= 0)
                        sum.add(reinterpret_cast<const uint4*>(a.rows)[size_t(rows[k]) * vectors + h]);
                uint8_t* destination = l.nvlink ?
                    a.peers[source] + l.reply_slot(a.rank) + size_t(token) * l.hidden * sizeof(uint16_t) :
                    a.base + l.partial(source, compact_token);
                reinterpret_cast<uint4*>(destination)[h] = sum.pack();
            }
            __syncthreads();
        }
        compact_token -= n;
    }
}

__device__ __forceinline__ void histogram_routes(const DeviceArgs& a) {
    const Layout l = a.layout;
    const Control* state = control(a.base);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int worker = a.rank;
    int i = tid;
    for (int source = 0; source < l.sources; ++source) {
        const int n = state->received_tokens[source];
        const uint8_t* packet = a.peers[source] + l.request;
        for (; i < n * l.topk; i += stride) {
            const int64_t expert = reinterpret_cast<const volatile int64_t*>(packet + l.packet_ids(n))[i];
            if (expert < -1 || expert >= l.experts) fail("invalid real expert", a);
            if (expert >= worker * l.local_experts() && expert < (worker + 1) * l.local_experts())
                atomicAdd(a.counts + expert - worker * l.local_experts(), 1);
        }
        i -= n * l.topk;
    }
}

__device__ __forceinline__ void prefix_nvlink(const DeviceArgs& a) {
    const Layout l = a.layout;
    // start[e] = sum(round_up(count[i], alignment), i < e).
    // Parallelize expert offsets; preserve the original unpadded ends.
    using Scan = cub::BlockScan<int, kThreadsPerBlock>;
    __shared__ Scan::TempStorage storage;
    __shared__ int base;
    if (threadIdx.x == 0) base = 0;
    __syncthreads();
    for (int tile = 0; tile < l.local_experts(); tile += blockDim.x) {
        const int e = tile + threadIdx.x;
        const int count = e < l.local_experts() ? a.counts[e] : 0;
        const int rounded = (count + l.alignment - 1) / l.alignment * l.alignment;
        int offset, total;
        Scan(storage).ExclusiveSum(rounded, offset, total);
        if (e < l.local_experts()) {
            a.starts[e] = base + offset;
            a.psum[e] = base + offset + count;
            a.cursors[e] = 0;
        }
        __syncthreads();
        if (threadIdx.x == 0) base += total;
        __syncthreads();
    }
}

// DeepEP's warp-per-token mapping, adapted to FFN-only receipts and source
// reply slots. Warp shuffles retain the original top-k reduction order.
template <int kHidden>
__device__ __forceinline__ void nvlink_warp_partials(const DeviceArgs& a) {
    extern __shared__ __align__(32) uint8_t reply_staging[];
    constexpr int kWarp = kWarpSize, kTopK = kNVLinkWarpTopK;
    constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(uint4);
    const Layout l = a.layout;
    const int lane = threadIdx.x % kWarp;
    auto* tile = reinterpret_cast<uint4*>(reply_staging) + (threadIdx.x / kWarp) * kHiddenVec;
    int token = (threadIdx.x / kWarp) * gridDim.x + blockIdx.x;
    const int stride = (blockDim.x / kWarp) * gridDim.x;
    for (int source = 0; source < l.sources; ++source) {
        const int n = control(a.base)->received_tokens[source];
        for (; token < n; token += stride) {
            const int row = lane < kTopK ? a.receipt[(source * a.bucket + token) * kTopK + lane] : -1;
            unsigned mask = __ballot_sync(0xffffffff, row >= 0);
            int slots[kTopK];
#pragma unroll
            for (int k = 0; k < kTopK; ++k) {
                const int selected = __ffs(mask) - 1;
                const int value = __shfl_sync(0xffffffff, row, max(selected, 0));
                slots[k] = selected >= 0 ? value : -1;
                mask &= mask - 1;
            }
            // Match DeepEP's reduce -> shared tile -> peer TMA store sequence.
            // The previous store must stop reading before this tile is reused.
            if constexpr (kHidden != kNVLinkWideHidden) {
                tma::tma_store_wait();
                __syncwarp();
            }
            deepep_warp_combine<kHiddenVec>(lane, slots, tile,
                [&](int selected, int h) {
                    return load_remote_vector(reinterpret_cast<const uint4*>(a.rows) +
                                              size_t(selected) * kHiddenVec + h);
                },
                []() {
                    // Register loads/reduction may overlap the prior tile store.
                    // The first shared write still waits for every issuer.
                    if constexpr (kHidden == kNVLinkWideHidden) {
                        tma::tma_store_wait();
                        __syncwarp();
                    }
                });
            tma::tma_store_fence();
            __syncwarp();
            if (lane == 0) {
                tma::tma_store_1d(a.peers[source] + l.reply_slot(a.rank) +
                                    size_t(token) * kHidden * sizeof(nv_bfloat16),
                                  tile, kHidden * sizeof(nv_bfloat16));
                tma::tma_store_commit();
            }
            __syncwarp();
        }
        token -= n;
    }
    tma::tma_store_wait();
    __syncwarp();
}

template <int kHidden>
__device__ __forceinline__ void nvlink_warp_collect(const DeviceArgs& a) {
    constexpr int kWarp = kWarpSize, kTopK = kNVLinkWarpTopK, kWorkers = 2;
    constexpr int kHiddenVec = kHidden * sizeof(nv_bfloat16) / sizeof(uint4);
    const Layout l = a.layout;
    Control* state = control(a.base);
    if (blockIdx.x == 0 && threadIdx.x == 0) state->lifecycle = state->epoch;
    const int lane = threadIdx.x % kWarp;
    const auto* ids = reinterpret_cast<const int64_t*>(a.base + l.request + l.packet_ids(state->source_tokens));
    const int warp = (threadIdx.x / kWarp) * gridDim.x + blockIdx.x;
    const int stride = (blockDim.x / kWarp) * gridDim.x;
    for (int token = warp; token < state->source_tokens; token += stride) {
        const int64_t expert = lane < kTopK ? ids[token * kTopK + lane] : -1;
        int slots[kWorkers];
#pragma unroll
        for (int worker = 0; worker < kWorkers; ++worker) {
            const bool selected = expert >= 0 && expert / l.local_experts() == worker;
            slots[worker] = __ballot_sync(0xffffffff, selected) ? worker : -1;
        }
        deepep_warp_combine<kHiddenVec>(lane, slots,
            reinterpret_cast<uint4*>(a.out) + size_t(token) * kHiddenVec,
            [&](int worker, int h) {
                return load_remote_vector(a.base + l.reply_slot(worker) +
                                          (size_t(token) * kHiddenVec + h) * sizeof(uint4));
            });
    }
}

// Adapted from DeepEP's dispatch_copy_epilogue_impl: one warp owns one
// staging buffer; selected expert lanes issue asynchronous fan-out stores.
__device__ __forceinline__ void nvlink_tma_pack(const DeviceArgs& a) {
    extern __shared__ __align__(32) uint8_t staging[];
    const int kWarps = blockDim.x / kWarpSize;
    const Layout l = a.layout;
    const int lane = threadIdx.x % kWarpSize, warp = threadIdx.x / kWarpSize;
    const int bytes = l.hidden * l.element_bytes();
    uint8_t* tile = staging + warp * bytes;
    auto* barrier = reinterpret_cast<tma::mbarrier*>(staging + kWarps * bytes) + warp;
    if (lane == 0) tma::mbarrier_init_with_fence(barrier, 1);
    __syncwarp();
    tma::arrival_phase phase = 0;
    int token = warp * gridDim.x + blockIdx.x;
    const int stride = kWarps * gridDim.x;
    const Control* state = control(a.base);
    for (int source = 0; source < l.sources; ++source) {
        const int n = state->received_tokens[source];
        const uint8_t* packet = a.peers[source] + l.request;
        for (; token < a.bucket; token += stride) {
            const int64_t expert = token < n && lane < l.topk ?
                reinterpret_cast<const volatile int64_t*>(packet + l.packet_ids(n))[token * l.topk + lane] : -1;
            const bool owned = expert >= a.rank * l.local_experts() && expert < (a.rank + 1) * l.local_experts();
            const unsigned mask = __ballot_sync(0xffffffff, owned);
            if (mask) {
                // Every issuing lane owns its TMA store group. All lanes must
                // drain before lane zero overwrites the shared tile.
                tma::tma_store_wait();
                __syncwarp();
                if (lane == 0) {
                    tma::tma_load_1d(tile, packet + l.packet_x() + size_t(token) * bytes, barrier, bytes);
                    tma::mbarrier_arrive_and_set_tx(barrier, bytes);
                }
            }
            int row = -1;
            if (owned) {
                const int e = expert - a.rank * l.local_experts();
                row = a.starts[e] + atomicAdd(a.cursors + e, 1);
                const int max_rows = l.sources * a.bucket * min(l.topk, l.local_experts()) +
                                     l.local_experts() * (l.alignment - 1);
                if (row < 0 || row >= max_rows) fail("expert row overflow", a);
                a.recv_weights[row] = reinterpret_cast<const volatile float*>(packet + l.packet_weights(n))[token * l.topk + lane];
            }
            if (lane < l.topk) a.receipt[(source * a.bucket + token) * l.topk + lane] = row;
            __syncwarp();
            if (mask) {
                if (lane == 0) tma::mbarrier_wait_and_flip_phase(barrier, phase);
                __syncwarp();
                if (row >= 0) {
                    tma::tma_store_1d(a.recv_x + size_t(row) * bytes, tile, bytes);
                    tma::tma_store_commit();
                }
                __syncwarp();
                if (l.fp8) {
                    float scale = 0;
                    if (lane < l.scale_groups())
                        scale = reinterpret_cast<const volatile float*>(packet + l.packet_scales(n))[token * l.scale_groups() + lane];
                    for (unsigned remaining = mask; remaining; remaining &= remaining - 1) {
                        const int selected = __ffs(remaining) - 1;
                        const int target = __shfl_sync(0xffffffff, row, selected);
                        if (lane < l.scale_groups()) a.recv_scales[target + lane * a.recv_scale_stride1] = scale;
                    }
                }
            }
        }
        token -= a.bucket;
    }
    tma::tma_store_wait();
    __syncwarp();
}

__device__ __forceinline__ NVLinkControl* nvlink_control(uint8_t* base) {
    return reinterpret_cast<NVLinkControl*>(base);
}

// Shared-domain directional protocol. The role-specific peer array maps a
// worker to source packets, and a source to worker-local reply partials.
// Kernel boundaries complete all copy/reduction CTAs before publishing credits.
template <int kOperation>
__device__ __forceinline__ void nvlink_operation(const DeviceArgs& a) {
    const Layout l = a.layout;
    Control* state = control(a.base);
    NVLinkControl* local = nvlink_control(a.base);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr (kOperation == BeginSource) {
        if (tid == 0) {
            const uint64_t next = state->epoch + 1;
            if (state->lifecycle != state->epoch) fail("NVLink requires complete source pairs", a);
            // Reply readiness from EVERY worker also acknowledges its request
            // read. The preceding complete collect owns the packet until here.
            for (int worker = 0; worker < l.workers; ++worker)
                wait_local(&nvlink_control(a.peers[worker])->reply_ready, next - 1, a);
            state->epoch = next;
            state->source_tokens = live_source_tokens(a);
        }
        if (a.bucket > kNVLinkFusedMetadataTokens) {
            __syncthreads();
            auto* histogram = reinterpret_cast<int*>(a.base + l.request +
                                                     l.packet_histogram(state->source_tokens));
            for (int e = threadIdx.x; e < l.experts; e += blockDim.x) histogram[e] = 0;
        }
    } else if constexpr (kOperation == SendRequest) {
        if (tid == 0) release(&state->request_ready, state->epoch);
    } else if constexpr (kOperation == BeginWorker) {
        for (int e = threadIdx.x; e < l.local_experts(); e += blockDim.x)
            a.counts[e] = 0;
        if (tid == 0) {
            if (local->reply_ready != state->epoch) fail("NVLink requires complete worker pairs", a);
            ++state->epoch;
            for (int source = 0; source < l.sources; ++source) {
                // Request N from EVERY source follows its entire collect N-1
                // on an ordered stream, acknowledging all previous reply reads.
                wait_local(&control(a.peers[source])->request_ready, state->epoch, a);
                const int n = packet_tokens(a.peers[source], l);
                if (n < 0 || n > a.bucket) fail("NVLink source tokens exceed bucket", a);
                state->received_tokens[source] = n;
            }
        }
        __syncthreads();
        if (a.bucket <= kNVLinkFusedMetadataTokens) {
            histogram_routes(a);
        } else {
            for (int e = threadIdx.x; e < l.local_experts(); e += blockDim.x) {
                int count = 0;
                for (int source = 0; source < l.sources; ++source) {
                    const auto* histogram = reinterpret_cast<const volatile int*>(
                        a.peers[source] + l.request + l.packet_histogram(state->received_tokens[source]));
                    count += histogram[a.rank * l.local_experts() + e];
                }
                a.counts[e] = count;
            }
        }
        __syncthreads();
        prefix_nvlink(a);
    } else if constexpr (kOperation == MakePartials) {
        make_partial_vectors<false>(a);
    } else if constexpr (kOperation == SendReply) {
        if (tid == 0) release(&local->reply_ready, state->epoch);
    } else if constexpr (kOperation == WaitReply) {
        if (tid == 0)
            for (int worker = 0; worker < l.workers; ++worker)
                wait_local(&nvlink_control(a.peers[worker])->reply_ready, state->epoch, a);
    } else if constexpr (kOperation == CollectReply) {
        if (threadIdx.x == 0)
            for (int worker = 0; worker < l.workers; ++worker)
                wait_local(&nvlink_control(a.peers[worker])->reply_ready, state->epoch, a);
        __syncthreads();
        if (tid == 0) state->lifecycle = state->epoch;
        const int vectors = l.hidden / BF16VectorSum::kValues;
        for (int token = blockIdx.x; token < state->source_tokens; token += gridDim.x)
            for (int h = threadIdx.x; h < vectors; h += blockDim.x) {
                // This backend has exactly two FFN workers. Both always
                // publish partials, including zeros for unselected routes.
                const size_t offset = (size_t(token) * vectors + h) * sizeof(uint4);
                const uint4 first = load_remote_vector(a.base + l.reply_slot(0) + offset);
                const uint4 second = load_remote_vector(a.base + l.reply_slot(1) + offset);
                reinterpret_cast<uint4*>(a.out)[size_t(token) * vectors + h] =
                    deepep_add_bf16_vectors(first, second);
            }
    }
}

// One final strong signal per active context. Each receiver waits every one;
// a signal on context zero cannot order transfers issued on other contexts.
__device__ __forceinline__ void transfer(const DeviceArgs& a, size_t src, size_t dst,
                                       size_t bytes, int peer, int signal, bool direct = false) {
    if (threadIdx.x != 0) return;
    const int q = blockIdx.x % a.stripes;
    const size_t units = (bytes + 15) / 16;
    const size_t begin = (units * q / a.stripes) * 16;
    const size_t end = min((units * (q + 1) / a.stripes) * 16, bytes);
    // Direct mode has one CTA per destination and stripe; destinations share
    // the stripe's context and therefore require device-wide resource sharing.
    ncclGin gin(a.comm, q, direct ? NCCL_GIN_RESOURCE_SHARING_GPU : NCCL_GIN_RESOURCE_SHARING_CTA);
    const auto action = ncclGin_StrongSignalInc{static_cast<ncclGinSignal_t>(signal)};
    if (end > begin) {
        gin.put(direct ? ncclTeamWorld(a.comm) : ncclTeamRail(a.comm), peer,
                a.window, dst + begin, a.window, src + begin, end - begin,
                action, ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                cuda::thread_scope_thread, cuda::thread_scope_device,
                ncclGinOptFlagsDefault);
    } else {
        // Empty sources still advance their phase so workers cannot deadlock.
        gin.signal(direct ? ncclTeamWorld(a.comm) : ncclTeamRail(a.comm), peer, action);
    }
    // The strong signal publishes remote readiness. RequestCredit/ReplyCredit
    // prevent this packet's source storage from being overwritten until every
    // receiver has consumed it, which also proves local DMA consumption.
    // Credits carry no reusable source payload. Drain active contexts at close,
    // rather than flushing each payload and credit on the execution path.
}

__device__ __forceinline__ int direct_signal(int phase, int peer) { return phase * 4 + peer; }

// Direct requests contain one activation per selected worker, even if several
// selected experts live there. All peers still exchange phase signals: an empty
// message is completion, not permission to skip an epoch or buffer-reuse credit.
// Worker loops carry residual grid-stride indices across actual-length source
// segments, retaining parallelism without scanning inactive bucket rows.
template <int kOperation>
__device__ __forceinline__ void direct_operation(const DeviceArgs& a) {
    const Layout l = a.layout;
    Control* state = control(a.base);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int worker = a.rank;

    if constexpr (kOperation == BeginSource) {
        if (tid == 0) {
            const uint64_t next = state->epoch + 1;
            for (int peer = 0; peer < 4; ++peer) {
                wait_signal(0, direct_signal(RequestCredit, peer), next - 1, a);
                state->received_tokens[peer] = 0;
            }
            state->epoch = next;
            state->source_tokens = live_source_tokens(a);
        }
    } else if constexpr (kOperation == PlanDirectRequests) {
        for (int token = tid; token < state->source_tokens; token += stride) {
            unsigned workers = 0;
            for (int k = 0; k < l.topk; ++k) {
                const int64_t expert = a.ids[token * l.topk + k];
                if (expert < -1 || expert >= l.experts) fail("invalid real expert", a);
                for (int previous = 0; expert >= 0 && previous < k; ++previous)
                    if (a.ids[token * l.topk + previous] == expert)
                        fail("duplicate expert for one token", a);
                if (expert >= 0) workers |= 1u << (expert / l.local_experts());
            }
            for (int peer = 0; peer < 4; ++peer)
                *reinterpret_cast<int*>(a.base + l.token_index(peer, token)) =
                    workers & (1u << peer) ? atomicAdd(state->received_tokens + peer, 1) : -1;
        }
    } else if constexpr (kOperation == PackSource) {
        const int source_tokens = state->source_tokens;
        if (tid < 4) {
            auto* packet = reinterpret_cast<Packet*>(a.base + l.request_slot(tid));
            packet->tokens = state->received_tokens[tid];
            packet->reserved = source_tokens;
        }
        for (int index = blockIdx.x; index < 4 * source_tokens; index += gridDim.x) {
            const int peer = index / source_tokens, token = index % source_tokens;
            const int slot = *reinterpret_cast<const int*>(a.base + l.token_index(peer, token));
            if (slot < 0) continue;
            const int n = state->received_tokens[peer];
            uint8_t* packet = a.base + l.request_slot(peer);
            if (threadIdx.x == 0)
                reinterpret_cast<int*>(packet + l.packet_bytes(n))[slot] = token;
            const int vectors = l.hidden * l.element_bytes() / 16;
            for (int h = threadIdx.x; h < vectors; h += blockDim.x)
                reinterpret_cast<uint4*>(packet + l.packet_x())[slot * vectors + h] =
                    reinterpret_cast<const uint4*>(a.x)[token * vectors + h];
            for (int k = threadIdx.x; k < l.topk; k += blockDim.x) {
                reinterpret_cast<int64_t*>(packet + l.packet_ids(n))[slot * l.topk + k] =
                    a.ids[token * l.topk + k];
                reinterpret_cast<float*>(packet + l.packet_weights(n))[slot * l.topk + k] =
                    a.weights[token * l.topk + k];
            }
            for (int group = threadIdx.x; group < l.scale_groups(); group += blockDim.x)
                reinterpret_cast<float*>(packet + l.packet_scales(n))[slot * l.scale_groups() + group] =
                    a.scales[token * a.scale_stride0 + group * a.scale_stride1];
        }
    } else if constexpr (kOperation == SendRequest) {
        const int peer = blockIdx.x / a.stripes;
        const int source = a.rank - 4;
        transfer(a, l.request_slot(peer), l.request_slot(source),
                 l.direct_packet_bytes(state->received_tokens[peer]), peer,
                 direct_signal(RequestReady, source), true);
    } else if constexpr (kOperation == BeginWorker) {
        if (tid == 0) {
            ++state->epoch;
            // Reply payloads are posted asynchronously. Protect their local
            // partial storage before this epoch can write it, not just the
            // remote inbox before the next SendReply.
            for (int source = 0; source < 4; ++source)
                wait_signal(0, direct_signal(ReplyCredit, source), state->epoch - 1, a);
            for (int source = 0; source < 4; ++source) {
                for (int q = 0; q < a.stripes; ++q)
                    wait_signal(q, direct_signal(RequestReady, source), state->epoch, a);
                const auto* packet = reinterpret_cast<const volatile Packet*>(
                    a.base + l.request_slot(source));
                if (packet->tokens > uint64_t(a.bucket) || packet->reserved > uint64_t(a.bucket) ||
                    packet->tokens > packet->reserved)
                    fail("direct source tokens exceed bucket", a);
                state->received_tokens[source] = static_cast<int>(packet->tokens);
            }
        }
    } else if constexpr (kOperation == Histogram) {
        int i = tid;
        for (int source = 0; source < 4; ++source) {
            const int n = state->received_tokens[source];
            const uint8_t* packet = a.base + l.request_slot(source);
            for (; i < n * l.topk; i += stride) {
                const int64_t expert = reinterpret_cast<const int64_t*>(packet + l.packet_ids(n))[i];
                if (expert < -1 || expert >= l.experts) fail("invalid real expert", a);
                if (expert >= worker * l.local_experts() && expert < (worker + 1) * l.local_experts())
                    atomicAdd(a.counts + expert - worker * l.local_experts(), 1);
            }
            i -= n * l.topk;
        }
    } else if constexpr (kOperation == PackRows) {
        __shared__ int slots[32];
        int compact_token = blockIdx.x;
        for (int source = 0; source < 4; ++source) {
            const int n = state->received_tokens[source];
            const uint8_t* packet = a.base + l.request_slot(source);
            for (; compact_token < n; compact_token += gridDim.x) {
                const int token = reinterpret_cast<const int*>(packet + l.packet_bytes(n))[compact_token];
                if (token < 0 || uint64_t(token) >= reinterpret_cast<const Packet*>(packet)->reserved)
                    fail("direct original token outside source extent", a);
                if (threadIdx.x == 0)
                    *reinterpret_cast<int*>(a.base + l.token_index(source, compact_token)) = token;
                if (threadIdx.x < l.topk) {
                    const int k = threadIdx.x;
                    const int64_t expert = reinterpret_cast<const int64_t*>(packet + l.packet_ids(n))[compact_token * l.topk + k];
                    int row = -1;
                    if (expert >= worker * l.local_experts() && expert < (worker + 1) * l.local_experts()) {
                        const int e = expert - worker * l.local_experts();
                        row = a.starts[e] + atomicAdd(a.cursors + e, 1);
                        const int max_rows = 4 * a.bucket * min(l.topk, l.local_experts()) +
                                             l.local_experts() * (l.alignment - 1);
                        if (row < 0 || row >= max_rows) fail("expert row overflow", a);
                        a.recv_weights[row] = reinterpret_cast<const float*>(packet + l.packet_weights(n))[compact_token * l.topk + k];
                        a.receipt[(source * a.bucket + token) * l.topk + k] = row;
                    }
                    slots[k] = row;
                }
                __syncthreads();
                copy_expert_rows<false>(a, packet, n, compact_token, slots);
                __syncthreads();
            }
            compact_token -= n;
        }
    } else if constexpr (kOperation == PublishPackAndCredit) {
        if (tid == 0) {
            // All request data needed by reply generation has been copied into
            // worker-owned maps and receipts before sources may reuse inboxes.
            for (int source = 0; source < 4; ++source)
                send_credit(direct_signal(RequestCredit, worker), source + 4, a, true);
        }
    } else if constexpr (kOperation == MakePartials) {
        make_partial_vectors<true>(a);
    } else if constexpr (kOperation == PublishConsumedAndWaitCredit) {
        if (tid == 0)
            for (int source = 0; source < 4; ++source)
                wait_signal(0, direct_signal(ReplyCredit, source), state->epoch - 1, a);
    } else if constexpr (kOperation == SendReply) {
        const int source = blockIdx.x / a.stripes;
        transfer(a, l.partial(source, 0), l.reply_slot(worker),
                 size_t(state->received_tokens[source]) * l.hidden * sizeof(uint16_t), source + 4,
                 direct_signal(ReplyReady, worker), true);
    } else if constexpr (kOperation == CollectReply) {
        if (threadIdx.x == 0)
            for (int peer = 0; peer < 4; ++peer)
                for (int q = 0; q < a.stripes; ++q)
                    wait_signal(q, direct_signal(ReplyReady, peer), state->epoch, a);
        __syncthreads();
        const int vectors = l.hidden / BF16VectorSum::kValues;
        for (int token = blockIdx.x; token < state->source_tokens; token += gridDim.x)
            for (int h = threadIdx.x; h < vectors; h += blockDim.x) {
                BF16VectorSum sum;
                for (int peer = 0; peer < 4; ++peer) {
                    const int slot = *reinterpret_cast<const int*>(a.base + l.token_index(peer, token));
                    if (slot >= 0)
                        sum.add(load_remote_vector(a.base + l.reply_slot(peer) +
                                                   (size_t(slot) * vectors + h) * sizeof(uint4)));
                }
                reinterpret_cast<uint4*>(a.out)[size_t(token) * vectors + h] = sum.pack();
            }
    } else if constexpr (kOperation == ReplyConsumedCredit) {
        if (tid == 0)
            for (int peer = 0; peer < 4; ++peer)
                send_credit(direct_signal(ReplyCredit, a.rank - 4), peer, a, true);
    }
}

template <bool kPublishCounts>
__device__ __forceinline__ void pack_source_packet(const DeviceArgs& a) {
    const Layout l = a.layout;
    Control* state = control(a.base);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int tokens = state->source_tokens;
    uint8_t* packet = a.base + l.request;
    if (tid == 0) {
        reinterpret_cast<Packet*>(packet)->tokens = tokens;
        reinterpret_cast<Packet*>(packet)->reserved = 0;
    }
    const size_t xbytes = size_t(tokens) * l.hidden * l.element_bytes();
    for (size_t i = tid; i < xbytes / 16; i += stride)
        reinterpret_cast<uint4*>(packet + l.packet_x())[i] =
            reinterpret_cast<const uint4*>(a.x)[i];
    for (int i = tid; i < tokens * l.topk; i += stride) {
        const int64_t expert = a.ids[i];
        if (expert < -1 || expert >= l.experts) fail("invalid real expert", a);
        for (int previous = 0; expert >= 0 && previous < i % l.topk; ++previous)
            if (a.ids[i - i % l.topk + previous] == expert)
                fail("duplicate expert for one token", a);
        reinterpret_cast<int64_t*>(packet + l.packet_ids(tokens))[i] = a.ids[i];
        reinterpret_cast<float*>(packet + l.packet_weights(tokens))[i] = a.weights[i];
        if constexpr (kPublishCounts)
            if (expert >= 0)
                atomicAdd(reinterpret_cast<int*>(packet + l.packet_histogram(tokens)) + expert, 1);
    }
    for (int i = tid; i < tokens * l.scale_groups(); i += stride) {
        const int token = i / l.scale_groups(), group = i % l.scale_groups();
        reinterpret_cast<float*>(packet + l.packet_scales(tokens))[i] =
            a.scales[token * a.scale_stride0 + group * a.scale_stride1];
    }
}

template <int kOperation>
__global__ void directional_kernel(DeviceArgs a) {
    if constexpr (kOperation == PackSourceCounts) {
        pack_source_packet<true>(a);
        return;
    }
    if constexpr (kOperation == TmaPackRows) {
        nvlink_tma_pack(a);
        return;
    }
    if constexpr (kOperation == WarpPartials) {
        nvlink_warp_partials<kNVLinkWarpHidden>(a);
        return;
    }
    if constexpr (kOperation == WarpCollect) {
        nvlink_warp_collect<kNVLinkWarpHidden>(a);
        return;
    }
    if constexpr (kOperation == TmaPackRowsWide) {
        nvlink_tma_pack(a);
        return;
    }
    if constexpr (kOperation == WarpPartialsWide) {
        nvlink_warp_partials<kNVLinkWideHidden>(a);
        return;
    }
    if constexpr (kOperation == WarpCollectWide) {
        nvlink_warp_collect<kNVLinkWideHidden>(a);
        return;
    }
    if constexpr (kOperation == Drain) {
        if (a.layout.nvlink) return;
        if (threadIdx.x == 0) {
            ncclGin gin(a.comm, blockIdx.x, a.layout.hybrid ?
                        NCCL_GIN_RESOURCE_SHARING_CTA : NCCL_GIN_RESOURCE_SHARING_GPU);
            gin.flush(ncclCoopThread());
        }
        return;
    }
    const Layout l = a.layout;
    if (l.nvlink) {
        if constexpr (kOperation == BeginSource || kOperation == SendRequest || kOperation == BeginWorker ||
                      kOperation == PublishPackAndCredit || kOperation == MakePartials ||
                      kOperation == SendReply || kOperation == WaitReply || kOperation == CollectReply || kOperation == ReplyConsumedCredit) {
            nvlink_operation<kOperation>(a);
            return;
        }
    }
    if (!l.nvlink && !l.hybrid && kOperation != Prefix) {
        direct_operation<kOperation>(a);
        return;
    }
    Control* state = control(a.base);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int worker = a.rank;

    if constexpr (kOperation == BeginSource) {
        if (tid == 0) {
            const uint64_t next = state->epoch + 1;
            wait_signal(0, RequestCredit, next - 1, a);
            state->epoch = next;
            state->source_tokens = live_source_tokens(a);
        }
    } else if constexpr (kOperation == PackSource) {
        pack_source_packet<false>(a);
    } else if constexpr (kOperation == SendRequest) {
        transfer(a, l.request, l.request, l.packet_bytes(state->source_tokens), 0, RequestReady);
    } else if constexpr (kOperation == BeginWorker) {
        if (tid == 0) {
            ++state->epoch;
            // The previous reply's local DMA source must be consumed before
            // ReduceReply can overwrite worker_reply in this epoch.
            wait_signal(0, ReplyCredit, state->epoch - 1, a);
            for (int q = 0; q < a.stripes; ++q)
                wait_signal(q, RequestReady, state->epoch, a);
            release(&state->request_ready, state->epoch);
            // One waiter per worker acquires peer readiness and headers.
            // The next kernel inherits this stream dependency; launching a
            // waiter on every histogram CTA floods the peer control lines.
            for (int source = 0; source < 4; ++source) {
                wait_local(&control(a.peers[source])->request_ready, state->epoch, a);
                const int n = packet_tokens(a.peers[source], l);
                if (n < 0 || n > a.bucket) fail("source tokens exceed bucket", a);
                state->received_tokens[source] = n;
            }
        }
    } else if constexpr (kOperation == Histogram) {
        histogram_routes(a);
    } else if constexpr (kOperation == Prefix) {
        if (l.nvlink) {
            prefix_nvlink(a);
        } else if (tid == 0) {
            int offset = 0;
            for (int e = 0; e < l.local_experts(); ++e) {
                offset = (offset + l.alignment - 1) / l.alignment * l.alignment;
                a.starts[e] = offset;
                offset += a.counts[e];
                a.psum[e] = offset;
                a.cursors[e] = 0;
            }
        }
    } else if constexpr (kOperation == PackRows) {
        __shared__ int slots[32];
        int token = blockIdx.x;
        for (int source = 0; source < l.sources; ++source) {
            const int n = state->received_tokens[source];
            const uint8_t* packet = a.peers[source] + l.request;
            const int extent = l.nvlink ? a.bucket : n;
            for (; token < extent; token += gridDim.x) {
                if (threadIdx.x < l.topk) {
                    const int k = threadIdx.x;
                    const int64_t expert = token < n ?
                        reinterpret_cast<const volatile int64_t*>(packet + l.packet_ids(n))[token * l.topk + k] : -1;
                    int row = -1;
                    if (l.nvlink) a.receipt[(source * a.bucket + token) * l.topk + k] = -1;
                    if (expert >= worker * l.local_experts() && expert < (worker + 1) * l.local_experts()) {
                        const int e = expert - worker * l.local_experts();
                        row = a.starts[e] + atomicAdd(a.cursors + e, 1);
                        const int max_rows = l.sources * a.bucket * min(l.topk, l.local_experts()) +
                                             l.local_experts() * (l.alignment - 1);
                        if (row < 0 || row >= max_rows) fail("expert row overflow", a);
                        a.recv_weights[row] = reinterpret_cast<const volatile float*>(packet + l.packet_weights(n))[token * l.topk + k];
                        a.receipt[(source * a.bucket + token) * l.topk + k] = row;
                    }
                    slots[k] = row;
                }
                __syncthreads();
                if (token < n) copy_expert_rows<true>(a, packet, n, token, slots);
                __syncthreads();
            }
            token -= extent;
        }
    } else if constexpr (kOperation == PublishPackAndCredit) {
        if (tid == 0) {
            // A distinct kernel launch follows the entire payload-copy grid.
            release(&state->pack_done, state->epoch);
            for (int i = 0; i < 4; ++i)
                wait_local(&control(a.peers[i])->pack_done, state->epoch, a);
            send_credit(RequestCredit, 1, a);
        }
    } else if constexpr (kOperation == MakePartials) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 4; ++i)
                wait_local(&control(a.peers[i])->partials_consumed, state->epoch - 1, a);
        }
        __syncthreads();
        make_partial_vectors<false>(a);
    } else if constexpr (kOperation == PublishPartialAndWait) {
        if (tid == 0) {
            release(&state->partial_done, state->epoch);
            for (int i = 0; i < 4; ++i)
                wait_local(&control(a.peers[i])->partial_done, state->epoch, a);
        }
    } else if constexpr (kOperation == ReduceReply) {
        const int n = state->received_tokens[worker];
        const int vectors = l.hidden / BF16VectorSum::kValues;
        for (int token = blockIdx.x; token < n; token += gridDim.x)
            for (int h = threadIdx.x; h < vectors; h += blockDim.x) {
                BF16VectorSum sum;
                for (int peer = 0; peer < 4; ++peer)
                    sum.add(load_remote_vector(a.peers[peer] + l.partial(worker, token) + h * sizeof(uint4)));
                reinterpret_cast<uint4*>(a.base + l.worker_reply)[size_t(token) * vectors + h] = sum.pack();
            }
    } else if constexpr (kOperation == PublishConsumedAndWaitCredit) {
        if (tid == 0) {
            // All reduction-grid readers finished before partials may be reused.
            release(&state->partials_consumed, state->epoch);
            wait_signal(0, ReplyCredit, state->epoch - 1, a);
        }
    } else if constexpr (kOperation == SendReply) {
        const size_t bytes = size_t(state->received_tokens[worker]) * l.hidden * sizeof(uint16_t);
        transfer(a, l.worker_reply, l.source_reply, bytes, 1, ReplyReady);
    } else if constexpr (kOperation == CollectReply) {
        if (threadIdx.x == 0) {
            for (int q = 0; q < a.stripes; ++q)
                wait_signal(q, ReplyReady, state->epoch, a);
        }
        __syncthreads();
        const size_t bytes = size_t(state->source_tokens) * l.hidden * sizeof(uint16_t);
        for (size_t i = tid; i < bytes / 16; i += stride)
            reinterpret_cast<uint4*>(a.out)[i] = load_remote_vector(
                a.base + l.source_reply + i * sizeof(uint4));
    } else if constexpr (kOperation == ReplyConsumedCredit) {
        if (tid == 0) send_credit(ReplyCredit, 0, a);
    }
}

}  // namespace afdep
