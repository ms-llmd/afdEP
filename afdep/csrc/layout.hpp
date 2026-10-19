// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#pragma once

#include <cstdint>
#include <cstddef>
#include <nccl.h>
#include <nccl_device.h>

#ifdef __CUDACC__
#define AFDEP_HD __host__ __device__
#else
#define AFDEP_HD
#endif

namespace afdep {

constexpr int kThreadsPerBlock = 256;
constexpr int kPayloadVectorBytes = 16;
constexpr int kNVLinkBulkGridMultiplier = 8;
constexpr int kNVLinkFusedMetadataTokens = 32;
constexpr int kNVLinkWarpMinTokens = 2048;
constexpr int kNVLinkWarpHidden = 2048;
constexpr int kNVLinkWideHidden = 4096;
constexpr int kNVLinkWarpTopK = 8;
constexpr int kWarpSize = 32;
constexpr int kNVLinkTmaThreads = 1024;
constexpr int kNVLinkTmaWideThreads = 512;
constexpr int kNVLinkTmaMaxWarps = 32;
constexpr int kNVLinkWarpGridMultiplier = 4;

// This protocol is deliberately independent of DeepEP expert/world layouts.
// Hybrid rails join source j to ingress j; direct mode addresses all workers.
struct Control {
    uint64_t epoch;
    uint64_t request_ready;
    uint64_t pack_done;
    uint64_t partial_done;
    uint64_t partials_consumed;
    uint64_t lifecycle;
    int source_tokens;
    int received_tokens[4];
};

enum class NVLinkReplyMode : uint64_t { AllWorkerPartials = 0, SelectedWorkerPartials = 1 };

struct NVLinkControl {
    Control state;
    // Each worker's reply acknowledges its request reads. Each source's
    // next request acknowledges its previous reply reads. Ordered full pairs
    // retain one logical credit per reader without separate credit stores.
    uint64_t reply_ready;
    // Current-epoch source capabilities, read after RequestReady acquisition.
    uint32_t sparse_collectors;
};
static_assert(sizeof(NVLinkControl) <= 128, "NVLink control must fit the registered prefix");

struct Packet {
    uint64_t tokens;
    // GIN direct: original source extent; GIN hybrid: zero.
    // NVLink: current source collector mode, owned until request-read acknowledgement.
    uint64_t reserved;
};

struct Layout {
    int capacity, hidden, topk, experts, alignment, fp8, hybrid;
    int sources, workers, nvlink;
    size_t request, partials, source_reply, worker_reply;
    size_t source_bytes, worker_bytes;
    size_t request_stride, reply_stride, token_map;

    AFDEP_HD static size_t aligned(size_t n) { return (n + 127) / 128 * 128; }
    AFDEP_HD bool nvlink_bulk_shape(int tokens) const {
        return nvlink && (hidden == kNVLinkWarpHidden || hidden == kNVLinkWideHidden) &&
               topk == kNVLinkWarpTopK &&
               tokens >= kNVLinkWarpMinTokens;
    }
    AFDEP_HD int local_experts() const { return experts / workers; }
    AFDEP_HD int element_bytes() const { return fp8 ? 1 : 2; }
    AFDEP_HD int scale_groups() const { return fp8 ? hidden / 128 : 0; }
    AFDEP_HD size_t packet_x() const { return aligned(sizeof(Packet)); }
    AFDEP_HD size_t packet_scales(int n) const {
        return aligned(packet_x() + size_t(n) * hidden * element_bytes());
    }
    AFDEP_HD size_t packet_ids(int n) const {
        return aligned(packet_scales(n) + size_t(n) * scale_groups() * sizeof(float));
    }
    AFDEP_HD size_t packet_weights(int n) const {
        return aligned(packet_ids(n) + size_t(n) * topk * sizeof(int64_t));
    }
    AFDEP_HD size_t packet_histogram(int n) const {
        return aligned(packet_weights(n) + size_t(n) * topk * sizeof(float));
    }
    AFDEP_HD size_t packet_bytes(int n) const {
        return packet_histogram(n) + (nvlink ? aligned(size_t(experts) * sizeof(int)) : 0);
    }
    AFDEP_HD size_t direct_packet_bytes(int n) const {
        return aligned(packet_bytes(n) + size_t(n) * sizeof(int));
    }
    AFDEP_HD size_t request_slot(int peer) const { return request + size_t(peer) * request_stride; }
    AFDEP_HD size_t reply_slot(int peer) const { return source_reply + size_t(peer) * reply_stride; }
    AFDEP_HD size_t token_index(int peer, int token) const {
        return token_map + (size_t(peer) * capacity + token) * sizeof(int);
    }
    AFDEP_HD size_t partial(int source, int token) const {
        return partials + (size_t(source) * capacity + token) * hidden * sizeof(uint16_t);
    }

    static Layout make(int capacity, int hidden, int topk, int experts,
                       int alignment, bool fp8, bool hybrid = true,
                       int workers = 4, int sources = 4, bool nvlink = false) {
        Layout l{};
        l.capacity = capacity; l.hidden = hidden; l.topk = topk;
        l.experts = experts; l.alignment = alignment; l.fp8 = fp8; l.hybrid = hybrid;
        l.sources = sources; l.workers = workers; l.nvlink = nvlink;
        l.request = aligned(nvlink ? sizeof(NVLinkControl) : sizeof(Control));
        if (nvlink) {
            // Sources own request packets and one incoming reply slot/worker.
            // Workers push completed partials; only their control is registered.
            l.reply_stride = size_t(capacity) * hidden * sizeof(uint16_t);
            l.source_reply = aligned(l.request + l.packet_bytes(capacity));
            l.source_bytes = aligned(l.source_reply + size_t(workers) * l.reply_stride);
            l.worker_bytes = l.request;
            return l;
        }
        if (!hybrid) {
            l.request_stride = l.direct_packet_bytes(capacity);
            l.reply_stride = size_t(capacity) * hidden * sizeof(uint16_t);
            l.source_reply = l.partials = aligned(l.request + 4 * l.request_stride);
            l.token_map = aligned(l.source_reply + 4 * l.reply_stride);
            l.source_bytes = l.worker_bytes = aligned(l.token_map + size_t(4) * capacity * sizeof(int));
            return l;
        }
        const size_t request_end = aligned(l.request + l.packet_bytes(capacity));
        const size_t reply_bytes = size_t(capacity) * hidden * sizeof(uint16_t);
        l.source_reply = request_end;
        l.source_bytes = aligned(l.source_reply + reply_bytes);
        l.partials = request_end;
        l.worker_reply = aligned(l.partials + 4 * reply_bytes);
        l.worker_bytes = aligned(l.worker_reply + reply_bytes);
        return l;
    }
};

enum Operation {
    BeginSource, PlanDirectRequests, PackSource, SendRequest, BeginWorker, Histogram, Prefix,
    PackRows, PublishPackAndCredit, MakePartials, PublishPartialAndWait,
    ReduceReply, PublishConsumedAndWaitCredit, SendReply, CollectReply,
    ReplyConsumedCredit, Drain, WaitReply, WarpPartials, WarpCollect, TmaPackRows, PackSourceCounts,
    WarpPartialsWide, WarpCollectWide, TmaPackRowsWide, NumOperations
};

enum Signal { RequestReady, ReplyReady, RequestCredit, ReplyCredit };
constexpr int kDirectSignalCount = 4 * 4;

// Passed by value to CUDA kernels. Peer pointers map role-specific NVLink
// peers. Network operations use explicit windows and byte offsets.
struct DeviceArgs {
    Layout layout;
    ncclDevComm_t comm;
    ncclWindow_t window;
    uint8_t* base;
    uint8_t* peers[4];
    const uint8_t* x;
    const float* scales;
    const int64_t* ids;
    const float* weights;
    const int* live_tokens;
    uint8_t* recv_x;
    float* recv_scales;
    float* recv_weights;
    int* counts;
    int* starts;
    int* cursors;
    int* psum;
    int* receipt;
    const uint16_t* rows;
    uint16_t* out;
    int rank, bucket, tokens, stripes;
    int64_t scale_stride0, scale_stride1, recv_scale_stride1;
    uint64_t timeout_cycles;
};

}  // namespace afdep

#undef AFDEP_HD
