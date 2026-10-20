// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#include "kernels.cuh"
#include "kernels.hpp"

namespace afdep {

template <bool kFP8>
__global__ void directional_wide_pack(DeviceArgs args) {
    nvlink_tma_pack<kNVLinkWideHidden, kFP8>(args);
}

__global__ void directional_direct_collect(DeviceArgs args) {
    direct_operation<CollectReply, true>(args);
}

cudaError_t get_wide_tma_attributes(cudaFuncAttributes* attributes, bool fp8) {
    return fp8 ? cudaFuncGetAttributes(attributes, directional_wide_pack<true>) :
                 cudaFuncGetAttributes(attributes, directional_wide_pack<false>);
}

cudaError_t configure_tma_kernel(int hidden, int wide_max_shared, bool fp8) {
    // A later FP8 buffer must not lower the per-device function limit needed
    // by a previously captured BF16 graph. Per-launch storage stays exact.
    if (hidden == kNVLinkWideHidden) {
        static_assert(sizeof(tma::mbarrier) == sizeof(uint64_t));
        const auto status = fp8 ?
            cudaFuncSetAttribute(directional_wide_pack<true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, wide_max_shared) :
            cudaFuncSetAttribute(directional_wide_pack<false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, wide_max_shared);
        if (status != cudaSuccess) return status;
        return cudaFuncSetAttribute(directional_kernel<WarpPartialsWide>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (kThreadsPerBlock / kWarpSize) * kNVLinkWideHidden * sizeof(nv_bfloat16));
    }
    if (hidden != kNVLinkWarpHidden) return cudaErrorInvalidValue;
    const int bytes = (kNVLinkTmaThreads / kWarpSize) *
                      (kNVLinkWarpHidden * sizeof(nv_bfloat16) + sizeof(tma::mbarrier));
    return cudaFuncSetAttribute(directional_kernel<TmaPackRows>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

cudaError_t launch_kernel(int operation, DeviceArgs args, int blocks, cudaStream_t stream,
                         int wide_tma_threads) {
    const bool tma_pack = operation == TmaPackRows || operation == TmaPackRowsWide;
    const bool partials = operation == WarpPartials || operation == WarpPartialsWide;
    const int tma_threads = operation == TmaPackRowsWide ? wide_tma_threads : kNVLinkTmaThreads;
    const size_t shared_bytes = tma_pack ?
        (tma_threads / kWarpSize) * (args.layout.hidden * args.layout.element_bytes() + sizeof(tma::mbarrier)) :
        partials ? (kThreadsPerBlock / kWarpSize) * args.layout.hidden * sizeof(nv_bfloat16) : 0;
#define AFDEP_LAUNCH(operation_name)                                                                         \
    case operation_name:                                                                                     \
        directional_kernel<operation_name><<<blocks, kThreadsPerBlock, shared_bytes, stream>>>(args);        \
        break
    switch (operation) {
        AFDEP_LAUNCH(BeginSource);
        AFDEP_LAUNCH(PlanDirectRequests);
        AFDEP_LAUNCH(PackSource);
        AFDEP_LAUNCH(PackSourceCounts);
        AFDEP_LAUNCH(SendRequest);
        AFDEP_LAUNCH(BeginWorker);
        AFDEP_LAUNCH(Histogram);
        AFDEP_LAUNCH(Prefix);
        AFDEP_LAUNCH(PackRows);
        AFDEP_LAUNCH(PublishPackAndCredit);
        AFDEP_LAUNCH(MakePartials);
        AFDEP_LAUNCH(PublishPartialAndWait);
        AFDEP_LAUNCH(ReduceReply);
        AFDEP_LAUNCH(PublishConsumedAndWaitCredit);
        AFDEP_LAUNCH(SendReply);
    case CollectReply:
        if (!args.layout.nvlink && !args.layout.hybrid)
            directional_direct_collect<<<blocks, kThreadsPerBlock, shared_bytes, stream>>>(args);
        else
            directional_kernel<CollectReply><<<blocks, kThreadsPerBlock, shared_bytes, stream>>>(args);
        break;
        AFDEP_LAUNCH(ReplyConsumedCredit);
        AFDEP_LAUNCH(Drain);
        AFDEP_LAUNCH(WaitReply);
        AFDEP_LAUNCH(WarpPartials);
        AFDEP_LAUNCH(WarpCollect);
        AFDEP_LAUNCH(WarpPartialsWide);
        AFDEP_LAUNCH(WarpCollectWide);
    case TmaPackRows:
        directional_kernel<TmaPackRows><<<blocks, kNVLinkTmaThreads, shared_bytes, stream>>>(args);
        break;
    case TmaPackRowsWide:
        if (args.layout.fp8)
            directional_wide_pack<true><<<blocks, wide_tma_threads, shared_bytes, stream>>>(args);
        else
            directional_wide_pack<false><<<blocks, wide_tma_threads, shared_bytes, stream>>>(args);
        break;
    default:
        return cudaErrorInvalidValue;
    }
#undef AFDEP_LAUNCH
    return cudaGetLastError();
}

} // namespace afdep
