// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#include "kernels.cuh"
#include "kernels.hpp"

namespace afdep {

cudaError_t configure_tma_kernel() {
    // A later FP8 buffer must not lower the per-device function limit needed
    // by a previously captured BF16 graph. Per-launch storage stays exact.
    const int bytes = (kNVLinkTmaThreads / kWarpSize) *
                      (kNVLinkWarpHidden * sizeof(nv_bfloat16) + sizeof(tma::mbarrier));
    return cudaFuncSetAttribute(directional_kernel<TmaPackRows>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

cudaError_t launch_kernel(int operation, DeviceArgs args, int blocks, cudaStream_t stream) {
    const size_t shared_bytes = operation == TmaPackRows ?
        (kNVLinkTmaThreads / kWarpSize) * (args.layout.hidden * args.layout.element_bytes() + sizeof(tma::mbarrier)) :
        operation == WarpPartials ? (kThreadsPerBlock / kWarpSize) * args.layout.hidden * sizeof(nv_bfloat16) : 0;
#define AFDEP_LAUNCH(operation_name)                                                                         \
    case operation_name:                                                                                     \
        directional_kernel<operation_name><<<blocks, kThreadsPerBlock, shared_bytes, stream>>>(args);        \
        break
    switch (operation) {
        AFDEP_LAUNCH(BeginSource);
        AFDEP_LAUNCH(PlanDirectRequests);
        AFDEP_LAUNCH(PackSource);
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
        AFDEP_LAUNCH(CollectReply);
        AFDEP_LAUNCH(ReplyConsumedCredit);
        AFDEP_LAUNCH(Drain);
        AFDEP_LAUNCH(WaitReply);
        AFDEP_LAUNCH(WarpPartials);
        AFDEP_LAUNCH(WarpCollect);
    case TmaPackRows:
        directional_kernel<TmaPackRows><<<blocks, kNVLinkTmaThreads, shared_bytes, stream>>>(args);
        break;
    default:
        return cudaErrorInvalidValue;
    }
#undef AFDEP_LAUNCH
    return cudaGetLastError();
}

} // namespace afdep
