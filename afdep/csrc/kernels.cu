// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#include "kernels.cuh"
#include "kernels.hpp"

namespace afdep {

cudaError_t launch_kernel(int operation, DeviceArgs args, int blocks, cudaStream_t stream) {
    constexpr int kThreadsPerBlock = 256;
#define AFDEP_LAUNCH(operation_name)                                                                         \
    case operation_name:                                                                                     \
        directional_kernel<operation_name><<<blocks, kThreadsPerBlock, 0, stream>>>(args);                   \
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
    default:
        return cudaErrorInvalidValue;
    }
#undef AFDEP_LAUNCH
    return cudaGetLastError();
}

} // namespace afdep
