// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#pragma once

#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

static_assert(NCCL_VERSION_CODE >= 23007, "AFDEP requires NCCL 2.30.7 or newer");

namespace afdep {

inline void check_cuda_status(cudaError_t code, const char *expression) {
    TORCH_CHECK(code == cudaSuccess, "AFDEP CUDA error: ", cudaGetErrorString(code), " (", expression, ")");
}

inline void check_nccl_status(ncclResult_t code, const char *expression) {
    TORCH_CHECK(code == ncclSuccess, "AFDEP NCCL error: ", ncclGetErrorString(code), " (", expression, ")");
}

} // namespace afdep

// Function arguments are evaluated in the caller's scope: a caller variable
// named `status` (or `code`) cannot be shadowed by a check's internal local.
#define AFDEP_CUDA_CHECK(expression) ::afdep::check_cuda_status((expression), #expression)
#define AFDEP_NCCL_CHECK(expression) ::afdep::check_nccl_status((expression), #expression)
