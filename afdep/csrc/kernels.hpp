// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#pragma once

#include <cuda_runtime_api.h>

#include "layout.hpp"

namespace afdep {

// Every operation launches on its owner's current PyTorch stream. Separate
// kernel boundaries retain the protocol's original grid-completion ordering.
cudaError_t configure_tma_kernel();
cudaError_t launch_kernel(int operation, DeviceArgs args, int blocks, cudaStream_t stream);

} // namespace afdep
