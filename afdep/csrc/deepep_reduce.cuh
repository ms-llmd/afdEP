// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Copyright contributors to the afdEP project
// Adapted from the pinned DeepEP V2 impls/combine_utils.cuh. See ORIGIN.json.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace afdep {

struct NoStagingWait {
    __device__ __forceinline__ void operator()() const {}
};

// Reuse the upstream two-input BF16 bypass for CTA-per-token collectors too.
__device__ __forceinline__ uint4 deepep_add_bf16_vectors(uint4 first, const uint4 second) {
    auto* a = reinterpret_cast<nv_bfloat162*>(&first);
    const auto* b = reinterpret_cast<const nv_bfloat162*>(&second);
#pragma unroll
    for (int pair = 0; pair < sizeof(uint4) / sizeof(nv_bfloat162); ++pair)
        a[pair] = __hadd2(a[pair], b[pair]);
    return first;
}

// DeepEP's four-vector unroll and two-input BF16 bypass, without bias or TMA
// output staging. The caller supplies coherent loads from directional buffers.
template <int kHiddenVec, int kSlots, class Load, class Wait = NoStagingWait>
__device__ __forceinline__ void deepep_warp_combine(
    int lane, const int (&slots)[kSlots], uint4* destination, const Load& load,
    const Wait& wait = Wait{}) {
    constexpr int kWarp = 32;
    constexpr int kUnroll = 4;
    constexpr int kPairs = sizeof(uint4) / sizeof(nv_bfloat162);
    static_assert(kHiddenVec % (kWarp * kUnroll) == 0);
    bool pair = true;
    if constexpr (kSlots > 2) pair = slots[2] < 0;
    for (int tile = 0; tile < kHiddenVec; tile += kWarp * kUnroll) {
        if (pair) {
            uint4 first[kUnroll] = {}, second[kUnroll] = {};
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                const int h = tile + j * kWarp + lane;
                if (slots[0] >= 0) first[j] = load(slots[0], h);
                if (slots[1] >= 0) second[j] = load(slots[1], h);
            }
            if (tile == 0) wait();
#pragma unroll
            for (int j = 0; j < kUnroll; ++j)
                destination[tile + j * kWarp + lane] = deepep_add_bf16_vectors(first[j], second[j]);
        } else {
            float2 reduced[kUnroll * kPairs] = {};
#pragma unroll
            for (int k = 0; k < kSlots; ++k) {
                if (slots[k] < 0) break;
                uint4 values[kUnroll];
#pragma unroll
                for (int j = 0; j < kUnroll; ++j)
                    values[j] = load(slots[k], tile + j * kWarp + lane);
                const auto* view = reinterpret_cast<const nv_bfloat162*>(values);
#pragma unroll
                for (int j = 0; j < kUnroll * kPairs; ++j) {
                    const float2 value = __bfloat1622float2(view[j]);
                    reduced[j].x += value.x;
                    reduced[j].y += value.y;
                }
            }
            if (tile == 0) wait();
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                uint4 value;
                auto* view = reinterpret_cast<nv_bfloat162*>(&value);
#pragma unroll
                for (int p = 0; p < kPairs; ++p)
                    view[p] = __float22bfloat162_rn(reduced[j * kPairs + p]);
                destination[tile + j * kWarp + lane] = value;
            }
        }
    }
}

}  // namespace afdep
