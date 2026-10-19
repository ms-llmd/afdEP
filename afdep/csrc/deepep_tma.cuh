// SPDX-License-Identifier: MIT
// Copyright (c) 2025 DeepSeek
// Pinned DeepEP V2 one-dimensional TMA primitives. See ORIGIN.json.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace afdep::tma {
struct alignas(8) mbarrier { uint64_t storage; };
using arrival_phase = uint32_t;

__forceinline__ __device__ void mbarrier_init_with_fence(mbarrier* ptr, const int& arrive_count = 1) {
    asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::
                 "r"(arrive_count), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
    asm volatile("fence.mbarrier_init.release.cluster;" ::);
}

__forceinline__ __device__ void mbarrier_arrive_and_set_tx(mbarrier* ptr, const int& num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t" ::
                 "r"(num_bytes), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))));
}

__forceinline__ __device__ void mbarrier_wait_and_flip_phase(mbarrier* ptr, arrival_phase& phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "r"(phase), "r"(0x989680));
    phase ^= 1;
}

__forceinline__ __device__ void tma_store_fence() {
    asm volatile("fence.proxy.async.shared::cta;");
}

template <int kNumRemainingWaits = 0>
__forceinline__ __device__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(kNumRemainingWaits) : "memory");
}

enum TMACacheHint: int64_t {
    kEvictFirst = 0x12f0000000000000ll,
    kEvictNormal = 0x1000000000000000ll
};

__forceinline__ __device__ void tma_load_1d(
    const void* dst_ptr, const void* src_ptr, mbarrier* ptr, const int& num_bytes,
    const TMACacheHint& hint = TMACacheHint::kEvictFirst) {
    // NOTES: normally, the loaded part will be evicted soon
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr))),
        "l"(src_ptr),
        "r"(num_bytes),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(ptr))),
        "l"(hint)
        : "memory");
}

__forceinline__ __device__ void tma_store_1d(
    const void* dst_ptr, const void* src_ptr, const int& num_bytes,
    const TMACacheHint& hint = TMACacheHint::kEvictNormal) {
    // NOTES: normally, the stored part will be used soon
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::
                 "l"(dst_ptr),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src_ptr))),
                 "r"(num_bytes),
                 "l"(hint)
                 : "memory");
}

__forceinline__ __device__ void tma_store_commit() {
    asm volatile("cp.async.bulk.commit_group;");
}

}  // namespace afdep::tma
