# SPDX-License-Identifier: Apache-2.0
"""Derive afdEP role kernels that must stay mechanical copies of shared kernels.

The FFN directional hybrid dispatch is hybrid_dispatch.cuh without its scale-out
warps, which an FFN domain never uses: the seven edits below (six functional,
plus the generated-file banner) are the whole difference.
tests/unit/test_deepep_build.py re-derives the copy and requires byte equality,
so a fix to the shared kernel fails the test until the copy is regenerated:
python -m tests.role_kernels
"""

from __future__ import annotations

from pathlib import Path

IMPLS = (
    Path(__file__).resolve().parents[1]
    / "afdep/third_party/deepep_v2/deep_ep/include/deep_ep/impls"
)
FFN_HYBRID_DISPATCH = "afdep_ffn_hybrid_dispatch.cuh"

FFN_HYBRID_DISPATCH_EDITS = (
    (
        "#pragma once\n",
        "#pragma once  // afdEP: generated from hybrid_dispatch.cuh by "
        "tests/role_kernels.py; do not edit\n",
    ),
    (
        "          int kNumChannelsPerSM = kNumScaleoutWarps,\n"
        "          int kNumChannels = kNumScaleoutWarps * kNumSMs,\n",
        "          int kNumChannelsPerSM = kNumForwardWarps,\n"
        "          int kNumChannels = kNumForwardWarps * kNumSMs,\n",
    ),
    (
        "          int kNumThreads = kNumNotifyThreads + kNumScaleoutSendThreads"
        " + kNumForwardThreads>\n",
        "          int kNumThreads = kNumNotifyThreads + 2 * kNumForwardThreads>"
        "  // the shared kernel's launch bound\n",
    ),
    ("hybrid_dispatch_impl(\n", "afdep_ffn_hybrid_dispatch_impl(\n"),
    (
        '    EP_STATIC_ASSERT(kNumScaleoutWarps == kNumForwardWarps, "Invalid warp'
        ' size");\n',
        "    EP_STATIC_ASSERT(kNumScaleoutWarps == 0 and kNumForwardWarps > 0 and "
        'kAfdFfnScaleoutRanks > 0, "FFN domain: forward warps only");\n',
    ),
    (
        "    const bool afd_ffn_domain = scaleout_rank_idx < kAfdFfnScaleoutRanks;\n",
        "    const bool afd_ffn_domain = scaleout_rank_idx < kAfdFfnScaleoutRanks; "
        "EP_DEVICE_ASSERT(afd_ffn_domain);\n",
    ),
    (
        "    const auto tma_buffer = layout::BufferLayout<true>(token_layout, "
        "kNumScaleoutWarps + kNumForwardWarps, 1,\n"
        "            math::advance_ptr<int>(smem, kNumSmemBytesForNotify))"
        ".get_rank_buffer(warp_idx - kNumNotifyWarps).get_token_buffer(0);\n",
        "    const auto tma_buffer = layout::BufferLayout<true>(token_layout, "
        "2 * kNumForwardWarps, 1,\n"
        "            math::advance_ptr<int>(smem, kNumSmemBytesForNotify))"
        ".get_rank_buffer(warp_idx - kNumNotifyWarps + kNumForwardWarps)"
        ".get_token_buffer(0);\n",
    ),
)


def ffn_hybrid_dispatch(shared: str) -> str:
    """Apply each edit exactly once (forward warp f keeps channel sm*C+f, QP and TMA
    slot C+f; the launch bound and every other line are the shared kernel's)."""
    for old, new in FFN_HYBRID_DISPATCH_EDITS:
        if shared.count(old) != 1:
            raise ValueError(f"edit anchor not unique: {old!r}")
        shared = shared.replace(old, new)
    return shared


if __name__ == "__main__":
    text = ffn_hybrid_dispatch((IMPLS / "hybrid_dispatch.cuh").read_text())
    (IMPLS / FFN_HYBRID_DISPATCH).write_text(text)
