# SPDX-License-Identifier: Apache-2.0
"""Derive afdEP role kernels that must stay mechanical copies of shared kernels.

The FFN dispatch receive of one NVLink domain is dispatch.cuh in which only the
notify warps trigger the PDL copy epilogue, so CTA 0 triggers once every source's
counts have arrived (the FFN sends nothing, so its idle payload warps otherwise
trigger at kernel entry and the epilogue occupies every SM for the whole wait).
The FFN directional hybrid dispatch is hybrid_dispatch.cuh without its scale-out
warps, which an FFN domain never uses: its seven edits (six functional, plus the
generated-file banner) are the whole difference.

tests/unit/test_deepep_build.py re-derives both copies and requires byte
equality, so a fix to a shared kernel fails the test until the copies are
regenerated: python -m tests.role_kernels
"""

from __future__ import annotations

from pathlib import Path

IMPLS = (
    Path(__file__).resolve().parents[1]
    / "afdep/third_party/deepep_v2/deep_ep/include/deep_ep/impls"
)
FFN_HYBRID_DISPATCH = "afdep_ffn_hybrid_dispatch.cuh"
FFN_DISPATCH = "afdep_ffn_dispatch.cuh"

FFN_DISPATCH_EDITS = (
    (
        "#pragma once\n",
        "#pragma once  // afdEP: generated from dispatch.cuh by "
        "tests/role_kernels.py; do not edit\n",
    ),
    ("dispatch_impl(\n", "afdep_ffn_dispatch_impl(\n"),
    (
        "    const bool afd_source = rank_idx >= kAfdFfnRanks;\n",
        "    const bool afd_source = rank_idx >= kAfdFfnRanks; "
        'EP_STATIC_ASSERT(kAfdDirectedExit, "FFN receive: directed exit"); '
        "EP_DEVICE_ASSERT(not afd_source);\n",
    ),
    (
        "    cudaTriggerProgrammaticLaunchCompletion();\n\n"
        "    // Clean atomic counters\n",
        "    if (warp_idx < kNumNotifyWarps) cudaTriggerProgrammaticLaunchCompletion();"
        "  // after the count wait\n\n    // Clean atomic counters\n",
    ),
)

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


def apply_edits(shared: str, edits) -> str:
    """Apply each edit exactly once."""
    for old, new in edits:
        if shared.count(old) != 1:
            raise ValueError(f"edit anchor not unique: {old!r}")
        shared = shared.replace(old, new)
    return shared


def ffn_hybrid_dispatch(shared: str) -> str:
    """Forward warp f keeps channel sm*C+f, QP and TMA slot C+f; the launch bound and
    every other line are the shared kernel's."""
    return apply_edits(shared, FFN_HYBRID_DISPATCH_EDITS)


def ffn_dispatch(shared: str) -> str:
    """Same template, launch shape and statements; only the notify warps trigger."""
    return apply_edits(shared, FFN_DISPATCH_EDITS)


if __name__ == "__main__":
    text = ffn_hybrid_dispatch((IMPLS / "hybrid_dispatch.cuh").read_text())
    (IMPLS / FFN_HYBRID_DISPATCH).write_text(text)
    (IMPLS / FFN_DISPATCH).write_text(
        ffn_dispatch((IMPLS / "dispatch.cuh").read_text())
    )
