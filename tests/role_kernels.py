# SPDX-License-Identifier: Apache-2.0
"""Derive afdEP role kernels that must stay mechanical copies of shared kernels.

The FFN dispatch receive of one NVLink domain is dispatch.cuh in which only the
notify warps trigger the PDL copy epilogue, so CTA 0 triggers once every source's
counts have arrived (the FFN sends nothing, so its idle payload warps otherwise
trigger at kernel entry and the epilogue occupies every SM for the whole wait).
The FFN directional hybrid dispatch is hybrid_dispatch.cuh without its scale-out
warps, which an FFN domain never uses: its seven edits (six functional, plus the
generated-file banner) are the whole difference.

Count-first dispatch (afdep_count_first.cuh; AFDEP_PAYLOAD_FIRST=1 is the
control) adds three copies: the Attention source dispatch (dispatch.cuh with
epoch-tagged counts, a payload gate and per-row flags), the FFN stream receive
(dispatch.cuh that records the epochs, triggers after the count wait and
releases P after the prefix sums) and the FFN stream copy epilogue
(dispatch_copy_epilogue.cuh with claimed units of flag-guarded rows).

tests/unit/test_deepep_build.py re-derives every copy and requires byte
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
SOURCE_DISPATCH = "afdep_source_dispatch.cuh"
FFN_STREAM_DISPATCH = "afdep_ffn_stream_dispatch.cuh"
FFN_STREAM_EPILOGUE = "afdep_ffn_stream_epilogue.cuh"
SOURCE_HYBRID_DISPATCH = "afdep_source_hybrid_dispatch.cuh"
FFN_SPARSE_HYBRID_DISPATCH = "afdep_ffn_sparse_hybrid_dispatch.cuh"
FFN_SPARSE_HYBRID_COMBINE = "afdep_ffn_sparse_hybrid_combine.cuh"
SPARSE_HYBRID_COLLECT = "afdep_sparse_hybrid_collect.cuh"
MIXED_HYBRID_DISPATCH = "afdep_mixed_hybrid_dispatch.cuh"
MIXED_HYBRID_COMBINE = "afdep_mixed_hybrid_combine.cuh"

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

COUNT_FIRST_INCLUDE = (
    "#include <deep_ep/impls/afdep_roles.cuh>\n",
    "#include <deep_ep/impls/afdep_roles.cuh>\n"
    "#include <deep_ep/impls/afdep_count_first.cuh>\n",
)
COUNTS_ISSUED = (
    "            // This is necessary, as the waited results will rewrite the "
    "shared memory\n"
    "            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);\n"
)
NVLINK_STORE = (
    "            if (dst_ptr != nullptr)\n"
    "                ptx::tma_store_1d(dst_ptr, tma_buffer.get_base_ptr(), "
    "tma_buffer.get_num_bytes<false>());\n"
)
SLOT_RESET = (
    "    if (not kReuseSlotIndices and clean_counters and thread_idx < kNumRanks)\n"
    "        workspace_layout.get_scaleup_atomic_sender_counter()[thread_idx] = 0;"
    "\n"
)
RELEASE_ROW = (
    "afdep_cf_release_row<kNumRanks, kNumMaxTokensPerRank, team_t>(\n"
    "{0}    gin, workspace_layout, afd_pending_dst, afd_pending_slot, "
    "rank_idx - kAfdFfnRanks, afd_epoch);\n"
)

# Attention source: epoch-tagged rank counts, the payload gate and per-row flags.
SOURCE_DISPATCH_EDITS = (
    FFN_DISPATCH_EDITS[0],
    COUNT_FIRST_INCLUDE,
    (
        "          int kAfdZeroCollectRowBytes = 0,\n",
        "          int kAfdZeroCollectRowBytes = 0,\n"
        "          // afdEP count-first: tokens stored before the counts are issued, "
        "and one token per notify thread\n"
        "          int kAfdUngatedTokens = 0, bool kAfdThreadCount = true,\n",
    ),
    ("dispatch_impl(\n", "afdep_source_dispatch_impl(\n"),
    (
        "    const bool afd_source = rank_idx >= kAfdFfnRanks;\n",
        "    const bool afd_source = rank_idx >= kAfdFfnRanks;\n"
        "    EP_STATIC_ASSERT(kIsScaleupNVLink and kAfdSkipEntryBarrier and "
        "kAfdDirectionalNotify and kAfdDirectedExit and\n"
        "                     kAfdRoleBuffers and kAfdPerCtaExit and not "
        "kReuseSlotIndices and kAfdFfnRanks <= kAfdCfMaxFfnRanks and\n"
        "                     kNumSourceRanks <= kAfdCfMaxSources and "
        "afdep_cf_flags_fit(kNumSourceRanks, kNumMaxTokensPerRank),\n"
        '                     "Count-first source: one-domain role path");\n'
        "    EP_DEVICE_ASSERT(afd_source);\n",
    ),
    (
        "    const auto host_workspace_layout = layout::WorkspaceLayout("
        "mapped_host_workspace, 1, kNumRanks, kNumExperts);\n",
        "    const auto host_workspace_layout = layout::WorkspaceLayout("
        "mapped_host_workspace, 1, kNumRanks, kNumExperts);\n"
        "    // afdEP count-first: this exchange's epoch, read before any CTA can "
        "commit it\n"
        "    const int64_t afd_epoch = afdep_cf_source_epoch(workspace_layout);\n",
    ),
    (
        "static_cast<int64_t>(rank_count[i]), i,",
        "afdep_cf_tag_count(rank_count[i], afd_epoch), i,",
    ),
    (
        COUNTS_ISSUED,
        COUNTS_ISSUED + "            // afdEP count-first: the counts are issued; "
        "open the payload gate\n"
        "            if (thread_idx == 0)\n"
        "                afdep_cf_open_gate(workspace_layout, afd_epoch);\n",
    ),
    (
        "        // Iterate all tokens\n",
        "        // afdEP count-first: the gate, and the row whose flag this lane "
        "releases next\n"
        "        bool afd_gate_open = false;\n"
        "        int afd_pending_dst = -1, afd_pending_slot = -1;\n\n"
        "        // Iterate all tokens\n",
    ),
    (
        "            // Issue SF TMA or cp.async\n",
        "            // afdEP count-first: the wait above completed this lane's "
        "previous row; release its flag\n"
        "            " + RELEASE_ROW.format("            ") + "\n"
        "            // Issue SF TMA or cp.async\n",
    ),
    (
        "            // Issue TMA NVLink stores\n",
        "            // afdEP count-first: from token kAfdUngatedTokens on, store once "
        "SM0 has issued the counts\n"
        "            if (not afd_gate_open and token_idx >= kAfdUngatedTokens) {\n"
        "                afdep_cf_wait_gate<kNumTimeoutCycles>(workspace_layout, "
        "afd_epoch, lane_idx, rank_idx);\n"
        "                afd_gate_open = true;\n"
        "            }\n\n"
        "            // Issue TMA NVLink stores\n",
    ),
    (
        NVLINK_STORE,
        NVLINK_STORE + "            if (dst_ptr != nullptr)\n"
        "                afd_pending_dst = stored_dst_rank_idx, "
        "afd_pending_slot = stored_dst_slot_idx;\n",
    ),
    (
        "        }\n    }\n\n    // Barrier to ensure data arrival\n",
        "        }\n\n"
        "        // afdEP count-first: complete and release this lane's last row "
        "before the arrival release\n"
        "        ptx::tma_store_wait();\n"
        "        __syncwarp();\n"
        "        " + RELEASE_ROW.format("        ") + "    }\n\n"
        "    // Barrier to ensure data arrival\n",
    ),
    (
        SLOT_RESET,
        SLOT_RESET
        + "    // afdEP count-first: the last CTA commits the epoch after every CTA's "
        "done atomic (and entry read)\n"
        "    if (clean_counters and thread_idx == 0)\n"
        "        afdep_cf_commit_source_epoch(workspace_layout, afd_epoch);\n",
    ),
)
# The notify warps count one token per thread; the original warp loop stays verbatim.
SOURCE_COUNT_SPAN = (
    "        for (int i = global_warp_idx; (not kAfdDirectionalNotify or afd_source) "
    "and i < num_tokens;\n",
    "                atomicAdd_block(rank_count + dst_rank_idx, 1);\n        }\n",
    "        // afdEP count-first: one token per notify thread over the same tokens "
    "(sm_idx mod kNumSMs)\n"
    "        if constexpr (kAfdThreadCount) {\n"
    "            afdep_cf_count_tokens<kNumSMs, kNumNotifyThreads, kNumTopk, "
    "kAfdFfnRanks * kNumExpertsPerRank,\n"
    "                                  kNumExpertsPerRank, kAfdFfnRanks>(\n"
    "                topk_idx, num_tokens, sm_idx, thread_idx, rank_count, "
    "expert_count);\n"
    "        } else {\n",
    "        }\n",
)

# FFN receive: record each source's epoch, trigger after the count wait, release P.
FFN_STREAM_DISPATCH_EDITS = (
    FFN_DISPATCH_EDITS[0],
    COUNT_FIRST_INCLUDE,
    ("dispatch_impl(\n", "afdep_ffn_stream_dispatch_impl(\n"),
    (
        "    const bool afd_source = rank_idx >= kAfdFfnRanks;\n",
        "    const bool afd_source = rank_idx >= kAfdFfnRanks;\n"
        "    EP_STATIC_ASSERT(kIsScaleupNVLink and kAfdSkipEntryBarrier and "
        "kAfdDirectionalNotify and kAfdDirectedExit and\n"
        "                     kAfdRoleBuffers and kAfdPerCtaExit and "
        "kNumSourceRanks <= kAfdCfMaxSources and\n"
        "                     afdep_cf_flags_fit(kNumSourceRanks, "
        "kNumMaxTokensPerRank),\n"
        '                     "Count-first FFN receive: one-domain role path");\n'
        "    EP_DEVICE_ASSERT(not afd_source);\n",
    ),
    (
        "                    const auto count = static_cast<int>(\n"
        "                        ptx::ld_volatile<int64_t>(workspace_layout."
        "get_scaleup_rank_expert_count_ptr<false>() + index));\n",
        "                    const auto afd_word = ptx::ld_volatile<int64_t>(\n"
        "                        workspace_layout."
        "get_scaleup_rank_expert_count_ptr<false>() + index);\n"
        "                    const auto count = static_cast<int>(afd_word);\n",
    ),
    (
        "                        workspace_layout.get_scaleup_rank_expert_count_ptr"
        "<false>()[index] = 0;\n"
        "                        rank_expert_count[index] = decoded;\n",
        "                        workspace_layout.get_scaleup_rank_expert_count_ptr"
        "<false>()[index] = 0;\n"
        "                        rank_expert_count[index] = decoded;\n"
        "                        // afdEP count-first: rank words carry the source's "
        "epoch in their high half\n"
        "                        if (i < kNumSourceRanks)\n"
        "                            afdep_cf_record_epoch<kAfdFfnRanks>("
        "workspace_layout, i, afd_word, rank_idx);\n",
    ),
    (
        "                }, start_clock);\n            }\n"
        "            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);\n",
        "                }, start_clock);\n            }\n"
        "            ptx::named_barrier<kNumNotifyThreads>(kNotifyBarrierIndex);\n"
        "            // afdEP count-first: every count has arrived; P, not this "
        "trigger, orders the epilogue's reads\n"
        "            cudaTriggerProgrammaticLaunchCompletion();\n",
    ),
    (
        "                do_psum(expert_count, psum_num_recv_tokens_per_expert, "
        "kNumExpertsPerRank, 1);\n"
        "            }\n        }\n    } else {\n",
        "                do_psum(expert_count, psum_num_recv_tokens_per_expert, "
        "kNumExpertsPerRank, 1);\n"
        "            }\n\n"
        "            // afdEP count-first: both prefix sums and every epoch are "
        "written; release P\n"
        "            afdep_cf_named_barrier_mem<kNumNotifyThreads>"
        "(kNotifyBarrierIndex);\n"
        "            if (thread_idx == 0)\n"
        "                afdep_cf_publish_prefix(workspace_layout, rank_idx);\n"
        "        }\n    } else {\n",
    ),
    (
        "    cudaTriggerProgrammaticLaunchCompletion();\n\n"
        "    // Clean atomic counters\n",
        "    // afdEP count-first: CTA 0's notify warps triggered after the count wait"
        "\n\n    // Clean atomic counters\n",
    ),
)

# FFN copy epilogue: wait on P, then claim units of rows that each wait on their flag.
FFN_STREAM_EPILOGUE_EDITS = (
    (
        "#pragma once\n",
        "#pragma once  // afdEP: generated from dispatch_copy_epilogue.cuh by "
        "tests/role_kernels.py; do not edit\n",
    ),
    (
        "#include <deep_ep/common/ptx.cuh>\n",
        "#include <deep_ep/common/ptx.cuh>\n"
        "#include <deep_ep/impls/afdep_count_first.cuh>\n",
    ),
    (
        "          int kAfdRegionOffset = 0,\n",
        "          int kAfdRegionOffset = 0,\n"
        "          // afdEP count-first: the prefix and row wait timeout, and the "
        "fewest rows per claimed unit\n"
        "          int64_t kAfdTimeoutCycles = 0, int kAfdUnitMinRows = 1,\n",
    ),
    ("dispatch_copy_epilogue_impl(", "afdep_ffn_stream_epilogue_impl("),
    (
        "    // Will block until the main dispatch kernel has finished and all data "
        "are visible\n"
        "    // NOTES: PDL is used, please do not use `__ldg`\n"
        "    cudaGridDependencySynchronize();\n",
        "    // afdEP count-first: no grid dependency sync at entry; P orders the "
        "prefix sums, epochs and\n"
        "    // per-expert cursors, and each row's flag orders its data "
        "(PDL is used, please do not use `__ldg`)\n"
        "    EP_STATIC_ASSERT(kDoExpand and not kCachedMode and kNumScaleoutRanks == 1 "
        "and kAfdRegionOffset > 0 and\n"
        "                     kAfdTimeoutCycles > 0 and kNumScaleupRanks - "
        "kAfdRegionOffset <= kAfdCfMaxSources,\n"
        '                     "Count-first copy epilogue: one-domain expanded role '
        'path");\n'
        "    const auto afd_ws = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, "
        "kNumScaleupRanks, kNumExperts);\n"
        "    afdep_cf_wait_prefix<kAfdTimeoutCycles>(afd_ws, thread_idx, rank_idx);\n",
    ),
    (
        "    // Maintain linked list's ending\n",
        "    // afdEP count-first: the last claimant waits for the dispatch, then "
        "clears P and CLAIM\n"
        "    afdep_cf_epilogue_end(afd_ws, afd_cursor.last, thread_idx);\n\n"
        "    // Maintain linked list's ending\n",
    ),
)
# The static row loop becomes the claim loop; the row body stays verbatim.
EPILOGUE_ROW_SPAN = (
    "    // Current rank indices should be maintained\n",
    "            .get_token_buffer(i - current_rank_start);\n",
    "    // afdEP count-first: CTAs claim units of rows (chunk-major, source-minor) "
    "until none is left;\n"
    "    // warp w copies a unit's row w once its flag carries the source's epoch\n"
    "    auto afd_sources = afdep_cf_load_sources<kNumScaleupRanks, kAfdRegionOffset, "
    "kNumMaxTokensPerRank>(\n"
    "        afd_ws, psum_num_recv_tokens_per_scaleup_rank, lane_idx);\n"
    "    EP_DEVICE_ASSERT(afd_sources.total == num_recv_tokens);\n"
    "    afdep_cf_unit_rows<kNumWarps, kAfdUnitMinRows>(afd_sources, "
    "static_cast<int>(gridDim.x));\n"
    "    const auto afd_claims = static_cast<int*>(\n"
    "        layout::BufferLayout<true>(token_layout, kNumWarps, 1, smem)"
    ".get_buffer_end_ptr());\n"
    "    AfdCfCursor afd_cursor = {};\n"
    "    for (auto unit = afdep_cf_claim(afd_ws, afd_sources, afd_cursor, "
    "afd_claims, thread_idx); unit.valid;\n"
    "         unit = afdep_cf_claim(afd_ws, afd_sources, afd_cursor, afd_claims, "
    "thread_idx)) {\n"
    "        if (warp_idx >= unit.len)\n"
    "            continue;\n"
    "        const int current_rank_idx = kAfdRegionOffset + unit.source;\n"
    "        const int current_rank_start = unit.base;\n"
    "        const int afd_slot = unit.first_slot + warp_idx;\n"
    "        const int i = current_rank_start + afd_slot;\n"
    "        const auto buffer_token = scaleup_buffer.get_rank_buffer(unit.source)"
    ".get_token_buffer(afd_slot);\n"
    "        afdep_cf_wait_row<kAfdTimeoutCycles, kNumMaxTokensPerRank, "
    "kAfdRegionOffset>(\n"
    "            afd_ws, unit.source, afd_slot, unit.epoch, lane_idx, rank_idx);\n",
)


# Sparse directional hybrid (afdep_sparse_hybrid.cuh), Attention send:
# - kAfdStaggerMaxTokens (AFDEP_HYBRID_STAGGER=1): a source of at most that many tokens
#   puts token i on channel (i + s * C / S) mod C instead of i mod C, so at small
#   batches the sources' first tokens ride different channels (and SMs and QPs)
#   instead of every source's token 0 queueing on FFN forward warp 0. A channel still
#   carries at most ceil(T / C) tokens of a source; tails, the copy epilogue and combine
#   are keyed by (source, channel) or by source token.
# - kAfdSparseDispatch (AFDEP_HYBRID_SPARSE_DISPATCH=1): SM 0 publishes the source's
#   token count to each FFN domain's rail peer at entry (QP 0, with the counts), and a
#   channel that carried no token sends no finish tail; the FFN copy below retires those
#   lanes from the count instead of waiting for C finish atomics per source.
SPARSE_INCLUDE = (
    "#include <deep_ep/impls/afdep_roles.cuh>\n",
    "#include <deep_ep/impls/afdep_roles.cuh>\n"
    "#include <deep_ep/impls/afdep_sparse_hybrid.cuh>\n",
)
SPARSE_TEMPLATE = (
    "          bool kAfdSkipAttentionEntrySync = false,\n",
    "          bool kAfdSkipAttentionEntrySync = false,\n"
    "          // afdEP sparse hybrid (afdep_sparse_hybrid.cuh): staggered channels"
    " and token counts.\n"
    "          int kAfdStaggerMaxTokens = 0, bool kAfdSparseDispatch = false, "
    "bool kAfdSparseCombine = false,\n",
)
SOURCE_HYBRID_DISPATCH_EDITS = (
    (
        "#pragma once\n",
        "#pragma once  // afdEP: generated from hybrid_dispatch.cuh by "
        "tests/role_kernels.py; do not edit\n",
    ),
    SPARSE_INCLUDE,
    SPARSE_TEMPLATE,
    ("hybrid_dispatch_impl(\n", "afdep_source_hybrid_dispatch_impl(\n"),
    (
        "    const bool afd_ffn_domain = scaleout_rank_idx < kAfdFfnScaleoutRanks;\n",
        "    const bool afd_ffn_domain = scaleout_rank_idx < kAfdFfnScaleoutRanks; "
        'EP_STATIC_ASSERT(kAfdDirectional, "sparse send: directional hybrid"); '
        "EP_DEVICE_ASSERT(not afd_ffn_domain);\n",
    ),
    (
        "    if (warp_idx < kNumNotifyWarps) {\n        // Assign shared memory\n",
        "    if (warp_idx < kNumNotifyWarps) {\n"
        "        // afdEP sparse dispatch: this exchange's token count + 1 to each"
        " FFN domain\n"
        "        if (kAfdSparseDispatch and sm_idx == 0 and thread_idx <"
        " kAfdFfnScaleoutRanks)\n"
        "           "
        " gin.put_value<ncclTeamTagRail>(afdep_sparse_tokens_ptr(workspace_layout, "
        "scaleout_rank_idx),\n"
        "                                           static_cast<int64_t>(num_tokens)"
        " + 1, "
        "thread_idx);\n\n"
        "        // Assign shared memory\n",
    ),
    (
        "        // Iterate all tokens\n"
        "        preload_next_token(channel_idx);\n"
        "        for (int token_idx = channel_idx; token_idx < num_tokens; "
        "token_idx += kNumChannels) {\n",
        "        // Iterate all tokens (afdEP: optionally staggered by source)\n"
        "        const int first_token_idx = afdep_first_token<kNumChannels, "
        "kNumScaleoutRanks - kAfdFfnScaleoutRanks, kAfdStaggerMaxTokens>(\n"
        "            channel_idx, scaleout_rank_idx - kAfdFfnScaleoutRanks, "
        "num_tokens);\n"
        "        preload_next_token(first_token_idx);\n"
        "        for (int token_idx = first_token_idx; token_idx < num_tokens; "
        "token_idx += kNumChannels) {\n",
    ),
    (
        "        // Flush unflushed tails\n        update_scaleout_tail(true);\n",
        "        // afdEP sparse combine: the FFN domains this channel put a token"
        " for\n"
        "        if constexpr (kAfdSparseCombine) {\n"
        "            const auto sent_mask = ptx::gather(lane_idx <"
        " kAfdFfnScaleoutRanks "
        "and stored_scaleout_tail > 0);\n"
        "            if (lane_idx == 0)\n"
        "                *afdep_sparse_sent_ptr(workspace_layout, channel_idx) ="
        " sent_mask;\n"
        "        }\n\n"
        "        // Flush unflushed tails (afdEP sparse dispatch: only on channels"
        " that carried a token)\n"
        "        if (not kAfdSparseDispatch or first_token_idx < num_tokens)\n"
        "            update_scaleout_tail(true);\n",
    ),
)

# - kAfdSparseCombine (AFDEP_HYBRID_SPARSE_COMBINE=1): each channel also records, in a
#   local word, the FFN domains it put a token for. The FFN combine (copy below) replies
#   on a channel only to the sources whose tokens it received there, so it signals only
#   those (source, channel) pairs, plus channel 0 to every source as a done witness:
#   every signal follows the FFN's dispatch kernel, so a source that sent nothing still
#   cannot start its next exchange (count, tails, inbox writes) before that kernel ends.
#   The collect (copy below) waits for exactly those signals.

# Sparse directional hybrid, FFN forwarders (the FFN hybrid copy plus): a source lane
# starts unknown and retires on its finish tail as before, or as soon as the source's
# token count shows it never uses this channel (its first token on the channel is at or
# beyond the count). Data never waits on the count. After the exit barrier, whose grid
# sync orders every forwarder's reads, SM 0 waits for each source's count (a source that
# filled every channel retired no lane by count) and clears it for the next exchange.
FFN_SPARSE_HYBRID_DISPATCH_EDITS = (
    SPARSE_INCLUDE,
    SPARSE_TEMPLATE,
    ("afdep_ffn_hybrid_dispatch_impl(\n", "afdep_ffn_sparse_hybrid_dispatch_impl(\n"),
    (
        "        int recv_scaleout_rank_idx = channel_idx % kNumScaleoutRanks;\n",
        "        int recv_scaleout_rank_idx = channel_idx % kNumScaleoutRanks;\n"
        "        // afdEP sparse dispatch: lanes retire by count when the source"
        " never uses this channel\n"
        "        bool stored_count_known = not kAfdSparseDispatch or"
        " stored_finish_flag, "
        "stored_skip = false;\n",
    ),
    (
        "                if (lane_idx < kNumScaleoutRanks and lane_idx >="
        " kAfdFfnScaleoutRanks) {\n"
        "                    const auto signaled_tail = ptx::ld_acquire_sys<int64_t>(\n"
        "                       "
        " workspace_layout.get_scaleout_channel_signaled_tail_ptr("
        "channel_idx, lane_idx));\n"
        "                    math::unpack2<int, int64_t>(signaled_tail,"
        " stored_finish_flag, "
        "stored_scaleout_tail_idx);\n"
        "                }\n",
        "                if (lane_idx < kNumScaleoutRanks and lane_idx >="
        " kAfdFfnScaleoutRanks) {\n"
        "                    if (not stored_count_known) {\n"
        "                        const auto tokens = ptx::ld_acquire_sys<int64_t>("
        "afdep_sparse_tokens_ptr(workspace_layout, lane_idx));\n"
        "                        if (tokens != 0) {\n"
        "                            EP_DEVICE_ASSERT(tokens - 1 <="
        " kNumMaxTokensPerRank);\n"
        "                            stored_count_known = true;\n"
        "                            stored_skip = afdep_first_token<kNumChannels, "
        "kNumScaleoutRanks - kAfdFfnScaleoutRanks, kAfdStaggerMaxTokens>(\n"
        "                                channel_idx, lane_idx - kAfdFfnScaleoutRanks, "
        "static_cast<int>(tokens - 1)) >= tokens - 1;\n"
        "                            if (stored_skip)\n"
        "                                stored_finish_flag = 1,"
        " stored_scaleout_tail_idx = 0;\n"
        "                        }\n"
        "                    }\n"
        "                    if (not stored_skip) {\n"
        "                        const auto signaled_tail ="
        " ptx::ld_acquire_sys<int64_t>(\n"
        "                           "
        " workspace_layout.get_scaleout_channel_signaled_tail_ptr("
        "channel_idx, lane_idx));\n"
        "                        math::unpack2<int, int64_t>(signaled_tail,"
        " stored_finish_flag, "
        "stored_scaleout_tail_idx);\n"
        "                    }\n"
        "                }\n",
    ),
    (
        "    // Trigger the copy epilogue kernel\n",
        "    // afdEP sparse dispatch: consume and clear every source's token count"
        " of this exchange\n"
        "    if (kAfdSparseDispatch and sm_idx == 0 and kAfdFfnScaleoutRanks <="
        " thread_idx and "
        "thread_idx < kNumScaleoutRanks) {\n"
        "        const auto tokens_ptr = afdep_sparse_tokens_ptr(workspace_layout,"
        " thread_idx);\n"
        "        comm::timeout_while<kNumTimeoutCycles>([=](const bool&"
        " is_last_check) {\n"
        "            if (ptx::ld_acquire_sys<int64_t>(tokens_ptr) != 0)\n"
        "                return true;\n"
        "            if (is_last_check)\n"
        '                printf("afdEP sparse dispatch token-count timeout, source'
        ' domain: %d\\n", '
        "thread_idx);\n"
        "            return false;\n"
        "        });\n"
        "        *tokens_ptr = 0;\n"
        "    }\n\n"
        "    // Trigger the copy epilogue kernel\n",
    ),
)


# Sparse combine, FFN: signal a source on a channel only if this channel replied to it
# (it received the source's tokens there in dispatch), and on channel 0 always.
# Channel 0's signal is the done witness, so it must also certify that every forward
# warp of this rank has finished reading the combine scale-up inbox: that inbox can
# overlap the dispatch inbox a source writes next, and a source waits only for its own
# channels and channel 0. Each forward warp releases a rank-local DONE count, and
# channel 0 waits for all C first.
FFN_SPARSE_HYBRID_COMBINE_EDITS = (
    (
        "#pragma once\n",
        "#pragma once  // afdEP: generated from hybrid_combine.cuh by "
        "tests/role_kernels.py; do not edit\n",
    ),
    (
        "#include <deep_ep/impls/combine_utils.cuh>\n",
        "#include <deep_ep/impls/combine_utils.cuh>\n"
        "#include <deep_ep/impls/afdep_sparse_hybrid.cuh>\n",
    ),
    ("hybrid_combine_impl(", "afdep_ffn_sparse_hybrid_combine_impl("),
    (
        "    const bool afd_ffn_domain = scaleout_rank_idx < kAfdFfnScaleoutRanks;\n",
        "    const bool afd_ffn_domain = scaleout_rank_idx < kAfdFfnScaleoutRanks; "
        'EP_STATIC_ASSERT(kAfdDirectional, "sparse combine: directional hybrid"); '
        "EP_DEVICE_ASSERT(afd_ffn_domain);\n",
    ),
    (
        "        int stored_num_tokens_recv[kNumScaleupRanksPerLane] = {}, "
        "stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};\n",
        "        int stored_num_tokens_recv[kNumScaleupRanksPerLane] = {}, "
        "stored_cached_scaleup_tail[kNumScaleupRanksPerLane] = {};\n"
        "        // afdEP sparse combine: the source domains this channel replies to\n"
        "        uint32_t afd_replied_domains = 0;\n",
    ),
    (
        "            if (src_token_global_idx < 0)\n                break;\n",
        "            if (src_token_global_idx < 0)\n                break;\n"
        "            afd_replied_domains |= 1u << src_scaleout_rank_idx;\n",
    ),
    (
        "        // Update, wait and clean\n",
        "        // afdEP sparse combine: this warp has read its last partial;"
        " channel 0's witness\n"
        "        // waits for every forward warp of this rank\n"
        "        if (ptx::elect_one_sync())\n"
        "            ptx::red_add_rel_sys(afdep_sparse_done_ptr(workspace_layout),"
        " 1);\n"
        "        __syncwarp();\n"
        "        if (channel_idx == 0) {\n"
        "            const auto done_ptr = afdep_sparse_done_ptr(workspace_layout);\n"
        "            comm::timeout_while<kNumTimeoutCycles>([=](const bool&"
        " is_last_check) {\n"
        "                if (ptx::ld_acquire_sys<int>(done_ptr) == kNumChannels)\n"
        "                    return true;\n"
        "                if (is_last_check)\n"
        '                    printf("afdEP sparse combine done-witness timeout,'
        " scale-out: "
        '%d, scale-up: %d\\n",\n'
        "                           scaleout_rank_idx, scaleup_rank_idx);\n"
        "                return false;\n"
        "            });\n"
        "            __syncwarp();\n"
        "            if (lane_idx == 0)\n"
        "                *done_ptr = 0;\n"
        "        }\n"
        "        __syncwarp();\n\n"
        "        // Update, wait and clean\n",
    ),
    (
        "            if (afd_signal) {\n",
        "            if (afd_signal and (channel_idx == 0 or "
        "((afd_replied_domains >> lane_idx) & 1u))) {\n",
    ),
)

# Sparse combine, Attention collect: wait only for the channels this rank's dispatch put
# a token on (per FFN domain), and for channel 0, the done witness.
SPARSE_HYBRID_COLLECT_EDITS = (
    (
        "#pragma once\n",
        "#pragma once  // afdEP: generated from afdep_hybrid_collect.cuh by "
        "tests/role_kernels.py; do not edit\n",
    ),
    (
        "#include <deep_ep/common/ptx.cuh>\n",
        "#include <deep_ep/common/ptx.cuh>\n"
        "#include <deep_ep/impls/afdep_sparse_hybrid.cuh>\n",
    ),
    ("afdep_hybrid_collect_impl(", "afdep_sparse_hybrid_collect_impl("),
    (
        "        const int channel_idx = i / kAfdFfnScaleoutRanks, "
        "lane_idx = i % kAfdFfnScaleoutRanks;\n",
        "        const int channel_idx = i / kAfdFfnScaleoutRanks, "
        "lane_idx = i % kAfdFfnScaleoutRanks;\n"
        "        // afdEP sparse combine: only the channels this rank sent on, and"
        " channel 0\n"
        "        if (channel_idx != 0 and ((*afdep_sparse_sent_ptr(workspace_layout, "
        "channel_idx) >> lane_idx) & 1) == 0)\n"
        "            continue;\n",
    ),
)


# Mixed-role directional hybrid (AFDEP_MIXED_HYBRID=1): FFN ranks [0, F), 0 < F < U,
# fill part of NVLink domain 0 and share it with U - F Attention ranks; the other
# domains hold only Attention ranks. The directional protocol (kAfdFfnScaleoutRanks
# = 1) applies with roles per rank: every domain-0 GPU stays its rail's forwarder
# (node-wide ingress), only ranks below F hold experts, and domain 0's Attention GPUs
# are also sources, whose own tokens take the local bypass into their forwarder.
MIXED_TEMPLATE = (
    "          bool kAfdSkipAttentionEntrySync = false,\n",
    "          bool kAfdSkipAttentionEntrySync = false,\n"
    "          // afdEP mixed (afdep_suites.hpp): FFN ranks [0, F) share domain 0 with "
    "Attention ranks.\n"
    "          int kAfdMixedFfnRanks = 0,\n",
)
MIXED_HYBRID_DISPATCH_EDITS = (
    (
        "#pragma once\n",
        "#pragma once  // afdEP: generated from hybrid_dispatch.cuh by "
        "tests/role_kernels.py; do not edit\n",
    ),
    MIXED_TEMPLATE,
    ("hybrid_dispatch_impl(\n", "afdep_mixed_hybrid_dispatch_impl(\n"),
    (
        "    const bool afd_ffn_domain = scaleout_rank_idx < kAfdFfnScaleoutRanks;\n",
        "    const bool afd_ffn_domain = scaleout_rank_idx < kAfdFfnScaleoutRanks;\n"
        "    // afdEP mixed: only ranks [0, F) hold experts; domain 0's other GPUs are "
        "sources and forwarders,\n"
        "    // whose rail lane 0 carries their own tokens.\n"
        "    EP_STATIC_ASSERT(kAfdFfnScaleoutRanks == 1 and 0 < kAfdMixedFfnRanks and "
        "kAfdMixedFfnRanks < kNumScaleupRanks,\n"
        '                     "mixed: one FFN-holding domain shared with Attention'
        ' ranks");\n'
        "    const bool afd_holder = rank_idx < kAfdMixedFfnRanks;\n"
        "    const bool afd_self_lane = afd_ffn_domain and not afd_holder;\n",
    ),
    (
        "                dst_expert_idx = afdep_checked_expert<kAfdFfnScaleoutRanks * "
        "kNumExpertsPerScaleout>(raw_expert, lane_idx);\n",
        "                dst_expert_idx = afdep_checked_expert<kAfdMixedFfnRanks * "
        "kNumExpertsPerRank>(raw_expert, lane_idx);\n",
    ),
    (
        "                dst_expert_idx = afdep_checked_expert<kAfdFfnScaleoutRanks * "
        "kNumExpertsPerScaleout>(\n",
        "                dst_expert_idx = afdep_checked_expert<kAfdMixedFfnRanks * "
        "kNumExpertsPerRank>(\n",
    ),
    (
        "                (not kAfdDirectional or (not afd_ffn_domain and thread_idx < "
        "kAfdFfnScaleoutRanks))) {\n",
        "                (not afd_holder and thread_idx < kAfdFfnScaleoutRanks)) {  "
        "// afdEP mixed: domain-0 sources count to themselves\n",
    ),
    (
        "                for (int j = kAfdFfnScaleoutRanks; j < kNumScaleoutRanks; ++"
        " j) {\n",
        "                for (int j = afd_self_lane ? 0 : kAfdFfnScaleoutRanks; "
        "j < kNumScaleoutRanks; ++ j) {\n",
    ),
    (
        "    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps and "
        "(kAfdDirectional and afd_ffn_domain)) {\n",
        "    } else if (warp_idx < kNumNotifyWarps + kNumScaleoutWarps and"
        " afd_holder) {\n",
    ),
    (
        "        int stored_finish_flag = lane_idx >= kNumScaleoutRanks or "
        "lane_idx < kAfdFfnScaleoutRanks;\n",
        "        // afdEP mixed: on a domain-0 Attention GPU, lane 0 carries its own"
        " tokens\n"
        "        int stored_finish_flag = lane_idx >= kNumScaleoutRanks or "
        "(lane_idx < kAfdFfnScaleoutRanks and not afd_self_lane);\n",
    ),
    (
        "                if (lane_idx < kNumScaleoutRanks and lane_idx >= "
        "kAfdFfnScaleoutRanks) {\n",
        "                if (lane_idx < kNumScaleoutRanks and (lane_idx >= "
        "kAfdFfnScaleoutRanks or afd_self_lane)) {\n",
    ),
)

# Mixed-role combine: the directional combine already has every domain-0 GPU reply and
# signal pure-Attention domains only. Domain 0's Attention GPUs also read replies (their
# own, written locally by their forward warps) in their reduce epilogue, and the default
# reply inbox can overlap the dispatch inbox a pure-Attention source writes for the next
# exchange, so every rank keeps its reply inbox in a slice reserved just before the
# combine staging slice at the window end (calculate_buffer_size).
MIXED_HYBRID_COMBINE_EDITS = (
    (
        "#pragma once\n",
        "#pragma once  // afdEP: generated from hybrid_combine.cuh by "
        "tests/role_kernels.py; do not edit\n",
    ),
    (
        "          int kAfdFfnScaleoutRanks = 0,\n",
        "          int kAfdFfnScaleoutRanks = 0,\n"
        "          // afdEP mixed (afdep_suites.hpp): FFN ranks [0, F) share domain 0"
        " with "
        "Attention ranks.\n"
        "          int kAfdMixedFfnRanks = 0,\n",
    ),
    ("hybrid_combine_impl(", "afdep_mixed_hybrid_combine_impl("),
    (
        "    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,\n"
        "                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, "
        "comm::kHybridCombineTag0",
        "    // afdEP mixed: every rank's reply inbox lives in the slice before the"
        " staging "
        "slice.\n"
        "    EP_STATIC_ASSERT(kAfdFfnScaleoutRanks == 1 and 0 < kAfdMixedFfnRanks and "
        "kAfdMixedFfnRanks < kNumScaleupRanks,\n"
        '                     "mixed: one FFN-holding domain shared with Attention'
        ' ranks");\n'
        "    EP_DEVICE_ASSERT(afd_scaleout_send_base != nullptr);\n"
        "    scaleout_recv_buffer = layout::BufferLayout<false>(\n"
        "        token_layout, kNumTokensInScaleoutLayout, kNumMaxTokensPerRank,\n"
        "        math::advance_ptr(afd_scaleout_send_base, "
        "-scaleout_recv_buffer.get_num_bytes()));\n"
        "    comm::gpu_barrier<true, kNumScaleoutRanks, kNumScaleupRanks,\n"
        "                      kNumSMs, kNumThreads, kNumQPs, kNumTimeoutCycles, "
        "comm::kHybridCombineTag0",
    ),
)


def apply_edits(shared: str, edits) -> str:
    """Apply each edit exactly once."""
    for old, new in edits:
        if shared.count(old) != 1:
            raise ValueError(f"edit anchor not unique: {old!r}")
        shared = shared.replace(old, new)
    return shared


def span(text: str, start: str, end: str) -> tuple[int, int]:
    """Bounds of the text from start through end; each anchor occurs exactly once."""
    for anchor in (start, end):
        if text.count(anchor) != 1:
            raise ValueError(f"span anchor not unique: {anchor!r}")
    first, last = text.index(start), text.index(end)
    if last < first + len(start):
        raise ValueError(f"span anchors out of order: {start!r}")
    return first, last + len(end)


def wrap_span(text: str, start: str, end: str, prefix: str, suffix: str) -> str:
    """Surround the span from start through end; the span text is kept verbatim."""
    first, last = span(text, start, end)
    return text[:first] + prefix + text[first:last] + suffix + text[last:]


def replace_span(text: str, start: str, end: str, new: str) -> str:
    """Replace the span from start through end."""
    first, last = span(text, start, end)
    return text[:first] + new + text[last:]


def ffn_hybrid_dispatch(shared: str) -> str:
    """Forward warp f keeps channel sm*C+f, QP and TMA slot C+f; the launch bound and
    every other line are the shared kernel's."""
    return apply_edits(shared, FFN_HYBRID_DISPATCH_EDITS)


def source_hybrid_dispatch(shared: str) -> str:
    """Attention directional hybrid send with optional stagger and token counts; every
    other line is the shared kernel's."""
    return apply_edits(shared, SOURCE_HYBRID_DISPATCH_EDITS)


def ffn_sparse_hybrid_dispatch(shared: str) -> str:
    """The FFN hybrid copy whose forwarders also retire lanes by source token count."""
    return apply_edits(ffn_hybrid_dispatch(shared), FFN_SPARSE_HYBRID_DISPATCH_EDITS)


def ffn_sparse_hybrid_combine(shared: str) -> str:
    """The FFN hybrid combine signalling only channels that replied, plus channel 0."""
    return apply_edits(shared, FFN_SPARSE_HYBRID_COMBINE_EDITS)


def sparse_hybrid_collect(collect: str) -> str:
    """The Attention collect waiting only for channels it sent on, plus channel 0."""
    return apply_edits(collect, SPARSE_HYBRID_COLLECT_EDITS)


def mixed_hybrid_dispatch(shared: str) -> str:
    """The hybrid dispatch with per-rank roles in one mixed FFN-holding domain."""
    return apply_edits(shared, MIXED_HYBRID_DISPATCH_EDITS)


def mixed_hybrid_combine(shared: str) -> str:
    """The hybrid combine with every rank's reply inbox in the reserved slice."""
    return apply_edits(shared, MIXED_HYBRID_COMBINE_EDITS)


def ffn_dispatch(shared: str) -> str:
    """Same template, launch shape and statements; only the notify warps trigger."""
    return apply_edits(shared, FFN_DISPATCH_EDITS)


def source_dispatch(shared: str) -> str:
    """Count-first Attention source: the shared template plus the ungated-token and
    counting arguments; every other statement is the shared kernel's."""
    return wrap_span(apply_edits(shared, SOURCE_DISPATCH_EDITS), *SOURCE_COUNT_SPAN)


def ffn_stream_dispatch(shared: str) -> str:
    """Count-first FFN receive: same template and launch shape as the shared kernel."""
    return apply_edits(shared, FFN_STREAM_DISPATCH_EDITS)


def ffn_stream_epilogue(shared: str) -> str:
    """Count-first copy epilogue: the shared row body over claimed units of rows."""
    return replace_span(
        apply_edits(shared, FFN_STREAM_EPILOGUE_EDITS), *EPILOGUE_ROW_SPAN
    )


if __name__ == "__main__":
    hybrid = (IMPLS / "hybrid_dispatch.cuh").read_text()
    (IMPLS / FFN_HYBRID_DISPATCH).write_text(ffn_hybrid_dispatch(hybrid))
    (IMPLS / SOURCE_HYBRID_DISPATCH).write_text(source_hybrid_dispatch(hybrid))
    (IMPLS / FFN_SPARSE_HYBRID_DISPATCH).write_text(ffn_sparse_hybrid_dispatch(hybrid))
    combine = (IMPLS / "hybrid_combine.cuh").read_text()
    (IMPLS / FFN_SPARSE_HYBRID_COMBINE).write_text(ffn_sparse_hybrid_combine(combine))
    (IMPLS / MIXED_HYBRID_DISPATCH).write_text(mixed_hybrid_dispatch(hybrid))
    (IMPLS / MIXED_HYBRID_COMBINE).write_text(mixed_hybrid_combine(combine))
    collect = (IMPLS / "afdep_hybrid_collect.cuh").read_text()
    (IMPLS / SPARSE_HYBRID_COLLECT).write_text(sparse_hybrid_collect(collect))
    direct = (IMPLS / "dispatch.cuh").read_text()
    (IMPLS / FFN_DISPATCH).write_text(ffn_dispatch(direct))
    (IMPLS / SOURCE_DISPATCH).write_text(source_dispatch(direct))
    (IMPLS / FFN_STREAM_DISPATCH).write_text(ffn_stream_dispatch(direct))
    epilogue = (IMPLS / "dispatch_copy_epilogue.cuh").read_text()
    (IMPLS / FFN_STREAM_EPILOGUE).write_text(ffn_stream_epilogue(epilogue))
