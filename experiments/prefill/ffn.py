# SPDX-License-Identifier: Apache-2.0
"""Experimental BF16 FFN workspace; deliberately outside the communication API.

One packed row pool is shared by all local experts. Device prefix sums select
valid rows; group starts retain DeepGEMM's alignment, so this is not padding-free.
After warmup, the pooled path performs no host count reads or per-call allocations.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import triton
import triton.language as tl

PROGRAMS_PER_SM = 4


@dataclass(frozen=True)
class ExpertWeights:
    w13: torch.Tensor  # [local experts, 2 * intermediate, hidden]
    w2: torch.Tensor  # [local experts, hidden, intermediate]

    @property
    def num_experts(self) -> int:
        return self.w13.shape[0]

    @property
    def hidden(self) -> int:
        return self.w13.shape[2]

    @property
    def intermediate(self) -> int:
        return self.w2.shape[2]


def deterministic_weights(
    num_experts: int,
    hidden: int,
    intermediate: int,
    seed: int = 20261018,
    device: torch.device | str = "cuda",
    first_expert: int = 0,
) -> ExpertWeights:
    """Expert-global seeds let a source regenerate one reference expert at a time."""
    if min(num_experts, hidden, intermediate) <= 0 or first_expert < 0:
        raise ValueError("expert counts and dimensions must be positive")
    generator = torch.Generator(device=device)
    w13 = torch.empty(
        (num_experts, 2 * intermediate, hidden), dtype=torch.bfloat16, device=device
    )
    w2 = torch.empty(
        (num_experts, hidden, intermediate), dtype=torch.bfloat16, device=device
    )
    for local_expert in range(num_experts):
        generator.manual_seed(seed + first_expert + local_expert)
        w13[local_expert].normal_(std=1 / math.sqrt(hidden), generator=generator)
        w2[local_expert].normal_(std=1 / math.sqrt(intermediate), generator=generator)
    return ExpertWeights(w13, w2)


def load_deepgemm() -> ModuleType:
    """Load an optional installed package or an explicitly staged extension.

    AFDEP_DEEPGEMM_MODULE may name an importable module or a prebuilt .so. For a
    raw extension also set AFDEP_DEEPGEMM_ROOT to its matching deep_gemm JIT tree.
    This experiment neither vendors nor downloads DeepGEMM at import time.
    """
    module_name = os.environ.get("AFDEP_DEEPGEMM_MODULE", "deep_gemm")
    if module_name.endswith(".so"):
        path = Path(module_name).expanduser().resolve()
        # The staged historical extension exports this pybind initialization name.
        spec = importlib.util.spec_from_file_location("_afd_deepgemm", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load the DeepGEMM extension at {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        root = os.environ.get("AFDEP_DEEPGEMM_ROOT")
        if root is None:
            raise RuntimeError("a raw DeepGEMM extension needs AFDEP_DEEPGEMM_ROOT")
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        module.init(str(Path(root).expanduser().resolve()), cuda_home)
    else:
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise RuntimeError(
                "This optional experiment needs DeepGEMM's BF16 grouped psum API. "
                "Install a compatible deep_gemm, or stage an extension with "
                "AFDEP_DEEPGEMM_MODULE and AFDEP_DEEPGEMM_ROOT."
            ) from error
    if not hasattr(module, "m_grouped_bf16_gemm_nt_contiguous"):
        raise RuntimeError("DeepGEMM lacks the BF16 contiguous grouped GEMM API")
    return module


@triton.jit
def _swiglu(
    gate_up,
    activated,
    psum,
    ROWS: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(1)
    previous_end = tl.load(psum + expert - 1, mask=expert > 0, other=0)
    start = tl.cdiv(previous_end, ALIGNMENT) * ALIGNMENT
    end = tl.minimum(tl.load(psum + expert), ROWS)
    row = start + tl.program_id(0)
    col = tl.arange(0, BLOCK)
    # Device counts control the loop; capacity does not determine the CTA count.
    while row < end:
        gate = tl.load(
            gate_up + row * (2 * INTERMEDIATE) + col, col < INTERMEDIATE, 0
        ).to(tl.float32)
        up = tl.load(
            gate_up + row * (2 * INTERMEDIATE) + INTERMEDIATE + col,
            col < INTERMEDIATE,
            0,
        ).to(tl.float32)
        value = gate * tl.sigmoid(gate) * up
        tl.store(activated + row * INTERMEDIATE + col, value, col < INTERMEDIATE)
        row += tl.num_programs(0)


@triton.jit
def _weight_outputs(
    out,
    weights,
    psum,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ALIGNMENT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(1)
    previous_end = tl.load(psum + expert - 1, mask=expert > 0, other=0)
    start = tl.cdiv(previous_end, ALIGNMENT) * ALIGNMENT
    end = tl.minimum(tl.load(psum + expert), ROWS)
    row = start + tl.program_id(0)
    col = tl.arange(0, BLOCK)
    while row < end:
        value = tl.load(out + row * HIDDEN + col, col < HIDDEN, 0).to(tl.float32)
        weight = tl.load(weights + row)
        tl.store(out + row * HIDDEN + col, value * weight, col < HIDDEN)
        row += tl.num_programs(0)


class PrefillFFN:
    """Stable reply storage with pooled or eager exact-sized FFN intermediates.

    The caller owns input storage, count production, stream ordering, and credits.
    Prefix sums must describe aligned expert groups within the receive capacity;
    the communication layer enforces that bound. Only valid rows are written.
    There is one in-flight execution per instance; no asynchronous reuse is safe.
    """

    def __init__(
        self,
        weights: ExpertWeights,
        capacity_rows: int,
        expert_alignment: int = 128,
        deepgemm_module: ModuleType | None = None,
        expected_rows_per_expert: int | None = None,
    ) -> None:
        if capacity_rows <= 0 or expert_alignment <= 0:
            raise ValueError("workspace capacity and alignment must be positive")
        if expert_alignment != 128:
            raise ValueError("this DeepGEMM psum experiment requires 128-row alignment")
        if weights.w13.dtype != torch.bfloat16 or weights.w2.dtype != torch.bfloat16:
            raise ValueError("this experiment implements BF16 weights only")
        if not weights.w13.is_cuda or weights.w13.device != weights.w2.device:
            raise ValueError("weights must be on the same CUDA device")
        if weights.w13.ndim != 3 or weights.w2.ndim != 3:
            raise ValueError("expert weights must be three-dimensional")
        if min(weights.num_experts, weights.hidden, weights.intermediate) <= 0:
            raise ValueError("expert counts and dimensions must be positive")
        if (
            weights.w2.shape
            != (
                weights.num_experts,
                weights.hidden,
                weights.w13.shape[1] // 2,
            )
            or weights.w13.shape[1] % 2
        ):
            raise ValueError("incompatible gate/up and down projection shapes")
        if not weights.w13.is_contiguous() or not weights.w2.is_contiguous():
            raise ValueError("weights must be contiguous")
        if weights.hidden % 128 or weights.intermediate % 128:
            raise ValueError("DeepGEMM dimensions must be multiples of 128")
        if expected_rows_per_expert is not None and expected_rows_per_expert <= 0:
            raise ValueError("the expected rows per expert hint must be positive")
        self.weights = weights
        self.capacity_rows = capacity_rows
        self.alignment = expert_alignment
        # A fixed host-known hint keeps GEMM tuning independent of allocation
        # policy. In particular, exact row_limit must not change tile selection.
        self.expected_rows_per_expert = (
            max(1, capacity_rows // weights.num_experts)
            if expected_rows_per_expert is None
            else expected_rows_per_expert
        )
        num_sms = torch.cuda.get_device_properties(
            weights.w13.device
        ).multi_processor_count
        self.programs_per_expert = max(
            1, triton.cdiv(PROGRAMS_PER_SM * num_sms, weights.num_experts)
        )
        self.deepgemm = deepgemm_module or load_deepgemm()
        self.deepgemm.set_mk_alignment_for_contiguous_layout(expert_alignment)
        self.reply = torch.empty(
            (capacity_rows, weights.hidden),
            dtype=torch.bfloat16,
            device=weights.w13.device,
        )
        self._arena: torch.Tensor | None = None

    @property
    def workspace_bytes(self) -> int:
        """Includes pooled gate/up, activation AND reply; excludes received inputs.

        This configured capacity is not additive with a separately reported
        persistent reply allocation, and intermediates are lazy until warmup.
        """
        return (
            self.capacity_rows
            * (3 * self.weights.intermediate + self.weights.hidden)
            * 2
        )

    def _workspace(self, rows: int, allocate_exact: bool):
        if allocate_exact:
            arena = torch.empty(
                rows * 3 * self.weights.intermediate,
                dtype=torch.bfloat16,
                device=self.reply.device,
            )
        else:
            if self._arena is None:
                self._arena = torch.empty(
                    self.capacity_rows * 3 * self.weights.intermediate,
                    dtype=torch.bfloat16,
                    device=self.reply.device,
                )
            arena = self._arena
        gate_size = rows * 2 * self.weights.intermediate
        gate_up = arena[:gate_size].view(rows, 2 * self.weights.intermediate)
        activated = arena[gate_size : gate_size + rows * self.weights.intermediate]
        return gate_up, activated.view(rows, self.weights.intermediate)

    def __call__(
        self,
        recv_x: torch.Tensor,
        recv_weights: torch.Tensor,
        recv_count_psum: torch.Tensor,
        *,
        row_limit: int | None = None,
        allocate_exact: bool = False,
    ) -> torch.Tensor:
        if recv_x.ndim != 2 or recv_x.shape[1] != self.weights.hidden:
            raise ValueError("received inputs must have shape [rows, hidden]")
        rows = recv_x.shape[0] if row_limit is None else row_limit
        if not 0 <= rows <= recv_x.shape[0] <= self.capacity_rows:
            raise ValueError("received rows exceed the configured workspace capacity")
        if recv_x.dtype != torch.bfloat16 or not recv_x.is_contiguous():
            raise ValueError("received activations must be contiguous BF16")
        if recv_count_psum.shape != (self.weights.num_experts,):
            raise ValueError("prefix sums must have one entry per local expert")
        if recv_count_psum.dtype != torch.int32 or not recv_count_psum.is_contiguous():
            raise ValueError("prefix sums must be contiguous int32")
        if recv_weights.shape != (recv_x.shape[0],):
            raise ValueError("received weights must have one entry per received row")
        if recv_weights.dtype != torch.float32 or not recv_weights.is_contiguous():
            raise ValueError("received weights must be contiguous float32")
        if any(
            tensor.device != self.reply.device
            for tensor in (
                recv_x,
                recv_weights,
                recv_count_psum,
            )
        ):
            raise ValueError("all FFN inputs must be on the workspace device")
        if allocate_exact and row_limit is None:
            raise ValueError("exact allocation requires a host-known row_limit")
        if rows == 0:
            return self.reply[: recv_x.shape[0]]
        gate_up, activated = self._workspace(rows, allocate_exact)
        grouped = self.deepgemm.m_grouped_bf16_gemm_nt_contiguous
        grouped(
            recv_x[:rows],
            self.weights.w13,
            gate_up,
            recv_count_psum,
            use_psum_layout=True,
            expected_m_for_psum_layout=self.expected_rows_per_expert,
        )
        activation_block = triton.next_power_of_2(self.weights.intermediate)
        grid = (self.programs_per_expert, self.weights.num_experts)
        _swiglu[grid](
            gate_up,
            activated,
            recv_count_psum,
            rows,
            self.weights.intermediate,
            self.alignment,
            activation_block,
            num_warps=4 if activation_block <= 4096 else 8,
        )
        grouped(
            activated,
            self.weights.w2,
            self.reply[:rows],
            recv_count_psum,
            use_psum_layout=True,
            expected_m_for_psum_layout=self.expected_rows_per_expert,
        )
        output_block = triton.next_power_of_2(self.weights.hidden)
        _weight_outputs[grid](
            self.reply,
            recv_weights,
            recv_count_psum,
            rows,
            self.weights.hidden,
            self.alignment,
            output_block,
            num_warps=4 if output_block <= 4096 else 8,
        )
        return self.reply[: recv_x.shape[0]]
