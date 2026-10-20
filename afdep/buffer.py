# SPDX-License-Identifier: Apache-2.0
"""Role-specific PyTorch endpoints over the standalone AFDEP data plane."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.distributed as dist

from ._directional_build import (
    CUDA_CAPABILITY,
    MIN_NCCL_VERSION,
    load_native,
    source_digest,
)
from .config import Config


@dataclass(frozen=True)
class ReceiveResult:
    """Expert-major rows, routing weights, count metadata, and a reply receipt."""

    recv_x: torch.Tensor
    recv_scale: torch.Tensor | None
    recv_topk_weights: torch.Tensor
    recv_count_psum: torch.Tensor
    recv_counts_host: list[int]
    receipt: torch.Tensor


def _gather(group: dist.ProcessGroup, value) -> list:
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


class Buffer:
    """One collectively initialized, single-lane directional request/reply buffer.

    FFN workers precede Attention sources in group rank order. All
    methods enqueue on the current PyTorch CUDA stream. Initialization and
    ``close`` are collective CPU operations and must be outside CUDA capture.
    The caller selects the CUDA device and configures NIC affinity before init.
    """

    def __init__(self, *, group: dist.ProcessGroup, config: Config) -> None:
        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before creating AFDEP")
        if str(dist.get_backend(group)) != "gloo":
            raise ValueError("AFDEP bootstrap requires a CPU Gloo process group")
        if dist.get_world_size(group) != config.world_size:
            raise ValueError(
                f"The current backend requires a group of {config.world_size} ranks"
            )
        self.config = config
        self._group = group
        # A serving runtime can own another default world with different ranks.
        # get_rank(group) translates through that world's global-rank mapping;
        # our independent communicator already owns the correct local rank.
        self.rank = group.rank()
        self.role: Literal["ffn", "attention"] = (
            "ffn" if self.rank < config.ffn_ranks else "attention"
        )
        self.role_rank = (
            self.rank if self.role == "ffn" else self.rank - config.ffn_ranks
        )
        self._closed = False
        local_error = None
        try:
            if platform.system() != "Linux" or not torch.cuda.is_available():
                raise RuntimeError("The current CUDA backend requires Linux and CUDA")
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            if (properties.major, properties.minor) != CUDA_CAPABILITY:
                raise ValueError("The current CUDA backend requires SM90")
            if config.num_sms > properties.multi_processor_count:
                raise ValueError("num_sms exceeds the selected GPU's SM count")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Create AFDEP buffers outside CUDA graph capture")
            if (
                config.transport == "gin"
                and os.environ.get("EP_DISABLE_GIN", "0") != "0"
            ):
                raise ValueError("AFDEP requires GIN; unset EP_DISABLE_GIN")
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        records = _gather(
            group,
            {
                "config": asdict(config),
                "source": source_digest(),
                "error": local_error,
            },
        )
        failures = [
            f"rank {r}: {v['error']}" for r, v in enumerate(records) if v["error"]
        ]
        if failures:
            raise RuntimeError("AFDEP preflight failed: " + "; ".join(failures))
        if any(record["config"] != records[0]["config"] for record in records):
            raise ValueError("Every AFDEP rank must use an identical Config")
        if any(record["source"] != records[0]["source"] for record in records):
            raise ValueError("Every AFDEP rank must use identical native sources")
        self.device = torch.cuda.current_device()
        local_error = None
        try:
            self._module = load_native()
            runtime_version = self._module.get_nccl_version()
            if runtime_version < MIN_NCCL_VERSION:
                raise RuntimeError("AFDEP requires NCCL >= 2.30.7 at runtime")
            if runtime_version != self._module.compiled_nccl_version:
                raise RuntimeError(
                    "NCCL headers and loaded runtime versions differ. "
                    "AFDEP_NCCL_ROOT must match the libnccl.so.2 loaded by PyTorch; "
                    "set LD_LIBRARY_PATH before starting Python if needed."
                )
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        loaded = _gather(
            group,
            {
                "error": local_error,
                "nccl": None if local_error else runtime_version,
            },
        )
        failures = [
            f"rank {rank}: {record['error']}"
            for rank, record in enumerate(loaded)
            if record["error"]
        ]
        if failures:
            raise RuntimeError("AFDEP native load failed: " + "; ".join(failures))
        if any(record["nccl"] != loaded[0]["nccl"] for record in loaded):
            raise RuntimeError("Every AFDEP rank must use the same NCCL version")
        ids = _gather(
            group, self._module.get_nccl_unique_id() if self.rank == 0 else None
        )
        self._comm = self._module.create_nccl_comm(ids[0], config.world_size, self.rank)
        try:
            self._runtime = self._module.DirectionalM2NBuffer(
                self._comm,
                self.rank,
                config.world_size,
                config.ffn_ranks,
                config.attention_ranks,
                config.capacity_tokens_per_rank,
                config.hidden_size,
                config.top_k,
                config.num_experts,
                config.expert_alignment,
                config.payload == "fp8",
                config.num_gin_contexts,
                config.gpu_timeout_s,
                config.rdma_traffic_class,
                config.transport_stripes,
                config.num_sms,
                config.allow_hybrid_mode,
                config.transport == "nvlink",
                config.nvlink_early_metadata,
            )
            self.num_buffer_bytes = int(self._runtime.num_buffer_bytes)
            torch.cuda.synchronize(self.device)
            dist.barrier(group=group)
        except Exception:
            self._module.destroy_nccl_comm(self._comm)
            self._closed = True
            raise

    def _check_live(self, role: Literal["ffn", "attention"]) -> None:
        if self._closed:
            raise RuntimeError("AFDEP buffer is closed")
        if self.role != role:
            raise RuntimeError(f"This operation requires the {role} role")
        if torch.cuda.current_device() != self.device:
            raise RuntimeError("AFDEP buffer used on a different CUDA device")

    def source_dispatch(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        x_scale: torch.Tensor | None = None,
        *,
        bucket: int,
        live_tokens: torch.Tensor | None = None,
    ) -> None:
        """Publish activations/routes, optionally using a GPU int32[1] live count.

        The count selects a leading prefix within the fixed input/output shape.
        Keep its storage alive and ordered on the buffer stream through dispatch.
        """
        self._check_live("attention")
        self._runtime.source_dispatch(x, x_scale, ids, weights, bucket, live_tokens)

    def worker_receive(
        self,
        *,
        bucket: int,
        read_host_counts: bool = False,
        out: ReceiveResult | None = None,
        exact_rows: bool = False,
    ) -> ReceiveResult:
        """Receive rows, optionally sizing fresh storage after reading host counts.

        ``exact_rows`` requires host counts, no ``out``, and eager execution.
        Default and caller-owned storage use the conservative bucket capacity.
        """
        self._check_live("ffn")
        if exact_rows and not read_host_counts:
            raise ValueError("Exact receive allocation requires read_host_counts=True")
        if exact_rows and out is not None:
            raise ValueError("Exact receive allocation cannot use caller-owned storage")
        if read_host_counts and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Host receive counts are unavailable during CUDA capture"
            )
        if out is not None:
            return ReceiveResult(
                *self._runtime.worker_receive_into(
                    bucket,
                    read_host_counts,
                    out.recv_x,
                    out.recv_scale,
                    out.recv_topk_weights,
                    out.recv_count_psum,
                    out.receipt,
                )
            )
        return ReceiveResult(
            *self._runtime.worker_receive(bucket, read_host_counts, exact_rows)
        )

    def reserve_worker_receive(self, *, bucket: int) -> ReceiveResult:
        """Allocate reusable receive storage without starting an exchange."""
        self._check_live("ffn")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Reserve receive storage outside CUDA capture")
        config = self.config
        if type(bucket) is not int or not 0 < bucket <= config.capacity_tokens_per_rank:
            raise ValueError("Receive bucket must be positive and within capacity")
        rows = config.attention_ranks * bucket * min(
            config.top_k, config.experts_per_rank
        ) + config.experts_per_rank * (config.expert_alignment - 1)
        device = torch.device("cuda", self.device)
        scales = None
        if config.payload == "fp8":
            scales = torch.empty_strided(
                (rows, config.hidden_size // 128),
                (1, (rows + 3) // 4 * 4),
                dtype=torch.float32,
                device=device,
            )
        return ReceiveResult(
            torch.empty(
                (rows, config.hidden_size),
                dtype=(
                    torch.float8_e4m3fn if config.payload == "fp8" else torch.bfloat16
                ),
                device=device,
            ),
            scales,
            torch.empty(rows, dtype=torch.float32, device=device),
            torch.empty(config.experts_per_rank, dtype=torch.int32, device=device),
            [],
            torch.empty(
                (config.attention_ranks, bucket, config.top_k),
                dtype=torch.int32,
                device=device,
            ),
        )

    def worker_reply(self, rows: torch.Tensor, receipt: torch.Tensor) -> None:
        """Return BF16 expert rows already weighted by recv_topk_weights."""
        self._check_live("ffn")
        self._runtime.worker_reply(rows, receipt)

    def source_collect(self, out: torch.Tensor) -> None:
        """Collect worker contributions into BF16 [tokens,hidden]."""
        self._check_live("attention")
        self._runtime.source_collect(out)

    def close(self) -> None:
        """Collectively drain peers and free resources outside graph capture.

        All ranks must participate; do not use for rank-local exception cleanup.
        """
        if self._closed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Close AFDEP buffers outside CUDA graph capture")
        torch.cuda.synchronize(self.device)
        self._runtime.drain()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self._group)
        self._runtime.destroy()
        self._module.destroy_nccl_comm(self._comm)
        self._closed = True
