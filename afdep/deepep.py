# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""AFD role endpoints over the pinned DeepEP V2 ElasticBuffer.

Adapted from afd-plugin 1198c52, deepep_m2n/{buffer,layout}.py. This backend
retains DeepEP's union-rank protocol and dummy Attention expert slots.
It specializes unused role buffers and epilogues while preserving transport
credits. Model code, quantization and expert computation stay with the caller.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from ._build import load_deepep, source_digest

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

MAX_RANKS = 1024
MAX_LOGICAL_EXPERTS = 2048
MAX_EXPERTS_PER_RANK = 256
MAX_TOP_K = 32
HIDDEN_ALIGNMENT = 256
FP8_GROUP_SIZE = 128
MIN_COMM_SMS = 4
DEFAULT_COMM_SM_FRACTION = 24 / 132
HYBRID_DEFAULT_QPS = 129
DIRECT_DEFAULT_QPS = 17
INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class Config:
    """Real expert ownership and fixed allocation capacity, FFN ranks first."""

    capacity_tokens_per_rank: int
    hidden_size: int
    num_experts: int
    top_k: int
    ffn_ranks: int
    attention_ranks: int
    expert_alignment: int = 128
    payload: Literal["bf16", "fp8"] = "bf16"
    allow_hybrid_mode: bool = True
    num_sms: int = 0
    num_qps: int = 0
    rdma_traffic_class: int = 0
    cpu_timeout_s: int = 120
    gpu_timeout_s: int = 120
    role_aware: bool = True

    def __post_init__(self) -> None:
        for name in (
            "capacity_tokens_per_rank",
            "hidden_size",
            "num_experts",
            "top_k",
            "ffn_ranks",
            "attention_ranks",
            "expert_alignment",
            "cpu_timeout_s",
            "gpu_timeout_s",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < INT32_MAX:
                raise ValueError(f"{name} must be a positive int32 integer")
        for name in ("num_sms", "num_qps"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < INT32_MAX:
                raise ValueError(f"{name} must be a nonnegative int32 integer")
        if type(self.allow_hybrid_mode) is not bool:
            raise ValueError("allow_hybrid_mode must be a boolean")
        if type(self.role_aware) is not bool:
            raise ValueError("role_aware must be a boolean")
        if (
            type(self.rdma_traffic_class) is not int
            or not 0 <= self.rdma_traffic_class <= 255
        ):
            raise ValueError("rdma_traffic_class must be an integer in [0,255]")
        if self.hidden_size % HIDDEN_ALIGNMENT:
            raise ValueError("hidden_size must be a multiple of 256")
        if self.num_experts % self.ffn_ranks:
            raise ValueError("num_experts must divide evenly across FFN ranks")
        if self.top_k > min(MAX_TOP_K, self.num_experts):
            raise ValueError("top_k must be <=32 and <=num_experts")
        if self.payload not in {"bf16", "fp8"}:
            raise ValueError("payload must be 'bf16' or 'fp8'")
        if (
            self.world_size > MAX_RANKS
            or self.logical_experts > MAX_LOGICAL_EXPERTS
            or self.experts_per_rank > MAX_EXPERTS_PER_RANK
        ):
            raise ValueError(
                "expert topology exceeds the pinned DeepEP workspace bounds"
            )
        if self.receive_rows(self.capacity_tokens_per_rank) > INT32_MAX:
            raise ValueError("receive capacity exceeds int32 indexing")

    @property
    def world_size(self) -> int:
        return self.ffn_ranks + self.attention_ranks

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.ffn_ranks

    @property
    def logical_experts(self) -> int:
        """DeepEP's uniform expert blocks, including unused Attention slots."""
        return self.world_size * self.experts_per_rank

    def receive_rows(self, bucket: int) -> int:
        sources = self.attention_ranks if self.role_aware else self.world_size
        rows = sources * bucket * min(self.top_k, self.experts_per_rank)
        rows += (self.expert_alignment - 1) * self.experts_per_rank
        return (
            (rows + self.expert_alignment - 1)
            // self.expert_alignment
            * self.expert_alignment
        )


@dataclass(frozen=True)
class DispatchHandle:
    """Opaque dispatch receipt. Retain it unchanged through the matching reply."""

    num_experts: int
    num_max_tokens_per_rank: int
    num_sms: int
    topk_idx: torch.Tensor = field(repr=False)
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor = field(repr=False)
    recv_src_metadata: torch.Tensor = field(repr=False)
    token_metadata_at_forward: torch.Tensor | None = field(repr=False)
    channel_linked_list: torch.Tensor | None = field(repr=False)
    recv_shape: tuple[int, int]
    _owner: object = field(repr=False)


@dataclass(frozen=True)
class ReceiveResult:
    """Expert-major valid rows and device prefix sums; padding is unspecified."""

    recv_x: torch.Tensor
    recv_scale: torch.Tensor | None
    recv_topk_weights: torch.Tensor
    recv_count_psum: torch.Tensor
    recv_counts_host: list[int]
    handle: DispatchHandle

    @property
    def receipt(self) -> DispatchHandle:
        return self.handle


class ElasticBuffer:
    """DeepEP V2 facade with explicitly owned NCCL bootstrap.

    Each instance owns one native communicator and ElasticBuffer. The caller
    owns the CPU Gloo group and must collectively call close/destroy outside
    CUDA capture. No process-global communicator registry or environment changes.
    EP_DISABLE_GIN=1 is an application choice for a single NVLink domain.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        capacity_tokens_per_rank: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        allow_hybrid_mode: bool,
        rdma_traffic_class: int,
        cpu_timeout_s: int,
        gpu_timeout_s: int,
        use_fp8_dispatch: bool = False,
        num_sms: int = 0,
        num_qps: int = 0,
        role_aware: bool = False,
        ffn_ranks: int | None = None,
    ) -> None:
        # Intentional optional imports: configuration/package inspection stays CPU-only.
        import torch
        import torch.distributed as dist

        self._torch, self._dist, self._group = torch, dist, group
        self.rank, self.world_size = group.rank(), group.size()
        self.capacity_tokens_per_rank = capacity_tokens_per_rank
        self.num_experts, self.hidden_size, self.top_k = num_experts, hidden_size, top_k
        self.allow_hybrid_mode, self._fp8 = allow_hybrid_mode, use_fp8_dispatch
        self.role_aware, self.ffn_ranks = role_aware, ffn_ranks
        self._owner, self._pending, self._closed = object(), None, False
        settings = {
            "capacity": capacity_tokens_per_rank,
            "hidden": hidden_size,
            "experts": num_experts,
            "top_k": top_k,
            "hybrid": allow_hybrid_mode,
            "traffic_class": rdma_traffic_class,
            "cpu_timeout": cpu_timeout_s,
            "gpu_timeout": gpu_timeout_s,
            "fp8": use_fp8_dispatch,
            "num_sms": num_sms,
            "num_qps": num_qps,
            "role_aware": role_aware,
            "ffn_ranks": ffn_ranks,
        }
        error, module, device_sms = None, None, 0
        try:
            if not dist.is_initialized() or str(dist.get_backend(group)) != "gloo":
                raise ValueError(
                    "DeepEP bootstrap requires an initialized CPU Gloo group"
                )
            if self.world_size < 2 or not 0 <= self.rank < self.world_size:
                raise ValueError(
                    "DeepEP requires a valid group with at least two ranks"
                )
            for key in (
                "capacity",
                "hidden",
                "experts",
                "top_k",
                "cpu_timeout",
                "gpu_timeout",
            ):
                if type(settings[key]) is not int or not 0 < settings[key] < INT32_MAX:
                    raise ValueError(f"{key} must be a positive int32 integer")
            for key in ("num_sms", "num_qps"):
                if type(settings[key]) is not int or not 0 <= settings[key] < INT32_MAX:
                    raise ValueError(f"{key} must be a nonnegative int32 integer")
            if (
                type(allow_hybrid_mode) is not bool
                or type(use_fp8_dispatch) is not bool
                or type(role_aware) is not bool
            ):
                raise ValueError("hybrid, FP8 and role_aware settings must be booleans")
            if ffn_ranks is not None and (
                type(ffn_ranks) is not int or not 0 < ffn_ranks < self.world_size
            ):
                raise ValueError("ffn_ranks must leave positive FFN and source ranks")
            if role_aware and ffn_ranks is None:
                raise ValueError("role_aware requires explicit ffn_ranks")
            if (
                type(rdma_traffic_class) is not int
                or not 0 <= rdma_traffic_class <= 255
            ):
                raise ValueError("rdma_traffic_class must be an integer in [0,255]")
            if hidden_size % HIDDEN_ALIGNMENT or top_k > min(MAX_TOP_K, num_experts):
                raise ValueError(
                    "hidden must divide by 256; top_k must be <=32 and <=experts"
                )
            if num_experts % self.world_size:
                raise ValueError("num_experts must divide evenly over the group")
            if role_aware and top_k > ffn_ranks * (num_experts // self.world_size):
                raise ValueError("top_k exceeds the real FFN expert count")
            if (
                self.world_size > MAX_RANKS
                or num_experts > MAX_LOGICAL_EXPERTS
                or num_experts // self.world_size > MAX_EXPERTS_PER_RANK
                or self.world_size * capacity_tokens_per_rank * top_k > INT32_MAX
            ):
                raise ValueError(
                    "dimensions exceed the pinned DeepEP workspace/index bounds"
                )
            if (
                not torch.cuda.is_available()
                or torch.cuda.is_current_stream_capturing()
            ):
                raise RuntimeError(
                    "Create DeepEP buffers on CUDA, outside graph capture"
                )
            self.device = torch.cuda.current_device()
            device_sms = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
            if num_sms > device_sms:
                raise ValueError("num_sms exceeds the selected device SM count")
            module = load_deepep()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        records = self._gather(
            {
                "settings": settings,
                "source": source_digest(),
                "error": error,
                "sms": device_sms,
                "disable_gin": os.environ.get("EP_DISABLE_GIN", "0"),
            }
        )
        failures = [
            f"rank {i}: {r['error']}" for i, r in enumerate(records) if r["error"]
        ]
        if failures:
            raise RuntimeError("DeepEP preflight failed: " + "; ".join(failures))
        for key in ("settings", "source", "disable_gin"):
            if any(r[key] != records[0][key] for r in records):
                raise ValueError(f"Every DeepEP rank must use the same {key}")
        self._module = module
        self._resolved_num_sms = num_sms or max(
            MIN_COMM_SMS,
            round(min(r["sms"] for r in records) * DEFAULT_COMM_SM_FRACTION),
        )
        self.num_allocated_qps = num_qps or (
            HYBRID_DEFAULT_QPS if allow_hybrid_mode else DIRECT_DEFAULT_QPS
        )
        self.topk_idx_dtype = module.topk_idx_t
        ids = self._gather(
            module.get_local_nccl_unique_id() if self.rank == 0 else None
        )
        self._nccl_comm = int(
            module.create_nccl_comm(ids[0], self.world_size, self.rank)
        )
        try:
            self.physical_domain = tuple(
                module.get_physical_domain_size(self._nccl_comm)
            )
            if self.physical_domain[0] > 1 and records[0]["disable_gin"] == "1":
                raise ValueError(
                    "Multiple NVLink domains require GIN; EP_DISABLE_GIN=1 is set"
                )
            self.num_buffer_bytes = int(
                module.calculate_elastic_buffer_size(
                    self._nccl_comm,
                    capacity_tokens_per_rank + 1,
                    hidden_size,
                    top_k,
                    use_fp8_dispatch,
                    allow_hybrid_mode,
                    True,
                    # Role-sized inboxes: only sources write FFN dispatch regions
                    # and only FFN ranks write Attention combine regions.
                    ffn_ranks if role_aware else 0,
                )
            )
            native_args = (
                self.rank,
                self.world_size,
                self._nccl_comm,
                self.num_buffer_bytes,
                False,
                allow_hybrid_mode,
                True,
                True,
                rdma_traffic_class,
                self.num_allocated_qps,
                cpu_timeout_s,
                gpu_timeout_s,
                True,
            )
            self._runtime = module.ElasticBuffer(
                *native_args, *([ffn_ranks] if role_aware else [])
            )
        except Exception:
            module.destroy_nccl_comm(self._nccl_comm)
            self._closed = True
            raise
        self.logical_domain = tuple(self._runtime.get_logical_domain_size())
        self.num_scaleout_ranks, self.num_scaleup_ranks = self.logical_domain
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)
        self._runtime.barrier(True, False)
        torch.cuda.synchronize(self.device)
        # Count-first dispatch changes the wire protocol (afdep_suites.hpp), so every
        # rank must latch the same configuration: mixed role-kernel runs set
        # AFDEP_PAYLOAD_FIRST=1 on every rank. Checked once every workspace is ready,
        # so the collective close is safe.
        count_first = self._gather(tuple(self._runtime.get_afd_count_first_config()))
        if any(value != count_first[0] for value in count_first):
            self.close()
            raise ValueError(
                "Every DeepEP rank must use the same count-first configuration"
            )
        # The native constructor latches the screen arm unchecked (one rank throwing
        # there would leave its peers in the collective window registration); every
        # rank sees the same gathered value, so every rank rejects it here together.
        if count_first[0][1] not in ("full", "source"):
            self.close()
            raise ValueError("AFDEP_COUNT_FIRST_ARM must be full or source")

    def _gather(self, value) -> list:
        values = [None] * self._group.size()
        self._dist.all_gather_object(values, value, group=self._group)
        return values

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        return self._runtime.get_comm_stream()

    def _check_live(self) -> None:
        if self._closed:
            raise RuntimeError("DeepEP buffer is closed")
        if self._torch.cuda.current_device() != self.device:
            raise RuntimeError("DeepEP buffer used on a different CUDA device")

    def _num_sms(self) -> int:
        return self._resolved_num_sms

    def _num_qps(self, num_sms: int) -> int:
        multiplier = 16 if self.allow_hybrid_mode else 1
        return min(num_sms * multiplier + 1, self.num_allocated_qps)

    def dispatch(
        self,
        x: torch.Tensor,
        x_scale: torch.Tensor | None,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        bucket: int,
        expert_alignment: int,
        do_cpu_sync: bool,
    ) -> ReceiveResult:
        """Original expanded dispatch contract; retain handle through combine."""
        self._check_live()
        if self._pending is not None:
            raise RuntimeError("Complete the previous exchange before dispatch")
        if type(bucket) is not int or not 0 < bucket <= self.capacity_tokens_per_rank:
            raise ValueError("bucket must be positive and within capacity")
        if type(expert_alignment) is not int or expert_alignment < 1:
            raise ValueError("expert_alignment must be a positive integer")
        if type(do_cpu_sync) is not bool:
            raise ValueError("do_cpu_sync must be a boolean")
        if do_cpu_sync and self._torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Host receive counts are unavailable during CUDA capture"
            )
        if x.shape[0] > bucket:
            raise ValueError("source input exceeds its bucket")
        if self.role_aware and self.rank < self.ffn_ranks and x.shape[0] != 0:
            raise ValueError("role-aware FFN ranks cannot originate tokens")
        num_sms = self._num_sms()
        result = self._runtime.dispatch(
            x.contiguous(),
            x_scale,
            topk_ids.to(self.topk_idx_dtype).contiguous(),
            topk_weights.float().contiguous(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            bucket,
            self.num_experts,
            expert_alignment,
            num_sms,
            self._num_qps(num_sms),
            None,
            None,
            False,
            False,
            True,
            do_cpu_sync,
            True,
            True,
        )
        (
            recv_x,
            recv_scale,
            _,
            recv_weights,
            copied_ids,
            counts,
            rank_psum,
            expert_psum,
            src_metadata,
            _,
            forward_metadata,
            channel_links,
            _,
        ) = result
        if x_scale is not None and recv_scale is None:
            raise RuntimeError("DeepEP dispatch did not return activation scales")
        handle = DispatchHandle(
            self.num_experts,
            bucket,
            num_sms,
            copied_ids,
            rank_psum,
            src_metadata,
            forward_metadata,
            channel_links,
            tuple(recv_x.shape),
            self._owner,
        )
        self._pending = handle
        return ReceiveResult(
            recv_x, recv_scale, recv_weights, expert_psum, list(counts), handle
        )

    def combine(
        self, expert_output: torch.Tensor, handle: DispatchHandle, *, view: bool = False
    ) -> torch.Tensor:
        """Sum already-weighted expert rows into their original token order.

        ``view=True`` may return a strided view of the registered transport buffer
        (one FFN rank in one NVLink domain); it is valid until the next dispatch.
        """
        self._check_live()
        if handle._owner is not self._owner or handle is not self._pending:
            raise ValueError("Handle does not belong to this pending exchange")
        if tuple(expert_output.shape) != handle.recv_shape:
            raise ValueError("Expert output must match its dispatch receive shape")
        combined, _, _ = self._runtime.combine(
            expert_output.contiguous(),
            None,
            None,
            None,
            handle.recv_src_metadata,
            handle.topk_idx,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            handle.token_metadata_at_forward,
            handle.channel_linked_list,
            handle.num_experts,
            handle.num_max_tokens_per_rank,
            handle.num_sms,
            self._num_qps(handle.num_sms),
            None,
            None,
            False,
            False,
            True,
            view,
        )
        self._pending = None
        return combined

    def close(self) -> None:
        """Collective, outside capture, after all pairs finish; group stays alive."""
        if self._closed:
            return
        self._check_live()
        if self._torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Close DeepEP buffers outside CUDA graph capture")
        if any(self._gather(self._pending is not None)):
            raise RuntimeError(
                "Finish every rank's pending exchange before collective close"
            )
        self._torch.cuda.synchronize(self.device)
        self._dist.barrier(group=self._group)
        self._runtime.destroy()
        self._module.destroy_nccl_comm(self._nccl_comm)
        self._closed = True

    destroy = close


class Buffer:
    """AFD roles over ElasticBuffer, FFN ranks first, one in-flight exchange.

    Route only real experts [0,num_experts), with -1 for unused choices and no
    duplicate expert per token. Pass prequantized FP8 plus FP32 block128 scales,
    or BF16. Expert rows must be weighted exactly once before worker_reply.

    All ranks use the same bucket and read_host_counts policy per exchange.
    Capture complete role pairs with read_host_counts=False. Captured allocations
    retain stable graph-pool addresses; source shapes still specialize captures.
    Constructor/close are collective CPU operations outside capture. The caller
    owns the Gloo group and must keep payload/metadata alive through GPU use.
    """

    def __init__(self, *, group: dist.ProcessGroup, config: Config) -> None:
        # Roles must agree as well as the native uniform expert count.
        import torch.distributed as dist

        self.config = config
        configs = [None] * group.size()
        dist.all_gather_object(configs, asdict(config), group=group)
        if any(value != configs[0] for value in configs):
            raise ValueError("Every AFD rank must use the same Config")
        if group.size() != config.world_size:
            raise ValueError("group size does not match the configured AFD topology")
        self.rank = group.rank()
        self.role: Literal["ffn", "attention"] = (
            "ffn" if self.rank < config.ffn_ranks else "attention"
        )
        self.role_rank = (
            self.rank if self.role == "ffn" else self.rank - config.ffn_ranks
        )
        self.transport = ElasticBuffer(
            group=group,
            capacity_tokens_per_rank=config.capacity_tokens_per_rank,
            hidden_size=config.hidden_size,
            num_experts=config.logical_experts,
            top_k=config.top_k,
            allow_hybrid_mode=config.allow_hybrid_mode,
            rdma_traffic_class=config.rdma_traffic_class,
            cpu_timeout_s=config.cpu_timeout_s,
            gpu_timeout_s=config.gpu_timeout_s,
            use_fp8_dispatch=config.payload == "fp8",
            num_sms=config.num_sms,
            num_qps=config.num_qps,
            role_aware=config.role_aware,
            ffn_ranks=config.ffn_ranks,
        )
        self.physical_domain = self.transport.physical_domain
        self.logical_domain = self.transport.logical_domain
        self.num_buffer_bytes = self.transport.num_buffer_bytes
        self.topk_idx_dtype = self.transport.topk_idx_dtype
        self._torch, self.device = self.transport._torch, self.transport.device
        self._payload_dtype = (
            self._torch.float8_e4m3fn
            if config.payload == "fp8"
            else self._torch.bfloat16
        )
        self._allocate_role_storage()

    def _allocate_role_storage(self) -> None:
        torch, c = self._torch, self.config
        device = torch.device("cuda", self.device)
        self._zero_output = None
        if self.role == "attention":
            self._zero_output = torch.zeros(
                (
                    0 if c.role_aware else c.receive_rows(c.capacity_tokens_per_rank),
                    c.hidden_size,
                ),
                dtype=torch.bfloat16,
                device=device,
            )
        else:
            self._empty_x = torch.empty(
                (0, c.hidden_size), dtype=self._payload_dtype, device=device
            )
            self._empty_ids = torch.empty(
                (0, c.top_k), dtype=self.topk_idx_dtype, device=device
            )
            self._empty_weights = torch.empty(
                (0, c.top_k), dtype=torch.float32, device=device
            )
            self._empty_scale = (
                torch.empty(
                    (0, c.hidden_size // FP8_GROUP_SIZE),
                    dtype=torch.float32,
                    device=device,
                )
                if c.payload == "fp8"
                else None
            )

    def _check_role(self, role: str) -> None:
        self.transport._check_live()
        if self.role != role:
            raise RuntimeError(f"This operation requires the {role} role")

    def _tensor(
        self, tensor: torch.Tensor, dtype, shape: tuple[int, ...], *, contiguous=True
    ) -> None:
        if (
            not tensor.is_cuda
            or tensor.get_device() != self.device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or (contiguous and not tensor.is_contiguous())
        ):
            raise ValueError(
                "Payload tensor dtype, shape, device or contiguity mismatch"
            )

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        return self.transport.comm_stream

    def source_dispatch(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        x_scale: torch.Tensor | None = None,
        *,
        bucket: int,
        read_host_counts: bool = False,
    ) -> None:
        """Publish prequantized activations and real expert IDs; no casts/copies."""
        self._check_role("attention")
        c, torch = self.config, self._torch
        if x.ndim != 2:
            raise ValueError("source input must have shape [tokens,hidden]")
        n = x.shape[0]
        self._tensor(x, self._payload_dtype, (n, c.hidden_size))
        self._tensor(ids, self.topk_idx_dtype, (n, c.top_k))
        self._tensor(weights, torch.float32, (n, c.top_k))
        if (x_scale is not None) != (c.payload == "fp8"):
            raise ValueError("FP8 dispatch requires FP32 block128 scales")
        if x_scale is not None:
            self._tensor(
                x_scale,
                torch.float32,
                (n, c.hidden_size // FP8_GROUP_SIZE),
                contiguous=False,
            )
        self.transport.dispatch(
            x,
            x_scale,
            ids,
            weights,
            bucket=bucket,
            expert_alignment=c.expert_alignment,
            do_cpu_sync=read_host_counts,
        )

    def worker_receive(
        self, *, bucket: int, read_host_counts: bool = False
    ) -> ReceiveResult:
        """Participate with zero source tokens and receive expert-major rows."""
        self._check_role("ffn")
        return self.transport.dispatch(
            self._empty_x,
            self._empty_scale,
            self._empty_ids,
            self._empty_weights,
            bucket=bucket,
            expert_alignment=self.config.expert_alignment,
            do_cpu_sync=read_host_counts,
        )

    def source_collect(self, *, view: bool = False) -> torch.Tensor:
        """Return the native BF16 token output without an extra output copy.

        With ``view=True`` and one FFN rank in one NVLink domain, the result is a
        strided view of the registered combine inbox: no reduce kernel runs, and it
        stays valid only until the next ``source_dispatch`` on this Buffer. Other
        layouts return the ordinary result.
        """
        self._check_role("attention")
        handle = self.transport._pending
        if handle is None:
            raise RuntimeError("source_collect requires a matching dispatch")
        assert self._zero_output is not None
        return self.transport.combine(
            self._zero_output[: handle.recv_shape[0]], handle, view=view
        )

    def worker_reply(self, rows: torch.Tensor, receipt: DispatchHandle) -> None:
        """Reply with BF16 rows weighted exactly once by recv_topk_weights."""
        self._check_role("ffn")
        self._tensor(rows, self._torch.bfloat16, receipt.recv_shape)
        self.transport.combine(rows, receipt)

    def close(self) -> None:
        self.transport.close()
        self._zero_output = None


# Names used by the pre-v1 reuse connector. These share the v1 implementation.
DeepEPBuffer = Buffer
DeepEPConfig = Config
