# SPDX-License-Identifier: Apache-2.0
"""Torch-only DeepEP role validation; no model, GEMM, or performance claims.

Example: EP_DISABLE_GIN=1 torchrun --standalone --nproc-per-node=4 \
    -m tests.gpu.deepep --payload fp8
Fixed-shape inactive graph inputs use -1 routes; this is not a GPU live-count API.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from afdep import Buffer, Config

DEFAULT_HIDDEN = 256
EXPERTS_PER_FFN = 4
DEFAULT_TOP_K = 2
DEFAULT_FFN_RANKS = 2
GRAPH_STEPS = 26
DELAY_CYCLES = 2_000_000
COLLECT_VIEW = False  # --collect-view: Attention reads the single-FFN collect view
# FFN rank 0 delays replies; the last Attention rank delays its next dispatch.


@dataclass(frozen=True)
class Case:
    name: str
    counts: tuple[int, ...]
    epoch: int
    routing: str = "mixed"


def inputs(case: Case, source: int, rows: int, config: Config):
    tokens = torch.arange(rows, device="cuda", dtype=torch.int64)
    columns = torch.arange(config.hidden_size, device="cuda", dtype=torch.int64)
    signature = (tokens + 1) | ((source + 1) << 8) | (case.epoch << 16)
    signs = 2 * ((signature[:, None] >> (columns[None, :] % 24)) & 1) - 1
    magnitudes = (2.0 ** (columns % 4).float()) / 16
    x = (signs.float() * magnitudes).bfloat16()
    slots = torch.arange(config.top_k, device="cuda", dtype=torch.int64)
    if case.routing == "skew":
        ids = (tokens[:, None] + slots) % config.experts_per_rank
        ids = torch.where(slots < config.experts_per_rank, ids, -1)
    else:
        ids = (tokens[:, None] + slots + source + case.epoch) % config.num_experts
    weights = (
        torch.tensor(
            [0.25, 0.5] if config.top_k == 2 else [1 / 8] * config.top_k,
            dtype=torch.float32,
            device="cuda",
        )
        .expand(rows, -1)
        .contiguous()
    )
    active = tokens < case.counts[source]
    if case.routing == "skip":
        active = torch.zeros_like(active)
    ids = torch.where(active[:, None], ids, -1).contiguous()
    weights = torch.where(active[:, None] & (ids >= 0), weights, 0).contiguous()
    return x, ids, weights


def expert_rows(received, worker: int, config: Config):
    """Expert e applies (e+1)*identity and its received weight, exactly in BF16."""
    ends = received.recv_count_psum
    previous = torch.cat((ends.new_zeros(1), ends[:-1]))
    starts = (
        (previous + config.expert_alignment - 1) // config.expert_alignment
    ) * config.expert_alignment
    positions = torch.arange(received.recv_x.shape[0], device="cuda")[:, None]
    belongs = (positions >= starts[None, :]) & (positions < ends[None, :])
    factors = torch.arange(config.experts_per_rank, device="cuda", dtype=torch.float32)
    factors += worker * config.experts_per_rank + 1
    factor = (belongs.float() * factors).sum(dim=1)
    decoded = received.recv_x.float()
    if received.recv_scale is not None:
        decoded = (
            decoded.view(-1, config.hidden_size // 128, 128)
            * received.recv_scale[:, :, None]
        ).reshape_as(decoded)
    weighted = decoded * received.recv_topk_weights[:, None] * factor[:, None]
    return torch.where(belongs.any(dim=1)[:, None], weighted, 0).bfloat16()


class Exchange:
    def __init__(self, buffer: Buffer, rows: int):
        self.buffer, self.config, self.rows = buffer, buffer.config, rows
        if buffer.role == "attention":
            dtype = (
                torch.float8_e4m3fn if self.config.payload == "fp8" else torch.bfloat16
            )
            self.input = torch.empty(
                (rows, self.config.hidden_size), dtype=torch.bfloat16, device="cuda"
            )
            self.x = (
                torch.empty((rows, self.config.hidden_size), dtype=dtype, device="cuda")
                if self.config.payload == "fp8"
                else self.input
            )
            self.ids = torch.empty(
                (rows, self.config.top_k), dtype=buffer.topk_idx_dtype, device="cuda"
            )
            self.weights = torch.empty(
                (rows, self.config.top_k), dtype=torch.float32, device="cuda"
            )
            # A deliberately strided scale view also exercises native stride handling.
            self.scale = (
                torch.full(
                    (rows, (self.config.hidden_size // 128) * 2),
                    float("nan"),
                    dtype=torch.float32,
                    device="cuda",
                )[:, ::2]
                if self.config.payload == "fp8"
                else None
            )
            if self.scale is not None:
                groups = torch.arange(self.config.hidden_size // 128, device="cuda")
                self.scale.copy_(2.0 ** (groups % 3 - 1).float())

    def _quantize(self) -> None:
        if self.scale is not None:
            grouped = self.input.float().view(
                self.rows, self.config.hidden_size // 128, 128
            )
            self.x.copy_((grouped / self.scale[:, :, None]).reshape_as(self.input))

    def update(self, case: Case) -> None:
        if self.buffer.role == "attention":
            x, ids, weights = inputs(
                case, self.buffer.role_rank, self.rows, self.config
            )
            self.input.copy_(x)
            self._quantize()
            self.ids.copy_(ids)
            self.weights.copy_(weights)

    def run(self, *, host_counts=False, delay=False, alternate=False):
        if self.buffer.role == "attention":
            if alternate:
                self.input.neg_()
                self._quantize()
            if delay and self.buffer.role_rank == self.config.attention_ranks - 1:
                # A lagging source must not let peers overwrite its prior-exchange
                # inputs or receive before every source has published.
                torch.cuda._sleep(DELAY_CYCLES)
            self.buffer.source_dispatch(
                self.x,
                self.ids,
                self.weights,
                self.scale,
                bucket=self.config.capacity_tokens_per_rank,
                read_host_counts=host_counts,
            )
            if self.config.role_aware:
                assert self.buffer.transport._pending.recv_shape == (
                    0,
                    self.config.hidden_size,
                )
            return self.buffer.source_collect(view=COLLECT_VIEW).clone(), None
        received = self.buffer.worker_receive(
            bucket=self.config.capacity_tokens_per_rank, read_host_counts=host_counts
        )
        if self.config.role_aware and not host_counts:
            assert received.recv_x.shape[0] == self.config.receive_rows(
                self.config.capacity_tokens_per_rank
            )
        rows = expert_rows(received, self.buffer.role_rank, self.config)
        if delay and self.buffer.role_rank == 0:
            torch.cuda._sleep(DELAY_CYCLES)
        snapshot = received.recv_count_psum.clone()
        self.buffer.worker_reply(rows, received.receipt)
        return snapshot, received.recv_counts_host if host_counts else None


def combined_reference(x, ids, weights, config: Config, scaleup_ranks: int | None):
    """BF16 combine result along DeepEP's reduction tree.

    Each FFN rank sums its experts' BF16 rows into one BF16 partial; with hybrid
    (multi-domain) transport each FFN domain's forwarder sums its ranks' partials
    into one BF16 partial; the source sums what it receives. Every stage is an
    exact FP32 sum rounded once to BF16, so this equals a single rounding whenever
    the intermediates are exact (one FFN domain, small expert factors).
    """
    contribution = (weights * (ids + 1).float()).masked_fill(ids < 0, 0)
    rows_bf16 = (x.float()[:, None, :] * contribution[:, :, None]).bfloat16().float()
    owner = torch.where(ids >= 0, ids // config.experts_per_rank, -1)
    partials = [
        (rows_bf16 * (owner == rank)[:, :, None]).sum(dim=1).bfloat16().float()
        for rank in range(config.ffn_ranks)
    ]
    if scaleup_ranks is not None:
        partials = [
            sum(partials[start : start + scaleup_ranks]).bfloat16().float()
            for start in range(0, config.ffn_ranks, scaleup_ranks)
        ]
    return sum(partials).bfloat16()


def verify(snapshot, case: Case, buffer: Buffer, rows: int, *, sign=1) -> None:
    value, host_counts = snapshot
    config = buffer.config
    if buffer.role == "attention":
        x, ids, weights = inputs(case, buffer.role_rank, rows, config)
        scaleout, scaleup = buffer.logical_domain
        expected = combined_reference(
            sign * x.float(), ids, weights, config, scaleup if scaleout > 1 else None
        )
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        return
    first = buffer.role_rank * config.experts_per_rank
    counts = torch.zeros(config.experts_per_rank, dtype=torch.int64, device="cuda")
    for source in range(config.attention_ranks):
        _, ids, _ = inputs(case, source, case.counts[source], config)
        selected = ids[(ids >= first) & (ids < first + config.experts_per_rank)] - first
        counts += torch.bincount(selected, minlength=config.experts_per_rank)
    previous = torch.cat((value.new_zeros(1), value[:-1]))
    starts = (
        (previous + config.expert_alignment - 1) // config.expert_alignment
    ) * config.expert_alignment
    torch.testing.assert_close((value - starts).long(), counts, rtol=0, atol=0)
    if host_counts is not None:
        padded = (
            (counts + config.expert_alignment - 1) // config.expert_alignment
        ) * config.expert_alignment
        assert host_counts == padded.cpu().tolist()


def graph_checks(buffer, cases, steps, *, delayed):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    exchange = Exchange(buffer, buffer.config.capacity_tokens_per_rank)
    # New source storage was initialized on the default stream.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        exchange.update(cases[0])
        for _ in range(2):
            exchange.run()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = [exchange.run(delay=delayed, alternate=True) for _ in range(steps)]
    snapshots = []
    with torch.cuda.stream(stream):
        for case in cases:
            exchange.update(case)
            graph.replay()
            # Preserve every layer and replay before the next input/arena reuse.
            snapshots.append([(value.clone(), counts) for value, counts in captured])
    torch.cuda.synchronize()
    for case, layers in zip(cases, snapshots, strict=True):
        for index, snapshot in enumerate(layers):
            verify(
                snapshot,
                case,
                buffer,
                exchange.rows,
                sign=(-1 if index % 2 == 0 else 1),
            )
    graph.reset()
    dist.barrier()


def invalid_route_check(buffer, kind, execution):
    """Destructive device-assert test; run in its own torchrun process."""
    exchange = Exchange(buffer, buffer.config.capacity_tokens_per_rank)
    exchange.update(
        Case("valid_before_invalid", (8,) * buffer.config.attention_ranks, 1)
    )
    exchange.run()
    torch.cuda.synchronize()
    dist.barrier()
    graph = None
    if execution == "graph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            exchange.run()
        torch.cuda.synchronize()
    if buffer.role == "attention" and buffer.role_rank == 0:
        if kind == "duplicate":
            exchange.ids[0, 1].copy_(exchange.ids[0, 0])
        else:
            exchange.ids[0, 0] = {
                "dummy": buffer.config.num_experts,
                "negative": -2,
                "wide": 1 << 32,
            }[kind]
    torch.cuda.synchronize()
    dist.barrier()
    print(
        json.dumps(
            {
                "rank": buffer.rank,
                "invalid_route": kind,
                "execution": execution,
                "expect_device_assert": True,
            }
        ),
        flush=True,
    )
    try:
        graph.replay() if graph is not None else exchange.run()
        torch.cuda.synchronize()
    except Exception as exc:
        print(json.dumps({"rank": buffer.rank, "guard_failure": str(exc)}), flush=True)
        # A failed CUDA context cannot participate in collective close. Torchrun
        # must terminate peers; require the route-guard assertion in the log.
        os._exit(2)
    raise AssertionError("invalid role route unexpectedly completed")


def queued_eager_checks(buffer, cases):
    """Reuse source host-counter storage without inter-case host sync."""
    exchange = Exchange(buffer, buffer.config.capacity_tokens_per_rank)
    snapshots = []
    for case in cases:
        exchange.update(case)
        snapshots.append(exchange.run(host_counts=True, delay=True))
    torch.cuda.synchronize()
    for case, snapshot in zip(cases, snapshots, strict=True):
        verify(snapshot, case, buffer, exchange.rows)
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--graph-steps", type=int, default=GRAPH_STEPS)
    parser.add_argument("--num-sms", type=int, default=8)
    parser.add_argument("--hybrid", choices=("on", "off"), default="on")
    parser.add_argument(
        "--generic", action="store_true", help="Control: disable role specialization"
    )
    parser.add_argument(
        "--hidden-size", type=int, choices=(256, 2048), default=DEFAULT_HIDDEN
    )
    parser.add_argument("--top-k", type=int, choices=(2, 8), default=DEFAULT_TOP_K)
    parser.add_argument("--ffn-ranks", type=int, default=DEFAULT_FFN_RANKS)
    parser.add_argument("--experts-per-ffn", type=int, default=EXPERTS_PER_FFN)
    parser.add_argument(
        "--invalid-route", choices=("dummy", "duplicate", "negative", "wide")
    )
    parser.add_argument(
        "--invalid-execution", choices=("eager", "graph"), default="graph"
    )
    parser.add_argument(
        "--collect-view",
        action="store_true",
        help="Attention collects through source_collect(view=True)",
    )
    args = parser.parse_args()
    global COLLECT_VIEW
    COLLECT_VIEW = args.collect_view
    if min(args.capacity, args.graph_steps, args.num_sms) < 1:
        parser.error("capacity, graph-steps and num-sms must be positive")
    if args.invalid_route and (args.generic or args.capacity < 8):
        parser.error("invalid-route requires role specialization and capacity>=8")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    world = dist.get_world_size()
    if not 0 < args.ffn_ranks < world:
        parser.error("ffn-ranks must leave at least one Attention rank")
    config = Config(
        args.capacity,
        args.hidden_size,
        args.experts_per_ffn * args.ffn_ranks,
        args.top_k,
        args.ffn_ranks,
        world - args.ffn_ranks,
        payload=args.payload,
        num_sms=args.num_sms,
        allow_hybrid_mode=args.hybrid == "on",
        role_aware=not args.generic,
        cpu_timeout_s=10 if args.invalid_route else 120,
        gpu_timeout_s=10 if args.invalid_route else 120,
    )
    buffer = Buffer(group=dist.group.WORLD, config=config)
    # One NVLink domain, or NVLink domains joined by GIN (recorded per rank).
    if buffer.physical_domain[0] * buffer.physical_domain[1] != config.world_size:
        raise ValueError("physical domain does not cover the world")
    if config.role_aware and buffer.role == "attention":
        native_dispatch = buffer.transport.dispatch

        def checked_source_host_counts(*args, **kwargs):
            received = native_dispatch(*args, **kwargs)
            expected = [0] * config.experts_per_rank if kwargs["do_cpu_sync"] else []
            assert received.recv_counts_host == expected
            return received

        buffer.transport.dispatch = checked_source_host_counts
    if args.invalid_route:
        invalid_route_check(buffer, args.invalid_route, args.invalid_execution)
    n, sources = args.capacity, config.attention_ranks
    full, empty = (n,) * sources, (0,) * sources
    # Source 1 (or the only source) is inactive; prefixes differ per source.
    inactive = tuple(0 if i == min(1, sources - 1) else n for i in range(sources))
    prefix = tuple(min((3, 1, 2)[i % 3], n) for i in range(sources))
    cases = [
        Case("mixed", full, 1),
        Case("changed_routes", full, 2),
        Case("skew", full, 3, "skew"),
        Case("inactive_source", inactive, 4),
        Case("all_skipped", full, 5, "skip"),
        Case("all_empty", empty, 6),
        Case("partial_prefix", prefix, 7),
    ]
    for host_counts in (False, True):
        for case in cases:
            rows = case.counts[buffer.role_rank] if buffer.role == "attention" else n
            exchange = Exchange(buffer, rows)
            exchange.update(case)
            snapshot = exchange.run(host_counts=host_counts, delay=True)
            torch.cuda.synchronize()
            verify(snapshot, case, buffer, rows)
            dist.barrier()
    for delayed in (False, True):
        graph_checks(buffer, cases * 2, args.graph_steps, delayed=delayed)
    queued_eager_checks(buffer, cases)
    summary = {
        "rank": buffer.rank,
        "role": buffer.role,
        "config": asdict(config),
        "physical_domain": buffer.physical_domain,
        "gpu": torch.cuda.get_device_name(),
        "registered_transport_bytes": buffer.num_buffer_bytes,
        "receive_bound_rows": (
            0
            if config.role_aware and buffer.role == "attention"
            else config.receive_rows(config.capacity_tokens_per_rank)
        ),
        "source_zero_output_bytes": (
            buffer._zero_output.numel() * buffer._zero_output.element_size()
            if buffer.role == "attention"
            else 0
        ),
        "eager_cases": len(cases) * 2,
        "queued_eager_host_count_cases": len(cases),
        "graph_replays": len(cases) * 2 * 2,
        "checked_graph_exchanges": len(cases) * 2 * 2 * args.graph_steps,
    }
    summaries = [None] * config.world_size
    dist.all_gather_object(summaries, summary)
    buffer.close()
    dist.destroy_process_group()
    if buffer.rank == 0:
        print(
            json.dumps(
                {"ok": True, "validation": "DeepEP role API", "ranks": summaries}
            )
        )


if __name__ == "__main__":
    main()
