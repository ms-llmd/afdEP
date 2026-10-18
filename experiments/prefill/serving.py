# SPDX-License-Identifier: Apache-2.0
"""Mixed-size one-layer component experiment, not an end-to-end serving test.

Dynamic eager allocates receive/reply storage per request and sizes compute from
GPU-produced host counts. Pooled variants share one maximum-capacity workspace
across buckets; graph mode caches communication/FFN graphs, with timed source
staging and padding. Only standard buckets are warmed, never future trace inputs.
Queued mode uses resident inputs and matched source graphs. Host enqueue can run
ahead while the one device lane stays ordered; GPU intervals are not response
latency. Capture, first replay and input generation precede the trace clock.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist

import afdep
from examples.round_trip import ATTENTION_RANKS, FFN_RANKS, initialize
from experiments.prefill.benchmark import (
    REFERENCE_ATOL,
    REFERENCE_RTOL,
    SourceInputs,
    reference_output,
)
from experiments.prefill.ffn import PrefillFFN, deterministic_weights, load_deepgemm

VARIANTS = ("dynamic_eager", "pooled_eager", "pooled_graph")
DEFAULT_BUCKETS = (2048,)
US_PER_SECOND = 1_000_000


@dataclass(frozen=True)
class Request:
    counts: tuple[int, ...]
    pattern: str
    hot_worker: int
    epoch: int

    @property
    def maximum(self) -> int:
        return max(self.counts)


@dataclass
class Payload:
    x: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor


def selected_bucket(tokens: int, buckets: tuple[int, ...]) -> int:
    return next(bucket for bucket in buckets if bucket >= tokens)


def trace(requests: int, seed: int, maximum: int) -> list[Request]:
    """Alternate small/medium/large bursts, with non-aligned live lengths."""
    rng = random.Random(seed)
    result = []
    for index in range(requests):
        phase = index // 10 % 3
        low, high = ((1, 128), (129, 512), (513, maximum))[phase]
        high = min(high, maximum)
        low = min(low, high)
        largest = rng.randint(low, high)
        if index == 1:
            largest = 1
        if index == 20:
            largest = maximum
        counts = [rng.randint(max(1, largest // 2), largest) for _ in range(4)]
        counts[index % ATTENTION_RANKS] = largest
        if index % 17 == 0:
            counts[(index + 1) % ATTENTION_RANKS] = 0
        if index % 97 == 0:
            counts = [0] * ATTENTION_RANKS
        result.append(
            Request(
                tuple(counts),
                "hot" if index // 5 % 4 == 1 else "uniform",
                rng.randrange(FFN_RANKS),
                seed * 1009 + index,
            )
        )
    return result


def payload(config: afdep.Config, request: Request, source: int) -> Payload:
    """GPU routing counts are unknown to host allocation policy until dispatch."""
    count = request.counts[source]
    generator = torch.Generator(device="cuda").manual_seed(request.epoch + source * 97)
    x = (
        torch.randn(
            (count, config.hidden_size),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.25
    )
    limit = config.experts_per_rank if request.pattern == "hot" else config.num_experts
    base = torch.randint(limit, (count, 1), generator=generator, device="cuda")
    offsets = torch.randperm(limit, generator=generator, device="cuda")
    width = min(config.top_k, limit)
    ids = torch.full((count, config.top_k), -1, device="cuda", dtype=torch.int64)
    chosen = (base + offsets[None, :width]).remainder(limit)
    if request.pattern == "hot":
        chosen += request.hot_worker * config.experts_per_rank
    omitted = torch.rand((count, width), generator=generator, device="cuda") < (1 / 11)
    ids[:, :width] = torch.where(omitted, -1, chosen)
    if request.pattern == "allskip":
        ids.fill_(-1)
    slots = torch.arange(config.top_k, device="cuda", dtype=torch.float32)
    weights = (1 / (slots[None, :] + 2)).expand(count, config.top_k).contiguous()
    return Payload(x.contiguous(), ids, weights)


def allocation_snapshot() -> dict:
    stats = torch.cuda.memory_stats()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "allocation_requests": stats.get("allocation.all.allocated", 0),
        "allocation_request_bytes": stats.get("allocated_bytes.all.allocated", 0),
        "num_device_alloc": stats.get("num_device_alloc", 0),
        "num_device_free": stats.get("num_device_free", 0),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_sync_all_streams": stats.get("num_sync_all_streams", 0),
        "num_ooms": stats.get("num_ooms", 0),
    }


def cpu_snapshot() -> dict:
    """Best-effort cgroup v2 counters; local ranks may share these counters."""
    stats = {}
    quota = None
    try:
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines():
            key, value = line.split()
            stats[key] = int(value)
    except (OSError, ValueError):
        pass
    try:
        quota = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError:
        pass
    return {"stat": stats, "cpu_max": quota}


class Pipeline:
    """One ordered device lane; stage/compute/reply precede the next reuse."""

    def __init__(
        self,
        buffer,
        config,
        rank,
        variant,
        buckets,
        weights,
        module,
        live_token_count=False,
        source_policy="variant",
    ):
        self.buffer = buffer
        self.config = config
        self.rank = rank
        self.variant = variant
        self.buckets = buckets
        self.fixed_bucket = buckets[0] if len(buckets) == 1 else None
        self.source_policy = source_policy
        self.dynamic_source = variant == "dynamic_eager" and source_policy == "variant"
        self.capture_for_role = (
            variant == "pooled_graph"
            if rank < FFN_RANKS
            else variant == "pooled_graph" or source_policy == "matched_graph"
        )
        self.receive = None
        self.views = {}
        self.graphs = {}
        self.executor = None
        self.stage = None
        self.output = None
        self.live_tokens = None
        self.logical_allocations = 0
        self.logical_allocation_bytes = 0
        if rank < FFN_RANKS:
            maximum_rows = self.rows(buckets[-1])
            self.executor = PrefillFFN(
                weights,
                capacity_rows=maximum_rows,
                expert_alignment=config.expert_alignment,
                deepgemm_module=module,
                expected_rows_per_expert=self.hint(buckets[-1]),
            )
            self.empty_reply = torch.empty(
                (0, config.hidden_size), device="cuda", dtype=torch.bfloat16
            )
            if variant == "dynamic_eager":
                self.executor.reply = self.empty_reply
            else:
                self.receive = buffer.reserve_worker_receive(bucket=buckets[-1])
                for bucket in buckets:
                    rows = self.rows(bucket)
                    # Native receipts are packed [sources,bucket,topk]. Taking
                    # [:,:bucket] on max storage would be noncontiguous/wrong.
                    receipt = self.receive.receipt.reshape(-1)[
                        : ATTENTION_RANKS * bucket * config.top_k
                    ].view(ATTENTION_RANKS, bucket, config.top_k)
                    self.views[bucket] = afdep.ReceiveResult(
                        self.receive.recv_x[:rows],
                        None,
                        self.receive.recv_topk_weights[:rows],
                        self.receive.recv_count_psum,
                        [],
                        receipt,
                    )
        elif not self.dynamic_source:
            maximum = buckets[-1]
            self.stage = Payload(
                torch.empty(
                    (maximum, config.hidden_size), device="cuda", dtype=torch.bfloat16
                ),
                torch.empty((maximum, config.top_k), device="cuda", dtype=torch.int64),
                torch.empty(
                    (maximum, config.top_k), device="cuda", dtype=torch.float32
                ),
            )
            self.output = torch.empty_like(self.stage.x)
            if live_token_count:
                self.live_tokens = torch.empty(1, device="cuda", dtype=torch.int32)

    def rows(self, bucket: int) -> int:
        config = self.config
        return ATTENTION_RANKS * bucket * min(
            config.top_k, config.experts_per_rank
        ) + config.experts_per_rank * (config.expert_alignment - 1)

    def hint(self, bucket: int) -> int:
        return max(
            1, ATTENTION_RANKS * bucket * self.config.top_k // self.config.num_experts
        )

    def stage_source(self, data: Payload, bucket: int) -> None:
        count = data.x.shape[0]
        if self.live_tokens is None:
            self.stage.ids[:bucket].fill_(-1)
            self.stage.weights[:bucket].zero_()
        self.stage.x[:count].copy_(data.x)
        self.stage.ids[:count].copy_(data.ids)
        self.stage.weights[:count].copy_(data.weights)
        if self.live_tokens is not None:
            self.live_tokens.fill_(count)

    def worker(self, bucket: int) -> None:
        received = self.buffer.worker_receive(bucket=bucket, out=self.views[bucket])
        reply = self.executor(
            received.recv_x, received.recv_topk_weights, received.recv_count_psum
        )
        self.buffer.worker_reply(reply, received.receipt)

    def source(self, bucket: int) -> None:
        self.buffer.source_dispatch(
            self.stage.x[:bucket],
            self.stage.ids[:bucket],
            self.stage.weights[:bucket],
            bucket=bucket,
            **(
                {"live_tokens": self.live_tokens}
                if self.live_tokens is not None
                else {}
            ),
        )
        self.buffer.source_collect(self.output[:bucket])

    def execute(self, request: Request, data: Payload | None) -> None:
        # Bucket selection, graph lookup, allocations and source staging are
        # deliberately part of the request's timed interval.
        bucket = (
            self.fixed_bucket
            if self.fixed_bucket is not None
            else selected_bucket(request.maximum, self.buckets)
        )
        if self.rank < FFN_RANKS:
            if self.fixed_bucket is None:
                self.executor.expected_rows_per_expert = self.hint(bucket)
            if self.variant == "dynamic_eager":
                self.executor.reply = self.empty_reply
                received = self.buffer.worker_receive(
                    bucket=(
                        bucket
                        if self.source_policy == "matched_graph"
                        else max(1, request.maximum)
                    ),
                    read_host_counts=True,
                    exact_rows=True,
                )
                rows = sum(received.recv_counts_host)
                self.executor.reply = torch.empty_like(received.recv_x)
                self.last_compute_rows = rows
                self.last_receive_rows = received.recv_x.shape[0]
                reply = self.executor(
                    received.recv_x,
                    received.recv_topk_weights,
                    received.recv_count_psum,
                    row_limit=rows,
                    allocate_exact=True,
                )
                self.buffer.worker_reply(reply, received.receipt)
            elif self.variant == "pooled_graph":
                self.graphs[bucket].replay()
            else:
                self.worker(bucket)
        elif self.dynamic_source:
            self.output = None
            self.output = torch.empty_like(data.x)
            self.buffer.source_dispatch(
                data.x,
                data.ids,
                data.weights,
                bucket=max(1, request.maximum),
            )
            self.buffer.source_collect(self.output)
        else:
            self.stage_source(data, bucket)
            if self.capture_for_role:
                self.graphs[bucket].replay()
            else:
                self.source(bucket)

    def account(self, request: Request) -> None:
        """Record logical allocation requests after the request clock stops."""
        if self.variant != "dynamic_eager":
            return
        config = self.config
        if self.rank >= FFN_RANKS:
            if not self.dynamic_source:
                return
            self.logical_allocations += 1
            self.logical_allocation_bytes += (
                request.counts[self.rank - FFN_RANKS] * config.hidden_size * 2
            )
        else:
            self.logical_allocations += 5 + int(self.last_compute_rows > 0)
            self.logical_allocation_bytes += (
                self.last_receive_rows * (4 * config.hidden_size + 4)
                + config.experts_per_rank * 4
                + ATTENTION_RANKS
                * (
                    selected_bucket(request.maximum, self.buckets)
                    if self.source_policy == "matched_graph"
                    else max(1, request.maximum)
                )
                * config.top_k
                * 4
                + self.last_compute_rows * 6 * self.executor.weights.intermediate
            )

    def release_last_request(self) -> None:
        if self.variant == "dynamic_eager":
            if self.executor is not None:
                self.executor.reply = self.empty_reply
            elif self.dynamic_source:
                self.output = None

    def storage_pointers(self) -> tuple[int, ...]:
        if (self.rank < FFN_RANKS and self.variant == "dynamic_eager") or (
            self.rank >= FFN_RANKS and self.dynamic_source
        ):
            return ()
        if self.receive is not None:
            tensors = (
                self.receive.recv_x,
                self.receive.recv_topk_weights,
                self.receive.recv_count_psum,
                self.receive.receipt,
                self.executor.reply,
                self.executor._arena,
            )
        else:
            tensors = (self.stage.x, self.stage.ids, self.stage.weights, self.output)
            if self.live_tokens is not None:
                tensors += (self.live_tokens,)
        return tuple(tensor.data_ptr() for tensor in tensors)

    def prepare(self, warmup: int) -> dict:
        warm_us = 0.0
        capture_us = 0.0
        first_replay_us = 0.0
        for bucket in self.buckets:
            for iteration in range(warmup):
                request = Request((bucket,) * 4, "uniform", 0, 9000 + iteration)
                data = (
                    payload(self.config, request, self.rank - FFN_RANKS)
                    if self.rank >= FFN_RANKS
                    else None
                )
                torch.cuda.synchronize()
                dist.barrier()
                started = time.perf_counter()
                # Graphs do not exist until the first eager warmup finishes.
                if self.capture_for_role:
                    if self.rank < FFN_RANKS:
                        self.executor.expected_rows_per_expert = self.hint(bucket)
                        self.worker(bucket)
                    else:
                        self.stage_source(data, bucket)
                        self.source(bucket)
                else:
                    self.execute(request, data)
                torch.cuda.synchronize()
                warm_us += (time.perf_counter() - started) * US_PER_SECOND
                dist.barrier()
            if self.capture_for_role:
                started = time.perf_counter()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    if self.rank < FFN_RANKS:
                        self.executor.expected_rows_per_expert = self.hint(bucket)
                        self.worker(bucket)
                    else:
                        self.source(bucket)
                self.graphs[bucket] = graph
                torch.cuda.synchronize()
                capture_us += (time.perf_counter() - started) * US_PER_SECOND
            # Some variants capture only the source role. All ranks must match
            # capture barriers and the subsequent source/worker prime exchange.
            dist.barrier()
            if self.variant == "pooled_graph" or self.source_policy == "matched_graph":
                started = time.perf_counter()
                if self.capture_for_role:
                    self.graphs[bucket].replay()
                else:
                    self.execute(request, data)
                torch.cuda.synchronize()
                first_replay_us += (time.perf_counter() - started) * US_PER_SECOND
                dist.barrier()
        self.release_last_request()
        self.logical_allocations = 0
        self.logical_allocation_bytes = 0
        return {
            "standard_bucket_warmup_us": warm_us,
            "capture_us": capture_us,
            "first_graph_replay_us": first_replay_us,
            "captured_graphs": len(self.graphs),
            "primed_graphs": len(self.graphs),
            "graph_strategy": (
                "single_capacity" if self.fixed_bucket is not None else "bucketed"
            ),
            "runtime_recaptures": 0,
            "one_maximum_receive_pool": self.receive is not None,
            "one_maximum_ffn_arena": self.executor is not None
            and self.variant != "dynamic_eager",
            "persistent_live_count_bytes": 4 if self.live_tokens is not None else 0,
        }

    def check(self, data: Payload | None, args) -> None:
        if data is None:
            return
        count = data.x.shape[0]
        expected = reference_output(
            SourceInputs(data.x, data.ids, data.weights, self.output[:count]),
            self.config,
            args.intermediate,
            args.weight_seed,
        )
        torch.testing.assert_close(
            self.output[:count], expected, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
        )

    def close(self) -> None:
        torch.cuda.synchronize()
        dist.barrier()
        self.graphs.clear()
        self.views.clear()
        self.receive = None
        self.stage = None
        self.output = None
        self.live_tokens = None
        self.executor = None
        gc.collect()
        torch.cuda.empty_cache()


def correctness(buffer, config, rank, args, weights, module) -> list[str]:
    maximum = args.buckets[-1]
    requests = [
        Request((maximum, 1, 0, min(17, maximum)), "uniform", 0, 7101),
        Request((1, 0, 1, 1), "hot", 2, 7102),
        Request((min(129, maximum), min(7, maximum), 1, 0), "uniform", 0, 7103),
        Request((min(31, maximum), 0, min(4, maximum), 0), "allskip", 0, 7104),
        Request((0, 0, 0, 0), "uniform", 0, 7105),
        Request((maximum,) * ATTENTION_RANKS, "hot", 1, 7106),
    ]
    checked = []
    for variant in args.variants:
        pipeline = Pipeline(
            buffer,
            config,
            rank,
            variant,
            args.buckets,
            weights,
            module,
            args.live_token_count,
            args.source_policy,
        )
        pipeline.prepare(args.warmup)
        pointers = pipeline.storage_pointers()
        resident = (
            [payload(config, request, rank - FFN_RANKS) for request in requests]
            if args.execution == "queued" and rank >= FFN_RANKS
            else None
        )
        snapshots = (
            [torch.empty_like(data.x) for data in resident]
            if resident is not None
            else None
        )
        torch.cuda.synchronize()
        dist.barrier()
        for index, request in enumerate(requests):
            data = (
                resident[index]
                if resident is not None
                else payload(config, request, rank - FFN_RANKS)
                if rank >= FFN_RANKS
                else None
            )
            if args.execution != "queued":
                torch.cuda.synchronize()
            if rank == 1 and index == 2:
                torch.cuda._sleep(5_000_000)
            pipeline.execute(request, data)
            if snapshots is not None:
                snapshots[index].copy_(pipeline.output[: data.x.shape[0]])
            if args.execution != "queued":
                torch.cuda.synchronize()
                pipeline.check(data, args)
            assert pipeline.storage_pointers() == pointers, "pooled storage moved"
        torch.cuda.synchronize()
        if snapshots is not None:
            for data, snapshot in zip(resident, snapshots, strict=True):
                expected = reference_output(
                    SourceInputs(data.x, data.ids, data.weights, snapshot),
                    config,
                    args.intermediate,
                    args.weight_seed,
                )
                torch.testing.assert_close(
                    snapshot, expected, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
                )
        dist.barrier()
        checked.append(variant)
        pipeline.close()
    return checked


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure(buffer, config, rank, args, weights, module, variant, repeat) -> dict:
    queued = args.execution == "queued"
    requests = trace(args.requests, args.trace_seed + repeat * 7919, args.buckets[-1])
    generation_started = time.perf_counter()
    resident = (
        [payload(config, request, rank - FFN_RANKS) for request in requests]
        if queued and rank >= FFN_RANKS
        else None
    )
    torch.cuda.synchronize()
    generation_us = (time.perf_counter() - generation_started) * US_PER_SECOND
    resident_bytes = (
        sum(
            tensor.numel() * tensor.element_size()
            for data in resident
            for tensor in (data.x, data.ids, data.weights)
        )
        if resident is not None
        else 0
    )
    gc.collect()
    torch.cuda.empty_cache()
    baseline = allocation_snapshot()
    started = time.perf_counter()
    pipeline = Pipeline(
        buffer,
        config,
        rank,
        variant,
        args.buckets,
        weights,
        module,
        args.live_token_count,
        args.source_policy,
    )
    construction_us = (time.perf_counter() - started) * US_PER_SECOND
    startup = pipeline.prepare(args.warmup)
    startup["construction_us"] = construction_us
    storage_pointers = pipeline.storage_pointers()
    events = []
    origin = torch.cuda.Event(enable_timing=True) if queued else None
    if args.cuda_events:
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in requests
        ]
        for begin, end in events:
            begin.record()
            end.record()
        if origin is not None:
            origin.record()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = allocation_snapshot()
    samples = []
    drain_samples = []
    dist.barrier()  # Exactly one initial barrier; no per-request Gloo control.
    cpu_before = cpu_snapshot()
    trace_started = time.perf_counter()
    if origin is not None:
        origin.record()
    for index, request in enumerate(requests):
        # Closed-loop generation is outside each request clock but inside the
        # trace. Queued inputs are already resident, outside both clocks.
        data = (
            resident[index]
            if resident is not None
            else payload(config, request, rank - FFN_RANKS)
            if rank >= FFN_RANKS
            else None
        )
        if not queued:
            torch.cuda.synchronize()
        context = (
            torch.cuda.nvtx.range(f"mixed/{variant}/request={index}/rank={rank}")
            if args.nvtx
            else nullcontext()
        )
        with context:
            started = time.perf_counter()
            if events:
                events[index][0].record()
            pipeline.execute(request, data)
            submit_us = (time.perf_counter() - started) * US_PER_SECOND
            if events:
                events[index][1].record()
            if not queued:
                torch.cuda.synchronize()
                wall_us = (time.perf_counter() - started) * US_PER_SECOND
        if not queued:
            pipeline.account(request)
        samples.append(
            {"host_enqueue_us": submit_us}
            if queued
            else {"wall_us": wall_us, "host_submit_us": submit_us}
        )
        if queued and args.queue_depth and (index + 1) % args.queue_depth == 0:
            drain_started = time.perf_counter()
            torch.cuda.synchronize()
            drain_samples.append((time.perf_counter() - drain_started) * US_PER_SECOND)
    if queued:
        drain_started = time.perf_counter()
        torch.cuda.synchronize()
        drain_samples.append((time.perf_counter() - drain_started) * US_PER_SECOND)
    trace_wall_s = time.perf_counter() - trace_started
    cpu_after = cpu_snapshot()
    after = allocation_snapshot()
    assert pipeline.storage_pointers() == storage_pointers, "pooled storage moved"
    # Validate final live token rows outside both the clock and allocator snapshot.
    pipeline.check(data, args)
    local = {
        "rank": rank,
        "startup": startup,
        "trace_wall_s": trace_wall_s,
        "cpu_cgroup": {
            "cpu_max": cpu_after["cpu_max"],
            "scope": "local_ranks_can_share_counters_do_not_sum",
            "deltas": {
                key: cpu_after["stat"][key] - cpu_before["stat"][key]
                if key in cpu_before["stat"] and key in cpu_after["stat"]
                else None
                for key in (
                    "nr_periods",
                    "nr_throttled",
                    "throttled_usec",
                    "usage_usec",
                )
            },
        },
        "samples": samples,
        "logical_workspace_allocations": (
            None if queued else pipeline.logical_allocations
        ),
        "logical_workspace_allocation_bytes": (
            None if queued else pipeline.logical_allocation_bytes
        ),
        "baseline_before_setup": baseline,
        "before_trace": before,
        "after_trace": after,
        "peak_allocated_increment_from_baseline_bytes": (
            after["peak_allocated_bytes"] - baseline["allocated_bytes"]
        ),
        "allocator_counter_deltas": {
            key: after[key] - before[key]
            for key in (
                "allocation_requests",
                "allocation_request_bytes",
                "num_device_alloc",
                "num_device_free",
                "num_alloc_retries",
                "num_sync_all_streams",
                "num_ooms",
            )
        },
        "registered_transport_bytes": buffer.num_buffer_bytes,
        "final_output_checked": rank >= FFN_RANKS,
        "pooled_storage_addresses_stable": bool(storage_pointers),
    }
    if events:
        local["cuda_event_us"] = [
            begin.elapsed_time(end) * 1000 for begin, end in events
        ]
    if queued:
        local.update(
            {
                "resident_input_bytes": resident_bytes,
                "allocation_accounting": (
                    "use_torch_allocator_deltas_no_per_request_logging"
                ),
                "resident_input_generation_us": generation_us,
                "host_window_drain_us": drain_samples,
                "gpu_completion_offset_us": [
                    origin.elapsed_time(end) * 1000 for _, end in events
                ],
                "gpu_interrequest_gap_us": [0.0]
                + [
                    events[index - 1][1].elapsed_time(events[index][0]) * 1000
                    for index in range(1, len(events))
                ],
            }
        )
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, local)
    maximum_wall = (
        [
            max(row["cuda_event_us"][i] for row in ranks[FFN_RANKS:])
            for i in range(len(requests))
        ]
        if queued
        else [
            max(row["samples"][i]["wall_us"] for row in ranks[FFN_RANKS:])
            for i in range(len(requests))
        ]
    )
    trace_wall_s = max(row["trace_wall_s"] for row in ranks)
    useful_tokens = sum(sum(request.counts) for request in requests)
    result = {
        "variant": variant,
        "repeat": repeat,
        "trace_seed": args.trace_seed + repeat * 7919,
        "live_token_count": args.live_token_count,
        "graph_strategy": "single_capacity" if len(args.buckets) == 1 else "bucketed",
        "max_rank_trace_wall_s": trace_wall_s,
        "useful_source_tokens": useful_tokens,
        "useful_source_tokens_per_trace_wall_second": useful_tokens / trace_wall_s,
        "max_attention_gpu_service_us" if queued else "max_attention_wall_us": {
            "p50": statistics.median(maximum_wall),
            "p95": percentile(maximum_wall, 0.95),
            "p99": percentile(maximum_wall, 0.99),
        },
        "requests": [
            {
                **asdict(request),
                "selected_bucket": selected_bucket(request.maximum, args.buckets),
                "dynamic_receive_bucket": (
                    selected_bucket(request.maximum, args.buckets)
                    if args.source_policy == "matched_graph"
                    else max(1, request.maximum)
                ),
                "pooled_padded_source_rows": (
                    0
                    if args.live_token_count
                    else ATTENTION_RANKS
                    * selected_bucket(request.maximum, args.buckets)
                    - sum(request.counts)
                ),
                "pooled_unused_capacity_source_rows": (
                    ATTENTION_RANKS * selected_bucket(request.maximum, args.buckets)
                    - sum(request.counts)
                ),
                "max_attention_gpu_service_us" if queued else "max_attention_wall_us": (
                    maximum_wall[index]
                ),
            }
            for index, request in enumerate(requests)
        ],
        "ranks": ranks,
    }
    if queued:
        result.update(
            {
                "execution": args.execution,
                "source_policy": args.source_policy,
                "configured_queue_depth": args.queue_depth,
                "maximum_host_enqueue_window_requests": min(
                    args.queue_depth or len(requests), len(requests)
                ),
                "concurrent_device_lanes": 1,
                "resident_producer": True,
                "timed_output_audit": (
                    "final_request_only_queued_burst_checked_separately"
                ),
            }
        )
    pipeline.close()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hybrid", "direct"), default="hybrid")
    parser.add_argument(
        "--execution", choices=("closed_loop", "queued"), default="closed_loop"
    )
    parser.add_argument(
        "--source-policy", choices=("variant", "matched_graph"), default="variant"
    )
    parser.add_argument("--queue-depth", type=int, default=0)
    parser.add_argument("--buckets", nargs="+", type=int, default=DEFAULT_BUCKETS)
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--trace-seed", type=int, default=20261018)
    parser.add_argument("--weight-seed", type=int, default=1729)
    parser.add_argument("--rdma-traffic-class", type=int, default=0)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--live-token-count", action="store_true")
    parser.add_argument("--cuda-events", action="store_true")
    parser.add_argument("--nvtx", action="store_true")
    args = parser.parse_args()
    args.buckets = tuple(sorted(set(args.buckets)))
    if min(args.buckets) <= 0 or min(args.requests, args.repeats, args.warmup) <= 0:
        parser.error("buckets, requests, repeats and warmup must be positive")
    if args.hidden <= 0 or args.hidden % 128 or args.intermediate <= 0:
        parser.error("hidden must be a positive multiple of 128")
    if args.intermediate % 128:
        parser.error("intermediate must be a positive multiple of 128")
    if args.queue_depth < 0:
        parser.error("queue depth must be nonnegative; zero drains only at trace end")
    if args.execution == "queued":
        if args.source_policy != "matched_graph" or not args.live_token_count:
            parser.error(
                "queued execution requires "
                "--source-policy matched_graph --live-token-count"
            )
        args.cuda_events = True
    return args


def main() -> None:
    args = parse_args()
    rank = initialize()
    config = afdep.Config(
        capacity_tokens_per_rank=args.buckets[-1],
        hidden_size=args.hidden,
        num_experts=args.experts,
        top_k=args.top_k,
        payload="bf16",
        allow_hybrid_mode=args.mode == "hybrid",
        rdma_traffic_class=args.rdma_traffic_class,
    )
    buffer = afdep.Buffer(group=dist.group.WORLD, config=config)
    weights = (
        deterministic_weights(
            config.experts_per_rank,
            args.hidden,
            args.intermediate,
            seed=args.weight_seed,
            first_expert=rank * config.experts_per_rank,
        )
        if rank < FFN_RANKS
        else None
    )
    module = load_deepgemm() if rank < FFN_RANKS else None
    checks = (
        []
        if args.skip_correctness
        else correctness(buffer, config, rank, args, weights, module)
    )
    results = []
    for repeat in range(args.repeats):
        shift = repeat % len(args.variants)
        variants = list(args.variants[shift:]) + list(args.variants[:shift])
        for variant in variants:
            results.append(
                measure(buffer, config, rank, args, weights, module, variant, repeat)
            )
    buffer.close()
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "experiment": (
                        f"{args.execution}_mixed_size_one_layer_component_study"
                    ),
                    "execution": args.execution,
                    "source_policy": args.source_policy,
                    "queue_depth": args.queue_depth,
                    "config": asdict(config),
                    "intermediate": args.intermediate,
                    "buckets": args.buckets,
                    "graph_strategy": (
                        "single_capacity" if len(args.buckets) == 1 else "bucketed"
                    ),
                    "runtime_recaptures": 0,
                    "requests_per_trace": args.requests,
                    "repeats": args.repeats,
                    "profiled": args.nvtx,
                    "live_token_count": args.live_token_count,
                    "correctness_variants": checks,
                    "per_expert_counts": "gpu_routing_unknown_to_pool_allocation",
                    "hint": (
                        "fixed_4_times_capacity_times_topk_div_num_experts_all_variants;"
                        "not_eager_retuned_for_each_actual_batch"
                        if len(args.buckets) == 1
                        else "same_4_times_selected_bucket_times_topk_div_num_experts"
                    ),
                    "warmup": "standard_buckets_only_no_future_trace_input_warmup",
                    "input_generation": (
                        "all_trace_inputs_gpu_resident_before_workspace_baseline;"
                        "generation_and_drain_excluded_from_trace_clock;"
                        "future_expert_counts_not_executed_or_warmed"
                        if args.execution == "queued"
                        else "gpu_random_generation_and_source_"
                        "drain_before_request_timer;"
                        "included_in_whole_trace_wall_and_source_allocator_counters"
                    ),
                    "timed": (
                        "bucket_selection_graph_lookup_source_staging_"
                        "device_count_write_or_padding_"
                        "dynamic_allocations_count_sync_exchange_compute_completion"
                    ),
                    "padding": (
                        "pooled_fixed_capacity_but_device_live_count_limits_payload;"
                        "only_actual_source_rows_staged_no_tail_fill"
                        if args.live_token_count
                        else "pooled_sources_send_bucket_rows_tail_ids_minus1_weights0;"
                        "hybrid_transfers_bucket_payload_direct_filters_unused_routes"
                    ),
                    "memory": (
                        "logical_workspace_requests_are_not_device_alloc_calls;"
                        "native_transport_separate_from_pytorch;ffn_peaks_for_workspace"
                    ),
                    "scope": (
                        "one_layer_one_ordered_device_lane_host_can_enqueue_ahead;"
                        "final_drain_tokens_per_second_not_full_serving_throughput;"
                        "gpu_intervals_are_service_not_host_response_latency;"
                        "exact_ffn_host_count_reads_naturally_block;"
                        "no_end_to_end_serving_or_deepep_performance_claim"
                        if args.execution == "queued"
                        else "one_layer_one_lane_closed_loop_"
                        "no_per_request_gloo_barriers;"
                        "host_lengths_known_expert_counts_device_derived;"
                        "no_end_to_end_serving_or_deepep_performance_claim"
                    ),
                    "cases": results,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
