# SPDX-License-Identifier: Apache-2.0
"""Coordinated receive/FFN/reply overlap for one BF16 MoE layer.

Compare a whole-request graph, serialized or overlapped microbatches, and an
optional two-request pipeline retaining full expert GEMMs. Graph capture, weights,
resident inputs and independent reference computation precede timing.
This is a component experiment, not a vLLM/full-model serving benchmark.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace

import torch
import torch.distributed as dist

import afdep
from afdep.pipeline import PipelineBuffer
from examples.round_trip import ATTENTION_RANKS, FFN_RANKS, initialize
from experiments.prefill.benchmark import (
    REFERENCE_ATOL,
    REFERENCE_RTOL,
    SourceInputs,
    reference_output,
)
from experiments.prefill.ffn import PrefillFFN, deterministic_weights, load_deepgemm
from experiments.prefill.serving import (
    Payload,
    Request,
    allocation_snapshot,
    cpu_snapshot,
    payload,
    percentile,
    trace,
)

DEFAULT_VARIANTS = ("whole_graph", "split_serial", "pipeline")
REQUEST_VARIANTS = ("request_serial", "request_pipeline")
VARIANTS = (*DEFAULT_VARIANTS, "pipeline_graph", *REQUEST_VARIANTS)
US_PER_SECOND = 1_000_000
DELAY_GPU_CYCLES = 5_000_000


def partitions(tokens: int, capacity: int, pieces: int) -> list[tuple[int, int]]:
    """Balanced contiguous slices preserve token order for uneven/empty sources."""
    if min(capacity, pieces) <= 0 or not 0 <= tokens <= capacity * pieces:
        raise ValueError("invalid microbatch partition")
    return [(i * tokens // pieces, (i + 1) * tokens // pieces) for i in range(pieces)]


@dataclass
class Resident:
    data: Payload
    output: torch.Tensor
    expected: torch.Tensor | None = None

    @property
    def bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.data.x, self.data.ids, self.data.weights, self.output)
        )


def resident_inputs(config, requests, rank, args, *, oracle=False):
    if rank < FFN_RANKS:
        return None
    result = []
    for index, request in enumerate(requests):
        data = payload(config, request, rank - FFN_RANKS)
        output = torch.empty_like(data.x)
        expected = (
            reference_output(
                SourceInputs(data.x, data.ids, data.weights, output),
                config,
                args.intermediate,
                args.weight_seed,
            )
            if oracle or index == len(requests) - 1
            else None
        )
        result.append(Resident(data, output, expected))
    torch.cuda.synchronize()
    return result


class Slot:
    """Fixed per-slot tensors and graphs; weights are shared across slots."""

    def __init__(self, buffer, config, rank, weights, module):
        self.buffer = buffer
        self.config = config
        self.rank = rank
        self.capacity = config.capacity_tokens_per_rank
        self.graphs = {}
        self.ready = torch.cuda.Event()
        self.computed = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.received = None
        self.executor = None
        self.stage = None
        self.output = None
        self.live_tokens = None
        if rank < FFN_RANKS:
            self.received = buffer.reserve_worker_receive(bucket=self.capacity)
            self.executor = PrefillFFN(
                weights,
                capacity_rows=self.received.recv_x.shape[0],
                expert_alignment=config.expert_alignment,
                deepgemm_module=module,
                expected_rows_per_expert=max(
                    1,
                    ATTENTION_RANKS
                    * self.capacity
                    * config.top_k
                    // config.num_experts,
                ),
            )
        else:
            self.stage = Payload(
                torch.zeros(
                    (self.capacity, config.hidden_size),
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                torch.full(
                    (self.capacity, config.top_k), -1, dtype=torch.int64, device="cuda"
                ),
                torch.zeros(
                    (self.capacity, config.top_k), dtype=torch.float32, device="cuda"
                ),
            )
            self.output = torch.empty_like(self.stage.x)
            self.live_tokens = torch.zeros(1, dtype=torch.int32, device="cuda")

    def stage_source(self, data, begin, end):
        count = end - begin
        self.stage.x[:count].copy_(data.x[begin:end])
        self.stage.ids[:count].copy_(data.ids[begin:end])
        self.stage.weights[:count].copy_(data.weights[begin:end])
        self.live_tokens.fill_(count)

    def receive(self):
        self.buffer.worker_receive(bucket=self.capacity, out=self.received)

    def compute(self):
        self.executor(
            self.received.recv_x,
            self.received.recv_topk_weights,
            self.received.recv_count_psum,
        )

    def reply(self):
        self.buffer.worker_reply(self.executor.reply, self.received.receipt)

    def dispatch(self):
        self.buffer.source_dispatch(
            self.stage.x,
            self.stage.ids,
            self.stage.weights,
            bucket=self.capacity,
            live_tokens=self.live_tokens,
        )

    def collect(self):
        self.buffer.source_collect(self.output)

    def whole(self):
        if self.rank < FFN_RANKS:
            self.receive()
            self.compute()
            self.reply()
        else:
            self.dispatch()
            self.collect()

    def pointers(self):
        tensors = (
            (
                self.received.recv_x,
                self.received.recv_topk_weights,
                self.received.recv_count_psum,
                self.received.receipt,
                self.executor.reply,
                self.executor._arena,
            )
            if self.received is not None
            else (
                self.stage.x,
                self.stage.ids,
                self.stage.weights,
                self.output,
                self.live_tokens,
            )
        )
        return tuple(t.data_ptr() for t in tensors if t is not None)


class Trial:
    def __init__(
        self, config, rank, args, weights, module, variant, microbatches, gemm_sms
    ):
        if variant == "pipeline_graph" and (microbatches != 2 or args.slots != 2):
            raise ValueError("pipeline_graph currently requires two chunks and slots")
        self.rank = rank
        self.variant = variant
        self.microbatches = (
            1
            if variant == "whole_graph" or variant in REQUEST_VARIANTS
            else microbatches
        )
        self.capacity = math.ceil(config.capacity_tokens_per_rank / self.microbatches)
        self.config = replace(config, capacity_tokens_per_rank=self.capacity)
        self.args = args
        self.gemm_sms = gemm_sms
        if module is not None:
            # This changes only DeepGEMM projections, not transport CTAs or the
            # short Triton activation/weighting kernels. Set before capture.
            module.set_num_sms(gemm_sms)
        self.count = 1 if variant == "whole_graph" else args.slots
        self.pool = PipelineBuffer(
            group=dist.group.WORLD, config=self.config, slots=self.count
        )
        self.slots = [
            Slot(self.pool.buffer(i), self.config, rank, weights, module)
            for i in range(self.count)
        ]
        self.source_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        self.previous_done = None
        self.next_slot = 0
        self.requests_submitted = 0
        self.pending_requests = deque()
        self.request_graph = None
        self.graph_fork = torch.cuda.Event()

    def range(self, label):
        label += (
            f"/rank={self.rank}/microbatches={self.microbatches}"
            f"/gemm_sms={self.gemm_sms}"
        )
        return torch.cuda.nvtx.range(label) if self.args.nvtx else nullcontext()

    def pointers(self):
        return tuple(slot.pointers() for slot in self.slots)

    def prepare(self):
        started = time.perf_counter()
        # Warm only standard full capacity inputs; never future trace payloads.
        request = Request((self.capacity,) * ATTENTION_RANKS, "uniform", 0, 771)
        data = (
            payload(self.config, request, self.rank - FFN_RANKS)
            if self.rank >= FFN_RANKS
            else None
        )
        for slot in self.slots:
            for _ in range(self.args.warmup):
                dist.barrier()
                if data is not None:
                    slot.stage_source(data, 0, self.capacity)
                slot.whole()
                torch.cuda.synchronize()
            if self.variant == "pipeline_graph":
                continue
            # Capturing each phase independently keeps event scheduling outside
            # the graphs. Capture itself executes no collective GPU protocol.
            phases = (
                ("whole",)
                if self.variant == "whole_graph"
                else ("receive", "compute", "reply")
                if self.rank < FFN_RANKS
                else ("dispatch", "collect")
            )
            for phase in phases:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    getattr(slot, phase)()
                slot.graphs[phase] = graph
            dist.barrier()
            for phase in phases:
                slot.graphs[phase].replay()
            torch.cuda.synchronize()
            dist.barrier()
        if self.variant == "pipeline_graph":
            self.capture_pipeline_graph()
            dist.barrier()
            with torch.cuda.stream(self.source_stream):
                self.request_graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
        self.source_stream.wait_stream(torch.cuda.current_stream())
        self.compute_stream.wait_stream(torch.cuda.current_stream())
        for index in range(self.count):
            self.pool.stream(index).wait_stream(torch.cuda.current_stream())
        return {
            "startup_us": (time.perf_counter() - started) * US_PER_SECOND,
            "graphs_per_rank": (
                1
                if self.request_graph is not None
                else sum(len(slot.graphs) for slot in self.slots)
            ),
            "runtime_recaptures": 0,
            "slot_capacity_tokens_per_source": self.capacity,
            "slots": self.count,
            "shared_expert_weights": self.rank < FFN_RANKS,
            "gemm_sms": self.gemm_sms,
        }

    def capture_pipeline_graph(self):
        """Capture one raw fork/join DAG; all storage/JIT work is warmed first."""
        self.request_graph = torch.cuda.CUDAGraph()
        root = self.source_stream
        root.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(self.request_graph, stream=root):
            if self.rank >= FFN_RANKS:
                for phase in ("dispatch", "collect"):
                    for chunk, slot in enumerate(self.slots):
                        with self.range(
                            f"pipeline_capture/pipeline_graph/chunk={chunk}/"
                            f"slot={chunk}/{phase}"
                        ):
                            getattr(slot, phase)()
            else:
                self.graph_fork.record(root)
                for chunk, slot in enumerate(self.slots):
                    stream = self.pool.stream(chunk)
                    with torch.cuda.stream(stream):
                        stream.wait_event(self.graph_fork)
                        with self.range(
                            f"pipeline_capture/pipeline_graph/chunk={chunk}/"
                            f"slot={chunk}/receive"
                        ):
                            slot.receive()
                        slot.ready.record(stream)
                with torch.cuda.stream(self.compute_stream):
                    self.compute_stream.wait_event(self.graph_fork)
                    for chunk, slot in enumerate(self.slots):
                        self.compute_stream.wait_event(slot.ready)
                        with self.range(
                            f"pipeline_capture/pipeline_graph/chunk={chunk}/"
                            f"slot={chunk}/gemm"
                        ):
                            slot.compute()
                        slot.computed.record(self.compute_stream)
                for chunk, slot in enumerate(self.slots):
                    stream = self.pool.stream(chunk)
                    with torch.cuda.stream(stream):
                        stream.wait_event(slot.computed)
                        with self.range(
                            f"pipeline_capture/pipeline_graph/chunk={chunk}/"
                            f"slot={chunk}/reply"
                        ):
                            slot.reply()
                        slot.done.record(stream)
                    root.wait_event(slot.done)

    def submit_pipeline_graph(self, item, request_id, events, delayed):
        """One replay, with all slot ownership and variable staging outside capture."""
        root = self.source_stream
        with torch.cuda.stream(root):
            if events is not None:
                events[0].record(root)
            leases = [self.pool.acquire(index) for index in range(self.count)]
            for lease in leases:
                lease.wait_on(root)
            if delayed:
                torch.cuda._sleep(DELAY_GPU_CYCLES)
            bounds = None
            if item is not None:
                bounds = partitions(item.data.x.shape[0], self.capacity, 2)
                for slot, (begin, end) in zip(self.slots, bounds, strict=True):
                    slot.stage_source(item.data, begin, end)
            with self.range(f"pipeline/pipeline_graph/request={request_id}/exchange"):
                self.request_graph.replay()
            if item is not None:
                for slot, (begin, end) in zip(self.slots, bounds, strict=True):
                    item.output[begin:end].copy_(slot.output[: end - begin])
            # FFN replay joins both reply streams; source copies consume both
            # output slots. Root completion therefore covers every slot user.
            for lease in leases:
                lease.release(stream=root)
            if events is not None:
                events[1].record(root)
        return events

    def worker_chunk(self, request_id, chunk, *, delayed=False):
        index = self.next_slot % self.count
        self.next_slot += 1
        slot = self.slots[index]
        stream = self.pool.stream(index)
        lease = self.pool.acquire(index)
        with torch.cuda.stream(stream):
            lease.wait_on(stream)
            if (
                self.variant not in ("pipeline", "request_pipeline")
                and self.previous_done is not None
            ):
                stream.wait_event(self.previous_done)
            if delayed:
                torch.cuda._sleep(DELAY_GPU_CYCLES)
            with self.range(
                f"pipeline/{self.variant}/request={request_id}/chunk={chunk}/slot={index}/receive"
            ):
                if self.variant == "whole_graph":
                    slot.graphs["whole"].replay()
                else:
                    slot.graphs["receive"].replay()
                    slot.ready.record(stream)
        if self.variant != "whole_graph":
            with torch.cuda.stream(self.compute_stream):
                self.compute_stream.wait_event(slot.ready)
                with self.range(
                    f"pipeline/{self.variant}/request={request_id}/chunk={chunk}/slot={index}/gemm"
                ):
                    slot.graphs["compute"].replay()
                slot.computed.record(self.compute_stream)
            with torch.cuda.stream(stream):
                stream.wait_event(slot.computed)
                with self.range(
                    f"pipeline/{self.variant}/request={request_id}/chunk={chunk}/slot={index}/reply"
                ):
                    slot.graphs["reply"].replay()
        lease.release(stream=stream)
        slot.done.record(stream)
        self.previous_done = slot.done

    def source_dispatch(self, item, request_id, chunk, begin, end, *, delayed=False):
        index = self.next_slot % self.count
        self.next_slot += 1
        slot = self.slots[index]
        stream = self.source_stream
        lease = self.pool.acquire(index)
        lease.wait_on(stream)
        if delayed:
            torch.cuda._sleep(DELAY_GPU_CYCLES)
        slot.stage_source(item.data, begin, end)
        with self.range(
            f"pipeline/{self.variant}/request={request_id}/chunk={chunk}/slot={index}/dispatch"
        ):
            slot.graphs["dispatch"].replay()
        return index, slot, lease, begin, end, request_id, chunk

    def source_collect(self, item, pending):
        index, slot, lease, begin, end, request_id, chunk = pending
        with self.range(
            f"pipeline/{self.variant}/request={request_id}/chunk={chunk}/slot={index}/collect"
        ):
            slot.graphs["collect"].replay()
        item.output[begin:end].copy_(slot.output[: end - begin])
        lease.release(stream=self.source_stream)

    def collect_request(self, pending):
        """Finish one original request before its full-capacity slot is reused."""
        item, exchange, events = pending
        self.source_collect(item, exchange)
        if events is not None:
            events[1].record(self.source_stream)

    def flush_requests(self):
        """Drain the final request; runtime completion remains inside timing."""
        with torch.cuda.stream(self.source_stream):
            while self.pending_requests:
                self.collect_request(self.pending_requests.popleft())

    def submit(self, item, *, events=None, delayed_source=False, delayed_worker=False):
        """Enqueue one original request; every output retains original token order."""
        request_id = self.requests_submitted
        self.requests_submitted += 1
        if self.variant == "pipeline_graph":
            return self.submit_pipeline_graph(
                item, request_id, events, delayed_source or delayed_worker
            )
        if self.rank < FFN_RANKS:
            for chunk in range(self.microbatches):
                self.worker_chunk(
                    request_id, chunk, delayed=delayed_worker and chunk == 0
                )
            return None
        with torch.cuda.stream(self.source_stream):
            if events is not None:
                events[0].record(self.source_stream)
            if self.variant == "request_pipeline":
                exchange = self.source_dispatch(
                    item,
                    request_id,
                    0,
                    0,
                    item.data.x.shape[0],
                    delayed=delayed_source,
                )
                self.pending_requests.append((item, exchange, events))
                # Publish the next original request before polling the earlier
                # reply. At most two full requests own a slot at any instant.
                if len(self.pending_requests) == self.count:
                    self.collect_request(self.pending_requests.popleft())
                return events
            if self.variant == "whole_graph":
                slot = self.slots[0]
                lease = self.pool.acquire(0)
                lease.wait_on(self.source_stream)
                if delayed_source:
                    torch.cuda._sleep(DELAY_GPU_CYCLES)
                slot.stage_source(item.data, 0, item.data.x.shape[0])
                with self.range(
                    f"pipeline/whole_graph/request={request_id}/chunk=0/slot=0/exchange"
                ):
                    slot.graphs["whole"].replay()
                item.output.copy_(slot.output[: item.data.x.shape[0]])
                lease.release(stream=self.source_stream)
            else:
                pending = deque()
                for chunk, (begin, end) in enumerate(
                    partitions(item.data.x.shape[0], self.capacity, self.microbatches)
                ):
                    if len(pending) == (
                        self.count if self.variant == "pipeline" else 1
                    ):
                        self.source_collect(item, pending.popleft())
                    pending.append(
                        self.source_dispatch(
                            item,
                            request_id,
                            chunk,
                            begin,
                            end,
                            delayed=delayed_source and chunk == 0,
                        )
                    )
                while pending:
                    self.source_collect(item, pending.popleft())
            if events is not None:
                events[1].record(self.source_stream)
        return events

    def close(self):
        if self.pending_requests:
            raise RuntimeError("request pipeline must be drained before closing")
        torch.cuda.synchronize()
        dist.barrier()
        self.request_graph = None
        self.slots.clear()
        self.pool.close()
        gc.collect()
        torch.cuda.empty_cache()


def correctness_requests(maximum):
    small = min(maximum, 31)
    return [
        Request((small, 1, 0, min(17, maximum)), "uniform", 0, 9191),
        Request((0, 0, 0, 0), "uniform", 0, 9192),
        Request((small, 0, 4 if maximum >= 4 else 1, 0), "allskip", 0, 9193),
        Request((maximum, maximum // 2, 1, 0), "hot", 2, 9194),
        Request((1, maximum, small, maximum // 3), "uniform", 0, 9195),
        Request((maximum, 1, 0, maximum), "hot", 1, 9196),
        Request((small, small, small, small), "hot", 3, 9197),
    ]


def check(trial, resident, cases):
    pointers = trial.pointers()
    torch.cuda.synchronize()
    dist.barrier()
    for index, _ in enumerate(cases):
        trial.submit(
            resident[index] if resident is not None else None,
            delayed_source=trial.rank == FFN_RANKS + 1 and index == 4,
            delayed_worker=trial.rank == 1 and index == 3,
        )
    trial.flush_requests()
    torch.cuda.synchronize()
    if resident is not None:
        for item in resident:
            torch.testing.assert_close(
                item.output, item.expected, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
            )
    if trial.pointers() != pointers:
        raise AssertionError("per-slot pooled storage changed")
    dist.barrier()
    return [
        "zero_tokens",
        "all_skipped_routes",
        "hot_expert_worker",
        "uneven_sources",
        "changing_routes",
        "delayed_source",
        "delayed_worker",
        "slot_reuse",
        "original_token_order",
        "independent_pytorch_oracle",
    ]


def summaries(values):
    return {f"p{q}": percentile(values, q / 100) for q in (50, 95, 99)}


def measure(trial, requests, resident, repeat, startup):
    torch.cuda.synchronize()
    pointers = trial.pointers()
    torch.cuda.reset_peak_memory_stats()
    before = allocation_snapshot()
    cpu_before = cpu_snapshot()
    events = (
        [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in requests
        ]
        if trial.rank >= FFN_RANKS
        else []
    )
    for begin, end in events:
        begin.record(trial.source_stream)
        end.record(trial.source_stream)
    torch.cuda.synchronize()
    submitted_at = []
    observed_host_us = []
    drains = []
    dist.barrier()
    started = time.perf_counter()
    pending = deque()
    for index, _ in enumerate(requests):
        submit_start = time.perf_counter()
        event_pair = trial.submit(
            resident[index] if resident is not None else None,
            events=events[index] if events else None,
        )
        submitted_at.append(submit_start)
        if event_pair is not None:
            pending.append((index, event_pair[1]))
            if trial.args.request_window and len(pending) >= trial.args.request_window:
                request_index, complete = pending.popleft()
                drain_start = time.perf_counter()
                complete.synchronize()
                now = time.perf_counter()
                drains.append((now - drain_start) * US_PER_SECOND)
                observed_host_us.append(
                    (now - submitted_at[request_index]) * US_PER_SECOND
                )
    trial.flush_requests()
    while pending:
        request_index, complete = pending.popleft()
        complete.synchronize()
        observed_host_us.append(
            (time.perf_counter() - submitted_at[request_index]) * US_PER_SECOND
        )
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - started
    after = allocation_snapshot()
    cpu_after = cpu_snapshot()
    if resident is not None:
        torch.testing.assert_close(
            resident[-1].output,
            resident[-1].expected,
            rtol=REFERENCE_RTOL,
            atol=REFERENCE_ATOL,
        )
    if trial.pointers() != pointers:
        raise AssertionError("per-slot pooled storage moved")
    local = {
        "rank": trial.rank,
        "trace_wall_s": wall_s,
        "registered_transport_bytes": trial.pool.num_buffer_bytes,
        "startup": startup,
        "before_trace": before,
        "after_trace": after,
        "resident_payload_and_output_bytes": sum(item.bytes for item in resident)
        if resident is not None
        else 0,
        "gpu_request_response_us": [
            begin.elapsed_time(end) * 1000 for begin, end in events
        ],
        "host_observed_request_completion_us": observed_host_us,
        "window_wait_us": drains,
        "cpu_cgroup": {
            "scope": "local_ranks_can_share_counters_do_not_sum",
            "cpu_max": cpu_after["cpu_max"],
            "deltas": {
                key: cpu_after["stat"].get(key, 0) - cpu_before["stat"].get(key, 0)
                for key in ("nr_periods", "nr_throttled", "throttled_usec")
            },
        },
        "allocation_counter_deltas": {
            key: after[key] - before[key]
            for key in (
                "allocation_requests",
                "num_device_alloc",
                "num_device_free",
                "num_alloc_retries",
                "num_ooms",
            )
        },
    }
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, local)
    response = [
        max(row["gpu_request_response_us"][i] for row in ranks[FFN_RANKS:])
        for i in range(len(requests))
    ]
    maximum_wall = max(row["trace_wall_s"] for row in ranks)
    useful_tokens = sum(sum(request.counts) for request in requests)
    return {
        "variant": trial.variant,
        "microbatches": trial.microbatches,
        "gemm_sms": trial.gemm_sms,
        "slots": trial.count,
        "slot_capacity_tokens_per_source": trial.capacity,
        "repeat": repeat,
        "useful_original_tokens": useful_tokens,
        "request_metadata": [asdict(request) for request in requests],
        "max_rank_trace_wall_s": maximum_wall,
        "useful_tokens_per_second": useful_tokens / maximum_wall,
        "max_attention_gpu_request_response_us": summaries(response),
        "per_request_max_attention_gpu_response_us": response,
        "timed_output_check": "last_original_request_independent_oracle_outside_clock",
        "ranks": ranks,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hybrid", "direct"), default="hybrid")
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--fixed-tokens", type=int, default=None)
    parser.add_argument("--microbatches", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument(
        "--variants", choices=VARIANTS, nargs="+", default=DEFAULT_VARIANTS
    )
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--requests", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--request-window", type=int, default=1)
    parser.add_argument("--num-sms", type=int, default=24)
    parser.add_argument("--gemm-sms", type=int, nargs="+", default=None)
    parser.add_argument("--num-gin-contexts", type=int, default=8)
    parser.add_argument("--transport-stripes", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--trace-seed", type=int, default=20261018)
    parser.add_argument("--weight-seed", type=int, default=1729)
    parser.add_argument("--rdma-traffic-class", type=int, default=0)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--nvtx", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.capacity,
            args.slots,
            args.requests,
            args.repeats,
            args.warmup,
            *args.microbatches,
        )
        <= 0
    ):
        parser.error(
            "capacity, slots, requests, repeats, warmup and microbatches "
            "must be positive"
        )
    if args.request_window < 0:
        parser.error("request window must be nonnegative; zero drains at trace end")
    if args.fixed_tokens is not None and not 0 <= args.fixed_tokens <= args.capacity:
        parser.error("fixed tokens must be between zero and capacity")
    args.microbatches = tuple(sorted(set(args.microbatches)))
    if args.gemm_sms is not None and any(
        budget <= 0 or budget % 2 for budget in args.gemm_sms
    ):
        parser.error("DeepGEMM SM budgets must be positive even integers")
    if "pipeline_graph" in args.variants and (
        2 not in args.microbatches or args.slots != 2
    ):
        parser.error("pipeline_graph requires --slots 2 and --microbatches including 2")
    if any(variant in REQUEST_VARIANTS for variant in args.variants) and (
        args.slots != 2 or args.request_window != 2
    ):
        parser.error("request variants require --slots 2 and --request-window 2")
    return args


def main():
    args = parse_args()
    rank = initialize(timeout_s=600)
    physical_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    args.gemm_sms = tuple(dict.fromkeys(args.gemm_sms or (physical_sms,)))
    if any(budget > physical_sms for budget in args.gemm_sms):
        raise ValueError("DeepGEMM SM budgets exceed this GPU's physical SM count")
    config = afdep.Config(
        capacity_tokens_per_rank=args.capacity,
        hidden_size=args.hidden,
        num_experts=args.experts,
        top_k=args.top_k,
        allow_hybrid_mode=args.mode == "hybrid",
        num_sms=args.num_sms,
        num_gin_contexts=args.num_gin_contexts,
        transport_stripes=args.transport_stripes,
        rdma_traffic_class=args.rdma_traffic_class,
    )
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
    cases = [
        (variant, pieces, gemm_sms)
        for variant in args.variants
        for pieces in (
            (1,)
            if variant == "whole_graph" or variant in REQUEST_VARIANTS
            else (2,)
            if variant == "pipeline_graph"
            else args.microbatches
        )
        for gemm_sms in args.gemm_sms
    ]
    checks = []
    if not args.skip_correctness:
        check_requests = correctness_requests(args.capacity)
        check_resident = resident_inputs(
            config, check_requests, rank, args, oracle=True
        )
        for variant, pieces, gemm_sms in cases:
            if rank == 0:
                print(
                    f"correctness {variant} microbatches={pieces} gemm_sms={gemm_sms}",
                    file=sys.stderr,
                    flush=True,
                )
            trial = Trial(
                config, rank, args, weights, module, variant, pieces, gemm_sms
            )
            startup = trial.prepare()
            checks.append(
                {
                    "variant": variant,
                    "microbatches": pieces,
                    "gemm_sms": gemm_sms,
                    "checks": check(trial, check_resident, check_requests),
                    "startup": startup,
                }
            )
            trial.close()
        del check_resident
    results = []
    if not args.correctness_only:
        for repeat in range(args.repeats):
            requests = trace(
                args.requests, args.trace_seed + repeat * 7919, args.capacity
            )
            if args.fixed_tokens is not None:
                requests = [
                    replace(
                        request,
                        counts=(args.fixed_tokens,) * ATTENTION_RANKS,
                        pattern="uniform",
                    )
                    for request in requests
                ]
            resident = resident_inputs(config, requests, rank, args)
            order = cases[repeat % len(cases) :] + cases[: repeat % len(cases)]
            for variant, pieces, gemm_sms in order:
                if rank == 0:
                    print(
                        f"timing repeat={repeat} {variant} microbatches={pieces} "
                        f"gemm_sms={gemm_sms}",
                        file=sys.stderr,
                        flush=True,
                    )
                baseline = allocation_snapshot()
                trial = Trial(
                    config, rank, args, weights, module, variant, pieces, gemm_sms
                )
                startup = trial.prepare()
                startup["baseline_before_setup"] = baseline
                result = measure(trial, requests, resident, repeat, startup)
                results.append(result)
                trial.close()
            del resident
            gc.collect()
            torch.cuda.empty_cache()
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "experiment": "coordinated_microbatch_single_layer_component",
                    "config": asdict(config),
                    "intermediate": args.intermediate,
                    "slots": args.slots,
                    "microbatch_sweep": args.microbatches,
                    "gemm_sm_sweep": args.gemm_sms,
                    "physical_gpu_sms": physical_sms,
                    "gemm_sm_scope": (
                        "DeepGEMM_projections_only_Triton_kernels_unchanged"
                    ),
                    "request_window": args.request_window,
                    "requests": args.requests,
                    "fixed_tokens_per_source": args.fixed_tokens,
                    "repeats": args.repeats,
                    "profiled": args.nvtx,
                    "runtime_recaptures": 0,
                    "scope": (
                        "one_BF16_DeepGEMM_SwiGLU_layer_no_Attention_compute;"
                        "no_full_model_serving_claim"
                    ),
                    "variants": {
                        "whole_graph": "one_max_capacity_graph_one_slot",
                        "split_serial": (
                            "same_partition_slots_phase_graphs_as_pipeline;"
                            "no_inter_chunk_overlap"
                        ),
                        "pipeline_graph": (
                            "C2_only_one_raw_multistream_fork_join_graph_per_request;"
                            "both_source_chunks_staged_before_replay;one_GEMM_stream;"
                            "same_two_slots_no_inter_request_pipeline"
                        ),
                        "pipeline": (
                            "ordered_source_dispatch_dispatch_collect;"
                            "per_slot_receive_reply_streams_one_GEMM_stream;"
                            "intra_original_request_overlap_no_inter_request_pipeline"
                        ),
                        "request_serial": (
                            "two_full_capacity_slots_phase_graphs_one_GEMM_stream;"
                            "source_dispatch_collect_and_worker_requests_serialized;"
                            "bounded_host_window2_shared_weights_no_token_splitting"
                        ),
                        "request_pipeline": (
                            "two_full_capacity_slots_phase_graphs_one_GEMM_stream;"
                            "source_dispatchA_dispatchB_collectA_bounded_window2;"
                            "inter_original_request_overlap_shared_weights_no_token_splitting"
                        ),
                    },
                    "timing": (
                        "resident_input_generation_weights_warmup_capture_"
                        "first_replay_reference_excluded;"
                        "runtime_staging_graph_replays_output_assembly_"
                        "completion_included;no_per_request_Gloo_barriers"
                    ),
                    "latency": (
                        "CUDA_first_staging_to_last_chunk_output_assembly;"
                        "max_attention_rank_per_original_request;"
                        "host_completion_upper_bound_with_window_gt1;"
                        "not_serving_response_latency"
                    ),
                    "memory": (
                        "native_registered_transport_separate_from_PyTorch;"
                        "weights_resident_inputs_outputs_in_setup_baseline;"
                        "independent_slot_arenas_shared_weights"
                    ),
                    "correctness": checks,
                    "cases": results,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
