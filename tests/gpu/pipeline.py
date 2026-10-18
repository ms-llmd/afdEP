# SPDX-License-Identifier: Apache-2.0
"""Validate two-slot reuse with eager and separately captured transport phases.

Launch with eight-rank torchrun as in ``tests.gpu.correctness``. Run both
``--payload bf16|fp8`` and ``--mode hybrid|direct``. This validates transport
ordering and weighted expert echo, not model performance or overlap speedup.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict

import torch
import torch.distributed as dist

import afdep
from examples.round_trip import (
    ATTENTION_RANKS,
    FFN_RANKS,
    HIDDEN_SIZE,
    NUM_EXPERTS,
    TOP_K,
    Case,
    Exchange,
    expert_rows,
    initialize,
    make_exchange,
    make_inputs,
)

NUM_SLOTS = 2
DEFAULT_CAPACITY = 32
DEFAULT_EPOCHS = 32
MIN_EPOCHS = 26
DELAY_SOURCE_RANK = FFN_RANKS + 1
DELAY_SOURCE_EPOCH = 5
DELAY_WORKER_RANK = 1
DELAY_WORKER_EPOCH = 8
DELAY_GPU_CYCLES = 5_000_000


def cases(capacity: int) -> list[Case]:
    """Odd case count makes each slot encounter every routing pattern."""
    small = min(7, capacity)
    return [
        Case("capacity", (capacity,) * ATTENTION_RANKS),
        Case("uneven", (capacity, 0, 1, small)),
        Case("empty", (0,) * ATTENTION_RANKS),
        Case("hot_worker_1", (small,) * ATTENTION_RANKS, worker=1),
        Case("rotate_workers", (small, 1, 0, small), rotate_worker=True),
        Case("all_skipped", (capacity,) * ATTENTION_RANKS, skip_all_experts=True),
        Case("decode", (1,) * ATTENTION_RANKS),
        Case("hot_worker_3", (capacity,) * ATTENTION_RANKS, worker=3),
        Case("zero_after_dense", (0,) * ATTENTION_RANKS),
    ]


class Slot:
    """Fixed source/receive/reply addresses shared by eager and phase graphs."""

    def __init__(self, buffer: afdep.Buffer, config: afdep.Config, rank: int):
        self.buffer = buffer
        self.config = config
        self.rank = rank
        full = Case(
            "fixed_storage",
            (config.capacity_tokens_per_rank,) * ATTENTION_RANKS,
            strided_scales=True,
        )
        self.exchange = make_exchange(buffer, config, rank, full)
        self.received = None
        self.rows = None
        self.live_tokens = None
        if rank < FFN_RANKS:
            self.received = buffer.reserve_worker_receive(bucket=full.bucket)
            self.rows = torch.empty(
                self.received.recv_x.shape, dtype=torch.bfloat16, device="cuda"
            )
        else:
            self.live_tokens = torch.full(
                (1,), full.bucket, dtype=torch.int32, device="cuda"
            )
        self.ready = torch.cuda.Event()
        self.computed = torch.cuda.Event()
        self.graphs: dict[str, torch.cuda.CUDAGraph] = {}

    def dispatch(self) -> None:
        inputs = self.exchange.inputs
        self.buffer.source_dispatch(
            inputs.x,
            inputs.ids,
            inputs.weights,
            x_scale=inputs.scale,
            bucket=self.config.capacity_tokens_per_rank,
            live_tokens=self.live_tokens,
        )

    def collect(self) -> None:
        self.buffer.source_collect(self.exchange.output)

    def receive(self) -> None:
        self.buffer.worker_receive(
            bucket=self.config.capacity_tokens_per_rank, out=self.received
        )

    def compute(self) -> None:
        self.rows.copy_(expert_rows(self.received, self.rank, self.config))

    def reply(self) -> None:
        self.buffer.worker_reply(self.rows, self.received.receipt)

    def whole(self) -> None:
        if self.rank < FFN_RANKS:
            self.receive()
            self.compute()
            self.reply()
        else:
            self.dispatch()
            self.collect()

    def pointers(self) -> tuple[int, ...]:
        if self.received is not None:
            tensors = (
                self.received.recv_x,
                self.received.recv_scale,
                self.received.recv_topk_weights,
                self.received.recv_count_psum,
                self.received.receipt,
                self.rows,
            )
        else:
            inputs = self.exchange.inputs
            tensors = (
                inputs.x,
                inputs.ids,
                inputs.weights,
                inputs.scale,
                self.live_tokens,
                self.exchange.output,
            )
        return tuple(tensor.data_ptr() for tensor in tensors if tensor is not None)

    def stage(self, sample) -> None:
        count = sample.x.shape[0]
        inputs = self.exchange.inputs
        # Invalid inactive routes catch reads beyond the GPU live-token count.
        inputs.ids.fill_(-99)
        inputs.weights.fill_(float("nan"))
        inputs.x[:count].copy_(sample.x)
        inputs.ids[:count].copy_(sample.ids)
        inputs.weights[:count].copy_(sample.weights)
        if inputs.scale is not None:
            inputs.scale.fill_(float("nan"))
            inputs.scale[:count].copy_(sample.scale)
        self.exchange.output.fill_(float("nan"))
        self.live_tokens.fill_(count)

    def phase(self, name: str, *, captured: bool) -> None:
        if captured:
            self.graphs[name].replay()
        else:
            getattr(self, name)()


def prepare(pool, slots, rank: int) -> None:
    """Warm complete exchanges, then capture each phase without GPU execution."""
    for index, slot in enumerate(slots):
        slot.whole()
        torch.cuda.synchronize()
        dist.barrier()
        phases = (
            ("receive", "compute", "reply")
            if rank < FFN_RANKS
            else ("dispatch", "collect")
        )
        for name in phases:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=pool.stream(index)):
                getattr(slot, name)()
            slot.graphs[name] = graph
        torch.cuda.synchronize()
        dist.barrier()


def check_variant(pool, slots, config, rank, epochs, *, captured: bool) -> dict:
    selected = cases(config.capacity_tokens_per_rank)
    # All samples/oracles are resident before queueing. Final snapshots prevent
    # later slot wraps from hiding stale payload, route, or output errors.
    samples = []
    snapshots = []
    if rank >= FFN_RANKS:
        for epoch in range(epochs):
            case = selected[epoch % len(selected)]
            sample = make_inputs(case, rank - FFN_RANKS, epoch + 1, config)
            samples.append(sample)
            snapshots.append(torch.empty_like(slots[0].exchange.output))
    pointers = tuple(slot.pointers() for slot in slots)
    source_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    dist.barrier()
    pending = deque()

    def collect_oldest() -> None:
        epoch, slot, lease = pending.popleft()
        with torch.cuda.stream(source_stream):
            slot.phase("collect", captured=captured)
            snapshots[epoch].copy_(slot.exchange.output)
            lease.release(stream=source_stream)

    for epoch in range(epochs):
        index = epoch % NUM_SLOTS
        slot = slots[index]
        lease = pool.acquire(index)
        if rank >= FFN_RANKS:
            with torch.cuda.stream(source_stream):
                lease.wait_on(source_stream)
                slot.stage(samples[epoch])
                if rank == DELAY_SOURCE_RANK and epoch == DELAY_SOURCE_EPOCH:
                    torch.cuda._sleep(DELAY_GPU_CYCLES)
                slot.phase("dispatch", captured=captured)
            pending.append((epoch, slot, lease))
            # Both slots have dispatched before waiting for the first reply.
            if len(pending) == NUM_SLOTS:
                collect_oldest()
        else:
            stream = pool.stream(index)
            with torch.cuda.stream(stream):
                slot.phase("receive", captured=captured)
                slot.ready.record(stream)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(slot.ready)
                if rank == DELAY_WORKER_RANK and epoch == DELAY_WORKER_EPOCH:
                    torch.cuda._sleep(DELAY_GPU_CYCLES)
                slot.phase("compute", captured=captured)
                slot.computed.record(compute_stream)
            with torch.cuda.stream(stream):
                stream.wait_event(slot.computed)
                slot.phase("reply", captured=captured)
                lease.release(stream=stream)
    while pending:
        collect_oldest()
    torch.cuda.synchronize()
    assert tuple(slot.pointers() for slot in slots) == pointers, "slot storage moved"
    if rank >= FFN_RANKS:
        for epoch, (sample, snapshot) in enumerate(
            zip(samples, snapshots, strict=True)
        ):
            case = selected[epoch % len(selected)]
            count = sample.x.shape[0]
            Exchange(
                pool.buffer(epoch % NUM_SLOTS),
                config,
                rank,
                case,
                sample,
                snapshot[:count],
            ).check_output()
            assert bool(torch.isnan(snapshot[count:]).all()), (
                "collection overwrote inactive output rows"
            )
    dist.barrier()
    return {
        "captured_phases": captured,
        "epochs": epochs,
        "minimum_exchanges_per_slot": epochs // NUM_SLOTS,
        "cases": [case.name for case in selected],
        "stable_storage": True,
        "changed_payload_routes_and_gpu_live_counts": True,
        "saved_every_output_for_independent_reference": True,
        "invalid_input_tail_ignored": True,
        "output_tail_untouched": True,
        "dispatch_two_slots_before_collect": True,
        "per_epoch_cpu_barrier": False,
        "delayed_source_epoch": DELAY_SOURCE_EPOCH,
        "delayed_worker_epoch": DELAY_WORKER_EPOCH,
        "delay_gpu_cycles": DELAY_GPU_CYCLES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--mode", choices=("hybrid", "direct"), default="hybrid")
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--num-sms", type=int, default=24)
    parser.add_argument("--num-gin-contexts", type=int, default=8)
    parser.add_argument("--transport-stripes", type=int, default=8)
    parser.add_argument("--rdma-traffic-class", type=int, default=0)
    args = parser.parse_args()
    if args.capacity < 1 or args.epochs < MIN_EPOCHS:
        parser.error(f"capacity must be positive; epochs must be >= {MIN_EPOCHS}")
    rank = initialize()
    config = afdep.Config(
        capacity_tokens_per_rank=args.capacity,
        hidden_size=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        payload=args.payload,
        allow_hybrid_mode=args.mode == "hybrid",
        num_sms=args.num_sms,
        num_gin_contexts=args.num_gin_contexts,
        transport_stripes=args.transport_stripes,
        rdma_traffic_class=args.rdma_traffic_class,
    )
    pool = afdep.PipelineBuffer(group=dist.group.WORLD, config=config, slots=NUM_SLOTS)
    slots = [Slot(pool.buffer(index), config, rank) for index in range(NUM_SLOTS)]
    prepare(pool, slots, rank)
    checked = [
        check_variant(pool, slots, config, rank, args.epochs, captured=captured)
        for captured in (False, True)
    ]
    registered = pool.num_buffer_bytes
    pool.close()
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "config": asdict(config),
                    "slots": NUM_SLOTS,
                    "registered_transport_bytes_per_rank": registered,
                    "expert_compute": "weighted_expert_echo_no_gemm",
                    "variants": checked,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
