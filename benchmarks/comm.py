# SPDX-License-Identifier: Apache-2.0
"""Paired in-process communication screen; no expert compute or serving claim.

Every arm owns one Buffer in the same processes. Timing blocks rotate the arm
order, so each block yields one paired comparison against the first arm.

Example: EP_DISABLE_GIN=1 torchrun --standalone --nproc-per-node=4 \
    -m benchmarks.comm --payload bf16 --routing random --output /tmp/screen.json
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from afdep import Buffer, Config
from afdep._build import source_digest

from .screen import DEFAULT_ARMS

METRIC = "per block: max Attention-rank CUDA-event us per exchange"
SCOPE = "communication only; init/capture excluded; graph replay included"
# Buffers coexisting in one process need separate DeepEP communication streams.
PER_BUFFER_COMM_STREAM_ENV = "AFD_DEEPEP_PER_BUFFER_COMM_STREAM"
# Arms may set AFDEP_BENCH_COLLECT_VIEW=1 (in "_env") to time source_collect(view=True).
COLLECT_VIEW = os.environ.get("AFDEP_BENCH_COLLECT_VIEW", "0") == "1"


def routes(args, rank, n):
    """Deterministic strided/skew patterns or seeded random top-k routes."""
    source = max(rank - args.ffn_ranks, 0)
    local_experts = args.experts // args.ffn_ranks
    route_experts = local_experts if args.routing == "skew" else args.experts
    stride = 5 if math.gcd(5, route_experts) == 1 else 1
    if args.routing == "random":
        generator = torch.Generator(device="cuda").manual_seed(
            args.routing_seed + source
        )
        return (
            torch.rand((n, args.experts), generator=generator, device="cuda")
            .topk(args.top_k, dim=1)
            .indices
        )
    return (
        torch.arange(n, device="cuda")[:, None] * 3
        + torch.arange(args.top_k, device="cuda")[None, :] * stride
        + source
    ) % route_experts


class Arm:
    """One buffer and one captured exchange per token extent."""

    def __init__(self, name, overrides, common):
        self.name = name
        self.config = Config(**(common | overrides))
        self.buffer = Buffer(group=dist.group.WORLD, config=self.config)
        self.graph = None
        self.result = None

    def prepare(self, n, x, ids, weights, scale, graph):
        buffer, expert = self.buffer, None

        def step():
            nonlocal expert
            if buffer.role == "ffn":
                received = buffer.worker_receive(bucket=n)
                if expert is None:
                    # Filled once outside capture/timing: every selected row
                    # contributes one, so each source receives exactly top_k.
                    expert = torch.ones(
                        received.recv_x.shape, dtype=torch.bfloat16, device="cuda"
                    )
                buffer.worker_reply(expert, received.receipt)
            else:
                buffer.source_dispatch(
                    x, ids.to(buffer.topk_idx_dtype), weights, scale, bucket=n
                )
                self.result = buffer.source_collect(view=COLLECT_VIEW)

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        dist.barrier()
        self.graph = None
        if graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                step()
        self.exchange = self.graph.replay if self.graph is not None else step

    def check(self, top_k):
        if self.buffer.role == "attention":
            torch.testing.assert_close(
                self.result, torch.full_like(self.result, top_k), rtol=0, atol=0
            )

    def release(self):
        self.exchange = self.graph = self.result = None


def timed(exchange, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        exchange()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--attention-ranks", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 128, 2048])
    parser.add_argument("--execution", choices=["eager", "graph"], default="graph")
    parser.add_argument("--hybrid", choices=["on", "off"], default="on")
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument(
        "--routing", choices=["strided", "random", "skew"], default="random"
    )
    parser.add_argument("--routing-seed", type=int, default=49001)
    parser.add_argument(
        "--arms",
        type=json.loads,
        default=DEFAULT_ARMS,
        help='JSON {"name": {Config overrides}}; the first arm is the control',
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens) < 1 or max(args.tokens) > args.capacity:
        parser.error("tokens must be in [1,capacity]")
    if min(args.iterations, args.blocks) < 1 or args.warmup < 0:
        parser.error("iterations and blocks must be positive; warmup nonnegative")
    if not isinstance(args.arms, dict) or len(args.arms) < 1:
        parser.error("--arms must be a nonempty JSON object")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    world = dist.get_world_size()
    if world != args.ffn_ranks + args.attention_ranks:
        raise ValueError("torchrun world size must equal FFN plus Attention ranks")
    common = {
        "capacity_tokens_per_rank": args.capacity,
        "hidden_size": args.hidden,
        "num_experts": args.experts,
        "top_k": args.top_k,
        "ffn_ranks": args.ffn_ranks,
        "attention_ranks": args.attention_ranks,
        "payload": args.payload,
        "allow_hybrid_mode": args.hybrid == "on",
        "num_sms": args.num_sms,
    }
    if len(args.arms) > 1:
        os.environ[PER_BUFFER_COMM_STREAM_ENV] = "1"
    arms = [Arm(name, overrides, common) for name, overrides in args.arms.items()]
    names = [arm.name for arm in arms]
    attention = arms[0].buffer.role == "attention"
    dtype = torch.bfloat16 if args.payload == "bf16" else torch.float8_e4m3fn
    cells = []
    for n in args.tokens:
        x = torch.ones((n, args.hidden), dtype=dtype, device="cuda")
        ids = routes(args, rank, n)
        weights = torch.full((n, args.top_k), 1 / args.top_k, device="cuda")
        scale = (
            torch.ones((n, args.hidden // 128), device="cuda")
            if args.payload == "fp8"
            else None
        )
        route_sha = hashlib.sha256(ids.long().cpu().numpy().tobytes()).hexdigest()
        for arm in arms:
            arm.prepare(n, x, ids, weights, scale, args.execution == "graph")
            for _ in range(args.warmup):
                arm.exchange()
            torch.cuda.synchronize()
            arm.check(args.top_k)
            dist.barrier()
        blocks = []
        for block in range(args.blocks):
            order = names[block % len(names) :] + names[: block % len(names)]
            local = {}
            for name in order:
                dist.barrier()
                local[name] = timed(arms[names.index(name)].exchange, args.iterations)
            gathered = [None] * world
            dist.all_gather_object(
                gathered, {"rank": rank, "attention": attention, "us": local}
            )
            blocks.append({"order": order, "ranks": gathered})
        for arm in arms:
            arm.check(args.top_k)
            arm.release()
        torch.cuda.synchronize()
        dist.barrier()
        per_arm = {
            name: [
                max(r["us"][name] for r in b["ranks"] if r["attention"]) for b in blocks
            ]
            for name in names
        }
        control = names[0]
        summary = {
            name: {
                "median_us": statistics.median(values),
                "paired_change_pct": [
                    100 * (v / c - 1)
                    for v, c in zip(values, per_arm[control], strict=True)
                ],
            }
            for name, values in per_arm.items()
        }
        for name in names:
            changes = summary[name]["paired_change_pct"]
            summary[name]["median_change_pct"] = statistics.median(changes)
            summary[name]["blocks_faster"] = sum(c < 0 for c in changes)
        cells.append(
            {
                "tokens": n,
                "route_sha256": route_sha,
                "blocks": blocks,
                "summary": summary,
            }
        )
        if rank == 0:
            print(
                json.dumps(
                    {"tokens": n}
                    | {
                        name: [
                            round(s["median_us"], 2),
                            round(s["median_change_pct"], 2),
                            s["blocks_faster"],
                        ]
                        for name, s in summary.items()
                    }
                ),
                flush=True,
            )
        del x, ids, weights, scale
    metadata = {
        "rank": rank,
        "arms": {arm.name: asdict(arm.config) for arm in arms},
        "registered_bytes": {arm.name: arm.buffer.num_buffer_bytes for arm in arms},
        "transport_sms": {
            arm.name: arm.buffer.transport._resolved_num_sms for arm in arms
        },
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "native_source_sha256": source_digest(),
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, metadata)
    for arm in reversed(arms):
        arm.buffer.close()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "args": vars(args) | {"output": str(args.output)},
                    "metadata": gathered,
                    "cells": cells,
                    "metric": METRIC,
                    "scope": SCOPE,
                    "time": time.time(),
                },
                indent=1,
            )
            + "\n"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
