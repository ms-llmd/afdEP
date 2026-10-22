# SPDX-License-Identifier: Apache-2.0
"""Exchanges with a concurrent GEMM load: does compute get the SMs a role frees?

Every rank runs a fixed batch of BF16 GEMMs on a side stream while its default
stream replays a fixed number of exchanges; both counts are arguments, so every
arm (one per process, paired by benchmarks.screen --module benchmarks.overlap)
does the same work. Per block and rank it reports the GEMM time per GEMM, the
exchange time per exchange and the makespan of both batches; lower is better.
Synthetic load: no serving claim.

Example: python -m benchmarks.screen --module benchmarks.overlap \
    --metric attention.gemm_us_per_gemm --output /tmp/overlap.json -- \
    --tokens 1 32 --gemms 200 300
"""

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from afdep._build import source_digest

from .comm import Arm, routes

SCOPE = "synthetic GEMM co-run; init/capture excluded; graph replay included"
ROLES = ("attention", "ffn")
GEMM_WARMUP = 20


def solo_us(fn, repeats):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def corun(exchange, iterations, gemms, a, b, out, side):
    """Both batches start together: (GEMM us per GEMM, exchange us, makespan us)."""
    main = torch.cuda.current_stream()
    begin, gemm_end, comm_end, both_end = (
        torch.cuda.Event(enable_timing=True) for _ in range(4)
    )
    begin.record(main)
    side.wait_event(begin)
    with torch.cuda.stream(side):
        for _ in range(gemms):
            torch.mm(a, b, out=out)
        gemm_end.record(side)
    for _ in range(iterations):
        exchange()
    comm_end.record(main)
    main.wait_event(gemm_end)
    both_end.record(main)
    both_end.synchronize()
    gemm = begin.elapsed_time(gemm_end) * 1000
    comm = begin.elapsed_time(comm_end) * 1000
    return gemm / gemms, comm / iterations, max(gemm, comm)


def role_worst(blocks, role, index):
    """Per block, the slowest rank of a role for one co-run metric."""
    return [max(r["corun"][index] for r in b if r["role"] == role) for b in blocks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=["bf16", "fp8"], default="bf16")
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--attention-ranks", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 512])
    parser.add_argument(
        "--gemms", type=int, nargs="+", required=True, help="GEMMs per block, per token"
    )
    parser.add_argument("--gemm-size", type=int, default=2048, help="square edge")
    parser.add_argument("--hybrid", choices=["on", "off"], default="on")
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument(
        "--routing", choices=["strided", "random", "skew"], default="random"
    )
    parser.add_argument("--routing-seed", type=int, default=49001)
    parser.add_argument("--arms", type=json.loads, required=True, help="one arm")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.tokens) < 1 or max(args.tokens) > args.capacity:
        parser.error("tokens must be in [1,capacity]")
    if len(args.gemms) != len(args.tokens) or min(args.gemms) < 1:
        parser.error("--gemms needs one positive count per token extent")
    if not isinstance(args.arms, dict) or len(args.arms) != 1:
        parser.error("--arms holds one arm; pair arms with benchmarks.screen")
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
    ((name, overrides),) = args.arms.items()
    arm = Arm(name, overrides, common)
    role = arm.buffer.role
    dtype = torch.bfloat16 if args.payload == "bf16" else torch.float8_e4m3fn
    a = torch.randn(args.gemm_size, args.gemm_size, dtype=torch.bfloat16, device="cuda")
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    side = torch.cuda.Stream()
    # cuBLAS initializes per stream on first use: warm both streams before timing.
    for stream in (torch.cuda.current_stream(), side):
        with torch.cuda.stream(stream):
            for _ in range(GEMM_WARMUP):
                torch.mm(a, b, out=out)
    torch.cuda.synchronize()
    gemm_solo = solo_us(lambda: torch.mm(a, b, out=out), 50)
    cells = []
    for n, gemms in zip(args.tokens, args.gemms, strict=True):
        x = torch.ones((n, args.hidden), dtype=dtype, device="cuda")
        ids = routes(args, rank, n)
        weights = torch.full((n, args.top_k), 1 / args.top_k, device="cuda")
        scale = (
            torch.ones((n, args.hidden // 128), device="cuda")
            if args.payload == "fp8"
            else None
        )
        arm.prepare(n, x, ids, weights, scale, True)
        for _ in range(args.warmup):
            arm.exchange()
        torch.cuda.synchronize()
        arm.check(args.top_k)
        dist.barrier()
        exchange_solo = solo_us(arm.exchange, args.iterations)
        blocks = []
        for _ in range(args.blocks):
            dist.barrier()
            torch.cuda.synchronize()
            result = corun(arm.exchange, args.iterations, gemms, a, b, out, side)
            gathered = [None] * world
            dist.all_gather_object(
                gathered, {"rank": rank, "role": role, "corun": result}
            )
            blocks.append(gathered)
        arm.check(args.top_k)
        arm.release()
        torch.cuda.synchronize()
        solos = [None] * world
        dist.all_gather_object(
            solos, {"rank": rank, "role": role, "exchange_us": exchange_solo}
        )
        summary = {}
        for r in ROLES:
            gemm, comm, span = (role_worst(blocks, r, i) for i in range(3))
            summary[r] = {
                "gemm_us_per_gemm": statistics.median(gemm),
                "exchange_us": statistics.median(comm),
                "makespan_us": statistics.median(span),
                "solo_exchange_us": max(
                    s["exchange_us"] for s in solos if s["role"] == r
                ),
            }
        # benchmarks.screen reads summary[arm]; median_us is the Attention makespan.
        summary["median_us"] = summary["attention"]["makespan_us"]
        cells.append(
            {"tokens": n, "gemms": gemms, "blocks": blocks, "summary": {name: summary}}
        )
        if rank == 0:
            print(json.dumps({"tokens": n, "gemms": gemms, name: summary}), flush=True)
        del x, ids, weights, scale
    metadata = {
        "rank": rank,
        "role": role,
        "arm": asdict(arm.config),
        "gemm_solo_us": gemm_solo,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "native_source_sha256": source_digest(),
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, metadata)
    arm.buffer.close()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "args": vars(args) | {"output": str(args.output)},
                    "metadata": gathered,
                    "cells": cells,
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
