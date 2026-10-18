# SPDX-License-Identifier: Apache-2.0
"""Communication-only round-trip benchmark with persistent BF16 one replies.

Run with torchrun on both hosts, using ``--module benchmarks.round_trip``.
No quantization, model GEMM, reply fill, or output check is timed. Transport
packs activations/routes/weights and reduces fixed expert replies. Host wall
samples include GPU completion and launch skew, and exclude the pre-iteration
Gloo barrier. Profile separately: tracing changes latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist

import afdep
from examples.round_trip import (
    ATTENTION_RANKS,
    FFN_RANKS,
    HIDDEN_SIZE,
    MICROSECONDS_PER_SECOND,
    NUM_EXPERTS,
    TOP_K,
    Case,
    initialize,
    make_exchange,
)

DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 20
DEFAULT_BUCKETS = (1, 128, 512, 2048)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--mode", choices=("hybrid", "direct"), default="hybrid")
    parser.add_argument("--tokens", nargs="+", type=int, default=DEFAULT_BUCKETS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--rdma-traffic-class", type=int, default=0)
    parser.add_argument(
        "--nvtx", action="store_true", help="annotate a separate Nsight Systems run"
    )
    args = parser.parse_args()
    if min(args.tokens) < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error(
            "tokens and iterations must be positive; warmup cannot be negative"
        )
    rank = initialize()
    config = afdep.Config(
        capacity_tokens_per_rank=max(args.tokens),
        hidden_size=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        payload=args.payload,
        allow_hybrid_mode=args.mode == "hybrid",
        rdma_traffic_class=args.rdma_traffic_class,
    )
    buffer = afdep.Buffer(group=dist.group.WORLD, config=config)
    all_cases = []
    for tokens in args.tokens:
        exchange = make_exchange(
            buffer, config, rank, Case(str(tokens), (tokens,) * ATTENTION_RANKS)
        )
        reply = None

        def run(tokens=tokens, exchange=exchange) -> None:
            nonlocal reply
            if rank < FFN_RANKS:
                received = buffer.worker_receive(bucket=tokens, read_host_counts=False)
                if reply is None:
                    reply = (
                        (received.recv_topk_weights != 0)[:, None]
                        .expand_as(received.recv_x)
                        .to(torch.bfloat16)
                        .contiguous()
                    )
                buffer.worker_reply(reply, received.receipt)
            else:
                buffer.source_dispatch(
                    exchange.inputs.x,
                    exchange.inputs.ids,
                    exchange.inputs.weights,
                    x_scale=exchange.inputs.scale,
                    bucket=tokens,
                )
                buffer.source_collect(exchange.output)

        run()
        torch.cuda.synchronize()
        dist.barrier()
        graph = None
        if args.graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            torch.cuda.synchronize()
            dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for iteration in range(args.warmup + args.iterations):
            torch.cuda.synchronize()
            dist.barrier()
            label = (
                f"afdEP/mode={args.mode}/rank={rank}/tokens={tokens}"
                f"/iteration={iteration}"
            )
            context = torch.cuda.nvtx.range(label) if args.nvtx else nullcontext()
            with context:
                start = time.perf_counter()
                if graph is None:
                    run()
                else:
                    graph.replay()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * MICROSECONDS_PER_SECOND
            if iteration >= args.warmup:
                samples.append(elapsed)
        if rank >= FFN_RANKS:
            expected = (exchange.inputs.ids >= 0).sum(dim=1, keepdim=True)
            torch.testing.assert_close(
                exchange.output,
                expected.expand_as(exchange.output).bfloat16(),
                rtol=0,
                atol=0,
            )
        local = {
            "rank": rank,
            "samples_us": samples,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "registered_transport_bytes": buffer.num_buffer_bytes,
        }
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, local)
        if rank == 0:
            maximum_source_samples = [
                max(row["samples_us"][iteration] for row in ranks[FFN_RANKS:])
                for iteration in range(args.iterations)
            ]
            all_cases.append(
                {
                    "tokens_per_source": tokens,
                    "median_iteration_max_attention_rank_wall_us": statistics.median(
                        maximum_source_samples
                    ),
                    "iteration_max_attention_rank_wall_us": maximum_source_samples,
                    "ranks": ranks,
                }
            )
    buffer.close()
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": args.mode,
                    "config": asdict(config),
                    "graph": args.graph,
                    "profiled": args.nvtx,
                    "iterations": args.iterations,
                    "warmup": args.warmup,
                    "expert_compute": "fixed_ones_no_gemm",
                    "cases": all_cases,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
