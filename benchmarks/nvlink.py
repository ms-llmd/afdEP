# SPDX-License-Identifier: Apache-2.0
"""Matched communication-only screen; no expert compute or serving claim."""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from afdep import Buffer as DeepEPBuffer
from afdep import Config as DeepEPConfig
from afdep import DirectionalBuffer as Buffer
from afdep import DirectionalConfig as Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["directional", "masked", "reuse"], required=True
    )
    parser.add_argument("--payload", choices=["bf16", "fp8"], required=True)
    parser.add_argument("--transport", choices=["nvlink", "gin"], default="nvlink")
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--attention-ranks", type=int, default=2)
    parser.add_argument("--gin-contexts", type=int, default=8)
    parser.add_argument("--transport-stripes", type=int, default=8)
    parser.add_argument("--rdma-traffic-class", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--execution", choices=["eager", "graph"], default="graph")
    parser.add_argument("--hybrid", choices=["on", "off"], default="on")
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512, 2048]
    )
    parser.add_argument(
        "--sms", type=int, default=132, help="Directional local-kernel base grid"
    )
    parser.add_argument(
        "--masked-sms",
        type=int,
        default=0,
        help="DeepEP transport grid; local epilogues use the full device",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--routing", choices=["strided", "random", "skew"], default="strided"
    )
    parser.add_argument("--routing-seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.tokens) < 1
        or max(args.tokens) > args.capacity
        or args.iterations < 1
        or args.repeats < 1
        or args.warmup < 0
    ):
        parser.error("tokens in [1,capacity]; iterations/repeats > 0; warmup >= 0")
    if (
        min(
            args.ffn_ranks,
            args.attention_ranks,
            args.gin_contexts,
            args.transport_stripes,
        )
        < 1
    ):
        parser.error("role ranks, GIN contexts and transport stripes must be positive")
    if args.backend == "reuse" and args.transport == "gin":
        parser.error(
            "role-aware DeepEP reuse requires one NVLink domain; use masked for GIN"
        )
    if args.experts % args.ffn_ranks:
        parser.error("experts must divide evenly across FFN ranks")
    if args.routing == "skew" and args.top_k > args.experts // args.ffn_ranks:
        parser.error("skew routing needs top-k <= experts per worker")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo")
    world_size = dist.get_world_size()
    if world_size != args.ffn_ranks + args.attention_ranks:
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
        "rdma_traffic_class": args.rdma_traffic_class,
    }
    directional = args.backend == "directional"
    config = (
        Config(
            **common,
            transport=args.transport,
            num_sms=args.sms,
            num_gin_contexts=args.gin_contexts,
            transport_stripes=args.transport_stripes,
        )
        if directional
        else DeepEPConfig(
            **common,
            num_sms=args.masked_sms,
            num_qps=args.gin_contexts if args.transport == "gin" else 0,
            role_aware=args.backend == "reuse",
        )
    )
    buffer = (Buffer if directional else DeepEPBuffer)(
        group=dist.group.WORLD, config=config
    )
    if not directional:
        domains, _ = buffer.physical_domain
        if (args.transport == "nvlink") != (domains == 1):
            buffer.close()
            raise ValueError("transport does not match the physical NVLink domains")
    worker = rank < args.ffn_ranks
    rows = []
    route_records = []
    local_experts = args.experts // args.ffn_ranks
    route_experts = local_experts if args.routing == "skew" else args.experts
    route_stride = 5 if math.gcd(5, route_experts) == 1 else 1
    for n in args.tokens:
        torch.cuda.reset_peak_memory_stats()
        dtype = torch.bfloat16 if args.payload == "bf16" else torch.float8_e4m3fn
        x = torch.ones((n, args.hidden), dtype=dtype, device="cuda")
        ids = (
            torch.arange(n, device="cuda")[:, None] * 3
            + torch.arange(args.top_k, device="cuda")[None, :] * route_stride
            + max(rank - args.ffn_ranks, 0)
        ) % route_experts
        if args.routing == "random":
            generator = torch.Generator(device="cuda").manual_seed(
                args.routing_seed + max(rank - args.ffn_ranks, 0)
            )
            ids = (
                torch.rand((n, args.experts), generator=generator, device="cuda")
                .topk(args.top_k, dim=1)
                .indices
            )
        ids = ids.to(torch.int64 if directional else buffer.topk_idx_dtype)
        canonical = ids.long().cpu().numpy()
        route_records.append(
            {
                "tokens": n,
                "sha256_int64": hashlib.sha256(canonical.tobytes()).hexdigest(),
                "tokens_per_worker": [
                    int(
                        (
                            (canonical >= w * local_experts)
                            & (canonical < (w + 1) * local_experts)
                        )
                        .any(axis=1)
                        .sum()
                    )
                    for w in range(args.ffn_ranks)
                ],
            }
        )
        weights = torch.full(
            (n, args.top_k), 1 / args.top_k, device="cuda", dtype=torch.float32
        )
        scale = (
            torch.ones((n, args.hidden // 128), device="cuda")
            if args.payload == "fp8"
            else None
        )
        received = (
            buffer.reserve_worker_receive(bucket=n) if worker and directional else None
        )
        expert = None
        result = torch.empty((n, args.hidden), dtype=torch.bfloat16, device="cuda")

        def step(n=n, received=received, x=x, ids=ids, weights=weights, scale=scale):
            nonlocal expert, result
            if worker:
                recv = (
                    buffer.worker_receive(bucket=n, out=received)
                    if directional
                    else buffer.worker_receive(bucket=n)
                )
                if expert is None:
                    # Filled once outside capture/timing. All selected expert rows
                    # contribute one, so each source must receive exactly top_k.
                    expert = torch.ones(
                        recv.recv_x.shape, dtype=torch.bfloat16, device="cuda"
                    )
                buffer.worker_reply(expert, recv.receipt)
            else:
                buffer.source_dispatch(x, ids, weights, scale, bucket=n)
                if directional:
                    buffer.source_collect(result)
                else:
                    result = buffer.source_collect()

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        dist.barrier()
        graph = None
        if args.execution == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
        exchange = graph.replay if graph is not None else step
        for _ in range(args.warmup):
            exchange()
        torch.cuda.synchronize()
        if not worker:
            torch.testing.assert_close(
                result, torch.full_like(result, args.top_k), rtol=0, atol=0
            )
        dist.barrier()
        for repeat in range(args.repeats):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            dist.barrier()
            if args.profile:
                torch.cuda.cudart().cudaProfilerStart()
            wall = time.perf_counter()
            start.record()
            for _ in range(args.iterations):
                exchange()
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall) * 1000 / args.iterations
            local = {
                "rank": rank,
                "gpu_ms": start.elapsed_time(end) / args.iterations,
                "wall_ms": wall_ms,
                "worker": worker,
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "torch_allocated_bytes": torch.cuda.memory_allocated(),
            }
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local)
            if args.profile:
                dist.barrier()
                torch.cuda.cudart().cudaProfilerStop()
            if not worker:
                torch.testing.assert_close(
                    result, torch.full_like(result, args.top_k), rtol=0, atol=0
                )
            rows.append({"tokens": n, "repeat": repeat, "ranks": gathered})
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "tokens": n,
                            "repeat": repeat,
                            "source_gpu_ms": max(
                                v["gpu_ms"] for v in gathered if not v["worker"]
                            ),
                        }
                    ),
                    flush=True,
                )
        del exchange, graph, step, expert, received, result, x, ids, weights, scale
        torch.cuda.synchronize()
        dist.barrier()
    metadata = {
        "rank": rank,
        "transport": args.transport,
        "world_size": world_size,
        "config": asdict(config),
        "registered_bytes": buffer.num_buffer_bytes,
        "launch_policy": {
            "directional_local_base_grid": config.num_sms if directional else None,
            "deepep_transport_sms": None
            if directional
            else buffer.transport._resolved_num_sms,
            "deepep_local_epilogue_sms": None
            if directional
            else torch.cuda.get_device_properties(local_rank).multi_processor_count,
        },
        "device": str(torch.cuda.get_device_properties(local_rank)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "endpoint_sha256": hashlib.sha256(
            Path(sys.modules[type(buffer).__module__].__file__).read_bytes()
        ).hexdigest(),
        "routing": route_records if not worker else None,
    }
    meta = [None] * world_size
    dist.all_gather_object(meta, metadata)
    buffer.close()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "args": vars(args) | {"output": str(args.output)},
                    "metadata": meta,
                    "measurements": rows,
                    "scope": (
                        "communication only; initialization/capture excluded; "
                        "eager enqueue or graph replay included; "
                        "expert output prepared outside timed path"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
