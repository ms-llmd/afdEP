# SPDX-License-Identifier: Apache-2.0
"""Matched four-GPU communication-only screen; no expert compute or serving claim."""

import argparse
import hashlib
import json
import os
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
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512, 2048]
    )
    parser.add_argument("--sms", type=int, default=132)
    parser.add_argument("--masked-sms", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--routing", choices=["strided", "random"], default="strided")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.tokens) < 1
        or max(args.tokens) > 2048
        or args.iterations < 1
        or args.repeats < 1
        or args.warmup < 0
    ):
        parser.error("tokens in [1,2048]; iterations/repeats > 0; warmup >= 0")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    common = {
        "capacity_tokens_per_rank": 2048,
        "hidden_size": 2048,
        "num_experts": 128,
        "top_k": 8,
        "ffn_ranks": 2,
        "attention_ranks": 2,
        "payload": args.payload,
    }
    directional = args.backend == "directional"
    config = (
        Config(**common, transport="nvlink", num_sms=args.sms)
        if directional
        else DeepEPConfig(
            **common, num_sms=args.masked_sms, role_aware=args.backend == "reuse"
        )
    )
    buffer = (Buffer if directional else DeepEPBuffer)(
        group=dist.group.WORLD, config=config
    )
    worker = rank < 2
    rows = []
    route_records = []
    for n in args.tokens:
        dtype = torch.bfloat16 if args.payload == "bf16" else torch.float8_e4m3fn
        x = torch.ones((n, 2048), dtype=dtype, device="cuda")
        ids = (
            torch.arange(n, device="cuda")[:, None] * 3
            + torch.arange(8, device="cuda")[None, :] * 5
            + max(rank - 2, 0)
        ) % 128
        if args.routing == "random":
            generator = torch.Generator(device="cuda").manual_seed(
                1234 + max(rank - 2, 0)
            )
            ids = (
                torch.rand((n, 128), generator=generator, device="cuda")
                .topk(8, dim=1)
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
                        ((canonical >= w * 64) & (canonical < (w + 1) * 64))
                        .any(axis=1)
                        .sum()
                    )
                    for w in range(2)
                ],
            }
        )
        weights = torch.full((n, 8), 1 / 8, device="cuda", dtype=torch.float32)
        scale = torch.ones((n, 16), device="cuda") if args.payload == "fp8" else None
        received = (
            buffer.reserve_worker_receive(bucket=n) if worker and directional else None
        )
        expert = None
        result = torch.empty((n, 2048), dtype=torch.bfloat16, device="cuda")

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
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        if not worker:
            torch.testing.assert_close(
                result, torch.full_like(result, 8), rtol=0, atol=0
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
                graph.replay()
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall) * 1000 / args.iterations
            local = {
                "rank": rank,
                "gpu_ms": start.elapsed_time(end) / args.iterations,
                "wall_ms": wall_ms,
                "worker": worker,
            }
            gathered = [None] * 4
            dist.all_gather_object(gathered, local)
            if args.profile:
                dist.barrier()
                torch.cuda.cudart().cudaProfilerStop()
            if not worker:
                torch.testing.assert_close(
                    result, torch.full_like(result, 8), rtol=0, atol=0
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
        del graph, expert, received, result
        torch.cuda.synchronize()
        dist.barrier()
    metadata = {
        "rank": rank,
        "config": asdict(config),
        "registered_bytes": buffer.num_buffer_bytes,
        "resolved_comm_sms": config.num_sms
        if directional
        else buffer.transport._resolved_num_sms,
        "device": str(torch.cuda.get_device_properties(rank)),
        "torch": torch.__version__,
        "routing": route_records if not worker else None,
    }
    meta = [None] * 4
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
                        "communication only; graph startup excluded, replay included; "
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
