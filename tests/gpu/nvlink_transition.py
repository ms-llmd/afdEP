# SPDX-License-Identifier: Apache-2.0
"""Alternate generic and bulk collectors on one buffer, eagerly or with graphs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace

import torch
import torch.distributed as dist

from afdep import DirectionalBuffer, DirectionalConfig
from tests.gpu.nvlink import (
    COUNTS,
    DELAY_CYCLES,
    expected_output,
    expert_output,
    inputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="keep worker bucket 2048; alternate source 0 storage 1536/2048",
    )
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    config = DirectionalConfig(
        2048,
        args.hidden,
        128,
        8,
        ffn_ranks=2,
        attention_ranks=2,
        payload=args.payload,
        transport="nvlink",
    )
    buffer = DirectionalBuffer(group=dist.group.WORLD, config=config)
    worker = rank < 2
    source = max(rank - 2, 0)
    paths = {}

    def prepare(extent):
        bucket = config.capacity_tokens_per_rank if args.asymmetric else extent
        source0_tokens = extent if args.asymmetric else None
        shape = replace(config, capacity_tokens_per_rank=bucket)
        x, scales, ids, weights, n = inputs(source, 0, shape, source0_tokens)
        live = torch.tensor([n], dtype=torch.int32, device="cuda")
        out = torch.empty_like(x, dtype=torch.bfloat16)
        receive = buffer.reserve_worker_receive(bucket=bucket) if worker else None

        def step():
            if worker:
                buffer.worker_receive(bucket=bucket, out=receive)
                value = expert_output(receive, shape, rank)
                if rank == 0:
                    torch.cuda._sleep(DELAY_CYCLES)
                buffer.worker_reply(value, receive.receipt)
            else:
                buffer.source_dispatch(
                    x, ids, weights, scales, bucket=bucket, live_tokens=live
                )
                if source == 1:
                    torch.cuda._sleep(DELAY_CYCLES)
                buffer.source_collect(out)

        step()
        torch.cuda.synchronize()
        dist.barrier()
        graph = None
        if args.graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
        return shape, source0_tokens, x, scales, ids, weights, live, out, step, graph

    extents = (1536, 2048) if args.asymmetric else (32, 2048)
    for extent in extents:
        paths[extent] = prepare(extent)
    dist.barrier()
    # No per-epoch CPU collective: workers enqueue ahead of delayed readers.
    small, large = extents
    sequence = (small, large, large, small, large, small)
    for epoch in range(2 * len(COUNTS)):
        extent = sequence[epoch % len(sequence)]
        case = epoch % len(COUNTS)
        shape, source0_tokens, x, scales, ids, weights, live, out, step, graph = paths[
            extent
        ]
        if not worker:
            updated = inputs(source, case, shape, source0_tokens)
            x.copy_(updated[0])
            if scales is not None:
                scales.copy_(updated[1])
            ids.copy_(updated[2])
            weights.copy_(updated[3])
            live.fill_(updated[4])
        if graph is None:
            step()
        else:
            graph.replay()
        if not worker:
            torch.cuda.synchronize()
            expected = expected_output(source, case, shape, source0_tokens)
            torch.testing.assert_close(out[: len(expected)], expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    dist.barrier()
    paths.clear()
    buffer.close()
    dist.destroy_process_group()
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "graph": args.graph,
                    "payload": args.payload,
                    "hidden": args.hidden,
                    "epochs": 2 * len(COUNTS),
                    "buckets": [2048] if args.asymmetric else list(extents),
                    "source0_storage_tokens": list(extents)
                    if args.asymmetric
                    else None,
                }
            )
        )


if __name__ == "__main__":
    main()
