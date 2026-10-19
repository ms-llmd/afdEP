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
    parser.add_argument("--short-source", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--stress",
        action="store_true",
        help="poison reply slots before changing routes, shapes and expert outputs",
    )
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="keep worker bucket 2048; alternate short-source storage 1536/2048",
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
    poison = torch.zeros((), dtype=torch.bool, device="cuda") if args.stress else None
    sign = torch.ones((), dtype=torch.bfloat16, device="cuda") if args.stress else None

    def prepare(extent):
        bucket = config.capacity_tokens_per_rank if args.asymmetric else extent
        shape = replace(config, capacity_tokens_per_rank=bucket)
        input_shape = (
            replace(shape, capacity_tokens_per_rank=extent)
            if args.asymmetric and source == args.short_source
            else shape
        )
        x, scales, ids, weights, n = inputs(source, 0, input_shape)
        live = torch.tensor([n], dtype=torch.int32, device="cuda")
        out = torch.empty_like(x, dtype=torch.bfloat16)
        receive = buffer.reserve_worker_receive(bucket=bucket) if worker else None

        def step():
            if worker:
                buffer.worker_receive(bucket=bucket, out=receive)
                value = expert_output(receive, shape, rank)
                if args.stress:
                    value = value * sign
                    value.masked_fill_(poison, float("nan"))
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
        return shape, input_shape, x, scales, ids, weights, live, out, step, graph

    extents = (1536, 2048) if args.asymmetric else (32, 2048)
    for extent in extents:
        paths[extent] = prepare(extent)
    dist.barrier()
    # No per-epoch CPU collective: workers enqueue ahead of delayed readers.
    small, large = extents
    sequence = (small, large, large, small, large, small)
    for epoch in range(2 * len(COUNTS)):
        if args.stress:
            # A valid full-size exchange writes NaNs into BOTH worker slots.
            # The next epoch must overwrite or avoid every unselected slot.
            poison.fill_(True)
            _, _, px, _, pids, pweights, plive, pout, pstep, pgraph = paths[large]
            if not worker:
                token = torch.arange(len(px), device="cuda")[:, None]
                lane = torch.arange(config.top_k, device="cuda")[None, :]
                pids.copy_(
                    (lane % config.ffn_ranks) * config.experts_per_rank
                    + (token + lane // config.ffn_ranks) % config.experts_per_rank
                )
                pweights.fill_(1 / config.top_k)
                plive.fill_(len(px))
            if pgraph is None:
                pstep()
            else:
                pgraph.replay()
            if not worker:
                torch.cuda.synchronize()
                assert torch.isnan(pout).all().item()
            poison.fill_(False)
            sign.fill_(-1 if epoch % 2 else 1)
        extent = sequence[epoch % len(sequence)]
        case = epoch % len(COUNTS)
        shape, input_shape, x, scales, ids, weights, live, out, step, graph = paths[
            extent
        ]
        skipped = args.stress and epoch % len(sequence) == 0
        if not worker:
            updated = inputs(source, case, input_shape)
            x.copy_(updated[0])
            if scales is not None:
                scales.copy_(updated[1])
            ids.copy_(updated[2])
            weights.copy_(updated[3])
            if skipped:
                ids.fill_(-1)
                weights.zero_()
            live.fill_(updated[4])
        if graph is None:
            step()
        else:
            graph.replay()
        if not worker:
            torch.cuda.synchronize()
            expected = expected_output(source, case, input_shape)
            if args.stress:
                expected = (
                    torch.zeros_like(expected)
                    if skipped
                    else expected * (-1 if epoch % 2 else 1)
                )
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
                    "stress": args.stress,
                    "poison_epochs": 2 * len(COUNTS) if args.stress else 0,
                    "epochs": 2 * len(COUNTS),
                    "buckets": [2048] if args.asymmetric else list(extents),
                    "short_source": args.short_source if args.asymmetric else None,
                    "short_source_storage_tokens": list(extents)
                    if args.asymmetric
                    else None,
                }
            )
        )


if __name__ == "__main__":
    main()
