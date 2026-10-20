# SPDX-License-Identifier: Apache-2.0
"""Check directional BF16 return sums across every BF16 operand bit pattern.

Four GPUs; eager or --graph, with --bulk for the warp/TMA path. Expert outputs
select the two halves of each source input, so the reference independently models
worker and source rounding.
NaN payloads and zero signs are not part of the numerical equality contract.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from afdep import DirectionalBuffer, DirectionalConfig


def operands(case: int, source: int, tokens: int, half: int):
    bits = (torch.arange(tokens * half, dtype=torch.int32) + source * 0x4321) & 0xFFFF
    other = (
        (bits * 13007 + 48251) & 0xFFFF,
        bits ^ 0x8000,
        bits,
        (bits + 0x80) & 0xFFFF,
    )[case]
    return (
        bits.to(torch.int16).view(torch.bfloat16).reshape(tokens, half),
        other.to(torch.int16).view(torch.bfloat16).reshape(tokens, half),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--bulk", action="store_true")
    parser.add_argument(
        "--hidden",
        type=int,
        choices=(2048, 4096),
        default=2048,
        help="hidden size for --bulk",
    )
    args = parser.parse_args()
    if args.hidden != 2048 and not args.bulk:
        parser.error("hidden size requires --bulk")
    tokens, half, experts = (2048, args.hidden // 2, 8) if args.bulk else (512, 128, 2)
    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    config = DirectionalConfig(
        tokens,
        2 * half,
        experts,
        experts,
        ffn_ranks=2,
        attention_ranks=2,
        transport="nvlink",
    )
    buffer = DirectionalBuffer(group=dist.group.WORLD, config=config)
    worker = rank < 2
    if worker:
        received = buffer.reserve_worker_receive(bucket=tokens)
        expert = torch.empty_like(received.recv_x)
        rows = torch.arange(expert.shape[0], device="cuda")
    else:
        x = torch.empty((tokens, 2 * half), device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(experts, device="cuda").expand(tokens, -1).contiguous()
        weights = torch.ones((tokens, experts), device="cuda")
        output = torch.empty_like(x)
        a, b = operands(0, rank - 2, tokens, half)
        x.copy_(torch.cat((a, b), dim=1))

    def step():
        if worker:
            buffer.worker_receive(bucket=tokens, out=received)
            values = received.recv_x[:, rank * half : (rank + 1) * half]
            expert[:, :half].copy_(values)
            expert[:, half:].copy_(values)
            # Only the first local expert returns values. The remaining experts
            # force the >2-input reduction path while keeping an exact reference.
            expert.masked_fill_((rows >= received.recv_count_psum[0])[:, None], 0)
            buffer.worker_reply(expert, received.receipt)
        else:
            buffer.source_dispatch(x, ids, weights, bucket=tokens)
            buffer.source_collect(output)

    step()
    torch.cuda.synchronize()
    dist.barrier()
    graph = None
    if args.graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
    for case in range(4):
        if not worker:
            a, b = operands(case, rank - 2, tokens, half)
            x.copy_(torch.cat((a, b), dim=1))
            # Each worker sums one nonzero row and any remaining zero rows into
            # FP32, then returns BF16. The source adds those two partials.
            zero = torch.zeros_like(a, dtype=torch.float32)
            first = (zero + a.float()).bfloat16()
            second = (zero + b.float()).bfloat16()
            expected = (zero + first.float() + second.float()).bfloat16()
            expected = torch.cat((expected, expected), dim=1).cuda()
        if graph is None:
            step()
        else:
            graph.replay()
        torch.cuda.synchronize()
        if not worker:
            torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)
        dist.barrier()
    del graph
    buffer.close()
    dist.destroy_process_group()
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "graph": args.graph,
                    "bulk": args.bulk,
                    "hidden": 2 * half,
                    "cases": 4,
                    "operand_patterns": 65536,
                }
            )
        )


if __name__ == "__main__":
    main()
