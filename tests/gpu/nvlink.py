# SPDX-License-Identifier: Apache-2.0
"""Four-GPU directional NVLink correctness, including changing graph inputs.

Launch with torchrun --nproc-per-node=4 -m tests.gpu.nvlink. No model or GEMM
dependency: a deterministic expert operation checks the entire token exchange.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist

from afdep import DirectionalBuffer as Buffer
from afdep import DirectionalConfig as Config

CAPACITY = 128
DELAY_CYCLES = 500_000
COUNTS = (
    (1, 0),
    (0, 1),
    (8, 1),
    (32, 8),
    (128, 128),
    (0, 0),
    (0, 128),
    (128, 0),
    (31, 33),
    (64, 65),
    (127, 1),
    (1, 127),
)


def inputs(source, case, config, source0_tokens=None):
    device = torch.device("cuda")
    capacity = config.capacity_tokens_per_rank
    if source == 0 and source0_tokens is not None:
        capacity = source0_tokens
    token = torch.arange(capacity, device=device)[:, None]
    column = torch.arange(config.hidden_size, device=device)[None, :]
    x = ((token * 13 + column * 7 + source * 11 + case * 17) % 97) - 48
    x = (x.float() / 32).to(torch.bfloat16)
    lane = torch.arange(config.top_k, device=device)[None, :]
    if case % 5 in (1, 2):
        ids = (token + lane) % config.experts_per_rank
        ids += (case % 5 - 1) * config.experts_per_rank
    else:
        ids = (token * 3 + lane * 5 + source + case * 7) % config.num_experts
    weights = (lane + 1).float().expand(capacity, -1).clone()
    weights /= config.top_k * (config.top_k + 1) / 2
    if case % 5 == 3:
        ids = torch.where((token + lane) % 3 == 0, -1, ids)
    if case % 5 == 4 and source == 1:
        ids.fill_(-1)
    weights.masked_fill_(ids < 0, 0)
    scale = None
    if config.payload == "fp8":
        blocks = x.float().reshape(capacity, -1, 128)
        scale = blocks.abs().amax(dim=2).clamp_min(1e-10) / 448
        x = (blocks / scale[:, :, None]).reshape_as(x).to(torch.float8_e4m3fn)
        # Exercise non-contiguous source scales independently of receive scales.
        scale = scale.t().contiguous().t()
    n = COUNTS[case][source]
    n = capacity if n == CAPACITY else min(n, capacity)
    return x, scale, ids.long(), weights, n


def decode(x, scale):
    value = x.float()
    return value if scale is None else value * scale.repeat_interleave(128, dim=1)


def expert_output(received, config, worker):
    row = torch.arange(received.recv_x.shape[0], device="cuda", dtype=torch.int32)
    local = torch.searchsorted(received.recv_count_psum, row, right=True)
    factor = (local + worker * config.experts_per_rank + 1).float()
    value = decode(received.recv_x, received.recv_scale)
    return (value * factor[:, None] * received.recv_topk_weights[:, None]).to(
        torch.bfloat16
    )


def expected_output(source, case, config, source0_tokens=None):
    x, scale, ids, weights, n = inputs(source, case, config, source0_tokens)
    value = decode(x[:n], None if scale is None else scale[:n])
    total = torch.zeros_like(value)
    for worker in range(config.ffn_ranks):
        partial = torch.zeros_like(value)
        for k in range(config.top_k):
            expert = ids[:n, k]
            owned = (expert >= worker * config.experts_per_rank) & (
                expert < (worker + 1) * config.experts_per_rank
            )
            term = value * (expert + 1).float()[:, None] * weights[:n, k, None]
            # Expert outputs enter the transport as BF16, before reduction.
            term = term.to(torch.bfloat16).float()
            partial += torch.where(owned[:, None], term, 0)
        total += partial.to(torch.bfloat16).float()
    return total.to(torch.bfloat16)


def check_worker(received, worker, case, config, source0_tokens=None):
    expected = torch.zeros(config.experts_per_rank, dtype=torch.int64, device="cuda")
    for source in range(config.attention_ranks):
        _, _, ids, _, n = inputs(source, case, config, source0_tokens)
        owned = ids[:n] - worker * config.experts_per_rank
        selected = owned[(owned >= 0) & (owned < config.experts_per_rank)]
        expected += torch.bincount(selected, minlength=config.experts_per_rank)
        assert (received.receipt[source, n:] == -1).all().item()
    end = received.recv_count_psum.long()
    previous = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int64), end[:-1]))
    start = (previous + config.expert_alignment - 1) // config.expert_alignment
    start *= config.expert_alignment
    torch.testing.assert_close(end - start, expected, rtol=0, atol=0)
    assert int((received.receipt >= 0).sum()) == int(expected.sum())
    row = torch.arange(received.recv_topk_weights.numel(), device="cuda")
    expert = torch.searchsorted(end, row, right=True)
    padding = expert >= config.experts_per_rank
    padding |= row < start[expert.clamp_max(config.experts_per_rank - 1)]
    weights = received.recv_topk_weights[padding]
    torch.testing.assert_close(weights, torch.zeros_like(weights), rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("bf16", "fp8"), required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--mixed-buffer", action="store_true")
    parser.add_argument("--early-metadata", action="store_true")
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--bucket", type=int)
    parser.add_argument("--source0-tokens", type=int)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--invalid", choices=("expert", "duplicate", "count"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.exact and args.graph:
        parser.error("exact allocation requires eager execution")
    bucket = args.bucket if args.bucket is not None else args.capacity
    if not 1 <= bucket <= args.capacity:
        parser.error("bucket must be within [1,capacity]")
    if args.source0_tokens is not None and not 1 <= args.source0_tokens <= bucket:
        parser.error("source0-tokens must be within [1,bucket]")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    config = Config(
        args.capacity,
        args.hidden,
        args.experts,
        args.top_k,
        ffn_ranks=2,
        attention_ranks=2,
        payload=args.payload,
        transport="nvlink",
        nvlink_early_metadata=args.early_metadata,
    )
    buffer = Buffer(group=dist.group.WORLD, config=config)
    allocation_config = config
    config = replace(config, capacity_tokens_per_rank=bucket)
    capacity = bucket
    worker = rank < config.ffn_ranks
    received = (
        buffer.reserve_worker_receive(bucket=capacity)
        if worker and not args.exact
        else None
    )
    source = rank - config.ffn_ranks
    prepared = inputs(max(source, 0), 0, config, args.source0_tokens)
    x, scale, ids, weights, n = prepared
    live = torch.tensor([n], dtype=torch.int32, device="cuda")
    out = torch.empty(
        (x.shape[0], config.hidden_size), dtype=torch.bfloat16, device="cuda"
    )

    def rejects_pair(operation):
        try:
            operation()
        except RuntimeError as error:
            assert "NVLink" in str(error), error
        else:
            raise AssertionError("Overlapping/duplicate pair was accepted")

    def step(check_pairs=False):
        nonlocal received
        if worker:
            received = (
                buffer.worker_receive(
                    bucket=capacity, exact_rows=True, read_host_counts=True
                )
                if args.exact
                else buffer.worker_receive(bucket=capacity, out=received)
            )
            if check_pairs:
                rejects_pair(
                    lambda: (
                        buffer.worker_receive(
                            bucket=capacity, exact_rows=True, read_host_counts=True
                        )
                        if args.exact
                        else buffer.worker_receive(bucket=capacity, out=received)
                    )
                )
            result = expert_output(received, config, rank)
            if rank == 0:
                torch.cuda._sleep(DELAY_CYCLES)
            buffer.worker_reply(result, received.receipt)
            if check_pairs:
                rejects_pair(lambda: buffer.worker_reply(result, received.receipt))
        else:
            buffer.source_dispatch(
                x, ids, weights, scale, bucket=capacity, live_tokens=live
            )
            if check_pairs:
                rejects_pair(
                    lambda: buffer.source_dispatch(
                        x, ids, weights, scale, bucket=capacity, live_tokens=live
                    )
                )
            if source == 1:
                torch.cuda._sleep(DELAY_CYCLES)
            buffer.source_collect(out)
            if check_pairs:
                rejects_pair(lambda: buffer.source_collect(out))

    if args.invalid:
        if not worker and source == 0:
            if args.invalid == "expert":
                ids[0, 0] = config.num_experts
            elif args.invalid == "duplicate":
                ids[0, 1] = ids[0, 0]
            else:
                live.fill_(capacity + 1)
        step()
        torch.cuda.synchronize()
        raise AssertionError(f"Invalid {args.invalid} was not rejected")

    step(check_pairs=True)
    torch.cuda.synchronize()
    dist.barrier()
    graph = None
    if args.graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        dist.barrier()
    if args.mixed_buffer:
        # Exercise graph replay after another payload's constructor has set
        # attributes on the same CUDA functions in this device/context.
        if graph is None:
            parser.error("mixed-buffer requires graph")
        other = Buffer(
            group=dist.group.WORLD,
            config=replace(config, payload="fp8" if args.payload == "bf16" else "bf16"),
        )
        other.close()
    records = []
    for case in range(len(COUNTS)):
        if not worker:
            updated = inputs(source, case, config, args.source0_tokens)
            x.copy_(updated[0])
            if scale is not None:
                scale.copy_(updated[1])
            ids.copy_(updated[2])
            weights.copy_(updated[3])
            live.fill_(updated[4])
        if graph is None:
            step()
        else:
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        if worker:
            check_worker(received, rank, case, config, args.source0_tokens)
        else:
            expected = expected_output(source, case, config, args.source0_tokens)
            torch.testing.assert_close(
                out[: expected.shape[0]],
                expected,
                rtol=0,
                atol=0,
                msg=lambda message, case=case, source=source: (
                    f"source={source}, case={case}: {message}"
                ),
            )
        torch.cuda.synchronize()
        dist.barrier()
        counts = tuple(
            inputs(source, case, config, args.source0_tokens)[4]
            for source in range(config.attention_ranks)
        )
        records.append({"case": case, "source_counts": counts, "ok": True})
    # Let FFN ranks enqueue ahead with no per-epoch CPU collective. Fast
    # workers must wait for delayed readers before reusing their partials.
    for case in range(len(COUNTS)):
        if not worker:
            updated = inputs(source, case, config, args.source0_tokens)
            x.copy_(updated[0])
            if scale is not None:
                scale.copy_(updated[1])
            ids.copy_(updated[2])
            weights.copy_(updated[3])
            live.fill_(updated[4])
        if graph is None:
            step()
        else:
            graph.replay()
        if not worker:
            torch.cuda.synchronize()
            expected = expected_output(source, case, config, args.source0_tokens)
            torch.testing.assert_close(
                out[: expected.shape[0]], expected, rtol=0, atol=0
            )
    torch.cuda.synchronize()
    dist.barrier()
    buffer.close()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "rank": rank,
        "ok": True,
        "config": asdict(config),
        "allocation_config": asdict(allocation_config),
        "graph": args.graph,
        "exact_allocation": args.exact,
        "mixed_payload_constructor_after_capture": args.mixed_buffer,
        "cases": records,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(torch.cuda.get_device_properties(rank)),
        "delay_cycles": DELAY_CYCLES,
        "queued_reuse_epochs": len(COUNTS),
        "source0_storage_tokens": args.source0_tokens,
        "pair_guards_and_recovery": True,
        "reuse_per_epoch_cpu_barrier": False,
        "performance_claim": False,
    }
    (args.output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rank": rank, "ok": True, "cases": len(records)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
