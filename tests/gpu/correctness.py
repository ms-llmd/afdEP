# SPDX-License-Identifier: Apache-2.0
"""Eight-rank standalone transport validation.

Launch on both hosts with the same torchrun arguments as examples.round_trip,
replacing its module with ``tests.gpu.correctness --payload bf16 --mode hybrid``.
Run both payloads and both modes; each invocation validates one combination.
This intentionally is not collected by ordinary CPU pytest runs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch
import torch.distributed as dist

import afdep
from examples.round_trip import (
    ATTENTION_RANKS,
    FFN_RANKS,
    FP8_BLOCK_SIZE,
    HIDDEN_SIZE,
    NUM_EXPERTS,
    TOP_K,
    Case,
    Exchange,
    initialize,
    make_exchange,
    make_inputs,
)

DEFAULT_CAPACITY = 128
DEFAULT_GRAPH_EPOCHS = 28
DELAYED_WORKER = 1
DELAYED_EPOCH = 8
DELAY_SECONDS = 0.02


def check_received(exchange: Exchange, epoch: int) -> None:
    """Verify every routed row, weight, expert boundary, and receipt mapping."""
    received = exchange.received
    config = exchange.config
    local_experts = config.num_experts // FFN_RANKS
    first_expert = exchange.rank * local_experts
    counts = torch.zeros(local_experts, dtype=torch.int64, device="cuda")
    real_rows = torch.zeros(received.recv_x.shape[0], dtype=torch.bool, device="cuda")
    row_experts = torch.full(
        (received.recv_x.shape[0],), -1, dtype=torch.int64, device="cuda"
    )
    decoded = received.recv_x.float()
    if received.recv_scale is not None:
        decoded = (
            decoded.reshape(
                decoded.shape[0], config.hidden_size // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE
            )
            * received.recv_scale[:, :, None]
        ).reshape(decoded.shape)
    for source in range(ATTENTION_RANKS):
        expected = make_inputs(exchange.case, source, epoch, config)
        count = expected.x.shape[0]
        receipt = received.receipt[source]
        selected = (expected.ids >= first_expert) & (
            expected.ids < first_expert + local_experts
        )
        torch.testing.assert_close(receipt[:count] >= 0, selected, rtol=0, atol=0)
        if count < exchange.case.bucket:
            assert bool((receipt[count:] == -1).all()), (
                "receipt retained absent source rows"
            )
        tokens, slots = selected.nonzero(as_tuple=True)
        rows = receipt[:count][selected].long()
        assert rows.numel() == rows.unique().numel(), "duplicate local receipt rows"
        assert not bool(real_rows[rows].any()), "different sources alias receive rows"
        real_rows[rows] = True
        row_experts[rows] = expected.ids[tokens, slots] - first_expert
        torch.testing.assert_close(
            decoded[rows], expected.decoded()[tokens], rtol=0, atol=0
        )
        torch.testing.assert_close(
            received.recv_topk_weights[rows],
            expected.weights[tokens, slots],
            rtol=0,
            atol=0,
        )
        counts += torch.bincount(
            expected.ids[selected] - first_expert, minlength=local_experts
        )
    count_list = counts.cpu().tolist()
    end = 0
    expected_ends = []
    expected_padded_counts = []
    for count in count_list:
        start = (
            (end + config.expert_alignment - 1)
            // config.expert_alignment
            * config.expert_alignment
        )
        end = start + count
        expected_ends.append(end)
        expected_padded_counts.append(
            (count + config.expert_alignment - 1)
            // config.expert_alignment
            * config.expert_alignment
        )
    assert received.recv_count_psum.cpu().tolist() == expected_ends
    assert received.recv_counts_host == expected_padded_counts
    positions = real_rows.nonzero(as_tuple=True)[0]
    actual_experts = torch.bucketize(positions, received.recv_count_psum, right=True)
    torch.testing.assert_close(actual_experts, row_experts[positions], rtol=0, atol=0)
    assert bool((received.recv_topk_weights[~real_rows] == 0).all()), (
        "padding weight is nonzero"
    )


def cases(capacity: int) -> list[Case]:
    uniform = min(16, capacity)
    return [
        Case("uniform", (uniform,) * ATTENTION_RANKS),
        Case("uneven", (capacity, 0, 1, min(7, capacity))),
        Case("empty", (0,) * ATTENTION_RANKS),
        Case("decode", (1,) * ATTENTION_RANKS),
        *(
            Case(f"only_worker_{worker}", (uniform,) * ATTENTION_RANKS, worker=worker)
            for worker in range(FFN_RANKS)
        ),
        Case(
            "rotating_worker_per_source",
            (uniform,) * ATTENTION_RANKS,
            rotate_worker=True,
        ),
        Case("all_sources_at_capacity", (capacity,) * ATTENTION_RANKS),
        # Fill one worker's direct-mode request slot from each source.
        Case(
            "rotating_worker_at_capacity",
            (capacity,) * ATTENTION_RANKS,
            rotate_worker=True,
        ),
        Case(
            "all_workers_per_token",
            (capacity,) * ATTENTION_RANKS,
            spread_workers=True,
        ),
        Case(
            "nonempty_all_routes_skipped",
            (uniform,) * ATTENTION_RANKS,
            skip_all_experts=True,
        ),
        Case(
            "strided_source_fp8_scales",
            (uniform,) * ATTENTION_RANKS,
            strided_scales=True,
        ),
    ]


def eager_checks(buffer: afdep.Buffer, config: afdep.Config, rank: int) -> list[str]:
    checked = []
    for case in cases(config.capacity_tokens_per_rank):
        exchange = make_exchange(buffer, config, rank, case)
        if case.strided_scales and config.payload == "fp8" and rank >= FFN_RANKS:
            assert not exchange.inputs.scale.is_contiguous()
        for epoch in range(2):
            exchange.update(epoch)
            exchange.run(read_host_counts=True)
            torch.cuda.synchronize()
            if rank < FFN_RANKS:
                check_received(exchange, epoch)
            else:
                exchange.check_output()
            # Eager tests are diagnostic and may perform host inspections.
            dist.barrier()
        checked.append(case.name)
    return checked


def graph_checks(
    buffer: afdep.Buffer, config: afdep.Config, rank: int, epochs: int
) -> dict:
    captures = []
    for case in cases(config.capacity_tokens_per_rank):
        exchange = make_exchange(buffer, config, rank, case)
        if case.strided_scales and config.payload == "fp8" and rank >= FFN_RANKS:
            assert not exchange.inputs.scale.is_contiguous()
        exchange.run()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            exchange.run()
        torch.cuda.synchronize()
        dist.barrier()
        captures.append((graph, exchange))
    # From here there is no per-epoch process-group control or barrier. Each
    # graph has a different bucket, while all epochs share transport windows.
    # Changed source/token/epoch bits catch stale and incorrectly routed data.
    for epoch in range(epochs):
        graph, exchange = captures[epoch % len(captures)]
        if rank == DELAYED_WORKER and epoch == DELAYED_EPOCH:
            time.sleep(DELAY_SECONDS)
        exchange.update(epoch + 1)
        graph.replay()
        if rank >= FFN_RANKS:
            torch.cuda.synchronize()
            exchange.check_output()
    torch.cuda.synchronize()
    dist.barrier()
    return {
        "epochs": epochs,
        "cases": [exchange.case.name for _, exchange in captures],
        "changed_payload_and_routes": True,
        "changing_destination_workers": True,
        "per_epoch_cpu_barrier": False,
        "delayed_worker_seconds": DELAY_SECONDS,
    }


def alignment_checks(buffer: afdep.Buffer, config: afdep.Config, rank: int) -> None:
    """Reject offset contiguous views before any protocol kernel is enqueued."""
    if rank >= FFN_RANKS:
        sample = make_inputs(
            Case("invalid_alignment", (1,) * ATTENTION_RANKS),
            rank - FFN_RANKS,
            0,
            config,
        )
        storage = torch.empty(
            config.hidden_size + 1, dtype=sample.x.dtype, device="cuda"
        )
        misaligned_x = storage[1:].reshape(1, config.hidden_size)
        assert misaligned_x.is_contiguous() and misaligned_x.data_ptr() % 16
        try:
            buffer.source_dispatch(
                misaligned_x,
                sample.ids,
                sample.weights,
                x_scale=sample.scale,
                bucket=1,
            )
        except RuntimeError as error:
            assert "align" in str(error).lower(), str(error)
        else:
            raise AssertionError("misaligned contiguous source activation was accepted")
        # The last successful graph can have any source row count. Reuse its
        # count by checking the output before the next valid eager exchange.
    dist.barrier()
    case = Case("alignment_recovery", (1,) * ATTENTION_RANKS)
    exchange = make_exchange(buffer, config, rank, case)
    exchange.run()
    torch.cuda.synchronize()
    exchange.check_output()
    if rank >= FFN_RANKS:
        storage = torch.empty(
            config.hidden_size + 1, dtype=torch.bfloat16, device="cuda"
        )
        misaligned_out = storage[1:].reshape(1, config.hidden_size)
        try:
            buffer.source_collect(misaligned_out)
        except RuntimeError as error:
            assert "align" in str(error).lower(), str(error)
        else:
            raise AssertionError("misaligned contiguous source output was accepted")
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--mode", choices=("hybrid", "direct"), default="hybrid")
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_GRAPH_EPOCHS)
    parser.add_argument("--rdma-traffic-class", type=int, default=0)
    args = parser.parse_args()
    if args.capacity < 1 or args.epochs <= DELAYED_EPOCH:
        parser.error(
            "capacity must be positive and epochs must exceed the delayed epoch "
            f"({DELAYED_EPOCH})"
        )
    rank = initialize()
    config = afdep.Config(
        capacity_tokens_per_rank=args.capacity,
        hidden_size=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        payload=args.payload,
        allow_hybrid_mode=args.mode == "hybrid",
        rdma_traffic_class=args.rdma_traffic_class,
    )
    buffer = afdep.Buffer(group=dist.group.WORLD, config=config)
    eager = eager_checks(buffer, config, rank)
    graph = graph_checks(buffer, config, rank, args.epochs)
    alignment_checks(buffer, config, rank)
    buffer.close()
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, {"rank": rank, "ok": True})
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": args.mode,
                    "config": asdict(config),
                    "eager_cases": eager,
                    "graph": graph,
                    "misaligned_contiguous_source_and_output_rejected": True,
                    "ranks": results,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
