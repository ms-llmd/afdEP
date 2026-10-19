# SPDX-License-Identifier: Apache-2.0
"""Eight-rank standalone transport validation.

Launch on both hosts with the same torchrun arguments as examples.directional,
replacing its module with ``tests.gpu.directional --payload bf16 --mode hybrid``.
Run both payloads and both modes; each invocation validates one combination.
This intentionally is not collected by ordinary CPU pytest runs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace

import torch
import torch.distributed as dist

from afdep import DirectionalBuffer as Buffer
from afdep import DirectionalConfig as Config
from afdep.buffer import ReceiveResult
from examples.directional import (
    ATTENTION_RANKS,
    FFN_RANKS,
    FP8_BLOCK_SIZE,
    HIDDEN_SIZE,
    NUM_EXPERTS,
    TOP_K,
    Case,
    Exchange,
    expert_rows,
    initialize,
    make_exchange,
    make_inputs,
)

DEFAULT_CAPACITY = 128
DEFAULT_GRAPH_EPOCHS = 26
DELAYED_WORKER = 1
DELAYED_EPOCH = 8
DELAY_SECONDS = 0.02


def check_received(exchange: Exchange, epoch: int, *, exact_rows: bool = False) -> None:
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
    if exact_rows:
        assert received.recv_x.shape[0] == sum(expected_padded_counts), (
            "exact receive storage does not match GPU-discovered aligned counts"
        )
        assert received.recv_topk_weights.shape == (sum(expected_padded_counts),)
        if received.recv_scale is not None:
            assert received.recv_scale.shape == (
                sum(expected_padded_counts),
                config.hidden_size // FP8_BLOCK_SIZE,
            )
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


def eager_checks(
    buffer: Buffer,
    config: Config,
    rank: int,
    *,
    exact_rows: bool = False,
) -> list[str]:
    checked = []
    for case in cases(config.capacity_tokens_per_rank):
        exchange = make_exchange(buffer, config, rank, case)
        if case.strided_scales and config.payload == "fp8" and rank >= FFN_RANKS:
            assert not exchange.inputs.scale.is_contiguous()
        for epoch in range(2):
            exchange.update(epoch)
            if exact_rows and rank < FFN_RANKS:
                if rank == DELAYED_WORKER and case.name == "uneven" and epoch == 1:
                    time.sleep(DELAY_SECONDS)
                exchange.received = buffer.worker_receive(
                    bucket=case.bucket, read_host_counts=True, exact_rows=True
                )
                buffer.worker_reply(
                    expert_rows(exchange.received, rank, config),
                    exchange.received.receipt,
                )
            else:
                exchange.run(read_host_counts=True)
            torch.cuda.synchronize()
            if rank < FFN_RANKS:
                check_received(exchange, epoch, exact_rows=exact_rows)
            else:
                exchange.check_output()
            # Eager tests are diagnostic and may perform host inspections.
            dist.barrier()
        checked.append(case.name)
    return checked


def exact_receive_checks(buffer: Buffer, config: Config, rank: int) -> dict:
    """Discover row capacity on the GPU, then allocate without stale epochs."""
    if rank < FFN_RANKS:
        reserved = buffer.reserve_worker_receive(bucket=1)
        invalid = (
            {"exact_rows": True},
            {"exact_rows": True, "read_host_counts": True, "out": reserved},
        )
        for options in invalid:
            try:
                buffer.worker_receive(bucket=1, **options)
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("invalid exact receive options were accepted")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            try:
                buffer.worker_receive(bucket=1, read_host_counts=True, exact_rows=True)
            except RuntimeError as error:
                assert "captur" in str(error).lower(), str(error)
            else:
                raise AssertionError("exact receive allowed host counts in a graph")
    dist.barrier()
    # A successful exchange after all invalid calls proves that they did not
    # advance the transport epoch or enqueue a receive waiting for absent peers.
    checked = eager_checks(buffer, config, rank, exact_rows=True)
    return {
        "cases": checked,
        "capacity_matches_discovered_aligned_counts": True,
        "zero_row_allocations": True,
        "invalid_options_rejected_before_exchange": True,
        "captured_count_sync_rejected_before_exchange": True,
        "changing_buckets_and_routes": True,
        "delayed_worker_seconds": DELAY_SECONDS,
    }


def graph_checks(buffer: Buffer, config: Config, rank: int, epochs: int) -> dict:
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


def alignment_checks(buffer: Buffer, config: Config, rank: int) -> None:
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


def reusable_receive_checks(
    buffer: Buffer, config: Config, rank: int, epochs: int
) -> dict:
    """Reuse each bucket's receive storage across changing eager/graph exchanges."""
    reservations = {}
    pointers = {}

    def storage_pointers(received: ReceiveResult) -> tuple[int, ...]:
        return tuple(
            tensor.data_ptr()
            for tensor in (
                received.recv_x,
                received.recv_scale,
                received.recv_topk_weights,
                received.recv_count_psum,
                received.receipt,
            )
            if tensor is not None
        )

    def run(exchange: Exchange, *, read_host_counts: bool = False) -> None:
        if rank >= FFN_RANKS:
            exchange.run()
            return
        bucket = exchange.case.bucket
        exchange.received = buffer.worker_receive(
            bucket=bucket,
            read_host_counts=read_host_counts,
            out=reservations[bucket],
        )
        assert storage_pointers(exchange.received) == pointers[bucket]
        buffer.worker_reply(
            expert_rows(exchange.received, rank, config), exchange.received.receipt
        )

    checked_cases = cases(config.capacity_tokens_per_rank)
    if rank < FFN_RANKS:
        for bucket in sorted({case.bucket for case in checked_cases}):
            reservations[bucket] = buffer.reserve_worker_receive(bucket=bucket)
            pointers[bucket] = storage_pointers(reservations[bucket])
        reserved = reservations[1]
        malformed = [
            replace(reserved, recv_x=reserved.recv_x[:-1]),
            replace(reserved, recv_topk_weights=reserved.recv_topk_weights[:-1]),
            replace(reserved, recv_count_psum=reserved.recv_count_psum[:-1]),
            replace(reserved, receipt=reserved.receipt[:, :0]),
        ]
        if reserved.recv_scale is not None:
            malformed.append(
                replace(reserved, recv_scale=reserved.recv_scale.contiguous())
            )
        for bad in malformed:
            try:
                buffer.worker_receive(bucket=1, out=bad)
            except RuntimeError:
                pass
            else:
                raise AssertionError("malformed reusable receive storage was accepted")
        if config.capacity_tokens_per_rank > 1:
            try:
                buffer.worker_receive(
                    bucket=config.capacity_tokens_per_rank, out=reserved
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("reusable storage accepted a different bucket")
    dist.barrier()
    captures = []
    for case in checked_cases:
        exchange = make_exchange(buffer, config, rank, case)
        if rank < FFN_RANKS:
            # Dense garbage must not survive a receive with empty/skipped routes.
            reservations[case.bucket].recv_topk_weights.fill_(7)
            reservations[case.bucket].receipt.fill_(0)
        run(exchange, read_host_counts=True)
        torch.cuda.synchronize()
        if rank < FFN_RANKS:
            check_received(exchange, 0)
        else:
            exchange.check_output()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(exchange)
        torch.cuda.synchronize()
        dist.barrier()
        captures.append((graph, exchange))
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
    if rank < FFN_RANKS:
        for bucket, reservation in reservations.items():
            assert storage_pointers(reservation) == pointers[bucket]
    dist.barrier()
    return {
        "cases": [case.name for case in checked_cases],
        "graph_epochs": epochs,
        "stable_receive_pointers": True,
        "poisoned_weights_and_receipts_reset": True,
        "malformed_storage_rejected_before_exchange": True,
        "delayed_worker_seconds": DELAY_SECONDS,
        "per_epoch_cpu_barrier": False,
    }


def live_token_checks(buffer: Buffer, config: Config, rank: int, epochs: int) -> dict:
    """Replay one graph while GPU source counts change inside fixed storage."""
    capacity = config.capacity_tokens_per_rank
    full = Case("live_capacity", (capacity,) * ATTENTION_RANKS, strided_scales=True)
    exchange = make_exchange(buffer, config, rank, full)
    reservation = (
        buffer.reserve_worker_receive(bucket=capacity) if rank < FFN_RANKS else None
    )
    live_tokens = (
        torch.full((1,), capacity, dtype=torch.int32, device="cuda")
        if rank >= FFN_RANKS
        else None
    )

    def run() -> None:
        if rank < FFN_RANKS:
            received = buffer.worker_receive(bucket=capacity, out=reservation)
            buffer.worker_reply(expert_rows(received, rank, config), received.receipt)
        else:
            buffer.source_dispatch(
                exchange.inputs.x,
                exchange.inputs.ids,
                exchange.inputs.weights,
                x_scale=exchange.inputs.scale,
                live_tokens=live_tokens,
                bucket=capacity,
            )
            buffer.source_collect(exchange.output)

    if rank >= FFN_RANKS:
        malformed = (
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int64, device="cuda"),
            torch.zeros(0, dtype=torch.int32, device="cuda"),
            torch.zeros(2, dtype=torch.int32, device="cuda"),
        )
        for count in malformed:
            try:
                buffer.source_dispatch(
                    exchange.inputs.x,
                    exchange.inputs.ids,
                    exchange.inputs.weights,
                    x_scale=exchange.inputs.scale,
                    live_tokens=count,
                    bucket=capacity,
                )
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("malformed GPU live-token scalar was accepted")
    dist.barrier()
    run()  # Successful recovery proves invalid counts did not advance an epoch.
    torch.cuda.synchronize()
    exchange.check_output()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()
    dist.barrier()
    small = min(17, capacity)
    changed = (
        Case("max_unequal", (capacity, capacity // 2, 0, small)),
        Case("one_hot_worker", (1, 0, 1, 1), worker=1),
        Case("zero", (0,) * ATTENTION_RANKS),
        Case("small_rotating", (small, 1, 0, min(7, capacity)), rotate_worker=True),
        Case("max_allskip", (capacity,) * ATTENTION_RANKS, skip_all_experts=True),
        Case("max_hot_worker", (capacity,) * ATTENTION_RANKS, worker=2),
        Case("zero_after_dense", (0,) * ATTENTION_RANKS),
        Case("capacity_after_zero", (capacity,) * ATTENTION_RANKS),
        Case("zero_after_capacity", (0,) * ATTENTION_RANKS),
    )
    scalar_pointer = live_tokens.data_ptr() if live_tokens is not None else None
    for epoch in range(epochs):
        case = changed[epoch % len(changed)]
        sample = None
        if rank >= FFN_RANKS:
            sample = make_inputs(case, rank - FFN_RANKS, epoch + 1, config)
            count = sample.x.shape[0]
            # Tail routes are intentionally invalid, not masked with -1. Any
            # capacity-based source scan instead of the GPU count must fail.
            exchange.inputs.ids.fill_(-99)
            exchange.inputs.weights.fill_(float("nan"))
            exchange.inputs.x[:count].copy_(sample.x)
            exchange.inputs.ids[:count].copy_(sample.ids)
            exchange.inputs.weights[:count].copy_(sample.weights)
            if sample.scale is not None:
                exchange.inputs.scale.fill_(float("nan"))
                exchange.inputs.scale[:count].copy_(sample.scale)
            exchange.output.fill_(float("nan"))
            live_tokens.fill_(count)
            assert live_tokens.data_ptr() == scalar_pointer
        if rank == DELAYED_WORKER and epoch == DELAYED_EPOCH:
            time.sleep(DELAY_SECONDS)
        graph.replay()
        if sample is not None:
            torch.cuda.synchronize()
            count = sample.x.shape[0]
            live = Exchange(buffer, config, rank, case, sample, exchange.output[:count])
            live.check_output()
            assert bool(torch.isnan(exchange.output[count:]).all()), (
                "reply collection wrote beyond the GPU live-token count"
            )
    torch.cuda.synchronize()
    dist.barrier()
    return {
        "graph_epochs": epochs,
        "one_fixed_capacity_graph": True,
        "cases": [case.name for case in changed],
        "malformed_counts_rejected_before_exchange": True,
        "invalid_tail_routes_not_read": True,
        "output_tail_untouched": True,
        "stable_gpu_count_pointer": True,
        "per_epoch_cpu_barrier": False,
        "delayed_worker_seconds": DELAY_SECONDS,
    }


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
    config = Config(
        capacity_tokens_per_rank=args.capacity,
        hidden_size=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        payload=args.payload,
        allow_hybrid_mode=args.mode == "hybrid",
        rdma_traffic_class=args.rdma_traffic_class,
    )
    buffer = Buffer(group=dist.group.WORLD, config=config)
    eager = eager_checks(buffer, config, rank)
    exact = exact_receive_checks(buffer, config, rank)
    graph = graph_checks(buffer, config, rank, args.epochs)
    alignment_checks(buffer, config, rank)
    reusable = reusable_receive_checks(buffer, config, rank, args.epochs)
    live_tokens = live_token_checks(buffer, config, rank, args.epochs)
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
                    "exact_receive": exact,
                    "graph": graph,
                    "reusable_receive": reusable,
                    "live_tokens": live_tokens,
                    "misaligned_contiguous_source_and_output_rejected": True,
                    "ranks": results,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
