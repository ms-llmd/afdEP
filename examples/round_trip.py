# SPDX-License-Identifier: Apache-2.0
"""Standalone weighted expert echo over AFDEP, without AFD or vLLM.

For the validated two-host, four-GPU-per-host configuration::

    torchrun --nnodes=2 --nproc-per-node=4 --node-rank=$NODE_RANK \
      --master-addr=$MASTER_ADDR --master-port=29500 \
      --module examples.round_trip --payload bf16 --mode hybrid

Node 0 owns experts, node 1 originates tokens. The worker operation is an
elementwise expert-dependent transform, not a model GEMM. Top-k weights must
be applied by the caller before ``worker_reply``; AFDEP sums those rows.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

import afdep

FFN_RANKS = 4
ATTENTION_RANKS = 4
WORLD_SIZE = FFN_RANKS + ATTENTION_RANKS
HIDDEN_SIZE = 2048
NUM_EXPERTS = 128
TOP_K = 8
FP8_BLOCK_SIZE = 128
SIGNATURE_BITS = 32
SOURCE_SHIFT = 16
EPOCH_SHIFT = 18
ROUTE_STRIDE = 7
WEIGHT_DENOMINATOR = 64
MICROSECONDS_PER_SECOND = 1_000_000


@dataclass(frozen=True)
class Case:
    name: str
    counts: tuple[int, ...]
    worker: int | None = None
    rotate_worker: bool = False
    skip_all_experts: bool = False
    strided_scales: bool = False
    spread_workers: bool = False

    @property
    def bucket(self) -> int:
        # Empty messages still need a nonzero allocation/capture bucket.
        return max(1, *self.counts)


@dataclass
class Inputs:
    x: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    scale: torch.Tensor | None

    def copy_(self, other: Inputs) -> None:
        self.x.copy_(other.x)
        self.ids.copy_(other.ids)
        self.weights.copy_(other.weights)
        if self.scale is not None:
            self.scale.copy_(other.scale)

    def decoded(self) -> torch.Tensor:
        if self.scale is None:
            return self.x.float()
        return (
            self.x.float().reshape(
                self.x.shape[0], self.x.shape[1] // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE
            )
            * self.scale[:, :, None]
        ).reshape(self.x.shape)


@dataclass
class Exchange:
    buffer: afdep.Buffer
    config: afdep.Config
    rank: int
    case: Case
    inputs: Inputs | None
    output: torch.Tensor | None
    received: afdep.ReceiveResult | None = None

    def run(self, *, read_host_counts: bool = False) -> None:
        if self.rank < FFN_RANKS:
            self.received = self.buffer.worker_receive(
                bucket=self.case.bucket, read_host_counts=read_host_counts
            )
            rows = expert_rows(self.received, self.rank, self.config)
            self.buffer.worker_reply(rows, self.received.receipt)
        else:
            self.buffer.source_dispatch(
                self.inputs.x,
                self.inputs.ids,
                self.inputs.weights,
                x_scale=self.inputs.scale,
                bucket=self.case.bucket,
            )
            self.buffer.source_collect(self.output)

    def update(self, epoch: int) -> None:
        if self.inputs is not None:
            self.inputs.copy_(
                make_inputs(
                    self.case,
                    self.rank - FFN_RANKS,
                    epoch,
                    self.config,
                )
            )

    def check_output(self) -> None:
        if self.inputs is None:
            return
        # Independent token-major reference. All chosen values are powers of
        # two, and the bounded sums are exactly representable in BF16.
        owner = self.inputs.ids.div(self.config.experts_per_rank, rounding_mode="floor")
        factors = torch.pow(2.0, (self.inputs.ids.remainder(2) + owner).float())
        factors = torch.where(self.inputs.ids >= 0, factors, 0)
        expected = self.inputs.decoded() * (self.inputs.weights * factors).sum(
            dim=1, keepdim=True
        )
        torch.testing.assert_close(self.output.float(), expected, rtol=0, atol=0)


def quantize_block128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference-only power-of-two block quantization for this example.

    This is intentionally ordinary PyTorch, outside timed transport regions.
    AFDEP accepts prequantized E4M3FN and FP32 dequantization scales; it does
    not choose a quantization policy. Powers of two keep this oracle exact.
    """
    blocks = x.float().reshape(x.shape[0], x.shape[1] // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE)
    maximum = blocks.abs().amax(dim=-1)
    scale = torch.pow(2.0, torch.ceil(torch.log2(maximum.clamp_min(1))))
    quantized = (blocks / scale[:, :, None]).to(torch.float8_e4m3fn)
    return quantized.reshape(x.shape).contiguous(), scale.contiguous()


def make_inputs(case: Case, source: int, epoch: int, config: afdep.Config) -> Inputs:
    count = case.counts[source]
    row = torch.arange(count, device="cuda", dtype=torch.int64)
    signature = (row + 1) | (source << SOURCE_SHIFT) | (epoch << EPOCH_SHIFT)
    column = torch.arange(config.hidden_size, device="cuda", dtype=torch.int64)
    bits = (signature[:, None] >> column.remainder(SIGNATURE_BITS)[None, :]) & 1
    # Different block scales ensure the FP8 scale payload is also exercised.
    magnitude = torch.pow(2.0, (column // FP8_BLOCK_SIZE).remainder(3).float())
    x = (bits * magnitude).bfloat16()
    slot = torch.arange(config.top_k, device="cuda", dtype=torch.int64)
    if case.spread_workers:
        # Interleave slots across workers so every token reaches all of them,
        # filling each direct-mode request slot and the full source reduction.
        slots_per_worker = config.top_k // FFN_RANKS
        local = (
            row[:, None] * slots_per_worker
            + slot[None, :] // FFN_RANKS
            + source
            + epoch * ROUTE_STRIDE
        ).remainder(config.experts_per_rank)
        ids = slot[None, :].remainder(FFN_RANKS) * config.experts_per_rank + local
    else:
        worker = (
            (source + epoch // FFN_RANKS) % FFN_RANKS
            if case.rotate_worker
            else case.worker
        )
        expert_limit = (
            config.experts_per_rank if worker is not None else config.num_experts
        )
        ids = (
            row[:, None] * config.top_k + slot[None, :] + source + epoch * ROUTE_STRIDE
        ).remainder(expert_limit)
        if worker is not None:
            ids += worker * config.experts_per_rank
    # Skipped choices test -1 semantics and force epoch-dependent zero counts.
    omitted = (row[:, None] + slot[None, :] + epoch).remainder(11) == 0
    ids = torch.where(omitted, -1, ids).contiguous()
    if case.skip_all_experts:
        ids.fill_(-1)
    weight = torch.pow(2.0, slot.remainder(3).float()) / WEIGHT_DENOMINATOR
    weights = weight[None, :].expand(count, config.top_k).contiguous()
    if config.payload == "fp8":
        quantized, scale = quantize_block128(x)
        if case.strided_scales:
            strided = torch.empty_strided(
                scale.shape,
                (1, max(4, count)),
                dtype=scale.dtype,
                device=scale.device,
            )
            strided.copy_(scale)
            scale = strided
        return Inputs(quantized, ids, weights, scale)
    return Inputs(x, ids, weights, None)


def expert_rows(
    received: afdep.ReceiveResult, rank: int, config: afdep.Config
) -> torch.Tensor:
    """Graph-safe transform using only device-side expert row boundaries."""
    if received.recv_scale is None:
        decoded = received.recv_x.float()
    else:
        decoded = (
            received.recv_x.float().reshape(
                -1, config.hidden_size // FP8_BLOCK_SIZE, FP8_BLOCK_SIZE
            )
            * received.recv_scale[:, :, None]
        ).reshape(received.recv_x.shape)
    row = torch.arange(received.recv_x.shape[0], device="cuda", dtype=torch.int32)
    local_expert = torch.bucketize(row, received.recv_count_psum, right=True)
    factor = (1 << rank) * torch.pow(2.0, local_expert.remainder(2).float())
    # AFDEP does not multiply routing weights in its reply reduction.
    weighted = decoded * factor[:, None] * received.recv_topk_weights[:, None]
    # Receive padding contains unspecified activation bytes. Selecting zero
    # explicitly also removes NaNs from unused rows (NaN * 0 remains NaN).
    return torch.where(received.recv_topk_weights[:, None] != 0, weighted, 0).bfloat16()


def make_exchange(
    buffer: afdep.Buffer, config: afdep.Config, rank: int, case: Case
) -> Exchange:
    inputs = (
        make_inputs(case, rank - FFN_RANKS, 0, config) if rank >= FFN_RANKS else None
    )
    output = (
        torch.empty(
            (inputs.x.shape[0], config.hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        if inputs is not None
        else None
    )
    return Exchange(buffer, config, rank, case, inputs, output)


def initialize(timeout_s: int = 180) -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=timeout_s))
    rank = dist.get_rank()
    if dist.get_world_size() != WORLD_SIZE:
        raise ValueError(
            "Run two hosts with four processes each; node 0 FFN, node 1 Attention"
        )
    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("The current CUDA backend requires SM90")
    return rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--mode", choices=("hybrid", "direct"), default="hybrid")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--rdma-traffic-class", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = initialize()
    config = afdep.Config(
        capacity_tokens_per_rank=max(1, args.tokens),
        hidden_size=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        payload=args.payload,
        allow_hybrid_mode=args.mode == "hybrid",
        rdma_traffic_class=args.rdma_traffic_class,
    )
    buffer = afdep.Buffer(group=dist.group.WORLD, config=config)
    exchange = make_exchange(
        buffer, config, rank, Case("example", (args.tokens,) * ATTENTION_RANKS)
    )
    exchange.run(read_host_counts=True)
    torch.cuda.synchronize()
    exchange.check_output()
    buffer.close()
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": args.mode,
                    "payload": args.payload,
                    "tokens_per_source": args.tokens,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
