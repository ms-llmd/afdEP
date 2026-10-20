# SPDX-License-Identifier: Apache-2.0
"""Focused 2A/2F NVLink readiness experiment; no expert GEMMs.

Run ``torchrun --standalone --nproc-per-node=4 -m tests.gpu.readiness
--stage timing --output results/timing``. Repeat with ``--stage diagnostic``
for injected-delay device-clock evidence. Run tests.gpu.nvlink separately,
with and without --early-metadata, for changing inputs and buffer reuse.

Timing uses preallocated zero expert outputs with unchanged routing/reduction.
CUDA graph replay launch cost is included; construction/JIT/capture are excluded.
The diagnostic is deliberately separate from the uninstrumented timing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from afdep import DirectionalBuffer as Buffer
from afdep import DirectionalConfig as Config
from afdep._directional_build import source_digest
from tests.gpu.nvlink import decode

ARMS = ("baseline", "early_gated", "early")
TRACE_FIELDS = (
    "prepare_done_ns",
    "source0_observed_ns",
    "source0_pack_start_ns",
    "source0_pack_end_ns",
    "source1_observed_ns",
    "source1_pack_start_ns",
    "source1_pack_end_ns",
    "source0_completed_chunks",
    "source0_chunks_before_peer_ready",
    "source1_completed_chunks",
    "source1_chunks_before_peer_ready",
    "epoch",
)
SEED = 20261020


def gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def sample(config, source, tokens, *, force_both_workers=False):
    generator = torch.Generator().manual_seed(SEED + source)
    ids = torch.rand(tokens, config.num_experts, generator=generator).argsort(dim=1)
    if force_both_workers:
        # Keep random, distinct experts while guaranteeing a selected token on
        # each worker even for a one-token diagnosis.
        first = torch.rand(
            tokens, config.experts_per_rank, generator=generator
        ).argsort(dim=1)[:, : config.top_k // 2]
        second = first + config.experts_per_rank
        ids = torch.cat((first, second), dim=1)
    else:
        ids = ids[:, : config.top_k]
    ids = ids.contiguous().to("cuda")
    row = torch.arange(tokens, device="cuda")[:, None]
    column = torch.arange(config.hidden_size, device="cuda")[None, :]
    x = ((row * 13 + column * 7 + source * 11) % 31 - 15).to(torch.bfloat16)
    weights = torch.full(
        (tokens, config.top_k), 1 / config.top_k, device="cuda", dtype=torch.float32
    )
    scale = None
    if config.payload == "fp8":
        # Integers are exact in E4M3FN here; different block scales exercise the
        # scale transport without introducing a tolerance-dependent oracle.
        scale = torch.pow(
            2.0,
            torch.arange(config.hidden_size // 128, device="cuda").remainder(3),
        ).float()[None, :].expand(tokens, -1).contiguous()
        x = x.to(torch.float8_e4m3fn)
    return x, scale, ids, weights


@dataclass
class Exchange:
    buffer: Buffer
    tokens: int
    force_both_workers: bool = False

    def __post_init__(self):
        config = self.buffer.config
        self.worker = self.buffer.rank < config.ffn_ranks
        self.received = None
        self.rows = None
        self.output = None
        self.inputs = None
        if self.worker:
            self.received = self.buffer.reserve_worker_receive(bucket=self.tokens)
            self.rows = torch.zeros(
                self.received.recv_x.shape, dtype=torch.bfloat16, device="cuda"
            )
        else:
            self.inputs = sample(
                config,
                self.buffer.role_rank,
                self.tokens,
                force_both_workers=self.force_both_workers,
            )
            self.output = torch.empty(
                (self.tokens, config.hidden_size), dtype=torch.bfloat16, device="cuda"
            )

    def step(self):
        if self.worker:
            self.buffer.worker_receive(bucket=self.tokens, out=self.received)
            self.buffer.worker_reply(self.rows, self.received.receipt)
        else:
            x, scale, ids, weights = self.inputs
            self.buffer.source_dispatch(x, ids, weights, scale, bucket=self.tokens)
            self.buffer.source_collect(self.output)

    def check(self):
        """Check all routed input rows and the zero-output communication echo."""
        if not self.worker:
            torch.testing.assert_close(
                self.output, torch.zeros_like(self.output), rtol=0, atol=0
            )
            return
        config = self.buffer.config
        received = self.received
        actual = decode(received.recv_x, received.recv_scale)
        seen = torch.zeros(actual.shape[0], dtype=torch.bool, device="cuda")
        expected_counts = torch.zeros(
            config.experts_per_rank, dtype=torch.int64, device="cuda"
        )
        first = self.buffer.rank * config.experts_per_rank
        for source in range(config.attention_ranks):
            x, scale, ids, weights = sample(
                config, source, self.tokens, force_both_workers=self.force_both_workers
            )
            owned = (ids >= first) & (ids < first + config.experts_per_rank)
            receipt = received.receipt[source]
            torch.testing.assert_close(receipt >= 0, owned, rtol=0, atol=0)
            token, slot = owned.nonzero(as_tuple=True)
            rows = receipt[owned].long()
            assert rows.numel() == rows.unique().numel()
            assert not bool(seen[rows].any())
            seen[rows] = True
            torch.testing.assert_close(
                actual[rows], decode(x, scale)[token], rtol=0, atol=0
            )
            torch.testing.assert_close(
                received.recv_topk_weights[rows], weights[token, slot], rtol=0, atol=0
            )
            expected_counts += torch.bincount(
                ids[owned] - first, minlength=config.experts_per_rank
            )
        ends = received.recv_count_psum.long()
        previous = torch.cat((torch.zeros_like(ends[:1]), ends[:-1]))
        starts = (
            (previous + config.expert_alignment - 1)
            // config.expert_alignment
            * config.expert_alignment
        )
        torch.testing.assert_close(ends - starts, expected_counts, rtol=0, atol=0)
        assert bool((received.recv_topk_weights[~seen] == 0).all())

    def capture(self):
        self.step()
        torch.cuda.synchronize()
        self.check()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.step()
        dist.barrier()
        return graph


def make_buffer(payload, early, hidden, capacity):
    return Buffer(
        group=dist.group.WORLD,
        config=Config(
            capacity,
            hidden,
            128,
            8,
            ffn_ranks=2,
            attention_ranks=2,
            payload=payload,
            transport="nvlink",
            nvlink_early_metadata=early,
        ),
    )


def write_result(path, result):
    if dist.get_rank() == 0:
        path.write_text(json.dumps(result, indent=2) + "\n")


def timing(args, result, output):
    for payload in args.payloads:
        buffers = {
            arm: make_buffer(payload, arm != "baseline", args.hidden, max(args.tokens))
            for arm in ARMS
        }
        buffers["early_gated"]._runtime._set_readiness_debug(force_payload_barrier=True)
        result["configs"][payload] = {
            arm: asdict(buffer.config) for arm, buffer in buffers.items()
        }
        for tokens in args.tokens:
            exchanges = {
                arm: Exchange(buffer, tokens) for arm, buffer in buffers.items()
            }
            graphs = {arm: exchange.capture() for arm, exchange in exchanges.items()}
            for repeat in range(args.repeats):
                order = ARMS[repeat % 3 :] + ARMS[: repeat % 3]
                for position, arm in enumerate(order):
                    graph = graphs[arm]
                    for _ in range(args.warmup):
                        graph.replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(args.iterations):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    local_us = begin.elapsed_time(end) * 1000 / args.iterations
                    durations = gather(local_us)
                    exchanges[arm].check()
                    dist.barrier()
                    record = {
                        "payload": payload,
                        "tokens": tokens,
                        "repeat": repeat,
                        "position": position,
                        "arm": arm,
                        "rank_microseconds": durations,
                        "critical_rank_microseconds": max(durations),
                    }
                    result["records"].append(record)
                    write_result(output, result)
                    if dist.get_rank() == 0:
                        print(json.dumps(record), flush=True)
        for buffer in buffers.values():
            buffer.close()
    summaries = []
    for payload in args.payloads:
        for tokens in args.tokens:
            records = [
                row
                for row in result["records"]
                if row["payload"] == payload and row["tokens"] == tokens
            ]
            by_key = {
                (row["repeat"], row["arm"]): row["critical_rank_microseconds"]
                for row in records
            }
            for arm in ARMS[1:]:
                ratios = [
                    by_key[repeat, arm] / by_key[repeat, "baseline"]
                    for repeat in range(args.repeats)
                ]
                summaries.append(
                    {
                        "payload": payload,
                        "tokens": tokens,
                        "arm": arm,
                        "paired_ratios_to_baseline": ratios,
                        "geomean_latency_change_percent": 100
                        * (math.exp(sum(map(math.log, ratios)) / len(ratios)) - 1),
                    }
                )
    result["descriptive_summary"] = summaries
    result["inference"] = "Three paired in-process repeats; no significance claim."


def diagnostic(args, result, output):
    for payload in args.payloads:
        for delayed_source in (0, 1):
            for arm in ARMS:
                buffer = make_buffer(
                    payload, arm != "baseline", args.hidden, args.diagnostic_tokens
                )
                trace = torch.zeros(len(TRACE_FIELDS), dtype=torch.int64, device="cuda")
                delay = (
                    args.delay_cycles
                    if buffer.role == "attention" and buffer.role_rank == delayed_source
                    else 0
                )
                buffer._runtime._set_readiness_debug(
                    delay_cycles=delay,
                    trace=trace if buffer.role == "ffn" else None,
                    force_payload_barrier=arm == "early_gated",
                )
                exchange = Exchange(buffer, args.diagnostic_tokens, True)
                graph = exchange.capture()
                trace.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                graph.replay()
                torch.cuda.synchronize()
                exchange.check()
                traces = gather(trace.cpu().tolist() if buffer.role == "ffn" else None)
                observations = []
                for worker, values in enumerate(traces[:2]):
                    named = dict(zip(TRACE_FIELDS, values, strict=True))
                    fast = 1 - delayed_source
                    completed = named[f"source{fast}_completed_chunks"]
                    before = named[f"source{fast}_chunks_before_peer_ready"]
                    for source in (0, 1):
                        assert (
                            named[f"source{source}_completed_chunks"]
                            == args.diagnostic_tokens
                        ), (arm, worker, named)
                    overlap = completed == before
                    if arm == "early":
                        assert overlap, (arm, worker, named)
                    else:
                        assert before == 0, (arm, worker, named)
                    observations.append(
                        {
                            "worker": worker,
                            "clock_scope": "single worker GPU globaltimer",
                            "clock_semantics": (
                                "Observed/start timestamps coincide after payload "
                                "acquire; they do not measure separate wait stages."
                            ),
                            "trace": named,
                            "all_fast_chunks_before_delayed_payload_ready": overlap,
                            "overlap_evidence": (
                                "Every completed fast-source CTA observed the "
                                "other source's request_ready below this epoch."
                            ),
                        }
                    )
                result["records"].append(
                    {
                        "payload": payload,
                        "arm": arm,
                        "delayed_source": delayed_source,
                        "delay_cycles": args.delay_cycles,
                        "tokens": args.diagnostic_tokens,
                        "workers": observations,
                        "correct": True,
                    }
                )
                write_result(output, result)
                buffer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("timing", "diagnostic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--payloads", nargs="+", choices=("bf16", "fp8"), default=["bf16", "fp8"]
    )
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--diagnostic-tokens", type=int, default=32)
    parser.add_argument("--delay-cycles", type=int, default=5_000_000)
    args = parser.parse_args()
    if not 1 <= min(args.tokens) <= max(args.tokens) <= 128:
        parser.error("tokens must be in [1,128]")
    if not 1 <= args.diagnostic_tokens <= 128:
        parser.error("diagnostic-tokens must be in [1,128]")
    if min(args.repeats, args.warmup, args.iterations, args.delay_cycles) <= 0:
        parser.error("repeats, warmup, iterations, and delay-cycles must be positive")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    if dist.get_world_size() != 4:
        raise ValueError("This experiment requires four GPUs: two FFN, two Attention.")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "complete": False,
        "stage": args.stage,
        "source_digest": source_digest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": gather(str(torch.cuda.get_device_properties(rank))),
        "seed": SEED,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "arms": {
            "baseline": "early metadata disabled in the same native build",
            "early_gated": (
                "early metadata enabled; preparation still waits for every payload"
            ),
            "early": "early metadata enabled; each gather waits on its own source",
        },
        "timing_scope": (
            "Communication only, preallocated zero expert outputs, graph replay "
            "launches included, constructor/JIT/capture excluded."
        ),
        "configs": {},
        "records": [],
    }
    output = args.output / f"{args.stage}.json"
    write_result(output, result)
    (timing if args.stage == "timing" else diagnostic)(args, result, output)
    result["complete"] = True
    write_result(output, result)
    if rank == 0:
        print(json.dumps({"complete": True, "output": str(output)}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
