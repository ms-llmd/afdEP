# SPDX-License-Identifier: Apache-2.0
"""Compare eager exact-compute allocations with pooled eager/graph FFN execution.

Run eight ranks using ``torchrun --module experiments.prefill.benchmark``.
This is an afdEP integration experiment, not a DeepEP benchmark: all variants
share the same conservative, reusable communication receive allocation. Only
eager_exact_compute reads host counts and allocates GEMM intermediates per call.
The graph is one receive/compute/reply microbatch, not an entire model/prompt.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as functional

import afdep
from examples.round_trip import ATTENTION_RANKS, FFN_RANKS, initialize
from experiments.prefill.ffn import PrefillFFN, deterministic_weights

VARIANTS = ("eager_exact_compute", "eager_pooled", "graph_pooled")
MICROSECONDS_PER_SECOND = 1_000_000
WEIGHT_SEED = 1729
REFERENCE_RTOL = 0.05
REFERENCE_ATOL = 0.003
DELAY_WORKER = 1
DELAY_EPOCH = 2
DELAY_GPU_CYCLES = 5_000_000


@dataclass
class SourceInputs:
    x: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    output: torch.Tensor

    def copy_(self, other: SourceInputs) -> None:
        self.x.copy_(other.x)
        self.ids.copy_(other.ids)
        self.weights.copy_(other.weights)


def make_inputs(
    config: afdep.Config,
    tokens: int,
    source: int,
    epoch: int,
    pattern: str = "uniform",
) -> SourceInputs:
    """Generate changing, distinct expert selections; disabled slots use -1."""
    generator = torch.Generator(device="cuda").manual_seed(4200 + source + epoch * 97)
    x = (
        torch.randn(
            (tokens, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.25
    )
    rows = torch.arange(tokens, device="cuda", dtype=torch.int64)[:, None]
    slots = torch.arange(config.top_k, device="cuda", dtype=torch.int64)[None, :]
    expert_limit = (
        config.experts_per_rank if pattern == "skewed" else config.num_experts
    )
    ids = (rows * config.top_k + slots + source + epoch * 7).remainder(expert_limit)
    ids = torch.where((rows + slots + epoch).remainder(11) == 0, -1, ids)
    if pattern == "skewed":
        ids = torch.where(slots < expert_limit, ids, -1)
    if pattern == "allskip":
        ids.fill_(-1)
    weights = (1.0 / (slots.float() + 2)).expand(tokens, config.top_k).contiguous()
    return SourceInputs(x.contiguous(), ids.contiguous(), weights, torch.empty_like(x))


def reference_output(
    inputs: SourceInputs,
    config: afdep.Config,
    intermediate: int,
    weight_seed: int,
) -> torch.Tensor:
    """Independent token-major PyTorch oracle, streaming one expert's weights."""
    result = torch.zeros_like(inputs.x, dtype=torch.float32)
    selected_experts = inputs.ids[inputs.ids >= 0].unique().cpu().tolist()
    for expert in selected_experts:
        weights = deterministic_weights(
            1,
            config.hidden_size,
            intermediate,
            seed=weight_seed,
            device="cuda",
            first_expert=expert,
        )
        tokens, slots = (inputs.ids == expert).nonzero(as_tuple=True)
        gate_up = functional.linear(inputs.x[tokens], weights.w13[0])
        gate, up = gate_up.float().chunk(2, dim=-1)
        activated = (functional.silu(gate) * up).bfloat16()
        output = functional.linear(activated, weights.w2[0])
        weighted = (output.float() * inputs.weights[tokens, slots, None]).bfloat16()
        result.index_add_(0, tokens, weighted.float())
        del weights, gate_up, gate, up, activated, output, weighted
    return result.bfloat16()


class Trial:
    """Own exactly one variant's workspace and optional captured graph."""

    def __init__(
        self,
        buffer: afdep.Buffer,
        config: afdep.Config,
        rank: int,
        variant: str,
        bucket: int,
        counts: tuple[int, ...],
        pattern: str,
        expert_weights,
        nvtx: bool = False,
        source_execution: str = "graph",
    ) -> None:
        self.buffer = buffer
        self.config = config
        self.rank = rank
        self.variant = variant
        self.bucket = bucket
        self.counts = counts
        self.pattern = pattern
        self.nvtx = nvtx
        self.source_execution = source_execution
        self.expected_rows_per_expert = max(
            1, ATTENTION_RANKS * bucket * config.top_k // config.num_experts
        )
        self.received = None
        self.executor = None
        self.inputs = None
        self.graph = None
        if rank < FFN_RANKS:
            self.received = buffer.reserve_worker_receive(bucket=bucket)
            self.executor = PrefillFFN(
                expert_weights,
                capacity_rows=self.received.recv_x.shape[0],
                expert_alignment=config.expert_alignment,
                expected_rows_per_expert=self.expected_rows_per_expert,
            )
        else:
            source = rank - FFN_RANKS
            self.inputs = make_inputs(config, counts[source], source, 0, pattern)

    def update(self, epoch: int) -> None:
        if self.inputs is not None:
            source = self.rank - FFN_RANKS
            self.inputs.copy_(
                make_inputs(
                    self.config, self.counts[source], source, epoch, self.pattern
                )
            )

    def run(self) -> None:
        if self.rank < FFN_RANKS:
            exact = self.variant == "eager_exact_compute"
            with self.stage("receive_countsync" if exact else "receive"):
                received = self.buffer.worker_receive(
                    bucket=self.bucket, out=self.received, read_host_counts=exact
                )
            with self.stage("expert_compute"):
                reply = self.executor(
                    received.recv_x,
                    received.recv_topk_weights,
                    received.recv_count_psum,
                    row_limit=sum(received.recv_counts_host) if exact else None,
                    allocate_exact=exact,
                )
            with self.stage("reply"):
                self.buffer.worker_reply(reply, received.receipt)
        else:
            with self.stage("source_exchange"):
                self.buffer.source_dispatch(
                    self.inputs.x,
                    self.inputs.ids,
                    self.inputs.weights,
                    bucket=self.bucket,
                )
                self.buffer.source_collect(self.inputs.output)

    def stage(self, name: str):
        label = f"prefill/{self.variant}/{name}/rank={self.rank}"
        return torch.cuda.nvtx.range(label) if self.nvtx else nullcontext()

    def pointers(self) -> tuple[int, ...]:
        if self.received is not None:
            tensors = (
                self.received.recv_x,
                self.received.recv_topk_weights,
                self.received.recv_count_psum,
                self.received.receipt,
                self.executor.reply,
            )
        else:
            tensors = (
                self.inputs.x,
                self.inputs.ids,
                self.inputs.weights,
                self.inputs.output,
            )
        return tuple(tensor.data_ptr() for tensor in tensors)

    def prepare(self) -> None:
        self.run()
        torch.cuda.synchronize()
        dist.barrier()
        capture = (
            self.variant == "graph_pooled"
            if self.rank < FFN_RANKS
            else self.source_execution == "graph"
        )
        if capture:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.run()
        # Source and FFN ranks may use different execution modes. Every rank
        # must enter the same collective sequence after optional capture.
        torch.cuda.synchronize()
        dist.barrier()

    def execute(self) -> None:
        if self.graph is None:
            self.run()
        else:
            self.graph.replay()

    def memory(self) -> dict:
        return {
            "torch_allocated_bytes": torch.cuda.memory_allocated(),
            "torch_reserved_bytes": torch.cuda.memory_reserved(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "receive_capacity_rows": (
                self.received.recv_x.shape[0] if self.received is not None else 0
            ),
            "pooled_compute_workspace_capacity_bytes": (
                self.executor.workspace_bytes
                if self.executor is not None and self.variant != "eager_exact_compute"
                else 0
            ),
            "persistent_reply_bytes": (
                self.executor.reply.numel() * self.executor.reply.element_size()
                if self.executor is not None
                else 0
            ),
        }

    def close(self) -> None:
        torch.cuda.synchronize()
        dist.barrier()
        self.graph = None
        self.received = None
        self.executor = None
        self.inputs = None
        gc.collect()
        torch.cuda.empty_cache()


def correctness(
    buffer: afdep.Buffer,
    config: afdep.Config,
    rank: int,
    args: argparse.Namespace,
    expert_weights,
) -> list[dict]:
    """Check genuine FFN results, empty sources, skew and changing graph routes."""
    count = min(max(args.tokens), 8)
    patterns = (
        ("uniform", (count,) * ATTENTION_RANKS),
        ("skewed", (count,) * ATTENTION_RANKS),
        ("allskip", (count,) * ATTENTION_RANKS),
        ("empty", (0, count, 0, count)),
        ("empty_all", (0,) * ATTENTION_RANKS),
    )
    checked = []
    for variant in args.variants:
        for pattern, counts in patterns:
            trial = Trial(
                buffer,
                config,
                rank,
                variant,
                count,
                counts,
                pattern,
                expert_weights,
                args.nvtx,
                args.source_execution,
            )
            trial.prepare()
            pointers = trial.pointers()
            for epoch in range(1, args.correctness_epochs + 1):
                trial.update(epoch)
                if rank == DELAY_WORKER and epoch == DELAY_EPOCH:
                    # Delay consumption on one worker without changing capture.
                    torch.cuda._sleep(DELAY_GPU_CYCLES)
                trial.execute()
                torch.cuda.synchronize()
                if trial.graph is not None:
                    assert trial.pointers() == pointers, (
                        "graph storage addresses changed"
                    )
                if trial.inputs is not None:
                    expected = reference_output(
                        trial.inputs, config, args.intermediate, args.weight_seed
                    )
                    torch.testing.assert_close(
                        trial.inputs.output,
                        expected,
                        rtol=REFERENCE_RTOL,
                        atol=REFERENCE_ATOL,
                    )
                dist.barrier()
            checked.append(
                {
                    "variant": variant,
                    "case": pattern,
                    "epochs": args.correctness_epochs,
                    "changed_routes_and_payload": True,
                    "graph_storage_addresses_stable_when_captured": True,
                    "ffn_graph": variant == "graph_pooled",
                    "source_execution": args.source_execution,
                    "delayed_worker_gpu_cycles": (
                        DELAY_GPU_CYCLES
                        if args.correctness_epochs >= DELAY_EPOCH
                        else 0
                    ),
                    "reference": "token_major_torch_bf16_swiglu_weighted",
                }
            )
            trial.close()
    return checked


def measure(
    buffer: afdep.Buffer,
    config: afdep.Config,
    rank: int,
    args: argparse.Namespace,
    expert_weights,
    tokens: int,
    variant: str,
    repeat: int,
) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    before = {
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
    }
    torch.cuda.reset_peak_memory_stats()
    trial = Trial(
        buffer,
        config,
        rank,
        variant,
        tokens,
        (tokens,) * ATTENTION_RANKS,
        "uniform",
        expert_weights,
        args.nvtx,
        args.source_execution,
    )
    trial.prepare()
    # Exercise every planned count/shape once before sampling. Exact allocation
    # can change Triton launch specializations as routes cross alignment bounds.
    for iteration in range(args.warmup + args.iterations):
        trial.update(100 + repeat * 1000 + iteration)
        trial.execute()
        torch.cuda.synchronize()
    dist.barrier()
    after_prepare = trial.memory()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for iteration in range(args.warmup + args.iterations):
        # Paired variants see identical inputs; generation and the Gloo barrier
        # are outside the timed interval. Exact mode's count wait stays inside.
        trial.update(100 + repeat * 1000 + iteration)
        torch.cuda.synchronize()
        dist.barrier()
        label = (
            f"prefill/{args.mode}/{variant}/tokens={tokens}"
            f"/repeat={repeat}/iteration={iteration}/rank={rank}"
        )
        context = torch.cuda.nvtx.range(label) if args.nvtx else nullcontext()
        with context:
            started = time.perf_counter()
            trial.execute()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * MICROSECONDS_PER_SECOND
        if iteration >= args.warmup:
            samples.append(elapsed)
    after_timing = trial.memory()
    if args.check_timed_output and trial.inputs is not None:
        expected = reference_output(
            trial.inputs, config, args.intermediate, args.weight_seed
        )
        torch.testing.assert_close(
            trial.inputs.output, expected, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
        )
    local = {
        "rank": rank,
        "samples_us": samples,
        "before_variant": before,
        "after_prepare": after_prepare,
        "after_timing": after_timing,
        "retained_allocated_delta_bytes": (
            after_timing["torch_allocated_bytes"] - before["torch_allocated_bytes"]
        ),
        "steady_peak_allocated_delta_bytes": (
            after_timing["torch_peak_allocated_bytes"] - before["torch_allocated_bytes"]
        ),
        # Native GIN windows are not necessarily tracked by PyTorch. Keep this
        # separate instead of summing potentially overlapping memory measures.
        "registered_transport_bytes": buffer.num_buffer_bytes,
        "final_numerical_output_checked": (
            args.check_timed_output and rank >= FFN_RANKS
        ),
    }
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, local)
    maximum_source_samples = [
        max(row["samples_us"][i] for row in ranks[FFN_RANKS:])
        for i in range(args.iterations)
    ]
    result = {
        "variant": variant,
        "repeat": repeat,
        "tokens_per_source": tokens,
        "expected_rows_per_expert_hint": trial.expected_rows_per_expert,
        "median_iteration_max_attention_rank_wall_us": statistics.median(
            maximum_source_samples
        ),
        "iteration_max_attention_rank_wall_us": maximum_source_samples,
        "ranks": ranks,
    }
    trial.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("hybrid", "direct"), default="hybrid")
    parser.add_argument(
        "--source-execution", choices=("graph", "eager"), default="graph"
    )
    parser.add_argument("--tokens", nargs="+", type=int, default=(16, 128, 512))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--alignment", type=int, default=128)
    parser.add_argument("--weight-seed", type=int, default=WEIGHT_SEED)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--correctness-epochs", type=int, default=3)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--check-timed-output", action="store_true")
    parser.add_argument("--order", choices=("rotated", "serialized"), default="rotated")
    parser.add_argument("--rdma-traffic-class", type=int, default=0)
    parser.add_argument("--nvtx", action="store_true")
    args = parser.parse_args()
    if (
        min(args.tokens) <= 0
        or args.intermediate <= 0
        or args.repeats <= 0
        or args.iterations <= 0
        or args.warmup < 0
        or args.correctness_epochs <= 0
    ):
        parser.error("sizes/repeats/iterations must be positive; warmup may be zero")
    if args.hidden <= 0 or args.hidden % 128 or args.intermediate % 128:
        parser.error("hidden and intermediate sizes must be positive multiples of 128")
    return args


def main() -> None:
    args = parse_args()
    rank = initialize()
    config = afdep.Config(
        capacity_tokens_per_rank=max(args.tokens),
        hidden_size=args.hidden,
        num_experts=args.experts,
        top_k=args.top_k,
        expert_alignment=args.alignment,
        payload="bf16",
        allow_hybrid_mode=args.mode == "hybrid",
        rdma_traffic_class=args.rdma_traffic_class,
    )
    buffer = afdep.Buffer(group=dist.group.WORLD, config=config)
    expert_weights = (
        deterministic_weights(
            config.experts_per_rank,
            config.hidden_size,
            args.intermediate,
            seed=args.weight_seed,
            device="cuda",
            first_expert=rank * config.experts_per_rank,
        )
        if rank < FFN_RANKS
        else None
    )
    checks = (
        []
        if args.skip_correctness
        else correctness(buffer, config, rank, args, expert_weights)
    )
    results = []
    for tokens in args.tokens:
        for repeat in range(args.repeats):
            shift = repeat % len(args.variants) if args.order == "rotated" else 0
            variants = list(args.variants[shift:]) + list(args.variants[:shift])
            for variant in variants:
                results.append(
                    measure(
                        buffer,
                        config,
                        rank,
                        args,
                        expert_weights,
                        tokens,
                        variant,
                        repeat,
                    )
                )
    buffer.close()
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "experiment": "bounded_ffn_workspace_not_deepep_comparison",
                    "config": asdict(config),
                    "intermediate_size": args.intermediate,
                    "weight_seed": args.weight_seed,
                    "expert_compute": "deepgemm_bf16_grouped_swiglu",
                    "precision": "bf16_activations_weights_and_replies",
                    "exact_baseline_scope": (
                        "host_count_sync_exact_compute_intermediates_"
                        "conservative_receive_storage"
                    ),
                    "graph_scope": "one_ffn_layer_microbatch_receive_compute_reply",
                    "memory_note": (
                        "torch_allocator_and_native_transport_"
                        "reported_separately_not_added; persistent_reply_is_"
                        "a_subset_of_pooled_capacity; source_peaks_include_"
                        "untimed_input_generation_use_ffn_peaks_for_workspace"
                    ),
                    "cold_jit_excluded": (
                        "every_planned_input_epoch_executed_once_before_sampling"
                    ),
                    "order": args.order,
                    "source_execution": args.source_execution,
                    "profiled": args.nvtx,
                    "timed_outputs_checked": args.check_timed_output,
                    "iterations": args.iterations,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "correctness": checks,
                    "cases": results,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
