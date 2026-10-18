# H100 experiment: directional EP versus Masked DeepEP

On eight H100 80GB GPUs, the tuned directional prototype had lower small/medium
communication latency than Masked DeepEP. At 2048 tokens/source, BF16 was near
parity and FP8 latency was about 12% lower. Registered transport capacity fell 68–72%.
These are communication-only measurements of the protocol that is now the default
hybrid mode; direct mode was not compared.

![H100 prototype latency and memory comparison](assets/h100-performance.svg)

## October 15 prototype comparison

CUDA-graph round trips in milliseconds; four Attention sources and four FFN workers:

| Payload | Tokens/source | Masked DeepEP | Directional, tuned |
|---|---:|---:|---:|
| BF16 | 1 | 0.635 | 0.432 |
| BF16 | 128 | 0.797 | 0.562 |
| BF16 | 512 | 1.606 | 1.119 |
| BF16 | 2048 | 3.791 | 3.698 |
| FP8 | 1 | 0.678 | 0.454 |
| FP8 | 128 | 0.773 | 0.562 |
| FP8 | 512 | 1.495 | 1.043 |
| FP8 | 2048 | 3.582 | 3.145 |

The **Masked DeepEP** baseline uses DeepEP's general EP dispatch/combine with
unused roles masked out. Its useful payload is already directional. afdEP's
protocol makes the source and expert-worker roles explicit.
This comparison predates extraction into the standalone afdEP package.
Both paths used hierarchical network/NVLink communication. These results do
not compare direct mode or establish the best DeepEP mode for each batch size.

| Setup | Value |
|---|---|
| GPUs / placement | Eight H100 80GB; two pure-role four-GPU NVLink domains |
| Workload | H=2048, E=128, top-k=8, alignment=128, capacity=8192/source |
| Dispatch / replies | BF16 or FP8 E4M3 block-128 + FP32 scales; BF16 replies |
| Transport | Eight GIN contexts; directional sends use eight stripes; one RoCE rail/lane |
| Software | PyTorch 2.13.0+cu130, CUDA 13.0, NCCL 2.30.7 |

The metric is the median across three runs of each run's median iteration
maximum across Attention ranks: five warmups and 100 measured exchanges/bucket.
Host-wall time includes launches, GPU completion, and readiness skew. Timed
expert outputs are persistent BF16 ones; quantization/model GEMMs are excluded.

The tuned directional local grids use 132 CTAs; GIN sends use eight. Earlier
24-CTA grids were about 2.2× slower at the largest bucket. Increasing local
parallelism reduced BF16 8.257→3.698 ms and FP8 8.217→3.145 ms without changing
the protocol. The baseline's local epilogues already used 132 CTAs.

These later tuning runs use earlier baseline references, not an equal-resource
randomized protocol-only ablation. A final large-bucket baseline check drifted
less than 1%; tiny-bucket drift was larger. Gains cannot be attributed precisely
to removed bookkeeping, and this does not establish aggregate NIC saturation.

## Memory

Separate sums across eight ranks, MiB; allocator peaks at 2048 tokens/source:

| Payload | Registered windows, baseline→directional | Graph allocator peaks | Eager allocator peaks |
|---|---:|---:|---:|
| BF16 | 3201.6→1030.0 | 9656→2419 | 1818→2419 |
| FP8 | 3201.6→906.0 | 7627→1892 | 1681→1892 |

Registered capacity fell 67.8%/71.7%; graph allocator peaks fell about 75%.
Eager allocator peaks instead increased 33.0%/12.5% with conservative receive
allocation. Neither metric is total VRAM; peaks need not coincide, and graph
pools/padded rows/retained caches affect them. Do not add the columns.

## Nsight findings

Nsight Systems 2026.5.1 covered all eight ranks and BF16/FP8 graph replays.
No DeepEP dispatch/combine kernels ran in the directional path. At 2048 tokens,
per-worker median stages changed with the local grid:

| Stage | BF16, 24→132 CTAs (ms) | FP8, 24→132 CTAs (ms) |
|---|---:|---:|
| Expert-row packing | 1.33–1.36→0.30–0.38 | 1.86–1.91→0.37–0.40 |
| Local partial reduction | 3.67–3.69→0.68 | 3.66–3.69→0.68 |
| Reply reduction | 0.72–0.84→0.14–0.16 | 0.73–0.84→0.14–0.16 |

GIN reply stages remained about 1.1 ms. Durations include waits/completion;
parallel ranks and overlapping stages must not be summed as elapsed time.
Baseline fused dispatch/combine durations do not isolate bookkeeping costs.
One tuned FP8 trace required graph-launch correlation for GPU-internal timing;
unaligned CPU/GPU offsets and cross-node timestamps were not interpreted.

Next targets are vectorized/reused local reductions that need fewer SMs,
activation reuse during expert expansion, FP8 scale packing, chunked pipelining
with independent credits, and smaller eager receive allocations. Full-device
local grids can compete with GEMMs; serving/compute overlap remains unmeasured.

## October 18 standalone H100 validation

The installed afdEP wheel compiled on both H100 hosts without AFD/vLLM/DeepEP
imports. Eight-rank BF16 and FP8 eager/graph correctness passed: eight cases run
twice, 18 changing-input graph replays, empty/skewed/all-skipped routes, full
source load, strided FP8 scales, delayed-peer reuse, and alignment rejection.
Exact artifacts/configuration are frozen in [the validation record](validation-2026-10-18.json).

The package checks used capacity 128 for correctness and 2048 for short
communication benchmarks (five warmups/20 samples). They passed BF16/FP8 in
eager and graph modes. **No DeepEP baseline was rerun**; these checks establish
package correctness, not an additional relative speedup.
Current hardware/topology support is listed in the [README](../README.md#support).

### Hybrid/direct extension

Both modes passed eight-H100 BF16/FP8 eager and graph correctness: 12 cases,
26 changing-input graph replays per combination, selective worker destinations,
empty/skewed routes, and delayed-peer reuse. All eight communication smoke runs
also passed (two modes × two payloads × eager/graph; five warmups/20 samples).
At capacity 2048/source, registered windows summed across eight ranks were
257.5/226.5 MiB for hybrid BF16/FP8 and 518.5/394.5 MiB for direct. Direct
reserves per-peer request/reply slots, so the cost falls mostly on Attention
ranks: per rank, BF16 hybrid uses 16.2 MiB per source and 48.2 MiB per worker,
while direct uses 64.8 MiB on every rank. At the experiment's 8192 capacity,
direct would total 2074.0/1578.0 MiB, 35%/51% below the Masked DeepEP windows
rather than 68–72%.

The benchmark routes each token to consecutive expert IDs, reaching about 1.2
workers per token. Direct sends one activation per selected worker, so these
timings sit near its best case; uniform top-8-of-128 routing reaches about 3.6.

Hybrid timings in this run were 0.7–1.4 ms higher than the October 18 package
record at 1–512 tokens/source, with the same capacity, warmups, and samples.
The cause has not been isolated, so the two records should not be compared, and
a same-host rerun of the previous commit in hybrid mode is still needed to rule
out a regression. No mode speedup or new DeepEP comparison is claimed.
[Mode validation record](validation-modes-2026-10-18.json).

The suite has since gained two cases that fill each direct-mode request slot and
route every token to all four workers; they have not yet run on GPUs.
