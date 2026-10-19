# Directional v2 · H100 experiment

**Independent communication tests:** four H100s, one NVLink domain,
2 Attention + 2 FFN, 128 experts, top-k 8. Each width has 90 fresh processes:
BF16/FP8 × three route patterns × five implementations × three balanced repeats.

| Hidden width | vs Masked24 | vs Masked132 | vs v1 reuse24 | vs v1 reuse132 |
|---|---:|---:|---:|---:|
| 2,048 | −18.5% | −14.2% | −19.4% | −15.4% |
| 4,096 | −32.1% | −24.9% | −32.6% | −25.3% |

Latency changes are equal-cell geometric means; negative is faster.
The 24/132 settings control DeepEP transport SMs; its local epilogues use the
full GPU. V2 uses a 132-block base grid. These are distinct launch policies.

![H2048 latency by payload, route and token count](assets/directional-primary.png)

![H4096 latency by payload, route and token count](assets/directional-robustness.png)

H2048 has two near-ties against Masked132, including a 0.2% median slowdown.
H4096 has four consistently slower cells against Masked132: BF16 random/8,192
**+0.3%**, BF16 strided/8,192 **+2.5%**, FP8 strided/2,048 **+7.2%**, and
FP8 strided/8,192 **+1.9%**. The FP8 strided regression is being profiled.
Three process pairs support descriptive observed ranges; they do not establish
universal non-degradation or serving gains.

Registered capacity falls **63–69%** across these configurations. At H4096,
capacity 8,192/source, it totals **386 vs 1,040 MiB** in BF16 and
**324 vs 1,032 MiB** in FP8 across four ranks. This excludes expert/receive
tensors, caller allocations and total VRAM.

## Where time goes

![Nsight communication phases](assets/directional-critical-path.png)

At 8,192 BF16 tokens, request/grouping is about 63 µs slower, return 64 µs faster
and source completion 7 µs slower in Nsight phase medians. Increasing the source
packing grid from 132 to 528 blocks cut its kernel from **90 to 58 µs**; device
instructions are unchanged. Worker packing starts at about the same time as
Masked DeepEP but still takes about 64 µs longer. Intervals include waits and
overlap, so they do not isolate network time or count bookkeeping.

Initialization, JIT and capture are excluded; **graph replay is included**.
Timing uses a graph per token extent, not one maximum-capacity graph. Expert
GEMMs are outside the timed path. Nsight runs are separate from latency runs.
This source passed 36 focused NVLink jobs and eight two-node GIN configurations,
covering BF16/FP8, eager/graphs, changing counts/routes, numerical edges and
delayed/poisoned reuse. Its parent kernel base passed 75 jobs, including three
expected invalid-input rejections. Exact source hashes are recorded below.

[All process/block measurements, source hashes and scope](validation-directional-2026-10-19.json)
· [Earlier H2048 record](validation-nvlink-2026-10-19.json)
· [Harness](../benchmarks/nvlink.py).

Use the [GPU environment](../README.md#use) and `EP_DISABLE_GIN=1`. Run each
payload/backend combination in a fresh four-rank process three times:

```bash
torchrun --standalone --nproc-per-node=4 -m benchmarks.nvlink \
  --backend directional --payload bf16 --hidden 4096 --capacity 8192 \
  --tokens 1 8 32 128 512 2048 8192 --routing random --routing-seed 59101 \
  --sms 132 --masked-sms 132 --warmup 1000 --iterations 1000 --repeats 5 \
  --output result.json
```

Use `--backend masked` or `--backend reuse` with `--masked-sms 24` or `132` for
the controls; repeat with FP8 and strided/skew routing. H2048 uses capacity
2,048, seed 49,001 and omits the 8,192 bucket. Two-node GIN performance and full
serving comparisons are in progress.
