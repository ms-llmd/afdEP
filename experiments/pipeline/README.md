# Coordinated microbatch overlap

One BF16 MoE layer with varying token counts. This measures communication plus
DeepGEMM SwiGLU, excluding Attention computation and full-model scheduling.

| Variant | Execution |
|---|---|
| `whole_graph` | Current one-slot maximum-capacity receive → GEMMs → reply graph |
| `split_serial` | Split the same original inputs; complete each chunk in order |
| `pipeline` | Same chunks and allocations; two slots overlap receive, GEMMs, reply |

```text
whole_graph:  [ receive all ][ GEMMs all ][ reply all ]
pipeline:     [ receive A ][ receive B ][ receive C ]
                           [ GEMMs A   ][ GEMMs B   ][ GEMMs C ]
                                        [ reply A ][ reply B ][ reply C ]
```

`split_serial` controls the cost of smaller GEMMs and extra graph launches.
Source tokens keep their original routes, weights, and output order. Each slot
has its own transport epochs, credits, receive pool, and compute arena. Slots
share expert weights; GEMMs run on one ordered compute stream. This first
experiment pipelines chunks within each original request; it does not pipeline
separate original requests.

```bash
torchrun <two-host options> --module experiments.pipeline.benchmark \
  --mode hybrid --capacity 2048 --microbatches 1 2 4 8 \
  --num-sms 24 --requests 90 --repeats 3
```

All graphs are captured and primed before timing. Runtime staging, graph replay,
output assembly, and completion are timed. Inputs are resident beforehand.
`--request-window 1` waits for each original request's completion on Attention;
larger windows admit more host submissions. `--fixed-tokens 512` replaces the
mixed-size trace with uniformly routed, fixed-size requests (per Attention rank).
There are no per-request Gloo barriers. CUDA response intervals start at first staging and end after all
chunk outputs are assembled. Host-observed completion with a larger window is
an upper bound; neither metric is full serving latency.

Correctness covers empty, skipped, skewed and changing routes, uneven sources,
delayed peers, and slot reuse against an independent PyTorch oracle. The final
timed request is also checked. Registered transport memory and PyTorch allocator
memory are reported separately. Use `--nvtx` for separate profiling runs.
