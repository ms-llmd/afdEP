# Installation and API

afdEP separates Attention token sources from expert workers. This reference
covers the current NCCL GIN/NVLink backend; see [support](../README.md#support)
for hardware and topology status.

## Install

GPU execution needs Linux, CUDA 13 PyTorch and a compatible `nvcc`, NCCL ≥2.30.7
with GIN device headers/RDMA, plus `setuptools` and `ninja`. The native extension
compiles on first use. Package import and configuration tests need no GPU.

```bash
# Preserve an existing CUDA/PyTorch environment:
python -m pip install --no-build-isolation --no-deps -e .
python -m afdep doctor
python -m afdep doctor --build  # Compile/load on the GPU host.
```

If the environment's NCCL is older, the tested isolated CUDA 13 setup is:

```bash
python -m pip install --no-deps --target "$HOME/.cache/afdep/dependencies" \
  nvidia-nccl-cu13==2.30.7
export AFDEP_NCCL_ROOT="$HOME/.cache/afdep/dependencies/nvidia/nccl"
export LD_LIBRARY_PATH="$AFDEP_NCCL_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m afdep doctor --build
```

Set library paths **before Python starts**, including every `torchrun` process.
Loaded NCCL and compile-time headers must match; changing NCCL rebuilds kernels.
`CUDA_HOME` or `CUDA_NVCC_EXECUTABLE` selects the compiler; `AFDEP_NCCL_ROOT`
selects NCCL; `AFDEP_CACHE_DIR` overrides the extension cache.

## Validate and run

From the checkout on each four-GPU host, set `NODE_RANK=0` for FFN and `1` for
Attention, and configure rendezvous and RDMA/NIC affinity:

```bash
torchrun --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  --module tests.gpu.correctness --payload bf16
```

Run all four combinations of `--payload bf16|fp8` and `--mode hybrid|direct`
(defaults `bf16` and `hybrid`). Use `--module examples.round_trip` for a weighted
expert example, or `--module benchmarks.round_trip --graph --iterations 100`
for communication timing. Add `--nvtx` only in a separate profiling run.

## API

Select the CUDA device and initialize an eight-rank CPU Gloo group first.
Every rank constructs the same configuration and closes buffers in the same order.

```python
import afdep

config = afdep.Config(
    capacity_tokens_per_rank=8192,
    hidden_size=2048,
    num_experts=128,
    top_k=8,
    payload="bf16",
    allow_hybrid_mode=True,  # False selects direct delivery.
)
buffer = afdep.Buffer(group=group, config=config)
# Attention role:
buffer.source_dispatch(x, ids, weights, x_scale=None, bucket=bucket)
buffer.source_collect(out)
# FFN role, instead of the two source calls:
r = buffer.worker_receive(bucket=bucket, read_host_counts=False)
buffer.worker_reply(weighted_expert_rows, r.receipt)
# Collectively, after the last completed exchange:
buffer.close()
```

The current backend orders FFN ranks 0–3 before Attention ranks 4–7. Each FFN
owns a contiguous quarter of the real experts. All ranks participate even when empty.

| Configuration | Current contract/default |
|---|---|
| Capacity / bucket | Positive; `0 ≤ actual tokens ≤ bucket ≤ capacity`; shared bucket |
| Hidden size / experts / top-k | H positive multiple of 128; E positive divisible by four; `1 ≤ K ≤ min(32,E)` |
| Roles / lanes | Four per role; one in-flight lane |
| `expert_alignment`, `payload` | 128; `"bf16"` or `"fp8"` |
| `num_gin_contexts`, `transport_stripes` | Eight each; positive stripes ≤ contexts |
| `num_sms` | 132 local CTAs; must not exceed device SM count; tune for compute overlap |
| `gpu_timeout_s`, `rdma_traffic_class` | 60 seconds; zero (the H100 experiment used traffic class 3) |

Gloo lifecycle timeouts are configured on the process group, independently.

| Source tensor | Shape / dtype |
|---|---|
| `x` | Contiguous CUDA `[T,H]`, BF16 or `float8_e4m3fn` |
| `ids`, `weights` | Contiguous `[T,K]`, INT64 and FP32 |
| `x_scale` | FP8 only: FP32 `[T,H/128]`; strides accepted |
| `out` | Contiguous BF16 `[T,H]`, including for FP8 dispatch |

All tensors use the buffer's CUDA device. `x`/`out` need 16-byte-aligned pointers;
contiguous views with offset storage can fail this check. IDs are real experts
`[0,E)` or `-1` for a disabled slot; nonnegative IDs must be unique per token.
FP8 is caller-prequantized block-128 data with FP32 dequantization scales;
afdEP transports it without choosing a quantizer.

## Communication modes

`allow_hybrid_mode` follows DeepEP's flag of the same name. It selects one
protocol and never falls back to the other.

| `allow_hybrid_mode` | Dispatch / reply organization |
|---|---|
| `True` (default) | Rail-aligned GIN to an ingress worker; NVLink distribution and reduction on the FFN side |
| `False` | Full-connectivity GIN directly to selected workers; each worker returns its partial, and the source reduces replies |

Mode is fixed at construction and must agree on every rank. Direct mode requires
full GIN connectivity; a rail-only fabric can support hybrid but reject direct.
Both modes currently require the same two pure-role four-GPU NVLink domains.

Direct mode reserves a request and a reply slot per peer. At equal capacity its
registered windows total about 1.7× (FP8) to 2× (BF16) hybrid's, and each
Attention rank's window is 4× larger. `Buffer.num_buffer_bytes` reports the
registered size of each rank.

## Receive rows and replies

`ReceiveResult` provides `recv_x`, optional `recv_scale`, `recv_topk_weights`,
`recv_count_psum`, optional `recv_counts_host`, and an opaque `receipt`.
Rows are expert-major; weights/scales are FP32, device ends/receipt INT32.
FP8 receive scales use padded column-major strides.
For local expert e and alignment A:

```text
start[e] = 0 if e == 0 else round_up(recv_count_psum[e-1], A)
end[e] = recv_count_psum[e]
allocated_rows = 4 * bucket * min(K, E/4) + (E/4) * (A-1)
```

Only `[start[e], end[e])` contains valid activations; gaps/tails are unspecified.
Padding weights are zero. Host counts, when requested, are aligned per-expert
counts; reading them synchronizes and is forbidden during graph capture.
Supply contiguous BF16 `[allocated_rows,H]` expert outputs, **already multiplied
by the received routing weights exactly once**. afdEP sums rows without applying
weights again, using the matching unchanged receipt. Padding outputs are ignored.
Each FFN worker rounds its local FP32 sum to a BF16 partial. Partials are then
summed in FP32 and rounded to the BF16 reply, on the FFN side in hybrid mode or
in `source_collect` in direct mode. This need not match one-rounding FP32
summation bit for bit.

## Ordering and lifetime

Use one ordered current PyTorch CUDA stream per buffer. Keep inputs, receive
rows, receipts, and expert outputs alive until their enqueued users finish;
manage events/lifetimes explicitly for other streams. No background stream or
implicit cross-stream dependency is supplied.
Keep the buffer's CUDA device current for every operation, including `close()`.

Complete dispatch→collect or receive→reply before the next exchange. Device
epochs and credits protect request, partial, and reply storage until all readers
finish; readiness is awaited on every active GIN context, including empty stripes.

Initialize/warm up outside CUDA capture. Capture a complete role-specific
exchange, use device counts, and retain fixed addresses/shapes/bucket. Values and
routes may change in place; different buckets use separate graphs replayed in
the same logical order on every rank. Concurrent use of one buffer is unsupported.

`close()` is collective, outside capture, after final GPU work; it drains peers
before releasing windows. Do not use collective close as rank-local exception
cleanup. A timeout/device error is fatal to the job: terminate and reconstruct
its participants rather than reuse partially advanced communication state.
