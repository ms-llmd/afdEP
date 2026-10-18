# afdEP

**Expert-parallel communication for Attention–FFN Disaggregation (AFD).**

afdEP connects Attention token sources to FFN expert workers: dispatch routed
tokens, receive expert inputs, and return combined results. Routing, quantization,
expert compute, and serving scheduling stay with the caller.

![DeepEP shared EP roles versus afdEP separate Attention and FFN roles](docs/assets/communication.svg)

Two backends use NCCL and NVLink; neither requires a DeepEP or vLLM package:

| API | Communication kernels |
|---|---|
| `DeepEPBuffer` | Vendored DeepEP V2 dispatch/combine; explicit AFD roles over the Masked DeepEP protocol |
| `Buffer` | Experimental directional request/reply kernels |

The DeepEP backend preserves its optimized kernels and graph execution.
It retains masking internally; extraction alone does not remove that overhead.
[Backend contracts](docs/api.md#deepep-backend).
The [vLLM serving comparison](https://github.com/ms-llmd/afd-plugin/blob/832e61e4a4acc24fa55e3fa08abdaae780b8b6f6/recipe/gpu/AfdEPConnector/qwen3_30b_fp8/README.md)
measures this backend separately on four H100s.

## Support

| Architecture | Backend | Validated hardware and modes |
|---|---|---|
| Hopper (SM90) | Directional | Eight H100 80GB GPUs; hybrid/direct; BF16/FP8; eager and CUDA graphs |
| Hopper (SM90) | DeepEP V2 | Four H100 80GB GPUs in one NVLink domain; BF16/FP8; eager and CUDA graphs |
| Blackwell | Directional: not implemented; DeepEP V2: unvalidated | — |

The directional backend implements **four Attention + four FFN ranks**, each role in
its own four-GPU NVLink domain, with one in-flight lane. Other N/M layouts and
concurrent lanes require further implementation and validation.

## Quick start

In a provisioned PyTorch/CUDA environment, from the checkout:

```bash
python -m pip install --no-build-isolation --no-deps -e .
python -m afdep doctor --build
```

After constructing a buffer on each rank, the roles use separate operations:

```python
# Attention ranks:
buffer.source_dispatch(x, expert_ids, weights, bucket=bucket)
buffer.source_collect(out)

# FFN ranks:
received = buffer.worker_receive(bucket=bucket)
buffer.worker_reply(weighted_expert_outputs, received.receipt)
```

Apply routing weights **once** before `worker_reply`.
[Installation and API](docs/api.md) · [Runnable example](examples/round_trip.py)

## H100 experiment

Directional prototype: eight H100s, four Attention + four FFN ranks,
CUDA graphs, communication only.

![H100 communication latency and registered buffer comparison](docs/assets/h100-performance.svg)

- At 512 tokens/source, BF16 and FP8 round-trip latencies were about **30% lower**.
- At 2048, BF16 was approximately tied; FP8 latency was **12% lower**.
- Registered transport buffers were **68–72% smaller**.

October 15: directional prototype vs **Masked DeepEP**,
with different tuned execution budgets. The standalone wheel passed separate
eight-H100 checks on October 18. [H100 experiment details](docs/performance.md).

## Development

[Contributing](CONTRIBUTING.md) · [CI](https://github.com/ms-llmd/afdEP/actions/workflows/ci.yml)

Optional: [FFN prefill workspace experiment](experiments/prefill/README.md).

Apache-2.0; upstream-derived components retain their [original notices](THIRD_PARTY_NOTICES.md).
