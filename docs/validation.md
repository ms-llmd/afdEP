# V1 validation

**2026-10-19 · 4 × H100 80 GB (Hopper SM90) · one NVLink domain · 2 Attention + 2 FFN.**
Python 3.12, PyTorch 2.13.0+cu130, CUDA 13.0, NCCL 2.30.7; GIN disabled for this
intra-domain validation.

| Check | Result |
|---|---|
| Unit tests / Ruff / distributions | 27 tests pass; formatting and source/license checks pass |
| Specialized GPU matrix | BF16/FP8 × hybrid on/off × capacities 32/2,048: 8 runs pass |
| Unspecialized control | BF16 and FP8: 2 runs pass |
| Nonuniform, strided FP8 scales | Large capacity, both hybrid settings: 2 runs pass |
| Installed wheel | Large FP8 eager and graph suite passes after a fresh native build |
| Invalid routes | Dummy, duplicate, negative and wide IDs assert in eager and graphs: 8 checks pass |

Every valid run checks mixed/changing/skewed routes, inactive sources, skipped
routes, empty messages, and partial prefixes. It covers host-count eager receives,
queued host-counter reuse, delayed FFN replies and changing CUDA graph replays.
Small configurations check 26 exchanges per captured graph; large ones check 4.
Large configurations use hidden size 2,048 and top-k 8, exceeding local expert count.

[Exact runtime fingerprints, matrix and log hashes](validation-v1.json).
These are correctness checks with synthetic expert outputs, **not performance or
serving measurements**. Other architectures, layouts and multi-domain role
specialization remain unvalidated.
