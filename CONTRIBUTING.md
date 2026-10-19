# Contributing

afdEP is an EP communication library for AFD. Keep routing policy, quantization,
model weights/GEMMs, and serving scheduling outside the core. Hardware/topology
support belongs to backends; update [support](README.md#support) with evidence.

```bash
python -m pytest tests/unit
python -m ruff check .
python -m ruff format --check .
```

Native changes need a build and GPU validation on the affected backend, including
BF16/FP8, eager/graphs, empty/skewed routes, changing epochs, delayed readers, and
buffer reuse. Preserve the [API/ordering contracts](docs/api.md); identify every
writer/reader and its completion handoff when changing synchronization.

Retain upstream licenses and provenance; mark adapted-code differences. Check
wheel/source contents. Label hardware-specific and communication-only timings,
keep profiling separate, and never publish credentials or private cluster data.

On four GPUs in one NVLink domain (set matching NCCL paths first):

```bash
EP_DISABLE_GIN=1 torchrun --standalone --nproc-per-node=4 \
    -m tests.gpu.deepep --payload fp8 --hybrid off --capacity 32
```

Repeat with BF16, both hybrid settings, and larger capacity/hidden/top-k.
Destructive route guards run in separate processes: `--invalid-route` accepts
`dummy`, `duplicate`, `negative`, or `wide`; require the route assertion in the
log. A timeout or unrelated nonzero exit is a failure, not a successful guard test.
