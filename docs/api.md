# API contract

`Buffer(group=..., config=Config(...))` owns one NCCL communicator and one
vendored DeepEP V2 ElasticBuffer. Initialization and `close()` are collective,
outside CUDA capture. The caller owns a CPU **Gloo** process group, sets the
current CUDA device first, and keeps the group alive until close.

## Roles and tensors

FFN ranks come first; each owns `num_experts / ffn_ranks` consecutive real experts.
Attention ranks follow. Every rank uses identical configuration, exchange order,
`bucket`, and `read_host_counts` policy. Only one unmatched dispatch per Buffer
is allowed. IDs must be unique per token, in `[0, num_experts)`, or `-1` to skip.
Use `buffer.topk_idx_dtype`; invalid real-role routes assert on the GPU, including
on graph replay. All payload tensors are on the selected CUDA device.

| Operation | Contract |
|---|---|
| `source_dispatch(x, ids, weights, x_scale=None, *, bucket, read_host_counts=False)` | Attention: contiguous `x [T,H]`, IDs/FP32 weights `[T,K]`, `T ≤ bucket ≤ capacity` |
| `worker_receive(*, bucket, read_host_counts=False)` | FFN: expert-major rows, weights, device expert prefix sums and opaque receipt |
| `worker_reply(rows, receipt)` | FFN: contiguous BF16 `[receive_rows,H]`, routing weight applied exactly once |
| `source_collect()` | Attention: BF16 `[T,H]` summed over selected experts, in original token order |

BF16 dispatch uses BF16 activations. FP8 uses `float8_e4m3fn` activations and
FP32 block-128 scales `[T,H/128]`; strided scale views are supported. The library
does not quantize or run GEMMs. `H` is a multiple of 256, `K ≤ min(32,E)`, and
real experts divide evenly over FFN ranks.

Receive prefix sum `ends[e]` is a device tensor: expert e occupies
`[align_up(ends[e-1], expert_alignment), ends[e])`, with `ends[-1]=0` for e=0.
Padding rows are unspecified and must not be treated as live tokens. Return the
entire receive shape to `worker_reply`, with valid expert rows in place.
`recv_counts_host`, when requested, reports **aligned** per-expert counts, not
raw live counts. Keep the receipt unchanged through the matching reply.

## Graphs, streams and lifetime

Capture **complete** dispatch/reply pairs with `read_host_counts=False` after
warming up the kernels on all ranks. Keep captured tensors, Buffer and graph
storage alive through every replay. Source tensor shapes and bucket specialize
the capture. Change payloads and IDs in place; encode inactive choices as `-1`.
This is capacity-bounded graph execution, not arbitrary-shape graph replay.
Graph capture/instantiation and replay are different operations; replay still
has a launch cost.

`read_host_counts=True` performs host synchronization for exact eager receive
allocation on FFN. Otherwise receive capacity is conservative and device counts
identify live rows. Attention has zero receive payload rows. FFN bounds count
Attention sources only. Registered transport buffers remain conservatively
sized by DeepEP and are distinct from receive tensors.

Operations order their internal communication stream with the current CUDA
stream using DeepEP events. Use caller stream dependencies before passing data
produced on another stream. Outputs/receipts and user inputs must remain alive
until dependent GPU work completes. A completed Python pair allows the next
pair to be enqueued; device-side protocol and stream ordering protect reuse.
Independent Buffers do not share ordering. `close()` synchronizes outstanding
work and frees registered storage; destroy graphs before closing the Buffer.

## Transport and tuning

V1 preserves DeepEP's uniform logical expert slots and collective participation.
Attention dummy slots are never legal routing targets. `role_aware=True` is the
public default. `role_aware=False` is an unspecialized control using the same
backend; `afdep.deepep.ElasticBuffer` retains the low-level connector interface.
Low-level callers must explicitly pass `role_aware=True, ffn_ranks=M` to enable
the specializations; the high-level `Buffer` does this from `Config`.

`allow_hybrid_mode`, `num_sms`, `num_qps`, timeouts and RDMA traffic class pass
through to the pinned transport. `num_sms=0` selects approximately 24/132 of the
smallest device's SM count; choose/tune explicitly for a workload. Both hybrid
settings are checked on one physical NVLink domain. Role specialization rejects
multiple physical domains; no inter-node v1 claim is made.

Native compilation is lazy and process-locked. `AFDEP_CACHE_DIR` controls the
extension cache; `EP_JIT_CACHE_DIR` controls DeepEP kernel JIT cache. Both must be
writable. Use matching NCCL headers/runtime before starting Python; already
loaded libraries cannot be replaced by changing environment variables later.
