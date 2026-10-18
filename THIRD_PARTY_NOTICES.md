# Third-party notices

afdEP's own code is licensed under Apache-2.0. Its small NCCL host runtime
contains code adapted from DeepEP and remains licensed under MIT.

## DeepEP host runtime

- Upstream: [DeepEP](https://github.com/deepseek-ai/DeepEP).
- Copyright: Copyright (c) 2025 DeepSeek.
- License: MIT; the complete text is shipped in
  [`afdep/third_party/deepep/LICENSE`](afdep/third_party/deepep/LICENSE).
- Adapted files: `afdep/csrc/nccl_runtime.hpp` and
  `afdep/csrc/nccl_runtime.cpp`.
- Origin: the DeepEP V2 snapshot vendored by
  [hao-ai-lab/FastAFD](https://github.com/hao-ai-lab/FastAFD) at commit
  [`3c7161949310b6d59d6b4cf9bf997a4935c8113b`](https://github.com/hao-ai-lab/FastAFD/commit/3c7161949310b6d59d6b4cf9bf997a4935c8113b),
  then copied into afd-plugin. The original DeepEP commit is not
  recorded in that snapshot. Exact upstream and local input SHA-256 hashes,
  primary-source verification, and the extraction differences are recorded in
  [`PROVENANCE.json`](afdep/third_party/deepep/PROVENANCE.json).

The license and runtime API header match that FastAFD revision byte-for-byte.
The AFD plugin runtime input includes its marked patch allowing single-node
use without GIN; its hash is retained separately from the pristine FastAFD
hash. afdEP requires GIN for its supported two-domain topology (rail
connectivity in hybrid mode, full connectivity in direct mode) and does not
retain that optional-GIN path.

The extracted runtime implements NCCL communicator bootstrap, topology queries,
registered windows, and a GPU-initiated rail or full-connectivity
communicator. afdEP does not ship or invoke DeepEP's expert dispatch/combine
kernels, dummy expert layouts, or its generic JIT compiler. No DeepGEMM, CUTLASS, or fmt source is included.

## AFD plugin directional protocol

afdEP's Apache-2.0 directional kernels, layout, and endpoint originated in the
[AFD plugin](https://github.com/ms-llmd/afd-plugin) experiments on
base commit `7cc7ebd2283eff0388a7aa9495a9445a1cbd8349`. The directional code was
an additional measured patch, rather than part of that commit itself. Input
hashes, the measured snapshot fingerprint, and the host launcher extraction
are recorded in [`afdep/csrc/ORIGIN.json`](afdep/csrc/ORIGIN.json).

The original AFD plugin attribution is retained in the extracted files.
afdEP compiles those kernels directly into its native extension instead of
using DeepEP's per-kernel runtime compiler.

## External runtime dependencies

PyTorch, NVIDIA CUDA, and NVIDIA NCCL are external build/runtime dependencies.
Their libraries and headers are not copied into this source distribution and
remain subject to their respective licenses.
