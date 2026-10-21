# SPDX-License-Identifier: Apache-2.0
"""Check vendored transport integrity and lazy, content-addressed build inputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

from afdep._build import SOURCE_NAMES, VENDOR_ROOT, source_digest


def test_pinned_origin_and_declared_role_specialization():
    provenance = json.loads((VENDOR_ROOT / "PROVENANCE.json").read_text())
    assert provenance["source_commit"] == "1198c52f8bed0c20a2b51ff550db5ccf2fda0906"
    assert provenance["local_changes"]
    changes = json.loads((VENDOR_ROOT / "LOCAL_CHANGES.json").read_text())
    expected = provenance["file_sha256"].copy()
    for name, entry in changes["modified_files"].items():
        assert entry["upstream_sha256"] == expected[name]
        expected[name] = entry["sha256"]
    expected.update(
        {name: entry["sha256"] for name, entry in changes["added_files"].items()}
    )
    actual = {
        path.relative_to(VENDOR_ROOT).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in VENDOR_ROOT.rglob("*")
        if path.is_file() and path.name not in {"PROVENANCE.json", "LOCAL_CHANGES.json"}
    }
    assert actual == expected
    assert all(name in expected for name in SOURCE_NAMES)
    assert "MIT License" in (VENDOR_ROOT / "LICENSE").read_text()
    # Direct and hybrid dispatch/combine add opt-in role specializations whose
    # defaults keep upstream kernels (see the template-default test). The
    # credit/barrier kernel remains the pinned input, independently reviewable here.
    path = "deep_ep/include/deep_ep/impls/barrier.cuh"
    assert expected[path] == provenance["file_sha256"][path]


def test_role_template_parameters_default_to_upstream_kernels():
    """Generic (Masked) instantiations omit role arguments and keep upstream code."""
    impls = VENDOR_ROOT / "deep_ep/include/deep_ep/impls"
    combine = (impls / "combine.cuh").read_text()
    assert "bool kAfdSkipEntryBarrier = false," in combine
    assert "int kAfdDirectedFfnRanks = 0," in combine
    assert "bool kAfdPerCtaExit = false," in combine
    assert "bool kAfdSingleFfnReduce = false," in combine
    dispatch = (impls / "dispatch.cuh").read_text()
    assert "int kAfdFfnRanks = 0," in dispatch
    assert "bool kAfdSkipEntryBarrier = false," in dispatch
    assert "bool kAfdDirectionalNotify = false," in dispatch
    assert "bool kAfdDirectedExit = false," in dispatch
    assert "bool kAfdRoleBuffers = false," in dispatch
    assert "bool kAfdPerCtaExit = false," in dispatch
    assert "int kAfdZeroCollectRowBytes = 0," in dispatch
    epilogue = (impls / "dispatch_copy_epilogue.cuh").read_text()
    assert "int kAfdRegionOffset = 0," in epilogue
    for hybrid in ("hybrid_dispatch.cuh", "hybrid_combine.cuh"):
        assert "int kAfdFfnScaleoutRanks = 0," in (impls / hybrid).read_text()
    hybrid_dispatch = (impls / "hybrid_dispatch.cuh").read_text()
    assert "bool kAfdDirectionalRoutes = false," in hybrid_dispatch
    assert "bool kAfdSkipAttentionEntrySync = false," in hybrid_dispatch


# Device sources of v2.1 (native digest 8205d02b). The role suites size launches of
# these kernels and add separate role kernels; they never edit a shared kernel, so
# Masked and every shared instantiation keep their code, include hash and cubin.
V21_KERNEL_SHA256 = {
    "dispatch.cuh": (
        "81e189fab1985fba3f8c772ecfe0b2efbfe4618744c0ae0195566ae20326ce6a"
    ),
    "combine.cuh": "8847dfb9b41cc0cb0c971ac0c35ca25d95cacc2443413ca3a6d6b7dcb43a51e1",
    "hybrid_dispatch.cuh": (
        "56ebe2ee6854380fa575881bf53e4a8d9392f7e4bf3ad1510c57b65854d1c313"
    ),
    "hybrid_combine.cuh": (
        "a6fd0f17475bf1b1783a9671cbe09218e982ae904ad86ff2e6aa12efe2c9ee8a"
    ),
    "dispatch_copy_epilogue.cuh": (
        "0b573abeb0dc2bb8d4c65d9a3ce28f82537e129d3dc4cf5e3709dd4e9cbf85cf"
    ),
    "afdep_roles.cuh": (
        "4f7ed0d81d7215753c99dfe1364e4ba6e707101cb5ede0d1b139e8b1923e937a"
    ),
}


def test_role_suites_leave_shared_kernels_untouched():
    provenance = json.loads((VENDOR_ROOT / "PROVENANCE.json").read_text())
    include = VENDOR_ROOT / "deep_ep/include/deep_ep"

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for name, sha in V21_KERNEL_SHA256.items():
        assert digest(include / "impls" / name) == sha, name
    upstream = [
        include / "impls/combine_reduce_epilogue.cuh",
        *sorted((include / "common").glob("*")),
    ]
    for path in upstream:
        name = path.relative_to(VENDOR_ROOT).as_posix()
        assert digest(path) == provenance["file_sha256"][name], name


def test_role_suite_shapes_are_host_launch_overrides():
    """The Attention hybrid send block relaunches the shared cubin, generated first."""
    elastic = VENDOR_ROOT / "csrc/kernels/elastic"
    suites = (elastic / "afdep_suites.hpp").read_text()
    for control in ("ROLE_KERNELS", "HYBRID_SEND", "HYBRID_FORWARD", "HYBRID_COLLECT"):
        assert f'"AFDEP_SHARED_{control}", 0) == 0' in suites
    dispatch = (elastic / "dispatch.hpp").read_text()
    generate = dispatch.index("const auto code = DispatchRuntime::generate(args);")
    assert generate < dispatch.index("shaped_args.launch_args = jit::LaunchArgs(")


def test_ffn_hybrid_dispatch_is_the_generated_copy():
    from tests.role_kernels import FFN_HYBRID_DISPATCH, IMPLS, ffn_hybrid_dispatch

    shared = (IMPLS / "hybrid_dispatch.cuh").read_text()
    assert (IMPLS / FFN_HYBRID_DISPATCH).read_text() == ffn_hybrid_dispatch(shared)


def test_runtime_header_contents_invalidate_build(tmp_path):
    header = tmp_path / "include/deep_ep/impls/dispatch.cuh"
    header.parent.mkdir(parents=True)
    header.write_text("first")
    os.utime(header, ns=(1_000_000_000, 1_000_000_000))
    first = source_digest(tmp_path)
    header.write_text("other")
    os.utime(header, ns=(1_000_000_000, 1_000_000_000))
    assert source_digest(tmp_path) != first


def test_loader_inspection_does_not_import_runtime_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import afdep; import afdep._build; "
            "api = (afdep.Config, afdep.Buffer); "
            "unexpected = {'torch', 'vllm', 'deep_gemm', 'afd_plugin', 'deep_ep'}; "
            "assert not unexpected & sys.modules.keys()",
        ],
        check=True,
    )
