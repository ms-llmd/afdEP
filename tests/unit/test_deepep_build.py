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
    # Direct dispatch adds opt-in role checks. The remaining main transport and
    # credit kernels remain the pinned input, independently reviewable here.
    for name in ("hybrid_dispatch", "combine", "hybrid_combine", "barrier"):
        path = f"deep_ep/include/deep_ep/impls/{name}.cuh"
        assert expected[path] == provenance["file_sha256"][path]


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
