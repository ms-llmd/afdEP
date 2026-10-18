# SPDX-License-Identifier: Apache-2.0
"""Validate source provenance and offline package inspection contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from afdep._build import PACKAGE_ROOT, SOURCE_NAMES, source_digest

# Hashes of the initial extraction; later changes update only current hashes.
EXTRACTED_FILE_SHA256 = {
    "afdep/csrc/layout.hpp": (
        "7141d10aabfb8f659f06933b87549a211e98d1d2b3ede9ecd1cc45e9867c1888"
    ),
    "afdep/csrc/kernels.cuh": (
        "05b50095f694231db3012821aed1fdf48e86b7a06d9c81eaeb8aee3e4db08bc0"
    ),
}


def test_standalone_sources_present():
    for name in SOURCE_NAMES:
        assert (PACKAGE_ROOT / "csrc" / name).is_file()
    for path in (PACKAGE_ROOT / "csrc").iterdir():
        if path.suffix in {".cpp", ".cu", ".cuh", ".hpp"}:
            source = path.read_text()
            assert "afd_plugin/" not in source
            assert "#include <deep_ep/" not in source
            assert "#include <deep_gemm/" not in source


def test_protocol_matches_recorded_provenance():
    origin = json.loads((PACKAGE_ROOT / "csrc/ORIGIN.json").read_text())
    # Future device protocol changes require an explicit provenance update,
    # separate from changes to its host packaging or launch adapter.
    assert len(origin["input_files"]) == 3
    assert origin["extracted_file_sha256"] == EXTRACTED_FILE_SHA256
    assert origin["protocol_changes"]
    for relative, expected_hash in origin["current_file_sha256"].items():
        path = PACKAGE_ROOT.parent / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
    assert len(source_digest()) == hashlib.sha256().digest_size * 2


def test_pinned_mit_attribution_ships():
    license_path = PACKAGE_ROOT / "third_party/deepep/LICENSE"
    assert "MIT License" in license_path.read_text()
    provenance = json.loads(license_path.with_name("PROVENANCE.json").read_text())
    assert provenance
    assert Path(license_path).is_file()
