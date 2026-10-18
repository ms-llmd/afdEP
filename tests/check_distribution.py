# SPDX-License-Identifier: Apache-2.0
"""Fail if a built wheel omits native runtime sources or third-party notices."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def source_files(root: str) -> set[str]:
    suffixes = {
        ".py",
        ".typed",
        ".cpp",
        ".cu",
        ".cuh",
        ".hpp",
        ".json",
        ".md",
        ".svg",
        ".txt",
    }
    return {
        path.as_posix()
        for path in Path(root).rglob("*")
        if path.is_file() and (path.suffix in suffixes or path.name == "LICENSE")
    }


def main() -> None:
    required = source_files("afdep")
    wheels = list(Path("dist").glob("*.whl"))
    sdists = list(Path("dist").glob("*.tar.gz"))
    assert wheels and sdists, "Build both wheel and sdist first"
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            assert required <= names, required - names
            assert any(name.endswith("/licenses/LICENSE") for name in names)
            assert any(
                name.endswith("/licenses/THIRD_PARTY_NOTICES.md") for name in names
            )
    for sdist in sdists:
        with tarfile.open(sdist) as archive:
            names = {name.split("/", 1)[-1] for name in archive.getnames()}
            assert required <= names, required - names
            documentation = source_files("docs")
            assert documentation <= names, documentation - names
    print("Wheel and sdist contain the standalone runtime and licenses")


if __name__ == "__main__":
    main()
