# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Lazy standalone build of the pinned DeepEP V2 transport.

Adapted from afd-plugin 1198c52, deepep_m2n/{buffer,jit_build}.py, and afdEP's
native builder. The vendored transport is unchanged; see its PROVENANCE.json.
AFDEP_CACHE_DIR selects the extension cache; EP_JIT_CACHE_DIR retains its
upstream meaning for DeepEP's later kernel compilations.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import ModuleType

from ._build import (
    DEFAULT_MAX_JOBS,
    PACKAGE_ROOT,
    cuda_wheel_include_flags,
    nccl_paths,
    resolve_nvcc,
)

VENDOR_ROOT = PACKAGE_ROOT / "third_party" / "deepep_v2"
EXTENSION_PREFIX = "_afdep_deepep_v2"
SOURCE_NAMES = (
    "csrc/python_api.cpp",
    "csrc/kernels/backend/nccl.cu",
    "csrc/kernels/backend/cuda_driver.cu",
)
CXX_FLAGS = (
    "-O3",
    "-Wno-deprecated-declarations",
    "-Wno-unused-variable",
    "-Wno-sign-compare",
    "-Wno-reorder",
    "-Wno-attributes",
    "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
)
NVCC_FLAGS = (
    "-O3",
    "-Xcompiler",
    "-O3",
    "--extended-lambda",
    "--diag-suppress=128,2417",
    "-rdc=true",
    "--ptxas-options=--register-usage-level=10",
    "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
)
SETUP_SCRIPT = """\
import json
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

spec = json.loads(Path(__file__).with_name("spec.json").read_text())
setup(
    name=spec["name"],
    ext_modules=[CUDAExtension(
        name=spec["name"],
        sources=spec["sources"],
        include_dirs=spec["include_dirs"],
        library_dirs=spec["library_dirs"],
        extra_compile_args={
            "cxx": spec["cxx_flags"],
            "nvcc": spec["nvcc_flags"],
            "nvcc_dlink": ["-dlink"],
        },
        extra_link_args=spec["link_flags"],
    )],
    cmdclass={"build_ext": BuildExtension},
)
"""


def source_digest(root: Path = VENDOR_ROOT) -> str:
    """Fingerprint file contents, including headers read later by DeepEP's JIT."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_signature(nvcc: Path, include: Path, lib: Path) -> dict:
    import torch

    header_digest = hashlib.sha256()
    for path in sorted(include.rglob("*.h")):
        header_digest.update(path.relative_to(include).as_posix().encode())
        header_digest.update(path.read_bytes())
    major, minor = torch.cuda.get_device_capability()
    library = (lib / "libnccl.so.2").resolve()
    return {
        "backend": "deepep_v2",
        "source": source_digest(),
        "vendor_root": str(VENDOR_ROOT),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "python_abi": sysconfig.get_config_var("SOABI"),
        "nvcc": str(nvcc),
        "nvcc_version": subprocess.check_output([str(nvcc), "--version"], text=True),
        "cxx": os.environ.get("CXX", "c++"),
        "nccl_include": str(include),
        "nccl_library": str(library),
        "nccl_library_mtime_ns": library.stat().st_mtime_ns,
        "nccl_headers": header_digest.hexdigest(),
        "cxx_flags": CXX_FLAGS,
        "nvcc_flags": NVCC_FLAGS,
        "system_headers": cuda_wheel_include_flags(),
        "setup_script": hashlib.sha256(SETUP_SCRIPT.encode()).hexdigest(),
        "source_names": SOURCE_NAMES,
        "link_flags": ["-lcuda", "-l:libnccl.so.2", f"-Wl,-rpath,{lib}"],
        "arch": f"{major}.{minor}",
    }


def load_deepep() -> ModuleType:
    """Build/load the native module without creating a communicator or buffer."""
    import fcntl

    import torch

    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("DeepEP V2 transport requires Linux with CUDA GPUs")
    nvcc = resolve_nvcc()
    include, lib = nccl_paths()
    signature = build_signature(nvcc, include, lib)
    fingerprint = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()
    name = f"{EXTENSION_PREFIX}_{fingerprint[:16]}"
    if name in sys.modules:
        return sys.modules[name]
    cache_root = Path(
        os.environ.get("AFDEP_CACHE_DIR", Path.home() / ".cache/afdep")
    ).expanduser()
    build_dir = cache_root / "deepep_v2" / fingerprint
    build_dir.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_path = build_dir / "lib" / f"{name}{suffix}"
    complete = build_dir / "build.complete"
    with (build_dir / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (
            not so_path.is_file()
            or not complete.is_file()
            or complete.read_text() != fingerprint
        ):
            spec = {
                "name": name,
                "sources": [str(VENDOR_ROOT / p) for p in SOURCE_NAMES],
                "include_dirs": [
                    str(VENDOR_ROOT / "deep_ep/include"),
                    str(VENDOR_ROOT / "csrc"),
                    str(include),
                ],
                "library_dirs": [str(lib)],
                "cxx_flags": [*CXX_FLAGS, *signature["system_headers"]],
                "nvcc_flags": [*NVCC_FLAGS, *signature["system_headers"]],
                "link_flags": signature["link_flags"],
            }
            (build_dir / "spec.json").write_text(json.dumps(spec, indent=2))
            (build_dir / "signature.json").write_text(json.dumps(signature, indent=2))
            setup = build_dir / "setup.py"
            setup.write_text(SETUP_SCRIPT)
            env = dict(os.environ)
            for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
                env.pop(key, None)
            env["CUDA_HOME"] = env["CUDA_PATH"] = str(nvcc.parent.parent)
            env["PYTORCH_NVCC"] = str(nvcc)
            env["TORCH_CUDA_ARCH_LIST"] = signature["arch"]
            env.setdefault("MAX_JOBS", DEFAULT_MAX_JOBS)
            subprocess.run(
                [
                    sys.executable,
                    str(setup),
                    "build_ext",
                    "--build-lib",
                    str(build_dir / "staging"),
                    "--build-temp",
                    str(build_dir / "temp"),
                ],
                cwd=build_dir,
                env=env,
                check=True,
            )
            artifact = build_dir / "staging" / so_path.name
            if not artifact.is_file():
                raise RuntimeError(f"DeepEP V2 build did not produce {artifact}")
            so_path.parent.mkdir(exist_ok=True)
            os.replace(artifact, so_path)
            complete.write_text(fingerprint)
        fcntl.flock(lock, fcntl.LOCK_UN)
    module_spec = importlib.util.spec_from_file_location(name, so_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not load DeepEP V2 extension at {so_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.init_jit(
        str(VENDOR_ROOT / "deep_ep"), str(nvcc.parent.parent), str(include.parent)
    )
    sys.modules[name] = module
    return module
