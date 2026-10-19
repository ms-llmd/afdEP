# SPDX-License-Identifier: Apache-2.0
"""Inspect AFDEP runtime requirements without creating a communicator."""

from __future__ import annotations

import argparse
import json
import platform

from . import __version__
from ._build import load_deepep, nccl_paths, resolve_nvcc, source_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor"])
    parser.add_argument(
        "--build", action="store_true", help="Compile/load CUDA extension"
    )
    args = parser.parse_args()
    report: dict = {
        "afdep": __version__,
        "platform": platform.platform(),
        "source_digest": source_digest(),
        "errors": [],
    }
    try:
        import torch

        report.update(
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            cuda_available=torch.cuda.is_available(),
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            report.update(
                gpu=props.name,
                capability=list(torch.cuda.get_device_capability()),
                sm_count=props.multi_processor_count,
            )
        else:
            report["errors"].append("No CUDA GPU is available")
    except ImportError as exc:
        report["errors"].append(str(exc))
    for key, operation in (("nvcc", resolve_nvcc), ("nccl", nccl_paths)):
        try:
            value = operation()
            report[key] = (
                [str(p) for p in value] if isinstance(value, tuple) else str(value)
            )
        except RuntimeError as exc:
            report["errors"].append(str(exc))
    if args.build:
        try:
            module = load_deepep()
            report["native_library"] = module.__file__
            report["nccl_version"] = module.get_nccl_version()
            report["compiled_nccl_version"] = module.compiled_nccl_version
            if report["nccl_version"] != report["compiled_nccl_version"]:
                report["errors"].append(
                    "NCCL headers differ from the loaded runtime. Match "
                    "AFDEP_NCCL_ROOT and LD_LIBRARY_PATH before starting Python."
                )
        except Exception as exc:
            report["errors"].append(f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
