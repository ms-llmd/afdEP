# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel communication for Attention-FFN Disaggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import Config

if TYPE_CHECKING:
    from .buffer import Buffer, ReceiveResult
    from .deepep import DeepEPBuffer, DeepEPConfig

__version__ = "0.1.0.dev0"
__all__ = [
    "Buffer",
    "Config",
    "DeepEPBuffer",
    "DeepEPConfig",
    "ReceiveResult",
    "__version__",
]


def __getattr__(name: str):
    # Keep configuration and package inspection usable without loading CUDA
    # libraries. Runtime classes load PyTorch only when explicitly requested.
    if name in {"Buffer", "ReceiveResult"}:
        from .buffer import Buffer, ReceiveResult

        return {"Buffer": Buffer, "ReceiveResult": ReceiveResult}[name]
    if name in {"DeepEPBuffer", "DeepEPConfig"}:
        from .deepep import DeepEPBuffer, DeepEPConfig

        return {"DeepEPBuffer": DeepEPBuffer, "DeepEPConfig": DeepEPConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
