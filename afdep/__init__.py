# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel communication for Attention-FFN Disaggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import Config

if TYPE_CHECKING:
    from .buffer import Buffer, ReceiveResult
    from .pipeline import PipelineBuffer, SlotLease

__version__ = "0.1.0.dev0"
__all__ = [
    "Buffer",
    "Config",
    "PipelineBuffer",
    "ReceiveResult",
    "SlotLease",
    "__version__",
]


def __getattr__(name: str):
    # Keep configuration and package inspection usable without loading CUDA
    # libraries. Runtime classes load PyTorch only when explicitly requested.
    if name in {"Buffer", "ReceiveResult"}:
        from .buffer import Buffer, ReceiveResult

        return {"Buffer": Buffer, "ReceiveResult": ReceiveResult}[name]
    if name in {"PipelineBuffer", "SlotLease"}:
        from .pipeline import PipelineBuffer, SlotLease

        return {"PipelineBuffer": PipelineBuffer, "SlotLease": SlotLease}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
