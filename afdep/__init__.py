# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel communication for Attention-FFN Disaggregation."""

from .config import Config as DirectionalConfig
from .deepep import Buffer, Config, ReceiveResult

__version__ = "1.0.0.dev0"
__all__ = [
    "Buffer",
    "Config",
    "ReceiveResult",
    "DirectionalBuffer",
    "DirectionalConfig",
    "__version__",
]


def __getattr__(name: str):
    if name == "DirectionalBuffer":
        from .buffer import Buffer as DirectionalBuffer

        return DirectionalBuffer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
