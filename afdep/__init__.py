# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel communication for Attention-FFN Disaggregation."""

from .deepep import Buffer, Config, ReceiveResult

__version__ = "1.0.0.dev0"
__all__ = ["Buffer", "Config", "ReceiveResult", "__version__"]
