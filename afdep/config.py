# SPDX-License-Identifier: Apache-2.0
"""Configuration of the current directional communication backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RANKS_PER_DOMAIN = 4
FP8_BLOCK_SIZE = 128
MAX_TOP_K = 32
INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class Config:
    """Allocation and launch configuration for the current backend."""

    capacity_tokens_per_rank: int
    hidden_size: int
    num_experts: int
    top_k: int
    ffn_ranks: int = RANKS_PER_DOMAIN
    attention_ranks: int = RANKS_PER_DOMAIN
    expert_alignment: int = 128
    payload: Literal["bf16", "fp8"] = "bf16"
    num_gin_contexts: int = 8
    transport_stripes: int = 8
    num_sms: int = 132
    gpu_timeout_s: int = 60
    rdma_traffic_class: int = 0
    allow_hybrid_mode: bool = True

    def __post_init__(self) -> None:
        if type(self.allow_hybrid_mode) is not bool:
            raise ValueError("allow_hybrid_mode must be a boolean")
        positive_fields = (
            "capacity_tokens_per_rank",
            "hidden_size",
            "num_experts",
            "top_k",
            "ffn_ranks",
            "attention_ranks",
            "expert_alignment",
            "num_gin_contexts",
            "transport_stripes",
            "num_sms",
            "gpu_timeout_s",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= INT32_MAX:
                raise ValueError(f"{name} must be a positive int32 integer")
        if (
            self.ffn_ranks != RANKS_PER_DOMAIN
            or self.attention_ranks != RANKS_PER_DOMAIN
        ):
            raise ValueError(
                "The current backend requires four FFN and four Attention ranks"
            )
        if self.hidden_size % FP8_BLOCK_SIZE:
            raise ValueError("hidden_size must be a multiple of 128")
        if self.num_experts % self.ffn_ranks:
            raise ValueError("num_experts must divide evenly across FFN ranks")
        if self.top_k > min(MAX_TOP_K, self.num_experts):
            raise ValueError("top_k must be <= 32 and <= num_experts")
        if self.payload not in {"bf16", "fp8"}:
            raise ValueError("payload must be 'bf16' or 'fp8'")
        if self.transport_stripes > self.num_gin_contexts:
            raise ValueError("transport_stripes must be <= num_gin_contexts")
        if (
            type(self.rdma_traffic_class) is not int
            or not 0 <= self.rdma_traffic_class <= 255
        ):
            raise ValueError("rdma_traffic_class must be an integer in [0,255]")
        max_rows = (
            self.attention_ranks * self.capacity_tokens_per_rank * self.top_k
            + self.experts_per_rank * (self.expert_alignment - 1)
        )
        if max_rows > INT32_MAX:
            raise ValueError("Configured receive row bound exceeds int32 indexing")

    @property
    def world_size(self) -> int:
        return self.ffn_ranks + self.attention_ranks

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.ffn_ranks

    def expert_range(self, ffn_rank: int) -> range:
        if type(ffn_rank) is not int or not 0 <= ffn_rank < self.ffn_ranks:
            raise ValueError("ffn_rank is outside the FFN domain")
        start = ffn_rank * self.experts_per_rank
        return range(start, start + self.experts_per_rank)
