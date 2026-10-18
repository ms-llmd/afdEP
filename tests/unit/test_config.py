# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported protocol/index geometries before allocating GPU memory."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict

import pytest

from afdep import Config

VALID = {
    "capacity_tokens_per_rank": 8192,
    "hidden_size": 2048,
    "num_experts": 128,
    "top_k": 8,
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capacity_tokens_per_rank", 0),
        ("capacity_tokens_per_rank", True),
        ("capacity_tokens_per_rank", 1.0),
        ("capacity_tokens_per_rank", 1 << 31),
        ("hidden_size", 129),
        ("hidden_size", -128),
        ("num_experts", 127),
        ("top_k", 33),
        ("top_k", 0),
        ("ffn_ranks", 8),
        ("attention_ranks", 2),
        ("expert_alignment", 0),
        ("payload", "fp16"),
        ("allow_hybrid_mode", 0),
        ("allow_hybrid_mode", 1),
        ("allow_hybrid_mode", "direct"),
        ("allow_hybrid_mode", None),
        ("num_gin_contexts", 0),
        ("transport_stripes", 9),
        ("num_sms", 0),
        ("gpu_timeout_s", -1),
        ("rdma_traffic_class", 256),
        ("rdma_traffic_class", True),
    ],
)
def test_unsupported_geometry(field, value):
    with pytest.raises(ValueError):
        Config(**(VALID | {field: value}))


def test_expert_ownership_is_real_only():
    config = Config(**VALID)
    assert config.world_size == 8
    assert config.experts_per_rank == 32
    assert [list(config.expert_range(r)) for r in range(4)] == [
        list(range(start, start + 32)) for start in range(0, 128, 32)
    ]
    with pytest.raises(ValueError):
        config.expert_range(4)
    with pytest.raises(ValueError):
        config.expert_range(True)


def test_top_k_cannot_exceed_experts():
    with pytest.raises(ValueError):
        Config(**(VALID | {"num_experts": 4, "top_k": 8}))


def test_native_index_bound_rejected():
    with pytest.raises(ValueError, match="int32 indexing"):
        Config(**(VALID | {"capacity_tokens_per_rank": 1 << 29}))


def test_configuration_is_immutable_and_serializable():
    config = Config(**VALID)
    assert asdict(config)["payload"] == "bf16"
    assert asdict(config)["allow_hybrid_mode"] is True
    assert Config(**(VALID | {"allow_hybrid_mode": False})).allow_hybrid_mode is False
    with pytest.raises(FrozenInstanceError):
        config.payload = "fp8"


def test_import_does_not_load_torch_or_cuda():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import afdep, sys; assert 'torch' not in sys.modules; "
            "assert afdep.Config(8, 128, 4, 1).world_size == 8",
        ],
        check=True,
    )
