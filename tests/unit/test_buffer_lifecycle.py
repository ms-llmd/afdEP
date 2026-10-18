# SPDX-License-Identifier: Apache-2.0
"""Protect peer window lifetimes when closing after deferred notifications."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def closing_buffer(monkeypatch):
    calls = []
    state = SimpleNamespace(capturing=False, fail_drain=False)
    torch = ModuleType("torch")
    distributed = ModuleType("torch.distributed")
    torch.distributed = distributed
    torch.cuda = SimpleNamespace(
        is_current_stream_capturing=lambda: state.capturing,
        synchronize=lambda device: calls.append("GPU complete"),
    )
    distributed.barrier = lambda *, group: calls.append("all ranks drained")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed)
    path = Path(__file__).parents[2] / "afdep" / "buffer.py"
    spec = importlib.util.spec_from_file_location("afdep._buffer_lifecycle_unit", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    def drain():
        calls.append("GIN drain submitted")
        if state.fail_drain:
            raise RuntimeError("injected local drain failure")

    buffer = object.__new__(module.Buffer)
    buffer._closed = False
    buffer.device = 0
    buffer._group = object()
    buffer._comm = 1
    buffer._runtime = SimpleNamespace(
        drain=drain, destroy=lambda: calls.append("windows released")
    )
    buffer._module = SimpleNamespace(
        destroy_nccl_comm=lambda comm: calls.append("communicator released")
    )
    return buffer, calls, state


def test_close_finishes_local_gin_before_peer_release_barrier(closing_buffer):
    buffer, calls, _ = closing_buffer
    buffer.close()
    assert calls == [
        "GPU complete",
        "GIN drain submitted",
        "GPU complete",
        "all ranks drained",
        "windows released",
        "communicator released",
    ]
    assert buffer._closed
    buffer.close()
    assert len(calls) == 6


def test_failed_local_drain_cannot_release_peer_windows(closing_buffer):
    buffer, calls, state = closing_buffer
    state.fail_drain = True
    with pytest.raises(RuntimeError, match="injected local drain failure"):
        buffer.close()
    assert calls == ["GPU complete", "GIN drain submitted"]
    assert not buffer._closed


def test_close_capture_rejected_before_drain(closing_buffer):
    buffer, calls, state = closing_buffer
    state.capturing = True
    with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
        buffer.close()
    assert not calls
