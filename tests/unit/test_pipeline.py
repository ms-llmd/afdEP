# SPDX-License-Identifier: Apache-2.0
"""Check host ownership/event ordering without pretending to test transport."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from afdep import Config


@pytest.fixture
def runtime(monkeypatch):
    log = []
    state = SimpleNamespace(device=0, capturing=False, counts=None, initialized=True)

    class Stream:
        def __init__(self, device):
            self.device = device

        def wait_event(self, event):
            log.append(("wait", self, event))

    class Event:
        def record(self, stream):
            log.append(("record", self, stream))

    class Buffer:
        def __init__(self, *, group, config):
            self.device = 0
            self.rank = 0
            self.role = "ffn"
            self.role_rank = 0
            self.num_buffer_bytes = 1024
            log.append(("init", self))

        def close(self):
            log.append(("close", self))

    def gather(out, value, *, group):
        out[:] = [value, value] if state.counts is None else state.counts

    dist = ModuleType("torch.distributed")
    dist.is_initialized = lambda: state.initialized
    dist.get_world_size = lambda group: 2
    dist.all_gather_object = gather
    torch = ModuleType("torch")
    torch.distributed = dist
    torch.cuda = SimpleNamespace(
        Stream=Stream,
        Event=Event,
        current_device=lambda: state.device,
        is_current_stream_capturing=lambda: state.capturing,
    )
    buffer_module = ModuleType("afdep.buffer")
    buffer_module.Buffer = Buffer
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setitem(sys.modules, "afdep.buffer", buffer_module)
    path = Path(__file__).parents[2] / "afdep" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("afdep._pipeline_unit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = Config(8, 128, 4, 1)
    return SimpleNamespace(
        module=module,
        state=state,
        log=log,
        config=config,
        Stream=Stream,
        Event=Event,
    )


def make_pool(runtime, slots=2):
    return runtime.module.PipelineBuffer(
        group=object(), config=runtime.config, slots=slots
    )


def test_slots_have_independent_buffers_streams_and_accounted_memory(runtime):
    pool = make_pool(runtime)
    assert pool.num_slots == 2
    assert pool.buffer(0) is not pool.buffer(1)
    assert pool.stream(0) is not pool.stream(1)
    assert pool.num_buffer_bytes == 2048
    pool.close()
    pool.close()
    assert [entry[1] for entry in runtime.log if entry[0] == "close"] == [
        entry[1] for entry in runtime.log if entry[0] == "init"
    ]


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_collective_slot_counts_fail_before_buffer_init(runtime, count):
    with pytest.raises(ValueError, match="positive integer"):
        make_pool(runtime, count)
    assert not runtime.log


def test_different_rank_counts_rejected_before_buffer_init(runtime):
    runtime.state.counts = [1, 2]
    with pytest.raises(ValueError, match="same slot count"):
        make_pool(runtime)
    assert not runtime.log


def test_initialization_required(runtime):
    runtime.state.initialized = False
    with pytest.raises(RuntimeError, match="Initialize"):
        make_pool(runtime)


@pytest.mark.parametrize("index", [-1, 2, True, 0.5])
def test_slot_indices_checked(runtime, index):
    pool = make_pool(runtime)
    for accessor in (pool.buffer, pool.stream, pool.acquire):
        with pytest.raises(ValueError, match="index"):
            accessor(index)


def test_leases_prevent_double_ownership_and_allow_other_slot(runtime):
    pool = make_pool(runtime)
    first = pool.acquire(0)
    with pytest.raises(RuntimeError, match="active lease"):
        pool.acquire(0)
    second = pool.acquire(1)
    with pytest.raises(RuntimeError, match="Release every"):
        pool.close()
    first.release()
    second.release()
    with pytest.raises(RuntimeError, match="already been released"):
        first.release()
    with pytest.raises(RuntimeError, match="already been released"):
        first.wait_on(pool.stream(0))
    pool.close()


def test_reuse_orders_streams_after_previous_completion_without_cpu_wait(runtime):
    pool = make_pool(runtime)
    first = pool.acquire(0)
    assert not [entry for entry in runtime.log if entry[0] == "wait"]
    final_stream = runtime.Stream(0)
    first.release(stream=final_stream)
    recorded = runtime.log[-1]
    assert recorded[0] == "record"
    assert recorded[2] is final_stream
    second = pool.acquire(0)
    assert runtime.log[-1] == ("wait", pool.stream(0), recorded[1])
    staging_stream = runtime.Stream(0)
    second.wait_on(staging_stream)
    assert runtime.log[-1] == ("wait", staging_stream, recorded[1])
    second.release()


def test_external_completion_event_is_retained_for_next_owner(runtime):
    pool = make_pool(runtime)
    lease = pool.acquire(0)
    done = runtime.Event()
    with pytest.raises(ValueError, match="either"):
        lease.release(stream=pool.stream(0), event=done)
    lease.release(event=done)
    next_lease = pool.acquire(0)
    assert runtime.log[-1] == ("wait", pool.stream(0), done)
    next_lease.release()


def test_acquire_release_are_host_operations_outside_capture(runtime):
    pool = make_pool(runtime)
    lease = pool.acquire(0)
    runtime.state.capturing = True
    # Raw native calls are intentionally available for startup phase capture.
    assert pool.buffer(0)
    with pytest.raises(RuntimeError, match="outside CUDA capture"):
        pool.acquire(1)
    with pytest.raises(RuntimeError, match="outside CUDA capture"):
        lease.release()
    runtime.state.capturing = False
    lease.release()


def test_closed_and_wrong_device_are_rejected(runtime):
    pool = make_pool(runtime)
    runtime.state.device = 1
    with pytest.raises(RuntimeError, match="different CUDA device"):
        pool.acquire(0)
    runtime.state.device = 0
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.buffer(0)
