# SPDX-License-Identifier: Apache-2.0
"""Host contracts for the DeepEP facade; these mocks do not validate transport."""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from afdep import deepep


class Tensor:
    def __init__(self, shape, dtype="bf16", device=0, contiguous=True):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim, self.is_cuda = len(shape), True
        self.device, self.contiguous_storage = device, contiguous
        self.view_parent = None

    def get_device(self):
        return self.device

    def is_contiguous(self):
        return self.contiguous_storage

    def contiguous(self):
        assert self.contiguous_storage, "test unexpectedly requested a payload copy"
        return self

    def to(self, dtype):
        assert self.dtype == dtype, "test unexpectedly requested a dtype conversion"
        return self

    def float(self):
        return self.to("fp32")

    def __getitem__(self, index):
        result = Tensor((index.stop, *self.shape[1:]), self.dtype, self.device)
        result.view_parent = self
        return result


@pytest.fixture
def runtime(monkeypatch):
    torch, dist = ModuleType("torch"), ModuleType("torch.distributed")
    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        capturing=False,
        is_current_stream_capturing=lambda: cuda.capturing,
        get_device_properties=lambda _: SimpleNamespace(multi_processor_count=132),
        synchronize=Mock(),
    )
    torch.cuda, torch.distributed = cuda, dist
    torch.bfloat16, torch.float32, torch.float8_e4m3fn = "bf16", "fp32", "fp8"
    torch.device = lambda _, index: index
    torch.empty = torch.zeros = lambda shape, dtype, device: Tensor(
        shape, dtype, device
    )
    dist.is_initialized, dist.get_backend = lambda: True, lambda _: "gloo"
    dist.barrier = Mock()

    def gather(values, value, group):
        values[:] = [value] * group.size()
        if value is None:
            values[0] = b"root-id"

    dist.all_gather_object = gather
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    group = SimpleNamespace(rank=lambda: 2, size=lambda: 4)
    native = Mock()
    native.get_logical_domain_size.return_value = (1, 4)
    native.combine.return_value = (Tensor((2, 256)), None, None)

    def dispatch(*args):
        constructor = module.ElasticBuffer.call_args.args
        rows = 0 if len(constructor) == 14 and group.rank() >= constructor[-1] else 4
        scale = (
            Tensor((rows, 2), "fp32", contiguous=False) if args[1] is not None else None
        )
        return (
            Tensor((rows, 256), args[0].dtype),
            scale,
            None,
            Tensor((rows,), "fp32"),
            args[2],
            [],
            Tensor((4,), "i32"),
            Tensor((8,), "i32"),
            Tensor((rows, 18), "i32"),
            Tensor((0,), "i32"),
            None,
            None,
            None,
        )

    native.dispatch.side_effect = dispatch
    module = SimpleNamespace(
        topk_idx_t="i64",
        get_local_nccl_unique_id=Mock(return_value=b"root-id"),
        create_nccl_comm=Mock(return_value=17),
        destroy_nccl_comm=Mock(),
        get_physical_domain_size=Mock(return_value=(1, 4)),
        calculate_elastic_buffer_size=Mock(return_value=4096),
        ElasticBuffer=Mock(return_value=native),
    )
    monkeypatch.setattr(deepep, "load_deepep", lambda: module)
    monkeypatch.setattr(deepep, "source_digest", lambda: "pinned-source")
    return SimpleNamespace(
        torch=torch, dist=dist, group=group, native=native, module=module
    )


def elastic(runtime, **kwargs):
    settings = dict(
        group=runtime.group,
        capacity_tokens_per_rank=16,
        hidden_size=256,
        num_experts=32,
        top_k=16,
        allow_hybrid_mode=True,
        rdma_traffic_class=0,
        cpu_timeout_s=120,
        gpu_timeout_s=120,
    )
    return deepep.ElasticBuffer(**(settings | kwargs))


def config(**kwargs):
    settings = dict(
        capacity_tokens_per_rank=16,
        hidden_size=256,
        num_experts=16,
        top_k=16,
        ffn_ranks=2,
        attention_ranks=2,
        role_aware=False,
    )
    return deepep.DeepEPConfig(**(settings | kwargs))


def test_real_and_logical_expert_counts():
    value = config()
    assert value.experts_per_rank == 8
    assert value.logical_experts == 32
    assert value.top_k > value.experts_per_rank
    with pytest.raises(ValueError, match="divide evenly"):
        config(num_experts=15)


def test_role_aware_receive_bounds_use_source_geometry():
    generic = config(capacity_tokens_per_rank=2048)
    roles = replace(generic, role_aware=True)
    assert generic.receive_rows(2048) == 66_560
    assert roles.receive_rows(2048) == 33_792
    assert roles.logical_experts == generic.logical_experts
    with pytest.raises(ValueError, match="role_aware must be a boolean"):
        replace(generic, role_aware=1)


def test_role_aware_requires_explicit_geometry_and_preserves_generic_ctor(runtime):
    with pytest.raises(RuntimeError, match="explicit ffn_ranks"):
        elastic(runtime, role_aware=True)
    runtime.module.ElasticBuffer.assert_not_called()
    generic = elastic(runtime)
    assert len(runtime.module.ElasticBuffer.call_args.args) == 13
    # Union sizing for the generic buffer; role-sized inboxes for the role path.
    assert runtime.module.calculate_elastic_buffer_size.call_args.args[-1] == 0
    generic.close()
    roles = elastic(runtime, role_aware=True, ffn_ranks=2)
    assert len(runtime.module.ElasticBuffer.call_args.args) == 14
    assert runtime.module.ElasticBuffer.call_args.args[-1] == 2
    assert runtime.module.calculate_elastic_buffer_size.call_args.args[-1] == 2
    roles.close()


def test_role_aware_source_owns_no_dummy_payload_arena(runtime):
    buffer = deepep.DeepEPBuffer(group=runtime.group, config=config(role_aware=True))
    assert buffer._zero_output.shape == (0, 256)
    x, ids, weights = Tensor((2, 256)), Tensor((2, 16), "i64"), Tensor((2, 16), "fp32")
    buffer.source_dispatch(x, ids, weights, bucket=8)
    assert buffer.transport._pending.recv_shape == (0, 256)
    buffer.source_collect()
    assert runtime.native.combine.call_args.args[0].shape == (0, 256)


def test_role_aware_worker_rejects_source_payload(runtime):
    runtime.group.rank = lambda: 0
    buffer = elastic(runtime, role_aware=True, ffn_ranks=2)
    with pytest.raises(ValueError, match="cannot originate tokens"):
        buffer.dispatch(
            Tensor((2, 256)),
            None,
            Tensor((2, 16), "i64"),
            Tensor((2, 16), "fp32"),
            bucket=8,
            expert_alignment=128,
            do_cpu_sync=False,
        )
    runtime.native.dispatch.assert_not_called()


def test_facade_accepts_topk_above_local_experts_and_owns_bootstrap(
    runtime, monkeypatch
):
    monkeypatch.delenv("EP_DISABLE_GIN", raising=False)
    buffer = elastic(runtime)
    assert buffer.top_k == 16  # Eight logical experts/rank; 16 choices is valid.
    runtime.module.create_nccl_comm.assert_called_once_with(b"root-id", 4, 2)
    assert buffer.physical_domain == (1, 4)
    assert buffer.logical_domain == (1, 4)
    assert "EP_DISABLE_GIN" not in deepep.os.environ
    assert runtime.module.calculate_elastic_buffer_size.call_args.args[1] == 17
    buffer.close()
    buffer.close()
    runtime.native.destroy.assert_called_once()
    runtime.module.destroy_nccl_comm.assert_called_once_with(17)


def test_explicit_disable_rejects_multidomain_before_buffer_creation(
    runtime, monkeypatch
):
    monkeypatch.setenv("EP_DISABLE_GIN", "1")
    runtime.module.get_physical_domain_size.return_value = (2, 2)
    with pytest.raises(ValueError, match="Multiple NVLink domains require GIN"):
        elastic(runtime)
    runtime.module.ElasticBuffer.assert_not_called()
    runtime.module.destroy_nccl_comm.assert_called_once_with(17)


def test_role_aware_accepts_multidomain_with_gin(runtime, monkeypatch):
    monkeypatch.delenv("EP_DISABLE_GIN", raising=False)
    runtime.module.get_physical_domain_size.return_value = (2, 2)
    buffer = elastic(runtime, role_aware=True, ffn_ranks=2)
    assert buffer.physical_domain == (2, 2)
    assert runtime.module.ElasticBuffer.call_args.args[-1] == 2
    buffer.close()


def test_opaque_handle_is_required_and_native_output_is_returned(runtime):
    buffer = elastic(runtime)
    x, ids, weights = Tensor((2, 256)), Tensor((2, 16), "i64"), Tensor((2, 16), "fp32")
    received = buffer.dispatch(
        x, None, ids, weights, bucket=8, expert_alignment=128, do_cpu_sync=False
    )
    args = runtime.native.dispatch.call_args.args
    assert args[:4] == (x, None, ids, weights)
    assert args[12:15] == (8, 32, 128)
    assert args[22:] == (False, True, True)
    assert received.receipt is received.handle
    with pytest.raises(ValueError, match="pending exchange"):
        buffer.combine(Tensor((4, 256)), replace(received.handle, _owner=object()))
    with pytest.raises(RuntimeError, match="previous exchange"):
        buffer.dispatch(
            x, None, ids, weights, bucket=8, expert_alignment=128, do_cpu_sync=False
        )
    rows = Tensor((4, 256))
    output = buffer.combine(rows, received.handle)
    assert output is runtime.native.combine.return_value[0]
    assert runtime.native.combine.call_args.args[0] is rows
    with pytest.raises(ValueError, match="pending exchange"):
        buffer.combine(rows, received.handle)


def test_capture_rejects_host_counts_and_close_with_pending_exchange(runtime):
    buffer = elastic(runtime)
    x, ids, weights = Tensor((2, 256)), Tensor((2, 16), "i64"), Tensor((2, 16), "fp32")
    runtime.torch.cuda.capturing = True
    with pytest.raises(RuntimeError, match="Host receive counts"):
        buffer.dispatch(
            x, None, ids, weights, bucket=8, expert_alignment=128, do_cpu_sync=True
        )
    runtime.native.dispatch.assert_not_called()
    runtime.torch.cuda.capturing = False
    buffer.dispatch(
        x, None, ids, weights, bucket=8, expert_alignment=128, do_cpu_sync=False
    )
    with pytest.raises(RuntimeError, match="pending exchange"):
        buffer.close()
    runtime.native.destroy.assert_not_called()


def test_role_source_returns_native_output_without_copy(runtime):
    buffer = deepep.DeepEPBuffer(group=runtime.group, config=config())
    x, ids, weights = Tensor((2, 256)), Tensor((2, 16), "i64"), Tensor((2, 16), "fp32")
    buffer.source_dispatch(x, ids, weights, bucket=8)
    output = buffer.source_collect()
    assert output is runtime.native.combine.return_value[0]
    assert runtime.native.combine.call_args.args[0].view_parent is buffer._zero_output
    assert runtime.native.combine.call_args.args[-1] is False
    buffer.source_dispatch(x, ids, weights, bucket=8)
    buffer.source_collect(view=True)
    assert runtime.native.combine.call_args.args[-1] is True
    with pytest.raises(RuntimeError, match="ffn role"):
        buffer.worker_receive(bucket=8)
    with pytest.raises(ValueError, match="dtype"):
        buffer.source_dispatch(x, ids, Tensor((2, 16)), bucket=8)


def test_role_worker_dispatches_empty_and_preserves_receipt(runtime):
    runtime.group.rank = lambda: 0
    buffer = deepep.DeepEPBuffer(group=runtime.group, config=config(payload="fp8"))
    received = buffer.worker_receive(bucket=8)
    args = runtime.native.dispatch.call_args.args
    assert args[0].shape == (0, 256)
    assert args[0].dtype == "fp8"
    assert args[1].shape == (0, 2)
    assert args[13] == 32
    assert received.recv_scale is not None
    rows = Tensor(received.recv_x.shape)
    buffer.worker_reply(rows, received.receipt)
    assert runtime.native.combine.call_args.args[0] is rows


def test_v1_public_api_enables_specialization_without_loading_torch():
    from afdep import Buffer, Config

    value = Config(2048, 4096, 128, 8, 2, 2)
    assert value.role_aware
    assert value.receive_rows(2048) < replace(value, role_aware=False).receive_rows(
        2048
    )
    assert Buffer is deepep.DeepEPBuffer
    assert Config is deepep.DeepEPConfig


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"capacity_tokens_per_rank": 0}, "positive int32"),
        ({"ffn_ranks": True}, "positive int32"),
        ({"attention_ranks": 0}, "positive int32"),
        ({"hidden_size": 257}, "multiple of 256"),
        ({"top_k": 17}, "top_k"),
        ({"num_sms": -1}, "nonnegative int32"),
        ({"num_qps": 1.5}, "nonnegative int32"),
        ({"allow_hybrid_mode": 1}, "boolean"),
        ({"payload": "fp16"}, "payload"),
        ({"rdma_traffic_class": 256}, "traffic_class"),
        ({"capacity_tokens_per_rank": (1 << 31) - 2}, "int32 indexing"),
    ],
)
def test_invalid_configuration_fails_before_gpu_initialization(settings, message):
    with pytest.raises(ValueError, match=message):
        config(**settings)
