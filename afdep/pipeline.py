# SPDX-License-Identifier: Apache-2.0
"""Independent transport slots for caller-scheduled microbatch pipelines."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .buffer import Buffer
from .config import Config


class SlotLease:
    """Host ownership of one slot until its final GPU operation is submitted.

    Phase streams, tensor lifetimes, and graph replay order remain caller-owned.
    Releasing a lease enqueues no CPU synchronization: the next owner inherits
    the recorded completion event as a GPU dependency.
    """

    def __init__(self, owner: PipelineBuffer, index: int) -> None:
        self._owner = owner
        self.index = index
        self._previous = owner._completions[index]
        self._active = True

    def _check_active(self) -> None:
        self._owner._check_open()
        if not self._active:
            raise RuntimeError("Pipeline slot lease has already been released")

    def wait_on(self, stream: torch.cuda.Stream) -> None:
        """Order another phase stream after the previous owner of this slot."""
        self._check_active()
        if self._previous is not None:
            stream.wait_event(self._previous)

    def release(
        self,
        *,
        stream: torch.cuda.Stream | None = None,
        event: torch.cuda.Event | None = None,
    ) -> None:
        """Release host ownership after recording or supplying final completion.

        ``event`` must already cover every slot reader/writer. Otherwise an
        event is recorded on ``stream`` (the slot stream by default). Join all
        other phase streams before releasing. Inputs, outputs, and graph storage
        must remain alive through their asynchronous users.
        """
        self._check_active()
        self._owner._check_outside_capture()
        if event is not None and stream is not None:
            raise ValueError("Supply either a completion event or a stream")
        if event is None:
            event = self._owner._events[self.index]
            event.record(self._owner.stream(self.index) if stream is None else stream)
        self._owner._completions[self.index] = event
        self._owner._leases[self.index] = None
        self._active = False


class PipelineBuffer:
    """Pool of independent single-lane buffers, not a microbatch scheduler.

    Construct and close collectively in identical order on every rank. Each
    slot owns a separate NCCL communicator, registered windows, native epochs,
    and reader credits. The caller supplies the same slot/exchange order on
    every rank and joins receive, compute, and reply streams explicitly.
    """

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        config: Config,
        slots: int = 2,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("Initialize torch.distributed before creating AFDEP")
        counts = [None] * dist.get_world_size(group)
        dist.all_gather_object(counts, slots, group=group)
        if any(type(count) is not int or count <= 0 for count in counts):
            raise ValueError("Every rank must request a positive integer slot count")
        if any(count != counts[0] for count in counts):
            raise ValueError("Every AFDEP rank must use the same slot count")
        self.config = config
        self._closed = False
        self._buffers = tuple(Buffer(group=group, config=config) for _ in range(slots))
        self.device = self._buffers[0].device
        self.rank = self._buffers[0].rank
        self.role = self._buffers[0].role
        self.role_rank = self._buffers[0].role_rank
        self._streams = tuple(
            torch.cuda.Stream(device=self.device) for _ in range(slots)
        )
        self._events = tuple(torch.cuda.Event() for _ in range(slots))
        self._completions: list[torch.cuda.Event | None] = [None] * slots
        self._leases: list[SlotLease | None] = [None] * slots
        self.num_buffer_bytes = sum(buffer.num_buffer_bytes for buffer in self._buffers)

    @property
    def num_slots(self) -> int:
        return len(self._buffers)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("AFDEP pipeline buffer is closed")
        if torch.cuda.current_device() != self.device:
            raise RuntimeError("AFDEP pipeline used on a different CUDA device")

    @staticmethod
    def _check_outside_capture() -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Acquire and release pipeline leases outside CUDA capture"
            )

    def _check_index(self, index: int) -> None:
        self._check_open()
        if type(index) is not int or not 0 <= index < self.num_slots:
            raise ValueError("Pipeline slot index is out of range")

    def buffer(self, index: int) -> Buffer:
        """Return a raw slot buffer for eager calls or startup phase capture.

        At runtime acquire a lease before replaying or enqueuing operations.
        Raw access does not automatically track ownership or tensor lifetimes.
        """
        self._check_index(index)
        return self._buffers[index]

    def stream(self, index: int) -> torch.cuda.Stream:
        """Return the slot's communication stream; other streams need events."""
        self._check_index(index)
        return self._streams[index]

    def acquire(self, index: int) -> SlotLease:
        """Own a slot, ordering its stream after the previous GPU completion.

        This does not wait on the CPU or join the current/default CUDA stream.
        Use ``lease.wait_on(other_stream)`` before accessing reusable slot
        storage from any additional stream, including input staging streams.
        """
        self._check_index(index)
        self._check_outside_capture()
        if self._leases[index] is not None:
            raise RuntimeError("Pipeline slot already has an active lease")
        lease = SlotLease(self, index)
        lease.wait_on(self._streams[index])
        self._leases[index] = lease
        return lease

    def close(self) -> None:
        """Collectively close all slots after every lease has been released.

        As with ``Buffer.close()``, this is not rank-local exception cleanup.
        Transport errors require terminating and reconstructing participants.
        """
        if self._closed:
            return
        self._check_open()
        self._check_outside_capture()
        if any(lease is not None for lease in self._leases):
            raise RuntimeError("Release every pipeline slot before collective close")
        for buffer in self._buffers:
            buffer.close()
        self._closed = True
