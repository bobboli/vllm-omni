# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import socket
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


@dataclass(frozen=True, slots=True)
class _SlotKey:
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    device_index: int


@dataclass(slots=True)
class _BufferSlot:
    output: torch.Tensor
    symm_handle: Any
    busy: bool = False


@dataclass(slots=True)
class AsyncUlyssesHandle:
    """An in-flight strict-Ulysses exchange issued by an exchanger."""

    _owner_token: object
    _slot: _BufferSlot | None
    _output: torch.Tensor | None
    _ready_event: torch.cuda.Event | None
    _joined: bool = False


class AsyncUlyssesExchange:
    """Overlap strict-Ulysses all-to-all with Q/K/V production.

    ``issue`` accepts ``[B, S_local, H, D]`` and immediately queues peer
    copies on a dedicated CUDA stream. ``join`` waits for all issued copies and
    returns ``[B, world_size * S_local, H / world_size, D]`` tensors in the
    order supplied to it.

    Symmetric output buffers are borrowed workspaces. Before a batch overwrites
    them, a cross-rank handshake follows its first producer event, which in
    turn follows every previous consumer on the compute stream. A second
    handshake makes the current peer writes visible before attention consumes
    them.

    All ranks must call ``issue`` and ``join`` in the same order. Tensors in an
    issue batch must have matching shapes and dtypes. This implementation
    supports CUDA tensors, rank-four inputs, and the strict Ulysses
    ``scatter_idx=2, gather_idx=1`` layout only.
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        *,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        barrier_channel: int = 0,
        barrier_timeout_ms: int = 60_000,
        device: torch.device | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before creating an async Ulysses exchanger")
        if scatter_idx != 2 or gather_idx != 1:
            raise ValueError(
                "Async Ulysses currently requires scatter_idx=2 and gather_idx=1, "
                f"got scatter_idx={scatter_idx}, gather_idx={gather_idx}"
            )
        if dist.get_backend(process_group) != "nccl":
            raise ValueError("Async Ulysses symmetric-memory exchange requires an NCCL process group")
        if barrier_timeout_ms <= 0:
            raise ValueError(f"barrier_timeout_ms must be positive, got {barrier_timeout_ms}")

        self._process_group = process_group
        self._rank = dist.get_rank(process_group)
        self._world_size = dist.get_world_size(process_group)
        self._barrier_channel = barrier_channel
        self._barrier_timeout_ms = barrier_timeout_ms

        self._owner_token = object()
        self._device: torch.device | None = None
        self._compute_stream: torch.cuda.Stream | None = None
        self._comm_stream: torch.cuda.Stream | None = None
        self._mem_pool: torch.cuda.MemPool | None = None
        self._slot_cache: dict[_SlotKey, list[_BufferSlot]] = {}
        self._outstanding: list[AsyncUlyssesHandle] = []
        self._batch_key: _SlotKey | None = None
        self._reuse_pending = False
        if device is not None:
            self._bind_device(device)

    @property
    def world_size(self) -> int:
        return self._world_size

    def _bind_device(self, device: torch.device) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"Async Ulysses requires a CUDA device, got device={device}")
        if device.index is None:
            device = torch.device("cuda", torch.accelerator.current_device_index())
        if self._device is None:
            self._validate_topology(device)
            backend = symm_mem.get_backend(device)
            if backend != "CUDA":
                raise RuntimeError(
                    "Async Ulysses requires the CUDA symmetric-memory backend, "
                    f"but torch.distributed._symmetric_memory is using {backend!r}"
                )
            self._device = device
            self._comm_stream = torch.cuda.Stream(device=device)
            self._mem_pool = torch.cuda.MemPool(
                symm_mem.get_mempool_allocator(device),
                use_on_oom=False,
                no_split=True,
            )
        elif device != self._device:
            raise ValueError(f"Async Ulysses exchanger is bound to {self._device}, got device={device}")

    def _validate_topology(self, device: torch.device) -> None:
        assert device.index is not None

        def stable_token(value: object) -> int:
            digest = hashlib.blake2b(str(value).encode(), digest_size=8).digest()
            return int.from_bytes(digest, byteorder="little", signed=True)

        def physical_device_id(index: int) -> object:
            properties = torch.cuda.get_device_properties(index)
            uuid = getattr(properties, "uuid", None)
            if uuid is not None:
                return uuid
            return (
                properties.pci_domain_id,
                properties.pci_bus_id,
                properties.pci_device_id,
            )

        hostname_token = stable_token(socket.gethostname())
        device_token = stable_token(physical_device_id(device.index))
        local_descriptor = torch.tensor(
            [hostname_token, device_token],
            dtype=torch.int64,
            device=device,
        )
        descriptors = [torch.empty_like(local_descriptor) for _ in range(self._world_size)]
        dist.all_gather(descriptors, local_descriptor, group=self._process_group)
        topology = [tuple(int(value) for value in descriptor.cpu().tolist()) for descriptor in descriptors]

        if len({hostname for hostname, _ in topology}) != 1:
            raise RuntimeError("Async Ulysses currently requires every process-group rank to be on one node")

        peer_device_tokens = [peer_device for _, peer_device in topology]
        if len(set(peer_device_tokens)) != self._world_size:
            raise RuntimeError("Async Ulysses requires one distinct physical CUDA device per process-group rank")

        local_device_indices = {
            stable_token(physical_device_id(index)): index for index in range(torch.accelerator.device_count())
        }
        missing_ranks = [rank for rank, token in enumerate(peer_device_tokens) if token not in local_device_indices]
        if missing_ranks:
            raise RuntimeError(
                "Async Ulysses requires every worker to expose every process-group CUDA device; "
                f"devices for group ranks {missing_ranks} are not visible"
            )
        inaccessible = [
            rank
            for rank, token in enumerate(peer_device_tokens)
            if local_device_indices[token] != device.index
            and not torch.cuda.can_device_access_peer(device.index, local_device_indices[token])
        ]
        if inaccessible:
            raise RuntimeError(
                "Async Ulysses requires CUDA peer access between every process-group device; "
                f"the local device cannot access devices for group ranks {inaccessible}"
            )

    def _validate_input(self, tensor: torch.Tensor) -> None:
        if tensor.device.type != "cuda":
            raise ValueError(f"Async Ulysses requires a CUDA tensor, got device={tensor.device}")
        if tensor.ndim != 4:
            raise ValueError(f"Async Ulysses requires a rank-four [B, S, H, D] tensor, got shape={tensor.shape}")
        if tensor.shape[2] % self._world_size != 0:
            raise ValueError(
                "Strict async Ulysses requires the head count to be divisible by the process-group size, "
                f"got heads={tensor.shape[2]}, world_size={self._world_size}"
            )

        self._bind_device(tensor.device)
        if tensor.device != self._device:
            raise ValueError(f"Async Ulysses exchanger is bound to {self._device}, got tensor on {tensor.device}")

        current_stream = torch.cuda.current_stream(tensor.device)
        if self._compute_stream is None:
            self._compute_stream = current_stream
        elif current_stream != self._compute_stream:
            raise RuntimeError("Async Ulysses issue and joined-output consumers must use one compute stream")

    def _acquire_slot(self, tensor: torch.Tensor) -> _BufferSlot:
        batch_size, local_seq_len, head_count, head_dim = tensor.shape
        key = _SlotKey(
            shape=(batch_size, local_seq_len, head_count, head_dim),
            dtype=tensor.dtype,
            device_index=tensor.device.index,
        )
        if self._batch_key is None:
            self._batch_key = key
        elif key != self._batch_key:
            raise ValueError(
                "Async Ulysses requires Q/K/V in one issue batch to have matching shapes, dtypes, and devices"
            )

        slots = self._slot_cache.get(key)
        if slots is None:
            # Allocate the complete V/Q/K bank before scheduling the first peer
            # copy. CUDA-IPC mappings must not be changed while a copy to an
            # earlier symmetric allocation is in flight.
            if self._slot_cache:
                assert not self._outstanding
                # The previous join completed all peer writes. Drain local
                # consumers before dropping the old pool; rendezvous below is
                # itself collective for the new shape.
                torch.cuda.current_stream(tensor.device).synchronize()
                self._slot_cache.clear()
                self._mem_pool = torch.cuda.MemPool(
                    symm_mem.get_mempool_allocator(tensor.device),
                    use_on_oom=False,
                    no_split=True,
                )
                self._reuse_pending = False
            slots = []
            for _ in range(3):
                assert self._mem_pool is not None
                with torch.cuda.use_mem_pool(self._mem_pool, device=tensor.device):
                    output = torch.empty(
                        (batch_size, self._world_size * local_seq_len, head_count // self._world_size, head_dim),
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                slots.append(
                    _BufferSlot(
                        output=output,
                        symm_handle=symm_mem.rendezvous(output, self._process_group),
                    )
                )
            self._slot_cache[key] = slots

        for slot in slots:
            if not slot.busy:
                slot.busy = True
                return slot
        raise RuntimeError("Async Ulysses supports at most three in-flight tensors per issue batch")

    @torch.compiler.disable
    def issue(self, tensor: torch.Tensor) -> AsyncUlyssesHandle:
        """Queue one forward Ulysses exchange and return its in-flight handle."""
        self._validate_input(tensor)

        if self._world_size == 1:
            handle = AsyncUlyssesHandle(
                _owner_token=self._owner_token,
                _slot=None,
                _output=tensor,
                _ready_event=None,
            )
            self._outstanding.append(handle)
            return handle

        assert self._comm_stream is not None
        slot = self._acquire_slot(tensor)
        batch_size, local_seq_len, head_count, head_dim = tensor.shape
        local_head_count = head_count // self._world_size

        try:
            # Pack the destination-rank head slices while staying on the
            # caller's compute stream. The following event covers both this
            # pack and any earlier consumer of a cached output slot.
            packed = (
                tensor.reshape(batch_size, local_seq_len, self._world_size, local_head_count, head_dim)
                .permute(2, 0, 1, 3, 4)
                .contiguous()
            )
            ready_event = torch.cuda.Event()
            assert self._compute_stream is not None
            ready_event.record(self._compute_stream)

            # The caching allocator must know about the side-stream read even
            # if scheduling a later peer copy raises and issue() exits early.
            packed.record_stream(self._comm_stream)
            handle = AsyncUlyssesHandle(
                _owner_token=self._owner_token,
                _slot=slot,
                _output=slot.output,
                _ready_event=ready_event,
            )
        except Exception:
            slot.busy = False
            if not self._outstanding:
                self._batch_key = None
            raise
        # Register the slot before scheduling peer work so abort() can drain a
        # partially issued exchange if any later operation raises.
        self._outstanding.append(handle)

        with torch.cuda.stream(self._comm_stream):
            self._comm_stream.wait_event(ready_event)
            if self._reuse_pending:
                # Every rank reaches this after its previous output consumers.
                # Complete the handshake before any peer overwrites the bank.
                slot.symm_handle.barrier(
                    channel=self._barrier_channel,
                    timeout_ms=self._barrier_timeout_ms,
                )
                self._reuse_pending = False
            output_batch_stride = self._world_size * local_seq_len * local_head_count * head_dim
            output_rank_stride = local_seq_len * local_head_count * head_dim
            handle_offset_bytes = slot.symm_handle.offset
            element_size = tensor.element_size()
            if handle_offset_bytes % element_size != 0:
                raise RuntimeError("Symmetric-memory allocation offset is not aligned to the tensor element size")
            for peer in range(self._world_size):
                for batch_idx in range(batch_size):
                    remote_offset = batch_idx * output_batch_stride + self._rank * output_rank_stride
                    remote_output = slot.symm_handle.get_buffer(
                        peer,
                        [local_seq_len, local_head_count, head_dim],
                        tensor.dtype,
                        handle_offset_bytes // element_size + remote_offset,
                    )
                    remote_output.copy_(packed[peer, batch_idx], non_blocking=True)

        return handle

    @torch.compiler.disable
    def join(self, handles: Sequence[AsyncUlyssesHandle]) -> tuple[torch.Tensor, ...]:
        """Join the current issue batch and return outputs in handle order."""
        handles = tuple(handles)
        if not handles:
            raise ValueError("Async Ulysses join requires at least one handle")
        if len(handles) != len(self._outstanding) or {id(handle) for handle in handles} != {
            id(handle) for handle in self._outstanding
        }:
            raise RuntimeError("Async Ulysses join must contain every outstanding handle exactly once")
        for handle in handles:
            if handle._owner_token is not self._owner_token:
                raise ValueError("Cannot join a handle issued by a different async Ulysses exchanger")
            if handle._joined:
                raise RuntimeError("Async Ulysses handle has already been joined")

        if self._world_size > 1:
            assert self._device is not None
            assert self._compute_stream is not None
            assert self._comm_stream is not None
            if torch.cuda.current_stream(self._device) != self._compute_stream:
                raise RuntimeError("Async Ulysses issue, join, and joined-output consumers must use one compute stream")

            # All copies share this stream, so one cross-rank handshake after
            # its FIFO covers the complete V/Q/K batch.
            with torch.cuda.stream(self._comm_stream):
                slot = self._outstanding[0]._slot
                assert slot is not None
                slot.symm_handle.barrier(
                    channel=self._barrier_channel,
                    timeout_ms=self._barrier_timeout_ms,
                )
                done_event = torch.cuda.Event()
                done_event.record(self._comm_stream)
            self._compute_stream.wait_event(done_event)
            self._reuse_pending = True

        outputs = []
        for handle in handles:
            assert handle._output is not None
            outputs.append(handle._output)
        for handle in self._outstanding:
            if handle._slot is not None:
                handle._slot.busy = False
            handle._joined = True
            handle._ready_event = None
            handle._slot = None
            handle._output = None
        self._outstanding.clear()
        self._batch_key = None
        return tuple(outputs)

    @torch.compiler.disable
    def abort(self) -> None:
        """Drain and discard a partially issued exchange batch."""
        if not self._outstanding:
            self._batch_key = None
            return

        self.join(tuple(self._outstanding))
        if self._device is not None:
            torch.accelerator.synchronize(self._device)

    @torch.compiler.disable
    def close(self) -> None:
        """Release symmetric-memory slots before their process group dies."""
        if self._outstanding:
            raise RuntimeError("Cannot close async Ulysses with outstanding exchanges")
        if self._device is not None:
            torch.accelerator.synchronize(self._device)
        self._slot_cache.clear()
        self._mem_pool = None
        self._batch_key = None
        self._reuse_pending = False
