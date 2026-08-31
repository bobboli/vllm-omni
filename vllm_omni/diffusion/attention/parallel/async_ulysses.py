# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory  # noqa: F401
import torch.nn as nn


class UlyssesAllGatherQKVLinear(nn.Module):
    """Dense QKV projection fused with a sequence all-gather.

    The input is sharded over packed sequence rows. Each Ulysses rank stores
    the Q, K, and V head rows assigned to that rank, packed as ``[Q | K | V]``.
    The symmetric-memory operator gathers the input rows while each rank
    computes only its local attention heads.
    """

    lora_unsupported_reason = (
        "the projection weight is sharded over Ulysses heads and consumed by a fused all-gather/GEMM operator"
    )

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        process_group: dist.ProcessGroup,
        *,
        params_dtype: torch.dtype = torch.bfloat16,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise ValueError(
                f"UlyssesAllGatherQKVLinear supports dense BF16 weights only, got params_dtype={params_dtype}"
            )
        if hidden_size <= 0 or head_size <= 0:
            raise ValueError(f"hidden_size and head_size must be positive, got {hidden_size=} and {head_size=}")
        if not hasattr(torch.ops.symm_mem, "fused_all_gather_matmul"):
            raise RuntimeError("this PyTorch build does not provide symm_mem.fused_all_gather_matmul")

        world_size = dist.get_world_size(process_group)
        rank = dist.get_rank(process_group)
        if world_size <= 1:
            raise ValueError("UlyssesAllGatherQKVLinear requires a Ulysses process group with more than one rank")
        if total_num_heads % world_size:
            raise ValueError(
                f"query head count must be divisible by the Ulysses world size: {total_num_heads} % {world_size} != 0"
            )
        if total_num_kv_heads % world_size:
            raise ValueError(
                "key/value head count must be divisible by the Ulysses world size: "
                f"{total_num_kv_heads} % {world_size} != 0"
            )

        group_name = getattr(process_group, "group_name", None)
        if group_name is None:
            raise ValueError("the Ulysses process group must expose a group_name")

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = total_num_heads // world_size
        self.num_kv_heads = total_num_kv_heads // world_size
        self.world_size = world_size
        self.rank = rank
        self.group_name = group_name
        self.prefix = prefix

        self._global_partition_sizes = (
            total_num_heads * head_size,
            total_num_kv_heads * head_size,
            total_num_kv_heads * head_size,
        )
        self._local_partition_sizes = tuple(size // world_size for size in self._global_partition_sizes)
        local_output_size = sum(self._local_partition_sizes)
        weight = nn.Parameter(
            torch.empty(local_output_size, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        self.register_parameter("weight", weight)

    def weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        """Load a full ``[Q | K | V]`` weight into this rank's head shard."""
        expected_shape = (sum(self._global_partition_sizes), self.hidden_size)
        if tuple(loaded_weight.shape) != expected_shape:
            raise ValueError(
                "QKV weight must be the full reordered [Q | K | V] tensor: "
                f"got {tuple(loaded_weight.shape)}, expected {expected_shape}"
            )
        if tuple(param.shape) != (sum(self._local_partition_sizes), self.hidden_size):
            raise ValueError(
                f"QKV destination has unexpected shape {tuple(param.shape)} for {self.prefix or '<unnamed>'}"
            )

        source_offset = 0
        destination_offset = 0
        with torch.no_grad():
            for global_size, local_size in zip(self._global_partition_sizes, self._local_partition_sizes):
                source_start = source_offset + self.rank * local_size
                param[destination_offset : destination_offset + local_size].copy_(
                    loaded_weight[source_start : source_start + local_size]
                )
                source_offset += global_size
                destination_offset += local_size

    def _validate_input(self, input_: torch.Tensor) -> None:
        if input_.ndim != 2 or input_.shape[1] != self.hidden_size:
            raise ValueError(
                f"UlyssesAllGatherQKVLinear input must be [local_tokens, hidden_size], got {tuple(input_.shape)}"
            )
        if input_.device.type != "cuda":
            raise ValueError(f"UlyssesAllGatherQKVLinear requires a CUDA input, got {input_.device}")
        if not input_.is_contiguous():
            raise ValueError("UlyssesAllGatherQKVLinear requires a contiguous input")
        if input_.dtype != torch.bfloat16 or self.weight.dtype != torch.bfloat16:
            raise ValueError(
                "UlyssesAllGatherQKVLinear supports dense BF16 inputs and weights only, "
                f"got input={input_.dtype}, weight={self.weight.dtype}"
            )

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, None]:
        self._validate_input(input_)
        _, outputs = torch.ops.symm_mem.fused_all_gather_matmul(
            input_,
            [self.weight.t()],
            gather_dim=0,
            group_name=self.group_name,
            return_A=False,
        )
        return outputs[0], None


__all__ = ["UlyssesAllGatherQKVLinear"]
