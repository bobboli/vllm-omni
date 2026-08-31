#!/usr/bin/env python3
"""Four-GPU correctness and compile smoke for Omni's Ulysses AGMM module."""

from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.parallel.async_ulysses import (
    UlyssesAllGatherQKVLinear,
)


def sequence_to_heads(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reference strict-Ulysses [S_local,H,D] -> [S_global,H_local,D]."""
    local_tokens, num_heads, head_dim = tensor.shape
    local_heads = num_heads // world_size
    send = (
        tensor.reshape(local_tokens, world_size, local_heads, head_dim)
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send)
    return receive.reshape(local_tokens * world_size, local_heads, head_dim)


def heads_to_sequence(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Reference strict-Ulysses reverse [S_global,H_local,D] -> [S_local,H,D]."""
    global_tokens, local_heads, head_dim = tensor.shape
    local_tokens = global_tokens // world_size
    send = (
        tensor.reshape(world_size, local_tokens, local_heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send)
    return (
        receive.reshape(world_size * local_heads, local_tokens, head_dim)
        .permute(1, 0, 2)
        .contiguous()
    )


def check(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    difference = actual_fp32 - expected_fp32
    max_abs = difference.abs().max()
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        expected_fp32
    )
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    dist.all_reduce(relative_l2, op=dist.ReduceOp.MAX)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    if dist.get_rank() == 0:
        print(
            f"PASS {name}: shape={tuple(actual.shape)} "
            f"max_abs={max_abs.item():.6g} rel_l2={relative_l2.item():.6g}",
            flush=True,
        )


def run_projection(
    projection: UlyssesAllGatherQKVLinear,
    local_x: torch.Tensor,
    full_weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    world_size: int,
) -> torch.Tensor:
    local_tokens = local_x.shape[0]
    inner_size = num_heads * head_dim
    local_heads = num_heads // world_size

    packed, bias = projection(local_x)
    assert bias is None
    candidate = tuple(
        part.reshape(local_tokens * world_size, local_heads, head_dim)
        for part in packed.split(inner_size // world_size, dim=-1)
    )

    local_projected = torch.mm(local_x, full_weight.t())
    local_qkv = tuple(
        part.reshape(local_tokens, num_heads, head_dim)
        for part in local_projected.split(inner_size, dim=-1)
    )
    reference = tuple(sequence_to_heads(part, world_size) for part in local_qkv)
    for label, actual, expected in zip(("q", "k", "v"), candidate, reference):
        check(f"eager/{label}", actual, expected)

    restored_q = heads_to_sequence(candidate[0], world_size)
    check("reverse_a2a_order", restored_q, local_qkv[0])
    return packed


def run_compile_and_graph(
    projection: UlyssesAllGatherQKVLinear,
    local_x: torch.Tensor,
    eager_output: torch.Tensor,
) -> None:
    compiled = torch.compile(projection, fullgraph=True, dynamic=False)
    compiled_output, compiled_bias = compiled(local_x)
    assert compiled_bias is None
    check("torch_compile/fullgraph", compiled_output, eager_output)

    static_x = local_x.clone()
    static_output, _ = projection(static_x)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output, _ = projection(static_x)
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    check("cuda_graph/replay", static_output, eager_output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-tokens", type=int, default=18_496)
    parser.add_argument("--hidden-size", type=int, default=5_376)
    parser.add_argument("--num-heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    if args.num_heads % world_size:
        raise ValueError("num_heads must be divisible by world size")

    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    inner_size = args.num_heads * args.head_dim
    torch.manual_seed(args.seed)
    full_weight = torch.randn(
        3 * inner_size,
        args.hidden_size,
        dtype=dtype,
        device=device,
    )
    full_weight.mul_(1.0 / math.sqrt(args.hidden_size))
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000 + local_rank)
    local_x = torch.randn(
        args.local_tokens,
        args.hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )

    projection = UlyssesAllGatherQKVLinear(
        hidden_size=args.hidden_size,
        head_size=args.head_dim,
        total_num_heads=args.num_heads,
        total_num_kv_heads=args.num_heads,
        process_group=group,
        params_dtype=dtype,
        prefix="blocks.0.attn.qkv_proj",
    ).to(device)
    projection.weight_loader(projection.weight, full_weight)
    expected_local_rows = 3 * inner_size // world_size
    assert projection.weight.shape == (expected_local_rows, args.hidden_size)
    wrapper = torch.nn.Module()
    wrapper.qkv_proj = projection
    assert tuple(wrapper.state_dict()) == ("qkv_proj.weight",)

    # Exercise symmetric-workspace creation and growth before the model shape.
    small_x = local_x[:64].contiguous()
    small_output, _ = projection(small_x)
    assert small_output.shape == (64 * world_size, expected_local_rows)
    dist.barrier()

    if local_rank == 0:
        print(
            f"shape local_x={tuple(local_x.shape)} full_weight={tuple(full_weight.shape)} "
            f"local_weight={tuple(projection.weight.shape)} group={group.group_name!r}",
            flush=True,
        )

    eager_output = run_projection(
        projection,
        local_x,
        full_weight,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        world_size=world_size,
    )
    run_compile_and_graph(projection, local_x, eager_output)
    dist.barrier()
    if local_rank == 0:
        print("ALL CHECKS PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
