"""Fused CUDA kernel for MiniMax H3 Q/K normalization and rotary embedding."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _rms_norm_rope_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    token_stride: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_half: tl.constexpr,
    eps: tl.constexpr,
    heads_per_program: tl.constexpr,
):
    token = tl.program_id(0)
    head_group = tl.program_id(1)
    heads = head_group * heads_per_program + tl.arange(0, heads_per_program)
    dims = tl.arange(0, head_dim)
    mask = heads[:, None] < num_heads
    offsets = token * token_stride + heads[:, None] * head_dim + dims[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + dims).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
    normalized = (x * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

    rotary_dim = rotary_half * 2
    pair_dims = tl.where(
        dims < rotary_half,
        dims + rotary_half,
        tl.where(dims < rotary_dim, dims - rotary_half, dims),
    )
    pair_offsets = token * token_stride + heads[:, None] * head_dim + pair_dims[None, :]
    pair_x = tl.load(x_ptr + pair_offsets, mask=mask, other=0.0).to(tl.float32)
    pair_weight = tl.load(weight_ptr + pair_dims).to(tl.float32)
    pair_normalized = (pair_x * inv_rms[:, None] * pair_weight[None, :]).to(tl.bfloat16)

    freq_dims = tl.where(dims < rotary_half, dims, tl.where(dims < rotary_dim, dims - rotary_half, 0))
    cos = tl.load(cos_ptr + token * rotary_half + freq_dims).to(tl.float32)
    sin = tl.load(sin_ptr + token * rotary_half + freq_dims).to(tl.float32)
    first = normalized.to(tl.float32) * cos - pair_normalized.to(tl.float32) * sin
    second = normalized.to(tl.float32) * cos + pair_normalized.to(tl.float32) * sin
    output = tl.where(
        dims < rotary_dim,
        tl.where(dims < rotary_half, first, second),
        normalized.to(tl.float32),
    )

    out_offsets = (token * num_heads + heads[:, None]) * head_dim + dims[None, :]
    tl.store(out_ptr + out_offsets, output, mask=mask)


@triton_op("vllm_omni::minimax_h3_rms_norm_rope", mutates_args={})
def minimax_h3_rms_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply MiniMax H3 per-head RMSNorm followed by its partial NeoX RoPE."""
    cos = cos.contiguous()
    sin = sin.contiguous()
    tokens, heads, head_dim = x.shape
    rotary_half = cos.shape[-1]
    output = torch.empty((tokens, heads, head_dim), dtype=x.dtype, device=x.device)
    heads_per_program = 8
    wrap_triton(_rms_norm_rope_kernel)[(tokens, triton.cdiv(heads, heads_per_program))](
        x,
        weight,
        cos,
        sin,
        output,
        x.stride(0),
        heads,
        head_dim,
        rotary_half,
        eps,
        heads_per_program,
        num_warps=8,
    )
    return output


__all__ = ["minimax_h3_rms_norm_rope"]
