# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_grouped_qkv_checkpoint_reorder():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        _reorder_grouped_qkv_to_qkv,
    )

    # Two groups with rows [q, k, v] become [q0, q1, k0, k1, v0, v1].
    grouped = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    reordered = _reorder_grouped_qkv_to_qkv(
        grouped,
        num_query_groups=2,
        heads_per_group=1,
        head_dim=1,
    )

    assert reordered[:, 0].tolist() == [0, 3, 1, 4, 2, 5]


def test_transformer_declares_cache_sp_layerwise_offload_and_hsdp():
    from cache_dit import ForwardPattern

    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    assert MiniMaxH3DiTModel._repeated_blocks == ["MiniMaxH3DiTBlock"]
    assert MiniMaxH3DiTModel._layerwise_offload_blocks_attrs == ["blocks"]
    assert MiniMaxH3DiTModel._cache_dit_adapter_config.block_forward_patterns["blocks"] == ForwardPattern.Pattern_3
    assert not MiniMaxH3DiTModel._cache_dit_adapter_config.has_separate_cfg
    assert set(MiniMaxH3DiTModel._sp_plan) == {"sp_prepare", "sp_gather"}

    model = object.__new__(MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.token_refiner = nn.Module()
    model.token_refiner.blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.final_layer = nn.Linear(4, 4)

    matched = [
        name
        for name, module in model.named_modules()
        if any(condition(name, module) for condition in MiniMaxH3DiTModel._hsdp_shard_conditions)
    ]
    assert matched == ["blocks.0", "blocks.1"]


def test_packed_attention_is_a_regional_compile_boundary():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    assert getattr(MiniMaxH3Attention._run_packed_attention, "_torchdynamo_disable", False)


def test_h3_fused_rope_matches_reference_and_preserves_unrotated_dims():
    from vllm_omni.diffusion.layers.rope import RotaryEmbedding
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    attention = object.__new__(MiniMaxH3Attention)
    nn.Module.__init__(attention)
    attention.rot_dim = 96
    attention.rope = RotaryEmbedding(is_neox_style=True, half_head_dim=False)
    attention.rope._forward_method = attention.rope.forward_native

    x = torch.randn(11, 3, 128, dtype=torch.bfloat16)
    freqs_half = torch.randn(11, 48)
    freqs = torch.cat((freqs_half, freqs_half), dim=-1)
    actual = attention._apply_rope(x, freqs)

    cos = torch.cos(freqs).to(x.dtype).unsqueeze(1)
    sin = torch.sin(freqs).to(x.dtype).unsqueeze(1)
    x_rot = x[..., :96]
    x1, x2 = x_rot.chunk(2, dim=-1)
    expected_rot = x_rot * cos + torch.cat((-x2, x1), dim=-1) * sin
    expected = torch.cat((expected_rot, x[..., 96:]), dim=-1)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[..., 96:], x[..., 96:], atol=0, rtol=0)


@pytest.mark.parametrize(
    ("tp_size", "message"),
    [
        (3, "num_attention_heads"),
        (5, "num_attention_heads"),
    ],
)
def test_tp_rejects_non_divisible_head_counts(tp_size, message):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    with pytest.raises(ValueError, match=message):
        model._validate_tp_config(
            arch=MiniMaxH3DiTArchConfig(),
            tp_size=tp_size,
        )


def test_tp_accepts_checkpoint_supported_sizes():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    arch = MiniMaxH3DiTArchConfig()
    for tp_size in (1, 2, 4, 7):
        model._validate_tp_config(arch=arch, tp_size=tp_size)


def test_async_ulysses_sp_plan_keeps_global_rope_rows():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    model.parallel_config = SimpleNamespace(
        async_ulysses=True,
        ulysses_degree=4,
        ulysses_mode="strict",
        ring_degree=1,
        use_hsdp=False,
    )
    model.od_config = SimpleNamespace(enable_distributed_layerwise_offload=False)

    model._configure_async_ulysses(tp_size=1, quant_config=None)

    assert set(model._sp_plan["sp_prepare"]) == {0, 2}
    assert set(MiniMaxH3DiTModel._sp_plan["sp_prepare"]) == {0, 1, 2}


@pytest.mark.parametrize(
    ("tp_size", "quant_config", "dlo", "message"),
    [
        (2, None, False, "tensor_parallel_size=1"),
        (1, object(), False, "unquantized model"),
        (1, None, True, "distributed layerwise offload"),
    ],
)
def test_async_ulysses_rejects_incompatible_projection_ownership(tp_size, quant_config, dlo, message):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3DiTModel,
    )

    model = object.__new__(MiniMaxH3DiTModel)
    model.parallel_config = SimpleNamespace(
        async_ulysses=True,
        ulysses_degree=4,
        ulysses_mode="strict",
        ring_degree=1,
        use_hsdp=False,
    )
    model.od_config = SimpleNamespace(enable_distributed_layerwise_offload=dlo)

    with pytest.raises(ValueError, match=message):
        model._configure_async_ulysses(tp_size=tp_size, quant_config=quant_config)


def test_async_ulysses_is_selected_only_for_main_blocks(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer as h3_module

    calls = []

    class _FakeAttention(nn.Module):
        def __init__(self, _arch, _quant_config, **kwargs):
            super().__init__()
            calls.append(kwargs)

    monkeypatch.setattr(h3_module, "_norm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(h3_module, "MiniMaxH3Attention", _FakeAttention)
    monkeypatch.setattr(h3_module, "MiniMaxH3MLP", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(h3_module, "MiniMaxH3AdalnProj", lambda *args, **kwargs: nn.Identity())
    arch = h3_module.MiniMaxH3DiTArchConfig()

    h3_module.MiniMaxH3DiTBlock(arch, None, prefix="blocks.0", async_ulysses=True)
    h3_module.MiniMaxH3TokenRefinerBlock(arch, None, prefix="token_refiner.blocks.0")

    assert calls[0]["async_ulysses"] is True
    assert "async_ulysses" not in calls[1]
    assert calls[1]["skip_sequence_parallel"] is True


def test_async_ulysses_requires_sequence_parallel_hooks(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.diffusion.registry as registry_module

    pipeline = nn.Module()
    pipeline._dit_modules = ("transformer",)
    pipeline.transformer = nn.Module()
    parallel_config = SimpleNamespace(
        sequence_parallel_size=4,
        async_ulysses=True,
        allgather_degree=1,
        ulysses_degree=4,
        ring_degree=1,
    )
    config = SimpleNamespace(parallel_config=parallel_config)
    context = SimpleNamespace(sp_plan_hooks_applied=None)
    monkeypatch.setattr(registry_module, "get_forward_context", lambda: context)

    with pytest.raises(RuntimeError, match="requires sequence-parallel hooks"):
        registry_module._apply_sequence_parallel_if_enabled(pipeline, config)
    assert context.sp_plan_hooks_applied is False
