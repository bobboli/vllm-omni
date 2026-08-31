# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.async_ulysses import (
    UlyssesAllGatherQKVLinear,
)
from vllm_omni.diffusion.attention.parallel.base import (
    NoParallelAttention,
    ParallelAttentionContext,
)
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeProcessGroup:
    group_name = "test-ulysses"


def _make_projection(monkeypatch: pytest.MonkeyPatch, *, rank: int = 1) -> UlyssesAllGatherQKVLinear:
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group: rank)
    return UlyssesAllGatherQKVLinear(
        hidden_size=2,
        head_size=1,
        total_num_heads=4,
        total_num_kv_heads=4,
        process_group=_FakeProcessGroup(),  # type: ignore[arg-type]
    )


def test_qkv_loader_packs_this_ranks_rows_from_each_partition(monkeypatch: pytest.MonkeyPatch):
    projection = _make_projection(monkeypatch, rank=1)
    full_weight = torch.arange(24, dtype=torch.bfloat16).view(12, 2)

    projection.weight_loader(projection.weight, full_weight)

    expected_rows = torch.tensor([2, 3, 6, 7, 10, 11])
    torch.testing.assert_close(projection.weight, full_weight.index_select(0, expected_rows))
    assert projection.weight.shape == (6, 2)
    assert not projection.weight.requires_grad
    assert projection.num_heads == projection.num_kv_heads == 2


def test_qkv_forward_dispatches_one_packed_weight_to_public_symm_op(monkeypatch: pytest.MonkeyPatch):
    projection = _make_projection(monkeypatch, rank=0)
    full_weight = torch.arange(24, dtype=torch.bfloat16).view(12, 2)
    projection.weight_loader(projection.weight, full_weight)
    input_ = torch.arange(6, dtype=torch.bfloat16).view(3, 2)
    call = {}

    def fake_fused_all_gather_matmul(input_shard, weights, gather_dim, group_name, **kwargs):
        call.update(
            A=input_shard,
            Bs=weights,
            gather_dim=gather_dim,
            group_name=group_name,
            return_A=kwargs["return_A"],
        )
        gathered = torch.cat((input_shard, input_shard + 1), dim=0)
        return None, [gathered @ weights[0]]

    monkeypatch.setattr(
        torch.ops.symm_mem,
        "fused_all_gather_matmul",
        fake_fused_all_gather_matmul,
    )
    monkeypatch.setattr(UlyssesAllGatherQKVLinear, "_validate_input", lambda self, input_: None)
    output, bias = projection(input_)

    assert bias is None
    assert call["A"] is input_
    assert len(call["Bs"]) == 1
    assert call["Bs"][0].untyped_storage().data_ptr() == projection.weight.untyped_storage().data_ptr()
    assert call["gather_dim"] == 0
    assert call["group_name"] == "test-ulysses"
    assert call["return_A"] is False
    expected = torch.cat((input_, input_ + 1), dim=0) @ projection.weight.t()
    torch.testing.assert_close(output, expected)


@dataclass
class _FakeSPGroup:
    ulysses_group: object
    ulysses_world_size: int = 2
    ring_world_size: int = 1


def test_pre_sharded_qkv_builds_only_reverse_context_and_preserves_metadata(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.diffusion.attention.parallel.ulysses as ulysses_module

    monkeypatch.setattr(ulysses_module, "get_ulysses_mode", lambda default: "strict")
    process_group = object()
    strategy = UlyssesParallelAttention(
        sp_group=_FakeSPGroup(process_group),  # type: ignore[arg-type]
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
        pre_sharded_qkv=True,
    )
    q = torch.randn(1, 8, 2, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = torch.ones(1, 8, dtype=torch.int32)
    metadata = AttentionMetadata(attn_mask=mask, extra={"cu_seqlens_q": torch.tensor([0, 8])})

    actual_q, actual_k, actual_v, actual_metadata, ctx = strategy.pre_attention_from_ulysses_qkv(
        q,
        k,
        v,
        attn_metadata=metadata,
    )

    assert actual_q is q and actual_k is k and actual_v is v
    assert actual_metadata is not metadata
    assert actual_metadata.extra is metadata.extra
    assert actual_metadata.attn_mask.dtype == torch.bool
    assert ctx.ulysses_pg is process_group
    assert (ctx.scatter_idx, ctx.gather_idx, ctx.joint_len) == (2, 1, 0)

    reverse_call = {}

    def fake_reverse(group, tensor, scatter_idx, gather_idx, use_sync):
        reverse_call.update(
            group=group,
            tensor=tensor,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
        )
        return tensor

    monkeypatch.setattr(ulysses_module.SeqAllToAll4D, "apply", staticmethod(fake_reverse))
    output = torch.randn_like(q)
    assert strategy.post_attention(output, ctx) is output
    assert reverse_call == {
        "group": process_group,
        "tensor": output,
        "scatter_idx": 1,
        "gather_idx": 2,
        "use_sync": False,
    }


@dataclass(frozen=True, slots=True)
class _TestContext(ParallelAttentionContext):
    pass


class _PreparedStrategy:
    name = "ulysses"

    def __init__(self):
        self.context = _TestContext(name=self.name)
        self.prepared = False
        self.posted_context = None

    def pre_attention_from_ulysses_qkv(self, query, key, value, *, attn_metadata):
        self.prepared = True
        return query, key, value, attn_metadata, self.context

    def post_attention(self, output, ctx):
        self.posted_context = ctx
        return output + 1


class _TestAttentionImpl(nn.Module):
    def forward(self, query, key, value, attn_metadata):
        del key, value, attn_metadata
        return query * 2


class _TestBackend:
    supports_piecewise_spans = True

    @staticmethod
    def get_name():
        return "TEST"


def _make_attention_dispatch(strategy) -> Attention:
    attention = object.__new__(Attention)
    nn.Module.__init__(attention)
    attention.skip_sequence_parallel = False
    attention.parallel_strategy = strategy
    attention._no_parallel_strategy = NoParallelAttention()
    attention.use_ring = False
    attention.attention = _TestAttentionImpl()
    attention.attn_backend = _TestBackend
    attention.backend_pref = "TEST"
    attention._kv_cache_dtype = None
    attention._disable_kv_quant = False
    return attention


def test_attention_dispatch_uses_pre_sharded_hook_and_reverse_context():
    strategy = _PreparedStrategy()
    attention = _make_attention_dispatch(strategy)
    q = torch.randn(1, 8, 2, 4, dtype=torch.bfloat16)

    output = attention.forward_from_ulysses_qkv(q, q, q)

    assert strategy.prepared
    assert strategy.posted_context is strategy.context
    torch.testing.assert_close(output, q * 2 + 1)


def test_attention_dispatch_rejects_inactive_sequence_parallelism():
    attention = _make_attention_dispatch(NoParallelAttention())
    q = torch.randn(1, 8, 2, 4, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="requires an active Ulysses"):
        attention.forward_from_ulysses_qkv(q, q, q)


def test_pre_sharded_qkv_rejects_joint_metadata(monkeypatch: pytest.MonkeyPatch):
    import vllm_omni.diffusion.attention.parallel.ulysses as ulysses_module

    monkeypatch.setattr(ulysses_module, "get_ulysses_mode", lambda default: "strict")
    strategy = UlyssesParallelAttention(
        sp_group=_FakeSPGroup(object()),  # type: ignore[arg-type]
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
        pre_sharded_qkv=True,
    )
    q = torch.randn(1, 8, 2, 4, dtype=torch.bfloat16)
    metadata = AttentionMetadata(joint_query=torch.randn(1, 2, 2, 4))

    with pytest.raises(ValueError, match="does not support joint attention metadata"):
        strategy.pre_attention_from_ulysses_qkv(q, q, q, attn_metadata=metadata)
