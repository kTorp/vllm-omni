# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.flux2 import flux2_transformer as flux2_mod

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Norm:
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        self.weight = weight
        self.variance_epsilon = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.variance_epsilon) * self.weight.float()).to(x.dtype)


def test_flux2_qk_norm_rope_fuses_without_an_environment_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_USE_FUSED_QK_NORM_ROPE", raising=False)
    assert not hasattr(flux2_mod, "_USE_FUSED_QK_NORM_ROPE")

    calls = []
    real = flux2_mod.fused_qk_norm_rope

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(flux2_mod, "fused_qk_norm_rope", spy)

    torch.manual_seed(0)
    batch, seq, heads, head_dim = 2, 6, 3, 16
    query = torch.randn(batch, seq, heads, head_dim)
    key = torch.randn(batch, seq, heads, head_dim)
    norm_q = _Norm(torch.rand(head_dim) + 0.5)
    norm_k = _Norm(torch.rand(head_dim) + 0.5)
    # Adjacent-pair RoPE carries one cos/sin value per rotated pair.
    cos = torch.randn(seq, head_dim // 2)
    sin = torch.randn(seq, head_dim // 2)
    rope = RotaryEmbedding(is_neox_style=False)

    fused_q, fused_k = flux2_mod._flux2_qk_norm_rope(query, key, norm_q, norm_k, rope, (cos, sin))
    assert len(calls) == 1 and calls[0]["interleaved"] is True

    # The platform kernel is a CPU-incompatible Triton launch on ROCm, so the
    # reference uses the portable PyTorch rotation.
    ref_q = rope.forward_native(norm_q(query), cos, sin)
    ref_k = rope.forward_native(norm_k(key), cos, sin)
    torch.testing.assert_close(fused_q, ref_q, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(fused_k, ref_k, rtol=1e-4, atol=1e-4)


def test_flux2_qk_norm_rope_without_rotary_embeddings_only_normalizes(monkeypatch):
    def forbid(*_args, **_kwargs):
        raise AssertionError("fused op must not run without rotary embeddings")

    monkeypatch.setattr(flux2_mod, "fused_qk_norm_rope", forbid)

    head_dim = 8
    query = torch.randn(1, 4, 2, head_dim)
    key = torch.randn(1, 4, 2, head_dim)
    norm_q = _Norm(torch.ones(head_dim))
    norm_k = _Norm(torch.ones(head_dim))
    out_q, out_k = flux2_mod._flux2_qk_norm_rope(
        query, key, norm_q, norm_k, RotaryEmbedding(is_neox_style=False), None
    )
    torch.testing.assert_close(out_q, norm_q(query))
    torch.testing.assert_close(out_k, norm_k(key))
