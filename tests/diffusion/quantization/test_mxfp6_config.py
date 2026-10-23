# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the ROCm W6A6 MXFP6 (E2M3) online quantization config (AITER A6W6).

The CPU tests mock the platform check, the gcnArchName probe and the AITER kernels, so they only cover
registration, dispatch, weight handling and the apply plumbing. The real AITER kernels are exercised by the
gfx950 test at the bottom of this file.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]
cpu = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _patch_tp_state(monkeypatch):
    """Let ModelWeightParameter be created on CPU without an initialized distributed group."""
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)


@pytest.fixture
def _rocm_platform(monkeypatch):
    """Make current_omni_platform report ROCm and stub the aiter op registration."""
    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization import mxfp6_config

    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(mxfp6_config, "_register_rocm_mxfp6_op", lambda: None)


def _patch_gcn_arch(monkeypatch, arch: str) -> None:
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: SimpleNamespace(gcnArchName=arch))


def _fake_linear_layer():
    from vllm.model_executor.layers.linear import LinearBase

    return MagicMock(spec=LinearBase)


# ---------------------------------------------------------------------------
# Config + registration
# ---------------------------------------------------------------------------


@cpu
def test_mxfp6_config_get_name():
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config

    assert DiffusionMXFP6Config.get_name() == "mxfp6"


@cpu
def test_mxfp6_is_registered_in_factory():
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.factory import SUPPORTED_QUANTIZATION_METHODS
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config

    assert "mxfp6" in SUPPORTED_QUANTIZATION_METHODS
    config = build_quant_config({"method": "mxfp6", "ignored_layers": ["blocks.0.attn1.to_qkv"]})
    assert isinstance(config, DiffusionMXFP6Config)
    assert config.ignored_layers == ["blocks.0.attn1.to_qkv"]


@cpu
def test_mxfp6_component_config_routes_per_transformer():
    """The Wan2.2 benchmark recipe: transformer = mxfp4, transformer_2 = mxfp6 (with its own ignored layers)."""
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.component_config import resolve_component_quant_config
    from vllm_omni.quantization.mxfp4_config import DiffusionMXFP4Config
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config

    components = build_quant_config(
        {
            "transformer": {"method": "mxfp4"},
            "transformer_2": {"method": "mxfp6", "ignored_layers": ["blocks.3.ffn.net_2"]},
        }
    )
    assert isinstance(resolve_component_quant_config(components, "transformer"), DiffusionMXFP4Config)
    t2 = resolve_component_quant_config(components, "transformer_2")
    assert isinstance(t2, DiffusionMXFP6Config)
    assert t2.ignored_layers == ["blocks.3.ffn.net_2"]


@cpu
def test_mxfp6_from_config_reads_modules_to_not_convert():
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config

    assert DiffusionMXFP6Config.from_config({"modules_to_not_convert": ["proj_out"]}).ignored_layers == ["proj_out"]
    assert DiffusionMXFP6Config.from_config({}).ignored_layers == []


@cpu
def test_mxfp6_rejects_serialized_checkpoints():
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config

    with pytest.raises(NotImplementedError, match="online-only"):
        DiffusionMXFP6Config.from_config({"is_checkpoint_mxfp6_serialized": True})


# ---------------------------------------------------------------------------
# get_quant_method dispatch
# ---------------------------------------------------------------------------


@cpu
def test_rocm_gfx950_dispatch_returns_online_method(_rocm_platform, monkeypatch):
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config, ROCmMxfp6OnlineLinearMethod

    _patch_gcn_arch(monkeypatch, "gfx950:sramecc+:xnack-")
    method = DiffusionMXFP6Config().get_quant_method(_fake_linear_layer(), "blocks.0.attn1.to_q")
    assert isinstance(method, ROCmMxfp6OnlineLinearMethod)


@cpu
def test_ignored_layer_returns_unquantized_before_platform_probe(monkeypatch):
    """Ignored layers must be skipped without touching the platform / arch probes."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config

    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: False)
    config = DiffusionMXFP6Config(ignored_layers=["blocks.0.attn1.to_qkv"])
    assert isinstance(config.get_quant_method(_fake_linear_layer(), "blocks.0.attn1.to_qkv"), UnquantizedLinearMethod)
    with pytest.raises(NotImplementedError):  # a non-ignored layer still reaches the platform check
        config.get_quant_method(_fake_linear_layer(), "blocks.1.attn1.to_qkv")


@cpu
def test_non_linear_layer_returns_none(_rocm_platform):
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config

    assert DiffusionMXFP6Config().get_quant_method(torch.nn.LayerNorm(8), "norm") is None


@cpu
def test_non_rocm_platform_raises(monkeypatch):
    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config

    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: False)
    with pytest.raises(NotImplementedError, match="ROCm"):
        DiffusionMXFP6Config().get_quant_method(_fake_linear_layer(), "blocks.0.attn1.to_q")


@cpu
def test_non_gfx950_raises(_rocm_platform, monkeypatch):
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config

    _patch_gcn_arch(monkeypatch, "gfx942:sramecc+:xnack-")
    with pytest.raises(NotImplementedError, match="gfx950"):
        DiffusionMXFP6Config().get_quant_method(_fake_linear_layer(), "blocks.0.attn1.to_q")


# ---------------------------------------------------------------------------
# create_weights / process_weights_after_loading / apply  (aiter mocked)
# ---------------------------------------------------------------------------

_K, _N = 256, 128


class _FakeLayer(torch.nn.Module):
    """Bare nn.Module that accepts register_parameter without a real weight_loader."""


@cpu
def test_create_weights_registers_meta_bf16_weight(_rocm_platform):
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config, ROCmMxfp6OnlineLinearMethod

    layer = _FakeLayer()
    ROCmMxfp6OnlineLinearMethod(DiffusionMXFP6Config()).create_weights(
        layer=layer,
        input_size_per_partition=_K,
        output_partition_sizes=[_N],
        input_size=_K,
        output_size=_N,
        params_dtype=torch.bfloat16,
    )
    assert layer.weight.is_meta
    assert layer.weight.shape == (_N, _K)
    assert layer.weight.dtype == torch.bfloat16


# Records what the fake aiter functions below were called with.
_CALLS: dict = {}


def _install_fake_aiter(monkeypatch):
    def quant_mxfp6_gemm(t):
        _CALLS.setdefault("quant", []).append((tuple(t.shape), t.dtype))
        return torch.full((7,), t.shape[0], dtype=torch.uint8), torch.full((3,), t.shape[1] // 32, dtype=torch.uint8)

    def gemm_a6w6(a, w, a_scale, w_scale, M, N, K, dtype=torch.bfloat16, **kw):
        _CALLS.setdefault("gemm", []).append((tuple(a.shape), tuple(w.shape), M, N, K, dtype))
        return torch.full((M, N), 2.0, dtype=dtype)

    mod = types.ModuleType("aiter.ops.gemm_op_a6w6")
    mod.quant_mxfp6_gemm = quant_mxfp6_gemm
    mod.gemm_a6w6 = gemm_a6w6
    monkeypatch.setitem(sys.modules, "aiter", sys.modules.get("aiter", types.ModuleType("aiter")))
    monkeypatch.setitem(sys.modules, "aiter.ops", sys.modules.get("aiter.ops", types.ModuleType("aiter.ops")))
    monkeypatch.setitem(sys.modules, "aiter.ops.gemm_op_a6w6", mod)


@cpu
def test_process_weights_replaces_weight_with_packed_buffers(_rocm_platform, monkeypatch):
    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config, ROCmMxfp6OnlineLinearMethod

    _install_fake_aiter(monkeypatch)
    _CALLS.clear()
    layer = _FakeLayer()
    layer.register_parameter("weight", torch.nn.Parameter(torch.randn(_N, _K).bfloat16(), requires_grad=False))
    method = ROCmMxfp6OnlineLinearMethod(DiffusionMXFP6Config())

    method.process_weights_after_loading(layer)

    assert _CALLS["quant"] == [((_N, _K), torch.bfloat16)]
    assert layer.weight is None
    assert layer.weight_packed.dtype == torch.uint8 and layer.weight_scale.dtype == torch.uint8
    assert (layer.mxfp6_out_features, layer.mxfp6_in_features) == (_N, _K)
    assert {"weight_packed", "weight_scale"} <= set(layer.state_dict())

    method.process_weights_after_loading(layer)  # idempotent
    assert len(_CALLS["quant"]) == 1


@cpu
@pytest.mark.parametrize("with_bias", [False, True])
def test_apply_runs_quant_gemm_bias_and_restores_shape_dtype(monkeypatch, with_bias):
    """apply(): flatten -> activation quant + gemm_a6w6(M, N, K) (+ bias) -> reshape -> original dtype."""
    from vllm_omni.quantization import mxfp6_config

    _install_fake_aiter(monkeypatch)
    _CALLS.clear()
    if not hasattr(torch.ops.vllm_omni, "rocm_mxfp6_gemm"):
        mxfp6_config._register_rocm_mxfp6_op()  # the op imports aiter lazily, at call time

    method = mxfp6_config.ROCmMxfp6OnlineLinearMethod(mxfp6_config.DiffusionMXFP6Config())
    layer = _FakeLayer()
    layer.register_parameter("weight", torch.nn.Parameter(torch.randn(_N, _K).bfloat16(), requires_grad=False))
    method.process_weights_after_loading(layer)
    _CALLS["quant"].clear()

    x = torch.randn(2, 5, _K).bfloat16()
    bias = torch.arange(_N, dtype=torch.bfloat16) if with_bias else None
    out = method.apply(layer, x, bias)

    assert out.shape == (2, 5, _N) and out.dtype == torch.bfloat16
    assert _CALLS["quant"] == [((10, _K), torch.bfloat16)]  # activation quantized as (M, K)
    assert _CALLS["gemm"][-1][2:5] == (10, _N, _K)
    expected = torch.full((10, _N), 2.0, dtype=torch.bfloat16)
    if with_bias:
        expected = expected + bias
    torch.testing.assert_close(out.reshape(10, _N), expected)


# ---------------------------------------------------------------------------
# Real kernels (gfx950 only)
# ---------------------------------------------------------------------------


def _is_gfx950() -> bool:
    if not (torch.cuda.is_available() and torch.version.hip):
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


@pytest.mark.skipif(not _is_gfx950(), reason="requires AMD gfx950 (MI355X) and AITER")
@pytest.mark.parametrize("shape", [(64, 1024, 512), (257, 5120, 3 * 5120), (4096, 5120, 13824)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_gfx950_mxfp6_linear_matches_bf16_reference(shape, with_bias):
    """Real AITER A6W6 GEMM through the omni linear method stays close to the BF16 linear."""
    import torch.nn.functional as F

    from vllm_omni.quantization.mxfp6_config import DiffusionMXFP6Config, ROCmMxfp6OnlineLinearMethod

    torch.manual_seed(0)
    M, K, N = shape
    dev = torch.device("cuda:0")
    weight = (torch.randn(N, K, device=dev) * 0.05).bfloat16()
    bias = torch.randn(N, device=dev).bfloat16() if with_bias else None
    x = torch.randn(M, K, device=dev).bfloat16()

    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(weight.clone(), requires_grad=False))
    method = ROCmMxfp6OnlineLinearMethod(DiffusionMXFP6Config())
    method.process_weights_after_loading(layer)
    assert layer.weight is None

    out = method.apply(layer, x, bias)
    ref = F.linear(x, weight, bias)

    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert cos > 0.995, f"cosine similarity {cos:.5f}"
    assert rel < 0.1, f"relative error {rel:.4f}"
