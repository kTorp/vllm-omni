# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""W6A6 MXFP6 (Microscaling FP6, E2M3) online quantization for diffusion transformers on ROCm.

Architecture mirrors the ROCm MXFP4 path in mxfp4_config.py:

  MXFPLinearMethodBase              – platform-agnostic skeleton (imported from mxfp8_config)
    ROCmMxfp6LinearMethod           – ROCm base class (AITER quant; online-only)
      ROCmMxfp6OnlineLinearMethod   – ROCm online (BF16 → MXFP6 via AITER)

  DiffusionMXFP6Config              – online config (quant_method="mxfp6")

MX (microscaling) format: groups of 32 K-dimension elements share one float8_e8m0fnu exponent
scale. Weights and activations are FP6 E2M3 (OCP MX spec).

Kernels: ``aiter.ops.gemm_op_a6w6`` (gfx950 / MI355X only).

  quant_mxfp6_gemm(t)  -> (packed uint8, packed_scale uint8)   works for activation and weight
  gemm_a6w6(a, w, a_scale, w_scale, M, N, K, dtype=bf16)       D = A @ W^T, bf16 output

The packed layout is opaque (not a plain FP6 tensor) and is only meaningful to ``gemm_a6w6``. AITER's
quantizer also applies a block-diagonal 32-point Hadamard rotation along K to both operands; this is part
of the packed-operand contract and leaves the GEMM result unchanged.

This is the same recipe as xDiT's ``xFuserMXFP6Linear`` (xfuser/model_executor/layers/mxfp6_linear.py),
which is what ``gemm_quantization low=fp4,high=fp6`` uses for Wan2.2's second transformer, so the two
frameworks run the same quantization and the same GEMM kernels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization._copy_missing_attrs import (
    copy_missing_attrs as _copy_missing_attrs,
)
from vllm_omni.quantization.mxfp8_config import (
    MXFPLinearMethodBase,
    _LazyWeightMixin,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DiffusionMXFP6Config(QuantizationConfig):
    """W6A6 MXFP6 (E2M3) online quantization config for diffusion transformers (ROCm gfx950).

    Online only: a BF16 checkpoint is quantized at load time. Pre-quantized MXFP6 checkpoints
    are not supported.
    """

    def __init__(self, ignored_layers: list[str] | None = None) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp6"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # AITER's A6W6 GEMM produces bf16 only.
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionMXFP6Config:
        if config.get("is_checkpoint_mxfp6_serialized"):
            raise NotImplementedError("Pre-quantized MXFP6 checkpoints are not supported; MXFP6 is online-only.")
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(ignored_layers=ignored_layers)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not current_omni_platform.is_rocm():
                raise NotImplementedError(
                    "DiffusionMXFP6Config (W6A6 MXFP6) is currently only supported on ROCm (AMD, gfx950)."
                )
            gcn_arch = torch.cuda.get_device_properties(torch.accelerator.current_device_index()).gcnArchName
            if "gfx950" not in gcn_arch:
                raise NotImplementedError(f"MXFP6 on ROCm requires gfx950 (MI355X). Detected: {gcn_arch}")
            return ROCmMxfp6OnlineLinearMethod(self)
        return None


# ---------------------------------------------------------------------------
# ROCm MXFP6 (AITER A6W6)
# ---------------------------------------------------------------------------


def _register_rocm_mxfp6_op() -> None:
    """Register the vllm_omni::rocm_mxfp6_gemm custom op for torch.compile.

    Wraps activation quantization + GEMM (+ bias) in a single opaque op so that
    torch.compile/inductor doesn't try to trace through AITER internals.
    """

    @torch.library.custom_op("vllm_omni::rocm_mxfp6_gemm", mutates_args=())
    def _rocm_mxfp6_gemm(
        a: torch.Tensor,
        w_packed: torch.Tensor,
        w_scale: torch.Tensor,
        out_features: int,
        in_features: int,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from aiter.ops.gemm_op_a6w6 import gemm_a6w6, quant_mxfp6_gemm

        if a.dtype != torch.bfloat16:
            a = a.to(torch.bfloat16)
        a_packed, a_scale = quant_mxfp6_gemm(a)
        out = gemm_a6w6(
            a_packed,
            w_packed,
            a_scale,
            w_scale,
            a.shape[0],
            out_features,
            in_features,
            dtype=torch.bfloat16,
        )
        if bias is not None:
            out = out + bias
        return out

    @_rocm_mxfp6_gemm.register_fake
    def _rocm_mxfp6_gemm_fake(
        a: torch.Tensor,
        w_packed: torch.Tensor,
        w_scale: torch.Tensor,
        out_features: int,
        in_features: int,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty(a.shape[0], out_features, dtype=torch.bfloat16, device=a.device)


class ROCmMxfp6LinearMethod(MXFPLinearMethodBase):
    """ROCm W6A6 MXFP6 linear method using AITER (gemm_a6w6).

    Buffers after process_weights_after_loading:
      weight_packed : uint8, 1-D, AITER A6W6 packed layout (opaque; see mxfp6_gemm_pack_size)
      weight_scale  : uint8, 1-D, packed per-group-of-32 e8m0 scales

    Forward path:
      _quantize_activation: pass-through (activation quant is inside the custom op)
      _quant_matmul: torch.ops.vllm_omni.rocm_mxfp6_gemm wraps activation quant,
                     gemm_a6w6 and the bias add in one custom op
    """

    def __init__(self, quant_config: DiffusionMXFP6Config) -> None:
        self.quant_config = quant_config
        self.out_dtype = torch.get_default_dtype()
        if not hasattr(torch.ops.vllm_omni, "rocm_mxfp6_gemm"):
            _register_rocm_mxfp6_op()

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        from aiter.ops.gemm_op_a6w6 import quant_mxfp6_gemm

        weight = layer.weight.data
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)
        out_features, in_features = weight.shape
        weight_packed, weight_scale = quant_mxfp6_gemm(weight.contiguous())

        layer.register_buffer("weight_packed", weight_packed, persistent=True)
        layer.register_buffer("weight_scale", weight_scale, persistent=True)
        layer.mxfp6_out_features = out_features
        layer.mxfp6_in_features = in_features

        # Remove the original weight parameter to free memory.
        if hasattr(layer, "weight") and isinstance(layer.weight, torch.nn.Parameter):
            delattr(layer, "weight")
            layer.register_parameter("weight", None)

        layer._already_called_process_weights_after_loading = True

    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Activation quantization is inside the custom op called by _quant_matmul,
        # so pass through the raw activation here.
        return x, None

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor | None,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.ops.vllm_omni.rocm_mxfp6_gemm(
            x_q,
            layer.weight_packed,
            layer.weight_scale,
            layer.mxfp6_out_features,
            layer.mxfp6_in_features,
            bias,
        )
        if output.dtype != ori_dtype:
            output = output.to(ori_dtype)
        return output


class ROCmMxfp6OnlineLinearMethod(_LazyWeightMixin, ROCmMxfp6LinearMethod):
    """ROCm W6A6 MXFP6 online linear method using AITER.

    MRO: ROCmMxfp6OnlineLinearMethod → _LazyWeightMixin → ROCmMxfp6LinearMethod
         → MXFPLinearMethodBase → LinearMethodBase

      create_weights  : _LazyWeightMixin          (meta device + patched loader)
      process_weights : ROCmMxfp6OnlineLinearMethod  (meta → materialize, then AITER quant)
      apply / ops     : ROCmMxfp6LinearMethod / MXFPLinearMethodBase (shared)
    """

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # Materialise from meta device if needed (same pattern as the ROCm MXFP4 online method).
        if layer.weight is not None and layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        # Delegate to the base class which does AITER quant.
        ROCmMxfp6LinearMethod.process_weights_after_loading(self, layer)
