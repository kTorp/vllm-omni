# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MHA v4 quantized diffusion attention backend."""

import torch

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.utils.aiter_mha_v4 import (
    get_forward_fn,
    require_mha_v4,
)
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none

_REQUIRED_HEAD_DIM = 128
_DEFAULT_FORMAT = "mxfp4"
_SUPPORTED_FORMATS = frozenset({"f4f4", "f6f4", "fp8", "i8fp8", "mxfp4", "mxfp6"})
_SUPPORTED_LAYOUTS = frozenset({"BSND", "BSHD"})


def _is_gfx95_supported() -> bool:
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName
    except (AttributeError, RuntimeError):
        return False
    return "gfx95" in arch.lower()


class AiterQuantBackend(AttentionBackend):
    supported_platforms = ("rocm",)

    @classmethod
    def validate_available(cls) -> None:
        if not _is_gfx95_supported():
            raise RuntimeError("AITER_QUANT_ATTN requires a gfx950-class ROCm GPU.")
        require_mha_v4()

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_REQUIRED_HEAD_DIM]

    @staticmethod
    def get_name() -> str:
        return "AITER_QUANT_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AiterQuantImpl"]:
        return AiterQuantImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError("AITER_QUANT_ATTN does not use a metadata builder.")


class AiterQuantImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        options = backend_kwargs or {}
        format_name = str(options.get("format", _DEFAULT_FORMAT)).lower()
        if format_name not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unknown AITER quant format {format_name!r}; "
                f"expected one of {sorted(_SUPPORTED_FORMATS)}."
            )
        if causal:
            raise NotImplementedError("AITER_QUANT_ATTN does not support causal attention.")
        if head_size != _REQUIRED_HEAD_DIM:
            raise NotImplementedError(
                f"AITER_QUANT_ATTN requires head_dim={_REQUIRED_HEAD_DIM}; got {head_size}."
            )
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise NotImplementedError("AITER_QUANT_ATTN does not support grouped-query attention.")
        if qkv_layout is not None and qkv_layout.upper() not in _SUPPORTED_LAYOUTS:
            raise ValueError(
                "AITER_QUANT_ATTN expects [B, S, H, D] tensors "
                f"(BSND/BSHD), not qkv_layout={qkv_layout!r}."
            )
        config = get_current_diffusion_config_or_none()
        if config is not None and config.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                "AITER_QUANT_ATTN requires float16 or bfloat16 model inputs; "
                f"got dtype={config.dtype}."
            )

        AiterQuantBackend.validate_available()
        self.format = format_name
        self.qkv_layout = qkv_layout
        self.softmax_scale = softmax_scale
        self.causal = causal
        self._forward = get_forward_fn(format_name)

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            raise NotImplementedError("AITER_QUANT_ATTN does not support attention masks.")
        return self._forward(
            query,
            key,
            value,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
