# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-safe wrappers for AITER MHA v4 MX attention recipes."""

from collections.abc import Callable

import torch

try:
    import aiter as _aiter
    from aiter.ops.mha_v4 import AttentionFormat as _AiterAttentionFormat
    from aiter.ops.mha_v4 import mha_v4_packed as _aiter_mha_v4_packed
    from aiter.ops.mha_v4 import mha_v4_q_multiplier as _aiter_mha_v4_q_multiplier
    from aiter.ops.mha_v4 import mxfp4_k_view as _aiter_mxfp4_k_view
    from aiter.ops.mha_v4 import mxfp4_v_view as _aiter_mxfp4_v_view
    from aiter.ops.mha_v4 import mxfp6_k_view as _aiter_mxfp6_k_view
    from aiter.ops.mha_v4 import native_fp8_format as _aiter_native_fp8_format
    from aiter.ops.mha_v4 import quantize_mxfp4_k as _aiter_quantize_mxfp4_k
    from aiter.ops.mha_v4 import quantize_mxfp4_q as _aiter_quantize_mxfp4_q
    from aiter.ops.mha_v4 import quantize_mxfp6_k as _aiter_quantize_mxfp6_k
    from aiter.ops.mha_v4 import quantize_mxfp6_q as _aiter_quantize_mxfp6_q
    from aiter.ops.mha_v4 import quantize_fp8 as _aiter_mha_v4_quantize_fp8
    from aiter.ops.mha_v4 import (
        quantize_fp8_rotated as _aiter_mha_v4_quantize_fp8_rotated,
    )
    from aiter.ops.mha_v4 import quantize_int8 as _aiter_mha_v4_quantize_int8
    from aiter.ops.mha_v4 import quantize_v_fp8 as _aiter_quantize_v_fp8
    from aiter.ops.mha_v4 import quantize_v_mxfp4 as _aiter_quantize_v_mxfp4
    from aiter.ops.mha_v4 import scale_modes_for_formats as _aiter_scale_modes_for_formats

    _MHA_V4_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _aiter = None
    _AiterAttentionFormat = None
    _aiter_mha_v4_packed = None
    _aiter_mha_v4_q_multiplier = None
    _aiter_mxfp4_k_view = None
    _aiter_mxfp4_v_view = None
    _aiter_mxfp6_k_view = None
    _aiter_native_fp8_format = None
    _aiter_quantize_mxfp4_k = None
    _aiter_quantize_mxfp4_q = None
    _aiter_quantize_mxfp6_k = None
    _aiter_quantize_mxfp6_q = None
    _aiter_mha_v4_quantize_fp8 = None
    _aiter_mha_v4_quantize_fp8_rotated = None
    _aiter_mha_v4_quantize_int8 = None
    _aiter_quantize_v_fp8 = None
    _aiter_quantize_v_mxfp4 = None
    _aiter_scale_modes_for_formats = None
    _MHA_V4_IMPORT_ERROR = exc


def require_mha_v4() -> None:
    if _MHA_V4_IMPORT_ERROR is not None:
        raise RuntimeError(
            "AITER_QUANT_ATTN requires an AITER build containing aiter.ops.mha_v4."
        ) from _MHA_V4_IMPORT_ERROR


@torch.library.custom_op("vllm_omni::aiter_fp8_attention", mutates_args=())
def _aiter_fp8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
) -> torch.Tensor:
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        fp8_format,
        fp8_format,
        fp8_format,
        *_aiter_scale_modes_for_formats(fp8_format, fp8_format, fp8_format),
    )


@_aiter_fp8_attention.register_fake
def _aiter_fp8_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
) -> torch.Tensor:
    del key, q_descale, k_descale, v_descale
    return query.new_empty(
        (query.shape[0], query.shape[1], query.shape[2], value.shape[-1]),
        dtype=torch.bfloat16,
    )


def _forward_fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    del softmax_scale, causal
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    quant_q, q_descale = _aiter_mha_v4_quantize_fp8_rotated(query)
    quant_k, k_descale = _aiter_mha_v4_quantize_fp8_rotated(key)
    quant_v, v_descale = _aiter_mha_v4_quantize_fp8(value)

    return _aiter_fp8_attention(
        quant_q,
        quant_k,
        quant_v,
        q_descale,
        k_descale,
        v_descale,
    )


@torch.library.custom_op("vllm_omni::aiter_i8fp8_quantize_q", mutates_args=())
def _aiter_i8fp8_quantize_q(
    query: torch.Tensor,
    clip: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_mha_v4_quantize_int8(query, clip)


@_aiter_i8fp8_quantize_q.register_fake
def _aiter_i8fp8_quantize_q_fake(
    query: torch.Tensor,
    clip: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del clip
    return query.new_empty(query.shape, dtype=torch.int8), query.new_empty(
        (1,),
        dtype=torch.float32,
    )


@torch.library.custom_op("vllm_omni::aiter_i8fp8_quantize_k", mutates_args=())
def _aiter_i8fp8_quantize_k(
    key: torch.Tensor,
    clip: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_mha_v4_quantize_int8(key, clip)


@_aiter_i8fp8_quantize_k.register_fake
def _aiter_i8fp8_quantize_k_fake(
    key: torch.Tensor,
    clip: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del clip
    return key.new_empty(key.shape, dtype=torch.int8), key.new_empty(
        (1,),
        dtype=torch.float32,
    )


@torch.library.custom_op("vllm_omni::aiter_i8fp8_quantize_v", mutates_args=())
def _aiter_i8fp8_quantize_v(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_mha_v4_quantize_fp8(value)


@_aiter_i8fp8_quantize_v.register_fake
def _aiter_i8fp8_quantize_v_fake(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return value.new_empty(value.shape, dtype=_aiter.dtypes.fp8), value.new_empty(
        (1,),
        dtype=torch.float32,
    )


@torch.library.custom_op("vllm_omni::aiter_i8fp8_attention", mutates_args=())
def _aiter_i8fp8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
) -> torch.Tensor:
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        _AiterAttentionFormat.INT8,
        _AiterAttentionFormat.INT8,
        fp8_format,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.INT8,
            _AiterAttentionFormat.INT8,
            fp8_format,
        ),
    )


@_aiter_i8fp8_attention.register_fake
def _aiter_i8fp8_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
) -> torch.Tensor:
    del key, q_descale, k_descale, v_descale
    return query.new_empty(
        (query.shape[0], query.shape[1], query.shape[2], value.shape[-1]),
        dtype=torch.bfloat16,
    )


def _forward_i8fp8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    del softmax_scale, causal
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    q_i8, q_descale = _aiter_i8fp8_quantize_q(query)
    k_i8, k_descale = _aiter_i8fp8_quantize_k(key)
    v_fp8, v_descale = _aiter_i8fp8_quantize_v(value)
    return _aiter_i8fp8_attention(
        q_i8,
        k_i8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
    )


@torch.library.custom_op("vllm_omni::aiter_mxfp4_quantize_q", mutates_args=())
def _aiter_mxfp4_quantize_q(
    query: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_mxfp4_q(
        query,
        _aiter_mha_v4_q_multiplier(softmax_scale),
    )


@_aiter_mxfp4_quantize_q.register_fake
def _aiter_mxfp4_quantize_q_fake(
    query: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_mxfp4_q(
        query,
        _aiter_mha_v4_q_multiplier(softmax_scale),
    )


@torch.library.custom_op("vllm_omni::aiter_mxfp4_quantize_k_raw", mutates_args=())
def _aiter_mxfp4_quantize_k_raw(
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_mxfp4_k(key)


@_aiter_mxfp4_quantize_k_raw.register_fake
def _aiter_mxfp4_quantize_k_raw_fake(
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_mxfp4_k(key)


@torch.library.custom_op("vllm_omni::aiter_mx_quantize_v", mutates_args=())
def _aiter_mx_quantize_v(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_v_fp8(value)


@_aiter_mx_quantize_v.register_fake
def _aiter_mx_quantize_v_fake(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_v_fp8(value)


@torch.library.custom_op("vllm_omni::aiter_f4_quantize_v_raw", mutates_args=())
def _aiter_f4_quantize_v_raw(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_v_mxfp4(value)


@_aiter_f4_quantize_v_raw.register_fake
def _aiter_f4_quantize_v_raw_fake(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_v_mxfp4(value)


@torch.library.custom_op("vllm_omni::aiter_mxfp4_kernel_raw", mutates_args=())
def _aiter_mxfp4_kernel_raw(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_buf: torch.Tensor,
    k_scale: torch.Tensor,
    v_fp8: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    k_fp4 = _aiter_mxfp4_k_view(k_buf, k_scale)
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        q_fp4,
        k_fp4,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        fp8_format,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP4,
            _AiterAttentionFormat.MXFP4,
            fp8_format,
        ),
        softmax_scale=softmax_scale,
    )


@_aiter_mxfp4_kernel_raw.register_fake
def _aiter_mxfp4_kernel_raw_fake(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_buf: torch.Tensor,
    k_scale: torch.Tensor,
    v_fp8: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    del q_scale, k_buf, k_scale, v_scale, softmax_scale
    batch, seq_len, num_heads, _ = q_fp4.shape
    return q_fp4.new_empty(
        (batch, seq_len, num_heads, v_fp8.shape[-1]),
        dtype=torch.bfloat16,
    )


def _forward_mxfp4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    del softmax_scale, causal
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    softmax_scale = query.shape[-1] ** -0.5

    q_fp4, q_scale = _aiter_mxfp4_quantize_q(query, softmax_scale)
    k_buf, k_scale = _aiter_mxfp4_quantize_k_raw(key)
    v_fp8, v_scale = _aiter_mx_quantize_v(value)
    return _aiter_mxfp4_kernel_raw(
        q_fp4,
        q_scale,
        k_buf,
        k_scale,
        v_fp8,
        v_scale,
        softmax_scale,
    )


@torch.library.custom_op("vllm_omni::aiter_f4f4_kernel_raw", mutates_args=())
def _aiter_f4f4_kernel_raw(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_buf: torch.Tensor,
    k_scale: torch.Tensor,
    v_buf: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    kv_len: int,
) -> torch.Tensor:
    k_fp4 = _aiter_mxfp4_k_view(k_buf, k_scale)
    v_fp4 = _aiter_mxfp4_v_view(v_buf, v_scale, kv_len)
    return _aiter_mha_v4_packed(
        q_fp4,
        k_fp4,
        v_fp4,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        _AiterAttentionFormat.MXFP4,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP4,
            _AiterAttentionFormat.MXFP4,
            _AiterAttentionFormat.MXFP4,
        ),
        softmax_scale=softmax_scale,
    )


@_aiter_f4f4_kernel_raw.register_fake
def _aiter_f4f4_kernel_raw_fake(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_buf: torch.Tensor,
    k_scale: torch.Tensor,
    v_buf: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    kv_len: int,
) -> torch.Tensor:
    del k_buf, k_scale, v_buf, v_scale, softmax_scale, kv_len
    batch, seq_len, num_heads, _ = q_fp4.shape
    return q_fp4.new_empty(
        (batch, seq_len, num_heads, q_scale.shape[-1] * 32),
        dtype=torch.bfloat16,
    )


def _forward_f4f4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    del softmax_scale, causal
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    softmax_scale = query.shape[-1] ** -0.5

    q_fp4, q_scale = _aiter_mxfp4_quantize_q(query, softmax_scale)
    k_buf, k_scale = _aiter_mxfp4_quantize_k_raw(key)
    v_buf, v_scale = _aiter_f4_quantize_v_raw(value)
    return _aiter_f4f4_kernel_raw(
        q_fp4,
        q_scale,
        k_buf,
        k_scale,
        v_buf,
        v_scale,
        softmax_scale,
        value.shape[1],
    )


@torch.library.custom_op("vllm_omni::aiter_mxfp6_quantize_q", mutates_args=())
def _aiter_mxfp6_quantize_q(
    query: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_mxfp6_q(
        query,
        _aiter_mha_v4_q_multiplier(softmax_scale),
    )


@_aiter_mxfp6_quantize_q.register_fake
def _aiter_mxfp6_quantize_q_fake(
    query: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_mxfp6_q(
        query,
        _aiter_mha_v4_q_multiplier(softmax_scale),
    )


@torch.library.custom_op("vllm_omni::aiter_mxfp6_quantize_k_raw", mutates_args=())
def _aiter_mxfp6_quantize_k_raw(
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_mxfp6_k(key)


@_aiter_mxfp6_quantize_k_raw.register_fake
def _aiter_mxfp6_quantize_k_raw_fake(
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _aiter_quantize_mxfp6_k(key)


@torch.library.custom_op("vllm_omni::aiter_mxfp6_kernel_raw", mutates_args=())
def _aiter_mxfp6_kernel_raw(
    k_buf: torch.Tensor,
    k_scale_buf: torch.Tensor,
    q_fp6: torch.Tensor,
    q_scale: torch.Tensor,
    v_fp8: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    batch, kv_len, num_heads, _ = v_fp8.shape
    k_fp6, k_scale = _aiter_mxfp6_k_view(
        k_buf,
        k_scale_buf,
        batch,
        kv_len,
        num_heads,
    )
    fp8_format = _aiter_native_fp8_format()
    return _aiter_mha_v4_packed(
        q_fp6,
        k_fp6,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP6,
        fp8_format,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP6,
            _AiterAttentionFormat.MXFP6,
            fp8_format,
        ),
        softmax_scale=softmax_scale,
    )


@_aiter_mxfp6_kernel_raw.register_fake
def _aiter_mxfp6_kernel_raw_fake(
    k_buf: torch.Tensor,
    k_scale_buf: torch.Tensor,
    q_fp6: torch.Tensor,
    q_scale: torch.Tensor,
    v_fp8: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    del k_buf, k_scale_buf, q_scale, v_scale, softmax_scale
    batch, seq_len, num_heads, _ = q_fp6.shape
    return q_fp6.new_empty(
        (batch, seq_len, num_heads, v_fp8.shape[-1]),
        dtype=torch.bfloat16,
    )


def _forward_mxfp6(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    del softmax_scale, causal
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    softmax_scale = query.shape[-1] ** -0.5

    q_fp6, q_scale = _aiter_mxfp6_quantize_q(query, softmax_scale)
    k_buf, k_scale_buf = _aiter_mxfp6_quantize_k_raw(key)
    v_fp8, v_scale = _aiter_mx_quantize_v(value)
    return _aiter_mxfp6_kernel_raw(
        k_buf,
        k_scale_buf,
        q_fp6,
        q_scale,
        v_fp8,
        v_scale,
        softmax_scale,
    )


@torch.library.custom_op("vllm_omni::aiter_f6f4_kernel_raw", mutates_args=())
def _aiter_f6f4_kernel_raw(
    k_buf: torch.Tensor,
    k_scale_buf: torch.Tensor,
    q_fp6: torch.Tensor,
    q_scale: torch.Tensor,
    v_buf: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    kv_len: int,
) -> torch.Tensor:
    v_fp4 = _aiter_mxfp4_v_view(v_buf, v_scale, kv_len)
    batch, _, num_heads, _ = v_fp4.shape
    k_fp6, k_scale = _aiter_mxfp6_k_view(
        k_buf,
        k_scale_buf,
        batch,
        kv_len,
        num_heads,
    )
    return _aiter_mha_v4_packed(
        q_fp6,
        k_fp6,
        v_fp4,
        q_scale,
        k_scale,
        v_scale,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP6,
        _AiterAttentionFormat.MXFP4,
        *_aiter_scale_modes_for_formats(
            _AiterAttentionFormat.MXFP6,
            _AiterAttentionFormat.MXFP6,
            _AiterAttentionFormat.MXFP4,
        ),
        softmax_scale=softmax_scale,
    )


@_aiter_f6f4_kernel_raw.register_fake
def _aiter_f6f4_kernel_raw_fake(
    k_buf: torch.Tensor,
    k_scale_buf: torch.Tensor,
    q_fp6: torch.Tensor,
    q_scale: torch.Tensor,
    v_buf: torch.Tensor,
    v_scale: torch.Tensor,
    softmax_scale: float,
    kv_len: int,
) -> torch.Tensor:
    del k_buf, k_scale_buf, v_buf, v_scale, softmax_scale, kv_len
    batch, seq_len, num_heads, _ = q_fp6.shape
    return q_fp6.new_empty(
        (batch, seq_len, num_heads, q_scale.shape[-1] * 32),
        dtype=torch.bfloat16,
    )


def _forward_f6f4(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    del softmax_scale, causal
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    softmax_scale = query.shape[-1] ** -0.5

    q_fp6, q_scale = _aiter_mxfp6_quantize_q(query, softmax_scale)
    k_buf, k_scale_buf = _aiter_mxfp6_quantize_k_raw(key)
    v_buf, v_scale = _aiter_f4_quantize_v_raw(value)
    return _aiter_f6f4_kernel_raw(
        k_buf,
        k_scale_buf,
        q_fp6,
        q_scale,
        v_buf,
        v_scale,
        softmax_scale,
        value.shape[1],
    )


_FORWARD_FNS: dict[str, Callable[..., torch.Tensor]] = {
    "f4f4": _forward_f4f4,
    "f6f4": _forward_f6f4,
    "fp8": _forward_fp8,
    "i8fp8": _forward_i8fp8,
    "mxfp4": _forward_mxfp4,
    "mxfp6": _forward_mxfp6,
}


def get_forward_fn(format_name: str) -> Callable[..., torch.Tensor]:
    try:
        return _FORWARD_FNS[format_name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown AITER quant format {format_name!r}; expected one of {sorted(_FORWARD_FNS)}."
        ) from None
