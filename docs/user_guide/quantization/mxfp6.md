# W6A6 MXFP6 Quantization (ROCm)

## Overview

W6A6 MXFP6 quantizes both weights and activations to FP6 (E2M3) using the OCP MX format: groups of 32
K-dimension elements share one `float8_e8m0fnu` exponent scale. The GEMM is AITER's A6W6 kernel
(`aiter.ops.gemm_op_a6w6`), which exists on AMD gfx950 (MI355X) only.

| Method | Mode | Platform |
| -------- | ------ | ---------- |
| `mxfp6` | Online only (BF16 checkpoint, quantized while loading) | ROCm gfx950 |

It follows the same structure as the ROCm [MXFP4](mxfp4.md) path:

- Weights are quantized once at load time with `aiter.ops.gemm_op_a6w6.quant_mxfp6_gemm` and stored as packed
  `uint8` buffers (`weight_packed`, `weight_scale`). The packed layout is opaque and only `gemm_a6w6` can
  consume it. The BF16 weight is freed.
- Activations are quantized on every forward pass with the same function, followed by `gemm_a6w6`. Both steps
  and the bias add run inside the custom op `vllm_omni::rocm_mxfp6_gemm`, so `torch.compile` treats them as one
  opaque node.
- AITER's quantizer applies a block-diagonal 32-point Hadamard rotation along K to both operands. This leaves
  the product unchanged and improves accuracy; it is part of the packed layout, not an option.
- Output is BF16, so only BF16 activations are supported.

This is the same recipe as xDiT's `gemm_quantization high=fp6` (`xFuserMXFP6Linear`).

## Usage

`mxfp6` is mainly useful for the second transformer of cascade models, as in xDiT's
`gemm_quantization low=fp4,high=fp6` for Wan2.2 A14B (high-noise expert on MXFP4, low-noise expert on MXFP6):

```bash
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni \
    --diffusion-quantization-config '{"transformer": {"method": "mxfp4"}, "transformer_2": {"method": "mxfp6"}}'
```

Layers can be kept in BF16 with `ignored_layers`, using exact module paths relative to the transformer, for
example `{"method": "mxfp6", "ignored_layers": ["blocks.0.attn1.to_qkv"]}`.

## Limitations

- ROCm gfx950 only; other platforms raise `NotImplementedError`.
- Online only. Pre-quantized MXFP6 checkpoints are not supported.
- BF16 activations only.
