# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig, validate_sp_plan
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    _RefinerSequence,
    _zimage_refiner_sp_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_refiner_plan_gathers_the_last_layer():
    plan = _zimage_refiner_sp_plan(2)
    validate_sp_plan(plan)
    assert set(plan) == {
        "noise_refiner_sequence",
        "noise_refiner_gather",
        "context_refiner_sequence",
        "context_refiner_gather",
    }
    assert [item.shard_group for item in plan["noise_refiner_gather"]] == ["zimage_noise_refiner"] * 4
    assert [item.expected_dims for item in plan["noise_refiner_gather"]] == [3, 2, 3, 3]
    assert [item.shard_group for item in plan["context_refiner_gather"]] == ["zimage_context_refiner"] * 4
    assert _zimage_refiner_sp_plan(0) == {}


def test_refiner_blocks_compile_separately_from_the_main_stack():
    from vllm_omni.diffusion.models.z_image.z_image_transformer import (
        ZImageContextRefinerBlock,
        ZImageNoiseRefinerBlock,
        ZImageTransformer2DModel,
        ZImageTransformerBlock,
    )

    assert ZImageTransformer2DModel._repeated_blocks == [
        "ZImageTransformerBlock",
        "ZImageNoiseRefinerBlock",
        "ZImageContextRefinerBlock",
    ]
    assert ZImageNoiseRefinerBlock.forward is not ZImageTransformerBlock.forward
    assert ZImageContextRefinerBlock.forward is not ZImageTransformerBlock.forward


def test_refiner_loop_keeps_every_layer_on_the_same_shard(monkeypatch):
    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen: list[tuple[int, int, int, int]] = []

        def forward(self, x, attn_mask, cos, sin, adaln_input=None):
            self.seen.append((x.shape[1], attn_mask.shape[1], cos.shape[1], sin.shape[1]))
            return x + 1

    class RefinerGroup(nn.Module):
        def __init__(self):
            super().__init__()
            self.noise_refiner_sequence = _RefinerSequence()
            self.noise_refiner = nn.ModuleList([Recorder(), Recorder()])
            self.noise_refiner_gather = _RefinerSequence()

        def forward(self, x, attn_mask, cos, sin):
            x, attn_mask, cos, sin = self.noise_refiner_sequence(x, attn_mask, cos, sin)
            for layer in self.noise_refiner:
                x = layer(x, attn_mask, cos, sin)
            return self.noise_refiner_gather(x, attn_mask, cos, sin)

    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.hooks.sequence_parallel.sp_shard",
        lambda tensor, dim, validate=True: tensor.chunk(2, dim=dim)[0],
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.hooks.sequence_parallel.sp_gather",
        lambda tensor, dim, validate=True: torch.cat((tensor, tensor), dim=dim),
    )

    model = RefinerGroup()
    full_plan = _zimage_refiner_sp_plan(2)
    plan = {
        "noise_refiner_sequence": full_plan["noise_refiner_sequence"],
        "noise_refiner_gather": full_plan["noise_refiner_gather"],
    }
    apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=2, ring_degree=1), plan)

    sequence = 8
    hidden = torch.arange(sequence, dtype=torch.float32).view(1, sequence, 1)
    mask = torch.ones(1, sequence, dtype=torch.bool)
    rope = torch.arange(sequence, dtype=torch.float32).view(1, sequence, 1)
    with override_forward_context(ForwardContext()):
        hidden_out, mask_out, cos_out, sin_out = model(hidden, mask, rope, rope)

    local = sequence // 2
    for layer in model.noise_refiner:
        assert layer.seen == [(local, local, local, local)]
    assert hidden_out.shape == hidden.shape
    assert mask_out.shape == mask.shape
    assert cos_out.shape == rope.shape
    assert sin_out.shape == rope.shape
    expected = hidden.chunk(2, dim=1)[0] + 2
    assert torch.equal(hidden_out, torch.cat((expected, expected), dim=1))
    assert torch.equal(mask_out, torch.cat((mask.chunk(2, dim=1)[0], mask.chunk(2, dim=1)[0]), dim=1))
