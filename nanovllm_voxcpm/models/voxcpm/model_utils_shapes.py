"""CPU-testable tensor bookkeeping helpers for the VoxCPM model."""

from __future__ import annotations

import torch

from nanovllm_voxcpm.models.voxcpm.model_utils import optimized_scale


def combine_feature_and_text_embeddings(
    feature_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    feature_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.where(feature_mask.unsqueeze(-1), feature_embeddings, text_embeddings)


def select_prefill_hidden(hidden_states: torch.Tensor, cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    return hidden_states[cu_seqlens_q[1:] - 1].contiguous()


def prepare_local_encoder_inputs(projected_inputs: torch.Tensor, special_token: torch.Tensor) -> torch.Tensor:
    special_tokens = special_token[0].expand(projected_inputs.size(0), 1, -1)
    return torch.cat([special_tokens, projected_inputs], dim=1)


def quantize_scalar_latents(hidden_states: torch.Tensor, scale: int) -> torch.Tensor:
    return torch.round(torch.tanh(hidden_states) * scale) / scale


def prepare_dit_decoder_input(
    sample: torch.Tensor,
    condition: torch.Tensor,
    conditioned_mu: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([conditioned_mu, condition, sample], dim=1)


def apply_classifier_free_guidance(
    positive: torch.Tensor,
    negative: torch.Tensor,
    cfg_value: torch.Tensor,
) -> torch.Tensor:
    batch_size = positive.size(0)
    scale = optimized_scale(positive.view(batch_size, -1), negative.view(batch_size, -1))
    expansion = (batch_size,) + (1,) * (positive.ndim - 1)
    scale = scale.view(expansion)
    cfg = cfg_value.view(expansion)
    return negative * scale + cfg * (positive - negative * scale)
