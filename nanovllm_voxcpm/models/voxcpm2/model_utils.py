"""Pure CPU-testable utilities extracted from voxcpm2/model.py.

All functions here are free of CUDA / flash-attn / Triton dependencies and
can be exercised in a standard CPU pytest session.
"""

import torch

from nanovllm_voxcpm.models.voxcpm2 import model_utils_rope as _rope

apply_rotary_emb = _rope.apply_rotary_emb
build_rope_cos_sin_cache = _rope.build_rope_cos_sin_cache
build_rope_inv_freq = _rope.build_rope_inv_freq
compute_rope_scaling_factor = _rope.compute_rope_scaling_factor
sinusoidal_pos_emb = _rope.sinusoidal_pos_emb

# ---------------------------------------------------------------------------
# Attention size calculations
# ---------------------------------------------------------------------------


def compute_attention_sizes(
    hidden_size: int,
    total_num_heads: int,
    total_num_kv_heads: int,
    head_dim: int | None,
    tp_size: int,
) -> dict[str, int]:
    """Derive per-rank attention tensor sizes from model config values.

    This mirrors the attribute math in Cpm4Attention.__init__ so the logic
    can be unit-tested without constructing the full nn.Module.

    Args:
        hidden_size: Model hidden dimension.
        total_num_heads: Total query attention heads across all TP ranks.
        total_num_kv_heads: Total key/value heads across all TP ranks.
        head_dim: Explicit head dimension; if ``None`` derived from
            ``hidden_size // total_num_heads``.
        tp_size: Tensor-parallel world size (typically 1 for unit tests).

    Returns:
        Dict with keys ``num_heads``, ``num_kv_heads``, ``head_dim``,
        ``q_size``, ``kv_size``, ``scaling``.
    """
    num_heads = total_num_heads // tp_size
    num_kv_heads = total_num_kv_heads // tp_size
    resolved_head_dim = head_dim if head_dim is not None else hidden_size // total_num_heads
    q_size = num_heads * resolved_head_dim
    kv_size = num_kv_heads * resolved_head_dim
    scaling = resolved_head_dim**-0.5
    return {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": resolved_head_dim,
        "q_size": q_size,
        "kv_size": kv_size,
        "scaling": scaling,
    }


# ---------------------------------------------------------------------------
# LoRA target-module parsing helpers
# ---------------------------------------------------------------------------


def parse_qkv_lora_targets(lora_targets: list[str]) -> list[str]:
    """Derive the qkv_proj LoRA sub-target list from raw target_modules_lm.

    Cpm4Attention strips the ``_proj`` suffix and keeps only q/k/v entries.

    Args:
        lora_targets: Raw ``target_modules_lm`` list (e.g. ``["q_proj", "o_proj"]``).

    Returns:
        Filtered list, e.g. ``["q", "k", "v"]`` (subset, order preserved).
    """
    return [t.replace("_proj", "") for t in lora_targets if t in ("q_proj", "k_proj", "v_proj")]


def parse_gate_up_lora_targets(lora_targets: list[str]) -> list[int]:
    """Derive the gate_up_proj LoRA index list from raw target_modules_lm.

    Cpm4MLP uses index 0 for gate_proj and index 1 for up_proj.

    Args:
        lora_targets: Raw ``target_modules_lm`` list.

    Returns:
        Sorted list of integer indices present in ``lora_targets``.
    """
    result: list[int] = []
    if "gate_proj" in lora_targets:
        result.append(0)
    if "up_proj" in lora_targets:
        result.append(1)
    return result


# ---------------------------------------------------------------------------
# VoxCPM2Model config-derivation helpers
# ---------------------------------------------------------------------------


def derive_encoder_config_fields(
    lm_config_hidden_size: int,
    lm_config_intermediate_size: int,
    lm_config_num_attention_heads: int,
    encoder_hidden_dim: int,
    encoder_ffn_dim: int,
    encoder_num_heads: int,
    encoder_num_layers: int,
    encoder_kv_channels: int | None,
) -> dict[str, int | None]:
    """Return the fields that VoxCPM2Model patches onto a deep-copy of lm_config
    to produce the encoder_config passed to VoxCPM2LocEnc.

    Args:
        lm_config_hidden_size: Original lm_config.hidden_size (unused by this
            function but kept for documentation symmetry).
        lm_config_intermediate_size: Original lm_config.intermediate_size.
        lm_config_num_attention_heads: Original lm_config.num_attention_heads.
        encoder_hidden_dim: encoder_config.hidden_dim override.
        encoder_ffn_dim: encoder_config.ffn_dim override.
        encoder_num_heads: encoder_config.num_heads override.
        encoder_num_layers: encoder_config.num_layers override.
        encoder_kv_channels: encoder_config.kv_channels override.

    Returns:
        Dict of field names → values that should be set on the encoder config copy.
    """
    return {
        "hidden_size": encoder_hidden_dim,
        "intermediate_size": encoder_ffn_dim,
        "num_attention_heads": encoder_num_heads,
        "num_hidden_layers": encoder_num_layers,
        "kv_channels": encoder_kv_channels,
        "vocab_size": 0,
    }


def derive_decoder_config_fields(
    dit_hidden_dim: int,
    dit_ffn_dim: int,
    dit_num_heads: int,
    dit_num_layers: int,
    dit_kv_channels: int | None,
) -> dict[str, int | None]:
    """Return the fields that VoxCPM2Model patches onto a deep-copy of lm_config
    to produce the decoder_config passed to VoxCPM2LocDiT.

    Args:
        dit_hidden_dim: dit_config.hidden_dim override.
        dit_ffn_dim: dit_config.ffn_dim override.
        dit_num_heads: dit_config.num_heads override.
        dit_num_layers: dit_config.num_layers override.
        dit_kv_channels: dit_config.kv_channels override.

    Returns:
        Dict of field names → values that should be set on the decoder config copy.
    """
    return {
        "hidden_size": dit_hidden_dim,
        "intermediate_size": dit_ffn_dim,
        "num_attention_heads": dit_num_heads,
        "num_hidden_layers": dit_num_layers,
        "kv_channels": dit_kv_channels,
        "vocab_size": 0,
    }


def compute_optimized_scale(
    positive_flat: torch.Tensor,
    negative_flat: torch.Tensor,
) -> torch.Tensor:
    """Compute the CFG optimised scale (UnifiedCFM.optimized_scale logic).

    Args:
        positive_flat: Conditioned flow estimate, shape ``(bsz, N)``.
        negative_flat: Unconditioned flow estimate, shape ``(bsz, N)``.

    Returns:
        Scale tensor of shape ``(bsz, 1)``.
    """
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm


def build_cfm_t_span(inference_timesteps: int, device: torch.device | None = None) -> torch.Tensor:
    """Build the cosine-adjusted time span used in UnifiedCFM.forward.

    Args:
        inference_timesteps: Number of Euler steps.
        device: Target device; defaults to CPU.

    Returns:
        Float32 tensor of shape ``(inference_timesteps + 1,)`` decreasing from
        approximately 1 to approximately 0 with a cosine adjustment applied.
    """
    t_span = torch.linspace(1, 0, inference_timesteps + 1, device=device, dtype=torch.float32)
    t_span = t_span + (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    return t_span


def compute_zero_init_steps(t_span_len: int) -> int:
    """Return the number of zero-initialised Euler steps (4 % of total span).

    Args:
        t_span_len: Length of the full t_span tensor (``inference_timesteps + 1``).

    Returns:
        ``max(1, int(t_span_len * 0.04))``.
    """
    return max(1, int(t_span_len * 0.04))
