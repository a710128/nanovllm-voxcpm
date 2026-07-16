"""Pure CPU-testable utilities extracted from voxcpm2/model.py.

All functions here are free of CUDA / flash-attn / Triton dependencies and
can be exercised in a standard CPU pytest session.
"""

import math

import torch


# ---------------------------------------------------------------------------
# RoPE / LongRoPE helpers
# ---------------------------------------------------------------------------


def compute_rope_scaling_factor(
    max_position_embeddings: int,
    original_max_position_embeddings: int,
) -> float:
    """Compute the LongRoPE amplitude-scaling factor.

    Args:
        max_position_embeddings: Extended context length.
        original_max_position_embeddings: Base (pre-scaling) context length.

    Returns:
        Multiplicative scaling factor applied to cos/sin caches.
    """
    scale = max_position_embeddings / original_max_position_embeddings
    return math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))


def build_rope_inv_freq(dim: int, base: float) -> torch.Tensor:
    """Compute inverse-frequency buffer for rotary embeddings.

    Args:
        dim: Head dimension (must be even).
        base: RoPE base frequency.

    Returns:
        1-D float32 tensor of shape ``(dim // 2,)``.
    """
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))


def build_rope_cos_sin_cache(
    seq_len: int,
    inv_freq: torch.Tensor,
    scaling_factor: float,
    short_factor: list[float],
    long_factor: list[float],
    original_max_position_embeddings: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the cos/sin embedding caches used by MiniCPMLongRoPE.

    Args:
        seq_len: Maximum sequence length to cache.
        inv_freq: Inverse-frequency tensor of shape ``(dim // 2,)``.
        scaling_factor: Amplitude multiplier (from :func:`compute_rope_scaling_factor`).
        short_factor: Per-frequency scaling factors for short sequences.
        long_factor: Per-frequency scaling factors for long sequences.
        original_max_position_embeddings: Threshold that decides short vs long factors.
        dtype: Target dtype for the cached tensors.

    Returns:
        Tuple ``(cos_cache, sin_cache)`` each of shape ``(seq_len, dim)``.
    """
    t = torch.arange(seq_len, dtype=inv_freq.dtype)
    ext_factors = (
        torch.tensor(long_factor, dtype=torch.float32)
        if seq_len > original_max_position_embeddings
        else torch.tensor(short_factor, dtype=torch.float32)
    )
    freqs = torch.mul(torch.outer(t, 1.0 / ext_factors), inv_freq.to(dtype=torch.float32))
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cache = emb.cos().to(dtype) * scaling_factor
    sin_cache = emb.sin().to(dtype) * scaling_factor
    return cos_cache, sin_cache


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to a tensor.

    This is the pure-math implementation used inside MiniCPMLongRoPE.

    Args:
        x: Float tensor of shape ``(num_tokens, num_heads, head_dim)``.
        cos: Cosine cache slice of shape ``(num_tokens, head_dim)``.
        sin: Sine cache slice of shape ``(num_tokens, head_dim)``.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    cos = cos.unsqueeze(1)  # (num_tokens, 1, head_dim)
    sin = sin.unsqueeze(1)
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    rotate_half_x = torch.cat((-x2, x1), dim=-1)
    result = x * cos.to(torch.float32) + rotate_half_x * sin.to(torch.float32)
    return result.to(orig_dtype)


# ---------------------------------------------------------------------------
# Sinusoidal positional embedding helper
# ---------------------------------------------------------------------------


def sinusoidal_pos_emb(x: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """Compute sinusoidal positional embeddings (SinusoidalPosEmb.forward logic).

    Args:
        x: Scalar or 1-D float tensor of timestep values.
        dim: Embedding dimension (must be even; each half is sin / cos).
        scale: Multiplier applied before computing sin/cos.

    Returns:
        Float tensor of shape ``(len(x), dim)``.
    """
    if x.ndim < 1:
        x = x.unsqueeze(0)
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=x.device) * -emb)
    emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


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
