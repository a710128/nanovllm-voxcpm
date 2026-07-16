"""CPU-testable pure helpers extracted from models/voxcpm/model.py.

All functions in this module are pure PyTorch tensor math — no flash_attn,
no Triton kernels, no CUDA graphs.  They can be exercised on CPU in unit
tests without a GPU.

Extraction map (model.py → model_utils.py):
  rotate_half                 — standalone free function (lines 37-40)
  apply_rotary_pos_emb        — standalone free function (lines 43-55)
  apply_rotary_emb_tokens     — body of MiniCPMLongRoPE._apply_rotary_emb (lines 135-153)
  compute_longrope_scaling_factor — math from MiniCPMLongRoPE.__init__ (line 81)
  compute_longrope_freqs      — body of MiniCPMLongRoPE._set_cos_sin_cache (lines 90-107)
  optimized_scale             — body of UnifiedCFM.optimized_scale (lines 671-676)
  sway_sampling_schedule      — body of UnifiedCFM.forward t_span logic (lines 665-667)
  sinusoidal_pos_emb          — body of SinusoidalPosEmb.forward (lines 494-503)
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Rotary Position Embedding helpers
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of *x*.

    Splits the last dimension into two equal halves ``(x1, x2)`` and returns
    ``cat(-x2, x1)``.  This is the standard RoPE rotation operator.

    Args:
        x: Input tensor whose last dimension is even.

    Returns:
        Tensor of the same shape as *x*.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors.

    Equivalent to the MiniCPM modeling implementation.  Computation is done
    in float32 and the result is cast back to the original key dtype.

    Args:
        q: Query tensor.
        k: Key tensor.  Determines the output dtype.
        cos: Cosine cache of shape ``[max_seq_len, head_dim]``.
        sin: Sine cache of shape ``[max_seq_len, head_dim]``.
        position_ids: Position indices used to index into *cos*/*sin*.
        unsqueeze_dim: Dimension along which to unsqueeze *cos*/*sin* for
            broadcasting.  Default is ``1`` (batch / head dimension).

    Returns:
        Tuple ``(q_embed, k_embed)`` with the same dtype as *k*.
    """
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


def apply_rotary_emb_tokens(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding per token (MiniCPMLongRoPE token-level variant).

    Matches the math inside ``MiniCPMLongRoPE._apply_rotary_emb``.
    Computation is promoted to float32 internally and cast back to the
    original dtype on return.

    Args:
        x: Input tensor of shape ``[num_tokens, num_heads, head_dim]``.
        cos: Cosine values of shape ``[num_tokens, head_dim]``.  Will be
            unsqueezed over the heads dimension for broadcasting.
        sin: Sine values of shape ``[num_tokens, head_dim]``.

    Returns:
        Tensor of the same shape and dtype as *x*.
    """
    orig_dtype = x.dtype
    cos = cos.unsqueeze(1).to(torch.float32)  # [num_tokens, 1, head_dim]
    sin = sin.unsqueeze(1).to(torch.float32)  # [num_tokens, 1, head_dim]
    x = x.to(torch.float32)

    x1, x2 = torch.chunk(x, 2, dim=-1)
    rotate_half_x = torch.cat((-x2, x1), dim=-1)
    result = x * cos + rotate_half_x * sin
    return result.to(orig_dtype)


# ---------------------------------------------------------------------------
# LongRoPE scheduling helpers
# ---------------------------------------------------------------------------


def compute_longrope_scaling_factor(
    max_position_embeddings: int,
    original_max_position_embeddings: int,
) -> float:
    """Compute the LongRoPE amplitude scaling factor.

    This reproduces the formula used in ``MiniCPMLongRoPE.__init__``:

    .. code-block:: text

        scale  = max_position_embeddings / original_max_position_embeddings
        factor = sqrt(1 + log(scale) / log(original_max_position_embeddings))

    Args:
        max_position_embeddings: Extended maximum sequence length.
        original_max_position_embeddings: Original (pre-extension) maximum
            sequence length.

    Returns:
        Floating-point scaling factor (>= 1.0).
    """
    scale = max_position_embeddings / original_max_position_embeddings
    return math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))


def compute_longrope_freqs(
    seq_len: int,
    inv_freq: torch.Tensor,
    ext_factors: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute LongRoPE cosine/sine caches for *seq_len* positions.

    Mirrors the body of ``MiniCPMLongRoPE._set_cos_sin_cache``.

    Args:
        seq_len: Number of positions to pre-compute.
        inv_freq: Inverse frequency tensor of shape ``[head_dim // 2]``.
        ext_factors: Per-frequency extension factors of shape
            ``[head_dim // 2]``.  Set to ``ones`` for standard RoPE.
        dtype: Target dtype for the returned caches.
        device: Target device for the returned caches.

    Returns:
        Tuple ``(cos_cache, sin_cache)`` each of shape
        ``[seq_len, head_dim]`` where ``head_dim = 2 * len(inv_freq)``.
    """
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs = torch.mul(
        torch.outer(t, 1.0 / ext_factors).to(device=device),
        inv_freq.to(device=device).to(dtype),
    )
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cache = emb.cos().to(dtype)
    sin_cache = emb.sin().to(dtype)
    return cos_cache, sin_cache


# ---------------------------------------------------------------------------
# Diffusion / CFM helpers
# ---------------------------------------------------------------------------


def optimized_scale(
    positive_flat: torch.Tensor,
    negative_flat: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample optimal CFG scale coefficient.

    Mirrors ``UnifiedCFM.optimized_scale``.  Given flattened positive and
    negative flow predictions, returns the scalar ``s*`` that projects the
    negative prediction onto the positive direction:

    .. code-block:: text

        s* = dot(pos, neg) / (||neg||² + eps)

    Args:
        positive_flat: Positive (conditioned) flow of shape ``[batch, -1]``.
        negative_flat: Negative (unconditioned) flow of shape ``[batch, -1]``.

    Returns:
        Tensor of shape ``[batch, 1]``.
    """
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm


def sway_sampling_schedule(
    n_timesteps: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the sway-sampling time schedule used by ``UnifiedCFM.forward``.

    Produces a decreasing sequence from 1 to 0 with ``n_timesteps + 1``
    points, then applies the sway warp:

    .. code-block:: text

        t' = t + (cos(pi/2 * t) - 1 + t)

    Args:
        n_timesteps: Number of Euler solver steps.
        device: Target device.
        dtype: Target dtype.

    Returns:
        1-D tensor of shape ``[n_timesteps + 1]``.
    """
    t_span = torch.linspace(1, 0, n_timesteps + 1, device=device, dtype=dtype)
    t_span = t_span + (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
    return t_span


# ---------------------------------------------------------------------------
# Sinusoidal position embedding (SinusoidalPosEmb.forward logic)
# ---------------------------------------------------------------------------


def sinusoidal_pos_emb(
    x: torch.Tensor,
    dim: int,
    scale: float = 1000,
) -> torch.Tensor:
    """Compute sinusoidal position embeddings.

    Mirrors ``SinusoidalPosEmb.forward``.

    Args:
        x: 1-D (or 0-D) position tensor.  0-D inputs are unsqueezed to 1-D.
        dim: Embedding dimension (must be even).
        scale: Frequency scale factor.  Default is ``1000``.

    Returns:
        Tensor of shape ``[len(x), dim]``.

    Raises:
        AssertionError: If *dim* is odd.
    """
    assert dim % 2 == 0, "sinusoidal_pos_emb requires dim to be even"
    if x.ndim < 1:
        x = x.unsqueeze(0)
    device = x.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=device) * -emb)
    emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb
