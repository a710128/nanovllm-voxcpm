"""RoPE and positional-embedding helpers for VoxCPM2.

These helpers are pure CPU math and are kept separate from the broader
VoxCPM2 utility module so the config and CFG helpers stay under the file-size
ceiling.
"""

import math

import torch


def compute_rope_scaling_factor(
    max_position_embeddings: int,
    original_max_position_embeddings: int,
) -> float:
    """Compute the LongRoPE amplitude-scaling factor."""
    scale = max_position_embeddings / original_max_position_embeddings
    return math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))


def build_rope_inv_freq(dim: int, base: float) -> torch.Tensor:
    """Compute inverse-frequency buffer for rotary embeddings."""
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
    """Build the cos/sin embedding caches used by MiniCPMLongRoPE."""
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
    """Apply rotary position embeddings to a tensor."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    rotate_half_x = torch.cat((-x2, x1), dim=-1)
    result = x * cos.to(torch.float32) + rotate_half_x * sin.to(torch.float32)
    return result.to(orig_dtype)


def sinusoidal_pos_emb(x: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """Compute sinusoidal positional embeddings."""
    if x.ndim < 1:
        x = x.unsqueeze(0)
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=x.device) * -emb)
    emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat((emb.sin(), emb.cos()), dim=-1)
