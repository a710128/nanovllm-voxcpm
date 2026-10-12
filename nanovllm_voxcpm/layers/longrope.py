"""Fused inference-only LongRoPE with the eager path's FP32 rounding."""

import torch
import triton
import triton.language as tl


@triton.jit
def _longrope_kernel(
    positions,
    query,
    key,
    cos_cache,
    sin_cache,
    query_out,
    key_out,
    positions_stride,
    query_stride,
    key_stride,
    query_inner_stride,
    key_inner_stride,
    cache_stride,
    HEAD_DIM: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    KEY_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dim = offset % HEAD_DIM
    paired_offset = offset - dim + (dim + HEAD_DIM // 2) % HEAD_DIM
    sign = tl.where(dim < HEAD_DIM // 2, -1.0, 1.0)
    position = tl.load(positions + token * positions_stride)
    cos = tl.load(cos_cache + position * cache_stride + dim).to(tl.float32)
    sin = tl.load(sin_cache + position * cache_stride + dim).to(tl.float32)
    q = tl.load(query + token * query_stride + offset * query_inner_stride, offset < QUERY_WIDTH, other=0).to(
        tl.float32
    )
    qr = tl.load(query + token * query_stride + paired_offset * query_inner_stride, offset < QUERY_WIDTH, other=0).to(
        tl.float32
    )
    k = tl.load(key + token * key_stride + offset * key_inner_stride, offset < KEY_WIDTH, other=0).to(tl.float32)
    kr = tl.load(key + token * key_stride + paired_offset * key_inner_stride, offset < KEY_WIDTH, other=0).to(
        tl.float32
    )
    # Keep the two products separately rounded, as in the eager FP32 path.
    # FMA changes bf16 rounding and can alter an autoregressive waveform.
    q_result = q * cos + (qr * sign) * sin
    k_result = k * cos + (kr * sign) * sin
    tl.store(query_out + token * QUERY_WIDTH + offset, q_result, offset < QUERY_WIDTH)
    tl.store(key_out + token * KEY_WIDTH + offset, k_result, offset < KEY_WIDTH)


def apply_longrope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate packed Q/K without materializing indexed caches or FP32 temporaries.

    Args:
        positions: One-dimensional position indices, one per flattened token.
        query: QKV projection slice, with heads packed within each token.
        key: K projection slice with the same token layout as query.
        cos_cache: Contiguous FP32 LongRoPE cosine cache.
        sin_cache: Contiguous FP32 LongRoPE sine cache, matching cos_cache.
        head_dim: Even head width; the two halves are rotated against each other.

    Returns:
        Contiguous query and key tensors with their original shapes and dtypes.
        This CUDA inference path does not provide autograd.
    """
    tokens = positions.numel()
    q = query.reshape(tokens, -1)
    k = key.reshape(tokens, -1)
    query_out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    key_out = torch.empty(key.shape, dtype=key.dtype, device=key.device)
    block_size = 256
    _longrope_kernel[(tokens, triton.cdiv(max(q.size(1), k.size(1)), block_size))](
        positions,
        q,
        k,
        cos_cache,
        sin_cache,
        query_out,
        key_out,
        positions.stride(0),
        q.stride(0),
        k.stride(0),
        q.stride(1),
        k.stride(1),
        cos_cache.stride(0),
        head_dim,
        q.size(1),
        k.size(1),
        block_size,
        enable_fp_fusion=False,
    )
    return query_out, key_out
