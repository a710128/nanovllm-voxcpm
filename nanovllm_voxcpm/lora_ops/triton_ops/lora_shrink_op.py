from __future__ import annotations

import torch
import triton
import triton.language as tl

from nanovllm_voxcpm.lora_ops.triton_ops.kernel_utils import do_shrink_kernel
from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_a_ptr, get_lora_op_configs, supports_pdl

_SMALL_M_THRESHOLD = 32
_SMALL_M_MAX_ACTIVE_LORAS = 32
_SMALL_M_MAX_BLOCK_RANK = 8
_SMALL_M_MAX_BLOCK_K = 1024


@triton.jit
def _lora_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_sk_m_n = tl.program_id(axis=0)
    pid_sk = pid_sk_m_n % SPLIT_K
    pid_m_n = pid_sk_m_n // SPLIT_K
    num_pid_in_group = GROUP_SIZE_M * cta_n_num
    group_id = pid_m_n // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(cta_m_num - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m_n % num_pid_in_group) % group_size_m)
    pid_n = (pid_m_n % num_pid_in_group) // group_size_m
    slice_id = tl.program_id(axis=1)
    lora_idx = tl.program_id(axis=2)

    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        return
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)
    cta_m_offset = pid_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        return
    cta_m_len = min(BLOCK_M, lora_m_size - cta_m_offset)
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    cta_lora_seq_indices = token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    offset_m = tl.arange(0, BLOCK_M) % cta_m_len
    ram = tl.load(cta_lora_seq_indices + offset_m)
    do_shrink_kernel(
        pid_n,
        pid_sk,
        slice_id,
        lora_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        cta_m_len,
        ram,
        input_d0_stride,
        input_d1_stride,
        lora_d0_stride,
        lora_d1_stride,
        lora_d2_stride,
        output_d0_stride,
        output_d1_stride,
        output_d2_stride,
        scaling,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        SLICE_NUM,
        USE_GDC,
    )


@triton.jit
def _lora_shrink_small_m_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_lora_mapping,
    scaling,
    input_d0_stride,
    input_d1_stride,
    lora_d0_stride,
    lora_d1_stride,
    lora_d2_stride,
    output_d0_stride,
    output_d1_stride,
    output_d2_stride,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
):
    pid_sk_mr = tl.program_id(axis=0)
    slice_id = tl.program_id(axis=1)

    pid_sk = pid_sk_mr % SPLIT_K
    pid_mr = pid_sk_mr // SPLIT_K
    pid_m = pid_mr % M
    pid_r = pid_mr // M

    lora_id = tl.load(token_lora_mapping + pid_m)
    if lora_id == -1:
        return

    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(tl.pointer_type(input_ptr.dtype.element_ty))

    offset_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offset_r < N
    load_r = tl.max_contiguous(tl.multiple_of(offset_r % N, BLOCK_R), BLOCK_R)

    accumulator = tl.zeros((1, BLOCK_R), dtype=tl.float32)
    split_k_start = pid_sk * BLOCK_K
    step_k = BLOCK_K * SPLIT_K
    a_ptr = input_ptr + pid_m * input_d0_stride + (split_k_start + tl.arange(0, BLOCK_K)) * input_d1_stride
    b_ptr = (
        cur_lora_ptr
        + lora_id * lora_d0_stride
        + load_r[None, :] * lora_d1_stride
        + (split_k_start + tl.arange(0, BLOCK_K))[:, None] * lora_d2_stride
    )
    for k_start in range(0, K, step_k):
        offset_k = k_start + split_k_start + tl.arange(0, BLOCK_K)
        mask_k = offset_k < K
        tiled_a = tl.load(a_ptr, mask=mask_k, other=0.0)
        tiled_b = tl.load(b_ptr, mask=mask_k[:, None] & mask_r[None, :], other=0.0)
        accumulator += tl.dot(tiled_a[None, :], tiled_b, out_dtype=tl.float32)
        a_ptr += step_k * input_d1_stride
        b_ptr += step_k * lora_d2_stride

    accumulator = tl.reshape(accumulator, (BLOCK_R,))
    accumulator *= scaling
    out_ptr = out_ptr + slice_id * output_d0_stride + pid_m * output_d1_stride + offset_r * output_d2_stride
    if SPLIT_K == 1:
        tl.store(out_ptr, accumulator, mask=mask_r)
    else:
        tl.atomic_add(out_ptr, accumulator, mask=mask_r, sem="relaxed")


def _use_small_m_shrink_path(M: int, N: int, num_active_loras: int, num_slices: int) -> bool:
    return M < _SMALL_M_THRESHOLD and N > 0 and 0 < num_active_loras <= _SMALL_M_MAX_ACTIVE_LORAS and num_slices > 0


def _select_small_m_kernel_config(K: int, N: int) -> tuple[int, int, int]:
    capped_k = min(K, _SMALL_M_MAX_BLOCK_K)
    if K >= 4096 and N <= 2:
        block_k = 512
        block_r = 1
        split_k = min(8, triton.cdiv(K, block_k))
    elif K >= 4096:
        block_k = 512
        block_r = min(4, N)
        split_k = min(8, triton.cdiv(K, block_k))
    elif capped_k >= 1024:
        block_k = 1024
        block_r = 1
        split_k = 1
    elif capped_k >= 512:
        block_k = 512
        block_r = 2
        split_k = 1
    elif capped_k >= 256:
        block_k = 256
        block_r = 4
        split_k = 1
    elif capped_k >= 128:
        block_k = 128
        block_r = 8
        split_k = 1
    else:
        block_k = 64
        block_r = 8
        split_k = 1
    return block_k, min(block_r, N, _SMALL_M_MAX_BLOCK_RANK), split_k


def _run_small_m_shrink(
    inputs: torch.Tensor,
    lora_ptr_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    M: int,
    N: int,
    K: int,
    token_lora_mapping: torch.Tensor,
    scaling: float,
    lora_strides_d0: int,
    lora_strides_d1: int,
    lora_strides_d2: int,
    num_slices: int,
) -> None:
    block_k, block_r, split_k = _select_small_m_kernel_config(K=K, N=N)
    num_warps = 8 if block_k >= 1024 else 4 if block_k >= 256 else 2
    output_tensor.zero_()
    grid = (split_k * M * triton.cdiv(N, block_r), num_slices)
    _lora_shrink_small_m_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        N,
        K,
        token_lora_mapping,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.stride(2),
        BLOCK_R=block_r,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        SLICE_NUM=num_slices,
        num_warps=num_warps,
        num_stages=2,
    )


@torch.inference_mode()
def lora_shrink(
    inputs: torch.Tensor,
    lora_a_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag: bool,
    num_active_loras: int,
    scaling: float,
) -> None:
    if no_lora_flag:
        return
    lora_ptr_tensor, lora_strides_d0, lora_strides_d1, lora_strides_d2 = _get_lora_a_ptr(lora_a_weights, inputs.device)
    M = inputs.size(0)
    N, K = lora_a_weights[0].shape[-2:]
    num_slices = len(lora_a_weights)
    max_loras = lora_ids.size(0)
    if _use_small_m_shrink_path(M=M, N=N, num_active_loras=num_active_loras, num_slices=num_slices):
        _run_small_m_shrink(
            inputs=inputs,
            lora_ptr_tensor=lora_ptr_tensor,
            output_tensor=output_tensor,
            M=M,
            N=N,
            K=K,
            token_lora_mapping=token_lora_mapping,
            scaling=scaling,
            lora_strides_d0=lora_strides_d0,
            lora_strides_d1=lora_strides_d1,
            lora_strides_d2=lora_strides_d2,
            num_slices=num_slices,
        )
        return

    output_tensor.zero_()
    kernel_config = get_lora_op_configs(
        "shrink", max_loras=max_loras, batch=M, hidden_size=K, rank=N, num_slices=num_slices
    )
    block_m = kernel_config["block_m"]
    block_n = kernel_config["block_n"]
    block_k = kernel_config["block_k"]
    split_k = kernel_config["split_k"]
    group_size_m = kernel_config["group_size_m"]
    if block_k is None or split_k is None or group_size_m is None:
        raise RuntimeError("Invalid shrink kernel config")
    even_k = K % (block_k * split_k) == 0
    grid = (split_k * triton.cdiv(M, block_m) * triton.cdiv(N, block_n), num_slices, num_active_loras)
    use_gdc = supports_pdl(inputs.device)
    _lora_shrink_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        inputs.stride(0),
        inputs.stride(1),
        lora_strides_d0,
        lora_strides_d1,
        lora_strides_d2,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor.stride(2),
        block_m,
        block_n,
        block_k,
        even_k,
        split_k,
        group_size_m,
        num_slices,
        use_gdc,
        num_warps=kernel_config["num_warps"],
        num_ctas=kernel_config["num_ctas"],
        num_stages=kernel_config["num_stages"],
        launch_pdl=use_gdc,
    )
