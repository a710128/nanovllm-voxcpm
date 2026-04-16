from __future__ import annotations

import torch
import triton
import triton.language as tl

from nanovllm_voxcpm.lora_ops.triton_ops.kernel_utils import do_shrink_kernel
from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_a_ptr, get_lora_op_configs, supports_pdl


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
    output_tensor.zero_()
    kernel_config = get_lora_op_configs(
        "shrink", max_loras=max_loras, batch=M, hidden_size=K, rank=N, num_slices=num_slices
    )
    block_m = kernel_config["block_m"]
    block_n = kernel_config["block_n"]
    block_k = kernel_config["block_k"]
    split_k = kernel_config["split_k"]
    group_size_m = kernel_config["group_size_m"]
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
