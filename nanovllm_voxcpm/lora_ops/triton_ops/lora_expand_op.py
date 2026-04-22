from __future__ import annotations

import torch
import triton
import triton.language as tl

from nanovllm_voxcpm.lora_ops.triton_ops.kernel_utils import do_expand_kernel
from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_b_ptr, get_lora_op_configs, supports_pdl


@triton.jit
def _lora_expand_kernel(
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
    slice_start_loc,
    input_d0_stride,
    input_d1_stride,
    input_d2_stride,
    ls_d0_ptr,
    ls_d1_ptr,
    ls_d2_ptr,
    output_d0_stride,
    output_d1_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    USE_GDC: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_n_num = tl.cdiv(N, BLOCK_N)
    cta_m_num = tl.cdiv(M, BLOCK_M)
    pid_mn = tl.program_id(axis=0)
    pid_m = pid_mn % cta_m_num
    pid_n = (pid_mn // cta_m_num) % cta_n_num
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
    do_expand_kernel(
        pid_n,
        lora_id,
        slice_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        cta_m_len,
        ram,
        slice_start_loc,
        input_d0_stride,
        input_d1_stride,
        input_d2_stride,
        ls_d0_ptr,
        ls_d1_ptr,
        ls_d2_ptr,
        output_d0_stride,
        output_d1_stride,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SAME_STRIDE,
        SLICE_NUM,
        EVEN_K,
        CAST_TYPE,
        ADD_INPUTS,
        USE_GDC,
    )


@torch.inference_mode()
def lora_expand(
    inputs: torch.Tensor,
    lora_b_weights: list[torch.Tensor],
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    no_lora_flag: bool,
    num_active_loras: int,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    if no_lora_flag:
        return
    (
        slice_start_tensor,
        lora_ptr_tensor,
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        hidden_sizes_tensor,
        same_stride,
        max_n,
    ) = _get_lora_b_ptr(lora_b_weights, offset_start, inputs.device)
    M = inputs.size(1)
    K = lora_b_weights[0].shape[-1]
    max_loras = lora_ids.size(0)
    num_slices = len(lora_b_weights)
    cast_type = inputs.dtype == torch.float32 and lora_b_weights[0].dtype in [torch.float16, torch.bfloat16]
    kernel_config = get_lora_op_configs(
        "expand", max_loras=max_loras, batch=M, hidden_size=max_n, rank=K, num_slices=num_slices, add_inputs=add_inputs
    )
    block_m = kernel_config["block_m"]
    block_n = kernel_config["block_n"]
    block_k = kernel_config["block_k"]
    if block_k is None:
        raise RuntimeError("Invalid expand kernel config")
    even_k = K % block_k == 0
    grid = (triton.cdiv(M, block_m) * triton.cdiv(max_n, block_n), num_slices, num_active_loras)
    use_gdc = supports_pdl(inputs.device)
    _lora_expand_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        M,
        max_n,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        slice_start_tensor,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        block_m,
        block_n,
        block_k,
        even_k,
        add_inputs,
        cast_type,
        num_slices,
        same_stride,
        use_gdc,
        num_warps=kernel_config["num_warps"],
        num_ctas=kernel_config["num_ctas"],
        num_stages=kernel_config["num_stages"],
        launch_pdl=use_gdc,
    )
