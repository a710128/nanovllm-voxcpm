from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def mm_k(
    a_ptr,
    b_ptr,
    ak_stride,
    bk_stride,
    offset_k,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    b_dtype: tl.constexpr,
    USE_GDC: tl.constexpr,
    base_k,
):
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    step_k = BLOCK_K * SPLIT_K
    num_iters = tl.cdiv(K, step_k)

    for k in range(num_iters):
        iter_k = k * step_k + base_k
        block_end = iter_k + BLOCK_K

        if EVEN_K:
            tiled_b = tl.load(b_ptr)
            if USE_GDC:
                tl.extra.cuda.gdc_wait()
            tiled_a = tl.load(a_ptr)
            if CAST_TYPE:
                tiled_a = tiled_a.to(b_dtype)
            accumulator += tl.dot(tiled_a, tiled_b)
        else:
            if iter_k >= K:
                pass
            elif block_end <= K:
                tiled_b = tl.load(b_ptr)
                if USE_GDC:
                    tl.extra.cuda.gdc_wait()
                tiled_a = tl.load(a_ptr)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)
                accumulator += tl.dot(tiled_a, tiled_b)
            else:
                k_offsets = tl.arange(0, BLOCK_K)
                mask = iter_k + k_offsets < K
                tiled_b = tl.load(b_ptr, mask=mask[:, None], other=0.0)
                if USE_GDC:
                    tl.extra.cuda.gdc_wait()
                tiled_a = tl.load(a_ptr, mask=mask[None, :], other=0.0)
                if CAST_TYPE:
                    tiled_a = tiled_a.to(b_dtype)
                accumulator += tl.dot(tiled_a, tiled_b)

        a_ptr += step_k * ak_stride
        b_ptr += step_k * bk_stride

    return accumulator


@triton.jit
def do_shrink_kernel(
    pid_n,
    pid_sk,
    slice_id,
    lora_index,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    if SLICE_NUM == 1:
        cur_lora_ptr = lora_ptr
    else:
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(tl.pointer_type(input_ptr.dtype.element_ty))

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    offset_k = pid_sk * BLOCK_K + tl.arange(0, BLOCK_K)
    a_ptr = input_ptr + ram[:, None] * input_d0_stride + offset_k[None, :] * input_d1_stride
    b_ptr = (
        cur_lora_ptr + lora_d0_stride * lora_index + rbn[None, :] * lora_d1_stride + offset_k[:, None] * lora_d2_stride
    )

    accumulator = mm_k(
        a_ptr,
        b_ptr,
        input_d1_stride,
        lora_d2_stride,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        False,
        cur_lora_ptr.dtype.element_ty,
        False,
        base_k=pid_sk * BLOCK_K,
    )
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_cm = tl.arange(0, BLOCK_M)
    cur_out_ptr = out_ptr if SLICE_NUM == 1 else out_ptr + slice_id * output_d0_stride
    c_ptr = cur_out_ptr + ram[:, None] * output_d1_stride + offset_cn[None, :] * output_d2_stride
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)
    accumulator *= scaling
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask, sem="relaxed")


@triton.jit
def do_expand_kernel(
    pid_n,
    lora_index,
    slice_id,
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    M_LEN,
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAME_STRIDE: tl.constexpr,
    SLICE_NUM: tl.constexpr,
    EVEN_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    USE_GDC: tl.constexpr,
):
    if SAME_STRIDE:
        cur_lora_d0_stride = ls_d0_ptr
        cur_lora_d1_stride = ls_d1_ptr
        cur_lora_d2_stride = ls_d2_ptr
    else:
        cur_lora_d0_stride = tl.load(ls_d0_ptr + slice_id)
        cur_lora_d1_stride = tl.load(ls_d1_ptr + slice_id)
        cur_lora_d2_stride = tl.load(ls_d2_ptr + slice_id)

    if SLICE_NUM == 1:
        cur_input_ptr = input_ptr
        cur_lora_ptr = lora_ptr
    else:
        cur_input_ptr = input_ptr + slice_id * input_d0_stride
        cur_lora_ptr = tl.load(lora_ptr + slice_id).to(tl.pointer_type(out_ptr.dtype.element_ty))

    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)
    a_ptr = cur_input_ptr + ram[:, None] * input_d1_stride + offset_k[None, :] * input_d2_stride
    b_ptr = (
        cur_lora_ptr
        + cur_lora_d0_stride * lora_index
        + offset_k[:, None] * cur_lora_d2_stride
        + rbn[None, :] * cur_lora_d1_stride
    )

    accumulator = mm_k(
        a_ptr,
        b_ptr,
        input_d2_stride,
        cur_lora_d2_stride,
        offset_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        1,
        CAST_TYPE,
        cur_lora_ptr.dtype.element_ty,
        USE_GDC,
        base_k=0,
    )

    tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
    cur_slice_start = slice_start_loc if SLICE_NUM == 1 else tl.load(slice_start_loc + slice_id)
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    offset_cm = tl.arange(0, BLOCK_M)
    c_ptr = out_ptr + ram[:, None] * output_d0_stride + offset_cn[None, :] * output_d1_stride
    c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < (cur_slice_start + N))
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)
