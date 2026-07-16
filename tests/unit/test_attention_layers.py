"""Tests for nanovllm_voxcpm.layers.attention – CPU-reachable non-kernel logic.

The GPU Triton kernel (store_kvcache_kernel) and all flash_attn call sites are
excluded from coverage via ``# pragma: no cover`` in the source.  Any test that
would actually *invoke* a GPU kernel must be marked ``@pytest.mark.gpu`` so it
is auto-skipped on CPU CI.
"""

import pytest

# ---------------------------------------------------------------------------
# Attention.__init__ (lines 52-66) – fully CPU-testable
# ---------------------------------------------------------------------------


class TestAttentionInit:
    """Attention.__init__ stores all parameters and registers empty caches."""

    def _make(self, **kwargs):
        from nanovllm_voxcpm.layers.attention import Attention

        defaults = dict(num_heads=8, head_dim=64, scale=0.125, num_kv_heads=8)
        defaults.update(kwargs)
        return Attention(**defaults)

    def test_stores_num_heads(self):
        attn = self._make(num_heads=16)
        assert attn.num_heads == 16

    def test_stores_head_dim(self):
        attn = self._make(head_dim=128)
        assert attn.head_dim == 128

    def test_stores_scale(self):
        attn = self._make(scale=0.0625)
        assert attn.scale == 0.0625

    def test_stores_num_kv_heads(self):
        attn = self._make(num_kv_heads=4)
        assert attn.num_kv_heads == 4

    def test_default_is_causal_true(self):
        attn = self._make()
        assert attn.is_causal is True

    def test_is_causal_false(self):
        attn = self._make(is_causal=False)
        assert attn.is_causal is False

    def test_k_cache_v_cache_initially_empty(self):
        attn = self._make()
        assert attn.k_cache.numel() == 0
        assert attn.v_cache.numel() == 0

    def test_is_nn_module(self):
        import torch.nn as nn

        attn = self._make()
        assert isinstance(attn, nn.Module)

    def test_parameters_have_no_trainable_params(self):
        """Attention layer holds no trainable weights; k/v cache are not parameters."""
        attn = self._make()
        params = list(attn.parameters())
        assert params == []

    def test_different_num_heads_and_kv_heads(self):
        """GQA config: num_kv_heads < num_heads."""
        attn = self._make(num_heads=32, num_kv_heads=8)
        assert attn.num_heads == 32
        assert attn.num_kv_heads == 8

    def test_k_cache_same_object_as_v_cache_on_init(self):
        """Both k_cache and v_cache are initialised from the same empty tensor."""
        attn = self._make()
        # They start as the *same* tensor object (chained assignment).
        assert attn.k_cache is attn.v_cache

    def test_repr_contains_class_name(self):
        attn = self._make()
        assert "Attention" in repr(attn)


# ---------------------------------------------------------------------------
# store_kvcache assertion guards (lines 41-46) – CPU-testable via bad inputs
#
# The actual Triton dispatch on line 47 is pragma'd.  We only reach that line
# when ALL asserts pass.  By passing bad tensors we exercise lines 41-46 while
# triggering AssertionError before the GPU call.
# ---------------------------------------------------------------------------


class TestStoreKvcacheAssertions:
    """store_kvcache validates tensor shapes/strides before dispatching kernel."""

    def _make_tensors(self, N=2, num_heads=2, head_dim=4, num_blocks=4, block_size=8):
        """Return well-formed key, value, k_cache, v_cache, slot_mapping tensors.

        k_cache / v_cache are shaped (num_blocks, block_size, num_heads, head_dim)
        so that stride(1) == num_heads * head_dim == D, matching the assertion in
        store_kvcache (lines 45-46 of attention.py).
        """
        import torch

        D = num_heads * head_dim
        key = torch.randn(N, num_heads, head_dim)
        value = torch.randn(N, num_heads, head_dim)
        k_cache = torch.empty(num_blocks, block_size, num_heads, head_dim)
        v_cache = torch.empty(num_blocks, block_size, num_heads, head_dim)
        slot_mapping = torch.zeros(N, dtype=torch.long)
        assert k_cache.stride(1) == D  # sanity-check helper is correct
        return key, value, k_cache, v_cache, slot_mapping

    def test_mismatched_slot_mapping_length_raises(self):
        import torch

        from nanovllm_voxcpm.layers.attention import store_kvcache

        key, value, k_cache, v_cache, _ = self._make_tensors(N=3)
        bad_slot = torch.zeros(2, dtype=torch.long)
        with pytest.raises(AssertionError):
            store_kvcache(key, value, k_cache, v_cache, bad_slot)

    def test_non_contiguous_key_raises(self):
        from nanovllm_voxcpm.layers.attention import store_kvcache

        key, value, k_cache, v_cache, slot_mapping = self._make_tensors()
        # Transpose to break head_dim stride assumption.
        bad_key = key.permute(0, 2, 1)  # shape (N, head_dim, num_heads)
        with pytest.raises(AssertionError):
            store_kvcache(bad_key, value, k_cache, v_cache, slot_mapping)

    def test_bad_cache_stride_raises(self):
        import torch

        from nanovllm_voxcpm.layers.attention import store_kvcache

        N, num_heads, head_dim, num_blocks, block_size = 2, 2, 4, 4, 8
        D = num_heads * head_dim
        key = torch.randn(N, num_heads, head_dim)
        value = torch.randn(N, num_heads, head_dim)
        bad_k_cache = torch.zeros(num_blocks, D)
        v_cache = torch.empty(num_blocks, block_size, num_heads, head_dim)
        slot_mapping = torch.zeros(N, dtype=torch.long)
        assert bad_k_cache.stride(1) != D
        with pytest.raises(AssertionError):
            store_kvcache(key, value, bad_k_cache, v_cache, slot_mapping)

    def test_shape_extraction_uses_key_shape(self):
        import torch

        from nanovllm_voxcpm.layers.attention import store_kvcache

        N, num_heads, head_dim, num_blocks, block_size = 2, 2, 4, 4, 8
        key = torch.randn(N, num_heads, head_dim)
        value = torch.randn(N, num_heads, head_dim)
        k_cache = torch.empty(num_blocks, block_size, num_heads, head_dim)
        v_cache = torch.empty(num_blocks, block_size, num_heads, head_dim)
        bad_slot = torch.zeros(99, dtype=torch.long)
        with pytest.raises(AssertionError):
            store_kvcache(key, value, k_cache, v_cache, bad_slot)


@pytest.mark.gpu
def test_store_kvcache_writes_valid_cuda_slots():
    import torch

    from nanovllm_voxcpm.layers.attention import store_kvcache

    num_tokens, num_heads, head_dim, block_size = 3, 2, 64, 8
    key = torch.randn(num_tokens, num_heads, head_dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    k_cache = torch.zeros(1, block_size, num_heads, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.arange(num_tokens, device="cuda", dtype=torch.long)

    store_kvcache(key, value, k_cache, v_cache, slot_mapping)
    torch.cuda.synchronize()

    assert torch.allclose(k_cache.view(-1, num_heads, head_dim)[slot_mapping], key)
    assert torch.allclose(v_cache.view(-1, num_heads, head_dim)[slot_mapping], value)
    assert k_cache.is_cuda
    assert v_cache.is_cuda
    assert k_cache.shape == (1, block_size, num_heads, head_dim)
    assert v_cache.shape == (1, block_size, num_heads, head_dim)
    assert torch.isfinite(k_cache).all()
    assert torch.isfinite(v_cache).all()


@pytest.mark.gpu
def test_causal_attention_prefill_and_decode_execute_real_cuda_kernels():
    import torch

    from nanovllm_voxcpm.layers.attention import Attention
    from nanovllm_voxcpm.utils.context import reset_context, set_context

    seq_len, num_heads, head_dim, block_size = 4, 2, 64, 256
    attention = Attention(
        num_heads=num_heads,
        head_dim=head_dim,
        scale=head_dim**-0.5,
        num_kv_heads=num_heads,
        is_causal=True,
    )
    attention.k_cache = torch.zeros(1, block_size, num_heads, head_dim, device="cuda", dtype=torch.float16)
    attention.v_cache = torch.zeros_like(attention.k_cache)
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    prefill_slots = torch.arange(seq_len, device="cuda", dtype=torch.long)
    cu_seqlens = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)

    try:
        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=prefill_slots,
            block_tables=None,
        )
        prefill_output = attention(q, k, v)
        torch.cuda.synchronize()

        assert prefill_output.is_cuda
        assert prefill_output.shape == q.shape
        assert prefill_output.dtype == q.dtype
        assert torch.isfinite(prefill_output).all()
        assert torch.allclose(attention.k_cache.view(-1, num_heads, head_dim)[prefill_slots], k)
        assert torch.allclose(attention.v_cache.view(-1, num_heads, head_dim)[prefill_slots], v)

        decode_q = torch.randn(1, num_heads, head_dim, device="cuda", dtype=torch.float16)
        decode_k = torch.randn_like(decode_q)
        decode_v = torch.randn_like(decode_q)
        decode_slot = torch.tensor([seq_len], device="cuda", dtype=torch.long)
        set_context(
            is_prefill=False,
            slot_mapping=decode_slot,
            context_lens=torch.tensor([seq_len + 1], device="cuda", dtype=torch.int32),
            block_tables=torch.tensor([[0]], device="cuda", dtype=torch.int32),
        )
        decode_output = attention(decode_q, decode_k, decode_v)
        torch.cuda.synchronize()

        assert decode_output.is_cuda
        assert decode_output.shape == (1, 1, num_heads, head_dim)
        assert decode_output.dtype == decode_q.dtype
        assert torch.isfinite(decode_output).all()
        assert torch.allclose(attention.k_cache.view(-1, num_heads, head_dim)[decode_slot], decode_k)
        assert torch.allclose(attention.v_cache.view(-1, num_heads, head_dim)[decode_slot], decode_v)
    finally:
        reset_context()


@pytest.mark.gpu
def test_allocate_kv_cache_allocates_bounded_real_cuda_cache():
    import torch
    import torch.nn as nn

    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.layers.attention import Attention

    class _Runner(model_runner.BaseModelRunner):
        @property
        def dtype(self):
            return torch.float16

    class _Config:
        num_kvcache_blocks = 1
        gpu_memory_utilization = 0.9

    previous_cuda_device = torch.cuda.current_device()
    previous_default_device = torch.get_default_device()
    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.cuda.set_device(0)
        torch.set_default_dtype(torch.float16)
        torch.set_default_device("cuda")
        num_kv_heads, head_dim, block_size = 2, 64, 8
        attention = Attention(
            num_heads=num_kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            is_causal=True,
        )
        runner = object.__new__(_Runner)
        runner.model = nn.Module()
        runner.model.add_module("attention", attention)
        runner.block_size = block_size
        runner._config = _Config()

        runner.allocate_kv_cache()
        torch.cuda.synchronize()

        assert attention.k_cache.is_cuda
        assert attention.v_cache.is_cuda
        assert attention.k_cache.shape == (1, block_size, num_kv_heads, head_dim)
        assert attention.v_cache.shape == (1, block_size, num_kv_heads, head_dim)
        # allocate_kv_cache uses torch.empty, so cache contents are uninitialized until a cache write.
    finally:
        torch.cuda.set_device(previous_cuda_device)
        torch.set_default_device(previous_default_device)
        torch.set_default_dtype(previous_default_dtype)
