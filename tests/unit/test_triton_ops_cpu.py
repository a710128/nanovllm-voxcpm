"""CPU-side tests for lora_kernel_metadata.py and utils.py.

These files contain pure Python metadata/parameter assembly logic that is
CPU-testable. Actual Triton JIT kernels (kernel_utils.py) are GPU-only and
are excluded from CPU coverage (kernel_utils.py 10% is GPU-only, excluded
from CPU coverage).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# lora_kernel_metadata.py – cover lines 52, 62-64
# ---------------------------------------------------------------------------


class TestLoRAKernelMeta:
    """Tests targeting LoRAKernelMeta.prepare_tensors uncovered paths."""

    def _make_meta(self, max_loras: int = 4, max_num_tokens: int = 16, captured_lora_counts=None):
        from nanovllm_voxcpm.lora_ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

        return LoRAKernelMeta.make(
            max_loras=max_loras,
            max_num_tokens=max_num_tokens,
            device=torch.device("cpu"),
            captured_lora_counts=captured_lora_counts,
        )

    # -- line 52: early-return branch when all tokens map to -1 ---------------

    def test_prepare_tensors_all_no_lora_sets_flag_and_returns_early(self):
        """When every token_lora_mapping entry is -1, no_lora_flag=True and
        the rest of the tensors are left at their reset state (line 52)."""
        meta = self._make_meta(max_loras=2, max_num_tokens=4)
        mapping = torch.tensor([-1, -1, -1], dtype=torch.int32)
        meta.prepare_tensors(mapping)

        assert meta.no_lora_flag is True
        assert meta.num_active_loras == 0
        # The token_lora_mapping buffer should NOT have been updated (early exit)
        # active_lora_ids should be all -1 (reset state)
        assert (meta.active_lora_ids == -1).all()

    def test_prepare_tensors_empty_mapping_sets_no_lora_flag(self):
        """Zero-length mapping → all-no-lora early exit (line 52)."""
        meta = self._make_meta(max_loras=2, max_num_tokens=4)
        mapping = torch.tensor([], dtype=torch.int32)
        meta.prepare_tensors(mapping)

        assert meta.no_lora_flag is True
        assert meta.num_active_loras == 0

    # -- lines 61-64: captured_lora_counts bisect branch ----------------------

    def test_prepare_tensors_captured_counts_rounds_up(self):
        """With captured_lora_counts, num_active_loras is rounded up to the
        next value in the list (lines 62-64)."""
        # captured list: [2, 4, 8]
        # active loras = 1 → bisect_left([2,4,8], 1) = 0 → captured[0] = 2
        meta = self._make_meta(max_loras=4, max_num_tokens=16, captured_lora_counts=[2, 4, 8])
        mapping = torch.tensor([0], dtype=torch.int32)  # 1 unique lora id
        meta.prepare_tensors(mapping)

        assert meta.num_active_loras == 2  # rounded up to first bucket

    def test_prepare_tensors_captured_counts_exact_match(self):
        """When num_active_loras exactly matches a captured count entry, it
        uses that entry (bisect_left returns index of exact match, lines 62-64)."""
        # captured list: [1, 2, 4]
        # active loras = 2 → bisect_left([1,2,4], 2) = 1 → captured[1] = 2
        meta = self._make_meta(max_loras=4, max_num_tokens=16, captured_lora_counts=[1, 2, 4])
        mapping = torch.tensor([0, 1], dtype=torch.int32)  # 2 unique lora ids
        meta.prepare_tensors(mapping)

        assert meta.num_active_loras == 2  # exact bucket match

    def test_prepare_tensors_captured_counts_exceeds_max_uses_actual(self):
        """When num_active_loras > max captured count, bisect returns
        len(captured_lora_counts) → idx >= len → skip, use actual count.
        Lines 61-64: the `if idx < len(...)` check prevents the override."""
        # captured list: [1, 2]
        # active loras = 3 → bisect_left([1,2], 3) = 2 = len(list) → no override
        meta = self._make_meta(max_loras=4, max_num_tokens=16, captured_lora_counts=[1, 2])
        mapping = torch.tensor([0, 1, 2], dtype=torch.int32)  # 3 unique lora ids
        meta.prepare_tensors(mapping)

        assert meta.num_active_loras == 3  # no capture override

    def test_prepare_tensors_no_captured_counts_uses_actual(self):
        """Without captured_lora_counts, num_active_loras equals the real count.
        Ensures the if-branch at line 61 is correctly skipped."""
        meta = self._make_meta(max_loras=4, max_num_tokens=16, captured_lora_counts=None)
        mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
        meta.prepare_tensors(mapping)

        assert meta.num_active_loras == 3

    def test_prepare_tensors_captured_counts_single_bucket_below(self):
        """Single-element captured list; active count below it → rounds up."""
        # captured list: [4]
        # active loras = 1 → bisect_left([4], 1) = 0 → captured[0] = 4
        meta = self._make_meta(max_loras=8, max_num_tokens=16, captured_lora_counts=[4])
        mapping = torch.tensor([0], dtype=torch.int32)
        meta.prepare_tensors(mapping)

        assert meta.num_active_loras == 4

    def test_make_with_captured_lora_counts_sorts_them(self):
        """make() should sort captured_lora_counts (unsorted input)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

        meta = LoRAKernelMeta.make(
            max_loras=4,
            max_num_tokens=8,
            device=torch.device("cpu"),
            captured_lora_counts=[8, 2, 4],
        )
        assert meta.captured_lora_counts == [2, 4, 8]

    def test_meta_args_specialize_active_lora_true(self):
        """meta_args with specialize_active_lora=True uses num_active_loras."""
        meta = self._make_meta(max_loras=2, max_num_tokens=8)
        mapping = torch.tensor([0, 1], dtype=torch.int32)
        meta.prepare_tensors(mapping)

        args = meta.meta_args(token_nums=2, specialize_active_lora=True)
        # 7th element (index 6) should be num_active_loras
        assert args[6] == meta.num_active_loras

    def test_meta_args_specialize_active_lora_false_uses_default(self):
        """meta_args with specialize_active_lora=False uses default_num_active_loras."""
        meta = self._make_meta(max_loras=2, max_num_tokens=8)
        mapping = torch.tensor([0, 1], dtype=torch.int32)
        meta.prepare_tensors(mapping)

        args = meta.meta_args(token_nums=2, specialize_active_lora=False)
        assert args[6] == meta.default_num_active_loras


# ---------------------------------------------------------------------------
# utils.py – cover lines 26-27, 40, 66-67, 99-103, 130-131
# ---------------------------------------------------------------------------


class TestGetLoraAPtr:
    """Tests for _get_lora_a_ptr uncovered paths (lines 26-27, 40)."""

    def setup_method(self):
        from nanovllm_voxcpm.lora_ops.triton_ops import utils as u

        u._LORA_A_PTR_DICT.clear()

    def test_4d_weight_is_squeezed(self):
        """4-D weight (n,1,rank,hidden) is squeezed to 3-D before caching (lines 26-27)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_a_ptr

        # shape: (num_loras, 1, rank, hidden) — 4-D with size(1)==1
        w = torch.randn(2, 1, 4, 8).contiguous()
        ptr, s0, s1, s2 = _get_lora_a_ptr([w], device=torch.device("cpu"))
        # After squeeze dim=1, shape is (2, 4, 8) so strides are those of [2,4,8]
        squeezed = w.squeeze(dim=1)
        assert s0 == squeezed.stride(0)
        assert s1 == squeezed.stride(1)
        assert s2 == squeezed.stride(2)

    def test_3d_weight_no_squeeze(self):
        """3-D weight (num_loras, rank, hidden) uses normal path (line 29)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_a_ptr

        w = torch.randn(2, 4, 8).contiguous()
        ptr, s0, s1, s2 = _get_lora_a_ptr([w], device=torch.device("cpu"))
        assert s0 == w.stride(0)
        assert s1 == w.stride(1)
        assert s2 == w.stride(2)

    def test_mismatched_strides_raises_value_error(self):
        """Non-uniform strides across multiple weights raises ValueError (line 40)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_a_ptr

        # Two weights with different innermost dimension → different strides
        w1 = torch.randn(2, 4, 8).contiguous()   # stride(2) = 1
        w2 = torch.randn(2, 4, 16).contiguous()  # stride(1) = 16 ≠ 8

        with pytest.raises(ValueError, match="same stride"):
            _get_lora_a_ptr([w1, w2], device=torch.device("cpu"))

    def test_cache_hit_returns_same_result(self):
        """Second call with same pointers returns cached tuple (line 17-18)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_a_ptr

        w = torch.randn(2, 4, 8).contiguous()
        r1 = _get_lora_a_ptr([w], device=torch.device("cpu"))
        r2 = _get_lora_a_ptr([w], device=torch.device("cpu"))
        assert r1 is r2


class TestGetLoraBPtr:
    """Tests for _get_lora_b_ptr uncovered paths (lines 66-67, 99-103)."""

    def setup_method(self):
        from nanovllm_voxcpm.lora_ops.triton_ops import utils as u

        u._LORA_B_PTR_DICT.clear()

    def test_4d_weight_is_squeezed(self):
        """4-D lora_b weight (n,1,hidden,rank) is squeezed (lines 66-67)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_b_ptr

        # shape: (num_loras, 1, hidden, rank) — 4-D with size(1)==1
        w = torch.randn(2, 1, 8, 4).contiguous()
        result = _get_lora_b_ptr([w], offset_start=0, device=torch.device("cpu"))
        # Should return 8-tuple without error
        assert len(result) == 8
        # same_stride should be True (single weight)
        assert result[6] is True

    def test_3d_weight_no_squeeze(self):
        """3-D lora_b weight uses normal path (line 69)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_b_ptr

        w = torch.randn(2, 8, 4).contiguous()
        result = _get_lora_b_ptr([w], offset_start=0, device=torch.device("cpu"))
        assert len(result) == 8

    def test_heterogeneous_strides_branch(self):
        """Multiple weights with different strides → same_stride=False and tensor outputs (lines 99-103)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_b_ptr

        # Two weights with different hidden sizes → different strides d1, and
        # different hidden_sizes → heterogeneous path
        w1 = torch.randn(2, 4, 4).contiguous()   # hidden=4
        w2 = torch.randn(2, 8, 4).contiguous()   # hidden=8 — different size(1)

        result = _get_lora_b_ptr([w1, w2], offset_start=0, device=torch.device("cpu"))
        (slice_start_tensor, lora_ptr_tensor, d0, d1, d2, hidden_sizes_tensor, same_stride, max_n) = result

        assert same_stride is False
        assert isinstance(d0, torch.Tensor)
        assert isinstance(d1, torch.Tensor)
        assert isinstance(d2, torch.Tensor)
        assert isinstance(hidden_sizes_tensor, torch.Tensor)
        assert max_n == 8  # max of [4, 8]

    def test_homogeneous_strides_branch(self):
        """Multiple weights with same strides → same_stride=True and scalar outputs."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_b_ptr

        w1 = torch.randn(2, 8, 4).contiguous()
        w2 = torch.randn(2, 8, 4).contiguous()

        result = _get_lora_b_ptr([w1, w2], offset_start=0, device=torch.device("cpu"))
        (slice_start_tensor, lora_ptr_tensor, d0, d1, d2, hidden_sizes_tensor, same_stride, max_n) = result

        assert same_stride is True
        assert isinstance(d0, int)
        assert isinstance(d1, int)
        assert isinstance(d2, int)
        assert isinstance(hidden_sizes_tensor, int)
        assert max_n == 8

    def test_offset_start_included_in_cache_key(self):
        """Different offset_start with same weights yields different cache entries."""
        from nanovllm_voxcpm.lora_ops.triton_ops import utils as u
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_b_ptr

        w = torch.randn(2, 8, 4).contiguous()
        r0 = _get_lora_b_ptr([w], offset_start=0, device=torch.device("cpu"))
        r8 = _get_lora_b_ptr([w], offset_start=8, device=torch.device("cpu"))

        assert len(u._LORA_B_PTR_DICT) == 2
        # slice_start differs
        assert r0[0] == 0
        assert r8[0] == 8

    def test_4d_weight_multiple_heterogeneous_hidden(self):
        """4-D weights with different hidden sizes trigger both squeeze AND heterogeneous strides."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import _get_lora_b_ptr

        w1 = torch.randn(2, 1, 4, 4).contiguous()   # 4-D, hidden=4
        w2 = torch.randn(2, 1, 8, 4).contiguous()   # 4-D, hidden=8

        result = _get_lora_b_ptr([w1, w2], offset_start=0, device=torch.device("cpu"))
        (_, _, d0, d1, d2, hidden_sizes_tensor, same_stride, max_n) = result

        assert same_stride is False
        assert max_n == 8


class TestGetLoraOpConfigs:
    """Tests for get_lora_op_configs uncovered paths (lines 130-131)."""

    def test_shrink_small_batch_config(self):
        """shrink with batch < 128 → split_k=64, block_k=256 (lines 129-141)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import get_lora_op_configs

        cfg = get_lora_op_configs(
            op_type="shrink",
            max_loras=4,
            batch=64,
            hidden_size=128,
            rank=8,
            num_slices=1,
        )
        assert cfg["split_k"] == 64
        assert cfg["block_k"] == 256

    def test_shrink_large_batch_config(self):
        """shrink with batch >= 128 → split_k=8, block_k=32 (lines 130-131)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import get_lora_op_configs

        cfg = get_lora_op_configs(
            op_type="shrink",
            max_loras=4,
            batch=128,
            hidden_size=128,
            rank=8,
            num_slices=1,
        )
        assert cfg["split_k"] == 8
        assert cfg["block_k"] == 32

    def test_shrink_batch_exactly_128_boundary(self):
        """Boundary: batch=128 uses large-batch path (split_k=8)."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import get_lora_op_configs

        cfg = get_lora_op_configs(
            op_type="shrink",
            max_loras=4,
            batch=128,
            hidden_size=64,
            rank=4,
            num_slices=1,
        )
        assert cfg["split_k"] == 8

    def test_shrink_large_batch_above_128(self):
        """batch=256 (> 128) also uses large-batch path."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import get_lora_op_configs

        cfg = get_lora_op_configs(
            op_type="shrink",
            max_loras=2,
            batch=256,
            hidden_size=256,
            rank=16,
            num_slices=1,
        )
        assert cfg["split_k"] == 8
        assert cfg["block_k"] == 32

    def test_expand_single_slice(self):
        """expand (non-shrink) with num_slices=1 → block_n=128."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import get_lora_op_configs

        cfg = get_lora_op_configs(
            op_type="expand",
            max_loras=4,
            batch=32,
            hidden_size=128,
            rank=8,
            num_slices=1,
        )
        assert cfg["block_n"] == 128

    def test_expand_multi_slice(self):
        """expand with num_slices > 1 → block_n=64."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import get_lora_op_configs

        cfg = get_lora_op_configs(
            op_type="expand",
            max_loras=4,
            batch=32,
            hidden_size=128,
            rank=8,
            num_slices=3,
        )
        assert cfg["block_n"] == 64

    def test_supports_pdl_returns_false_on_cpu(self):
        """supports_pdl() always returns False in non-GPU environment."""
        from nanovllm_voxcpm.lora_ops.triton_ops.utils import supports_pdl

        assert supports_pdl() is False
