import math

import pytest

torch = pytest.importorskip("torch")

from nanovllm_voxcpm.models.voxcpm2.model_utils import (
    apply_rotary_emb,
    build_rope_cos_sin_cache,
    build_rope_inv_freq,
    compute_rope_scaling_factor,
    sinusoidal_pos_emb,
)


class TestComputeRopeScalingFactor:
    def test_identity_when_equal(self):
        result = compute_rope_scaling_factor(32768, 32768)
        assert result == pytest.approx(1.0)

    def test_extended_context_greater_than_one(self):
        result = compute_rope_scaling_factor(131072, 32768)
        expected = math.sqrt(1 + math.log(4) / math.log(32768))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_double_context(self):
        result = compute_rope_scaling_factor(65536, 32768)
        expected = math.sqrt(1 + math.log(2) / math.log(32768))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_return_type_is_float(self):
        result = compute_rope_scaling_factor(32768, 32768)
        assert isinstance(result, float)


class TestBuildRopeInvFreq:
    def test_shape(self):
        inv_freq = build_rope_inv_freq(dim=16, base=10000.0)
        assert inv_freq.shape == (8,)

    def test_all_positive(self):
        inv_freq = build_rope_inv_freq(dim=16, base=10000.0)
        assert (inv_freq > 0).all()

    def test_decreasing(self):
        inv_freq = build_rope_inv_freq(dim=16, base=10000.0)
        assert (inv_freq[:-1] >= inv_freq[1:]).all()

    def test_first_element_is_one(self):
        inv_freq = build_rope_inv_freq(dim=8, base=10000.0)
        assert inv_freq[0].item() == pytest.approx(1.0)

    def test_base_affects_values(self):
        inv_a = build_rope_inv_freq(dim=8, base=10000.0)
        inv_b = build_rope_inv_freq(dim=8, base=100.0)
        assert (inv_b[1:] > inv_a[1:]).all()


class TestBuildRopeCosSinCache:
    def _make_inv_freq(self, dim=8, base=10000.0):
        return build_rope_inv_freq(dim, base)

    def test_output_shapes(self):
        inv_freq = self._make_inv_freq(dim=8)
        sf = compute_rope_scaling_factor(32, 32)
        cos, sin = build_rope_cos_sin_cache(
            seq_len=32,
            inv_freq=inv_freq,
            scaling_factor=sf,
            short_factor=[1.0] * 4,
            long_factor=[2.0] * 4,
            original_max_position_embeddings=32,
        )
        assert cos.shape == (32, 8)
        assert sin.shape == (32, 8)

    def test_scaling_factor_applied(self):
        inv_freq = self._make_inv_freq(dim=8)
        cos1, _ = build_rope_cos_sin_cache(
            seq_len=32,
            inv_freq=inv_freq,
            scaling_factor=1.0,
            short_factor=[1.0] * 4,
            long_factor=[1.0] * 4,
            original_max_position_embeddings=32,
        )
        cos2, _ = build_rope_cos_sin_cache(
            seq_len=32,
            inv_freq=inv_freq,
            scaling_factor=2.0,
            short_factor=[1.0] * 4,
            long_factor=[1.0] * 4,
            original_max_position_embeddings=32,
        )
        torch.testing.assert_close(cos2, cos1 * 2.0)

    def test_short_vs_long_factor_selection(self):
        inv_freq = self._make_inv_freq(dim=8)
        short_factor = [1.0] * 4
        long_factor = [3.0] * 4
        cos_short, _ = build_rope_cos_sin_cache(
            seq_len=32,
            inv_freq=inv_freq,
            scaling_factor=1.0,
            short_factor=short_factor,
            long_factor=long_factor,
            original_max_position_embeddings=32,
        )
        cos_long, _ = build_rope_cos_sin_cache(
            seq_len=64,
            inv_freq=inv_freq,
            scaling_factor=1.0,
            short_factor=short_factor,
            long_factor=long_factor,
            original_max_position_embeddings=32,
        )
        assert not torch.allclose(cos_short, cos_long[:32])

    def test_dtype_propagated(self):
        inv_freq = self._make_inv_freq(dim=8)
        cos, sin = build_rope_cos_sin_cache(
            seq_len=8,
            inv_freq=inv_freq,
            scaling_factor=1.0,
            short_factor=[1.0] * 4,
            long_factor=[1.0] * 4,
            original_max_position_embeddings=8,
            dtype=torch.float16,
        )
        assert cos.dtype == torch.float16
        assert sin.dtype == torch.float16


class TestApplyRotaryEmb:
    def _make_tensors(self, num_tokens=4, num_heads=2, head_dim=8):
        x = torch.randn(num_tokens, num_heads, head_dim)
        cos = torch.ones(num_tokens, head_dim)
        sin = torch.zeros(num_tokens, head_dim)
        return x, cos, sin

    def test_output_shape(self):
        x, cos, sin = self._make_tensors()
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_identity_when_cos1_sin0(self):
        x, cos, sin = self._make_tensors()
        out = apply_rotary_emb(x, cos, sin)
        torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-5)

    def test_dtype_preserved_float16(self):
        x = torch.randn(4, 2, 8, dtype=torch.float16)
        cos = torch.ones(4, 8, dtype=torch.float16)
        sin = torch.zeros(4, 8, dtype=torch.float16)
        out = apply_rotary_emb(x, cos, sin)
        assert out.dtype == torch.float16

    def test_rotate_half_negate_pattern(self):
        x = torch.randn(2, 1, 4)
        cos = torch.zeros(2, 4)
        sin = torch.ones(2, 4)
        out = apply_rotary_emb(x, cos, sin)
        x1, x2 = x.chunk(2, dim=-1)
        expected = torch.cat((-x2, x1), dim=-1)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_non_unit_cos_sin_values(self):
        torch.manual_seed(42)
        x = torch.randn(3, 2, 8)
        angle = torch.rand(3, 8) * 2 * math.pi
        out = apply_rotary_emb(x, angle.cos(), angle.sin())
        assert not torch.allclose(out, x)
        assert out.shape == x.shape


class TestSinusoidalPosEmb:
    def test_output_shape_1d(self):
        x = torch.tensor([0.0, 0.5, 1.0])
        out = sinusoidal_pos_emb(x, dim=8)
        assert out.shape == (3, 8)

    def test_output_shape_scalar(self):
        x = torch.tensor(0.5)
        out = sinusoidal_pos_emb(x, dim=8)
        assert out.shape == (1, 8)

    def test_first_half_is_sin_second_half_is_cos(self):
        x = torch.tensor([0.0])
        out = sinusoidal_pos_emb(x, dim=8, scale=1000.0)
        torch.testing.assert_close(out[0, :4], torch.zeros(4), atol=1e-5, rtol=0)
        torch.testing.assert_close(out[0, 4:], torch.ones(4), atol=1e-5, rtol=0)

    def test_scale_affects_output(self):
        x = torch.tensor([1.0])
        out1 = sinusoidal_pos_emb(x, dim=8, scale=1.0)
        out2 = sinusoidal_pos_emb(x, dim=8, scale=100.0)
        assert not torch.allclose(out1, out2)

    def test_different_positions_give_different_embeddings(self):
        x = torch.tensor([0.1, 0.9])
        out = sinusoidal_pos_emb(x, dim=8)
        assert not torch.allclose(out[0], out[1])


class TestHelpersConsistentWithModule:
    def test_scaling_factor_matches_module(self):
        from nanovllm_voxcpm.models.voxcpm2.model import MiniCPMLongRoPE

        rope = MiniCPMLongRoPE(
            head_size=8,
            rotary_dim=8,
            max_position_embeddings=64,
            base=10000.0,
            short_factor=[1.0] * 4,
            long_factor=[1.0] * 4,
            original_max_position_embeddings=32,
        )
        assert rope.scaling_factor == pytest.approx(compute_rope_scaling_factor(64, 32))

    def test_inv_freq_matches_module(self):
        from nanovllm_voxcpm.models.voxcpm2.model import MiniCPMLongRoPE

        rope = MiniCPMLongRoPE(8, 8, 32, 10000.0, original_max_position_embeddings=32)
        torch.testing.assert_close(rope.inv_freq, build_rope_inv_freq(8, 10000.0))

    def test_apply_rotary_emb_matches_module_method(self):
        from nanovllm_voxcpm.models.voxcpm2.model import MiniCPMLongRoPE

        rope = MiniCPMLongRoPE(8, 8, 32, 10000.0, original_max_position_embeddings=32)
        x = torch.randn(4, 2, 8)
        cos = torch.randn(4, 8)
        sin = torch.randn(4, 8)
        torch.testing.assert_close(rope._apply_rotary_emb(x, cos, sin), apply_rotary_emb(x, cos, sin))

    def test_sinusoidal_pos_emb_matches_module(self):
        from nanovllm_voxcpm.models.voxcpm2.model import SinusoidalPosEmb

        module = SinusoidalPosEmb(16)
        x = torch.tensor([0.1, 0.5, 0.9])
        torch.testing.assert_close(module(x, scale=1000), sinusoidal_pos_emb(x, 16, scale=1000.0))
