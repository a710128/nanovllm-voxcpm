"""Unit tests for nanovllm_voxcpm/models/voxcpm2/model_utils.py.

All tests run on CPU and exercise the pure-math helpers extracted from model.py.
No GPU, flash-attn, or triton dependencies are required.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from nanovllm_voxcpm.models.voxcpm2.model_utils import (
    apply_rotary_emb,
    build_cfm_t_span,
    build_rope_cos_sin_cache,
    build_rope_inv_freq,
    compute_attention_sizes,
    compute_optimized_scale,
    compute_rope_scaling_factor,
    compute_zero_init_steps,
    derive_decoder_config_fields,
    derive_encoder_config_fields,
    parse_gate_up_lora_targets,
    parse_qkv_lora_targets,
    sinusoidal_pos_emb,
)

# ---------------------------------------------------------------------------
# compute_rope_scaling_factor
# ---------------------------------------------------------------------------


class TestComputeRopeScalingFactor:
    def test_identity_when_equal(self):
        # scale = 1.0 → log(1) = 0 → factor = sqrt(1) = 1.0
        result = compute_rope_scaling_factor(32768, 32768)
        assert result == pytest.approx(1.0)

    def test_extended_context_greater_than_one(self):
        result = compute_rope_scaling_factor(131072, 32768)
        # scale = 4, log(4)/log(32768) = log(4)/log(32768)
        expected = math.sqrt(1 + math.log(4) / math.log(32768))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_double_context(self):
        result = compute_rope_scaling_factor(65536, 32768)
        expected = math.sqrt(1 + math.log(2) / math.log(32768))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_return_type_is_float(self):
        result = compute_rope_scaling_factor(32768, 32768)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# build_rope_inv_freq
# ---------------------------------------------------------------------------


class TestBuildRopeInvFreq:
    def test_shape(self):
        inv_freq = build_rope_inv_freq(dim=16, base=10000.0)
        assert inv_freq.shape == (8,)

    def test_all_positive(self):
        inv_freq = build_rope_inv_freq(dim=16, base=10000.0)
        assert (inv_freq > 0).all()

    def test_decreasing(self):
        inv_freq = build_rope_inv_freq(dim=16, base=10000.0)
        # Higher frequency indices should have smaller inv_freq values
        assert (inv_freq[:-1] >= inv_freq[1:]).all()

    def test_first_element_is_one(self):
        # For index 0: base ** (0 / dim) = 1.0, so inv_freq[0] = 1.0
        inv_freq = build_rope_inv_freq(dim=8, base=10000.0)
        assert inv_freq[0].item() == pytest.approx(1.0)

    def test_base_affects_values(self):
        inv_a = build_rope_inv_freq(dim=8, base=10000.0)
        inv_b = build_rope_inv_freq(dim=8, base=100.0)
        # Smaller base → larger inv_freq for indices > 0
        assert (inv_b[1:] > inv_a[1:]).all()


# ---------------------------------------------------------------------------
# build_rope_cos_sin_cache
# ---------------------------------------------------------------------------


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
        """seq_len <= original_max → short_factor; seq_len > original_max → long_factor."""
        inv_freq = self._make_inv_freq(dim=8)
        short_factor = [1.0] * 4
        long_factor = [3.0] * 4  # deliberately different

        # seq_len == original_max → short branch
        cos_short, _ = build_rope_cos_sin_cache(
            seq_len=32,
            inv_freq=inv_freq,
            scaling_factor=1.0,
            short_factor=short_factor,
            long_factor=long_factor,
            original_max_position_embeddings=32,
        )
        # seq_len > original_max → long branch
        cos_long, _ = build_rope_cos_sin_cache(
            seq_len=64,
            inv_freq=inv_freq,
            scaling_factor=1.0,
            short_factor=short_factor,
            long_factor=long_factor,
            original_max_position_embeddings=32,
        )
        # Different factors → different values (first 32 rows compared)
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


# ---------------------------------------------------------------------------
# apply_rotary_emb
# ---------------------------------------------------------------------------


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
        # cos=1, sin=0 → rotate_half zeroed out → output should equal input
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
        """With cos=0, sin=1 the output should be rotate_half(x)."""
        num_tokens, num_heads, head_dim = 2, 1, 4
        x = torch.randn(num_tokens, num_heads, head_dim)
        cos = torch.zeros(num_tokens, head_dim)
        sin = torch.ones(num_tokens, head_dim)
        out = apply_rotary_emb(x, cos, sin)
        x1, x2 = x.chunk(2, dim=-1)
        expected = torch.cat((-x2, x1), dim=-1)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_non_unit_cos_sin_values(self):
        torch.manual_seed(42)
        num_tokens, num_heads, head_dim = 3, 2, 8
        x = torch.randn(num_tokens, num_heads, head_dim)
        angle = torch.rand(num_tokens, head_dim) * 2 * math.pi
        cos = angle.cos()
        sin = angle.sin()
        out = apply_rotary_emb(x, cos, sin)
        # Output must differ from input when sin != 0
        assert not torch.allclose(out, x)
        # Shape preserved
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# sinusoidal_pos_emb
# ---------------------------------------------------------------------------


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
        # At x=0, sin(0)=0 and cos(0)=1 → first half ~0, second half ~1
        x = torch.tensor([0.0])
        out = sinusoidal_pos_emb(x, dim=8, scale=1000.0)
        # First 4 elements should be ~0 (sin of small angles at t=0)
        torch.testing.assert_close(out[0, :4], torch.zeros(4), atol=1e-5, rtol=0)
        # Last 4 elements should be ~1 (cos of small angles at t=0)
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


# ---------------------------------------------------------------------------
# compute_attention_sizes
# ---------------------------------------------------------------------------


class TestComputeAttentionSizes:
    def test_basic_no_tp(self):
        sizes = compute_attention_sizes(
            hidden_size=64, total_num_heads=8, total_num_kv_heads=4, head_dim=None, tp_size=1
        )
        assert sizes["num_heads"] == 8
        assert sizes["num_kv_heads"] == 4
        assert sizes["head_dim"] == 8  # 64 // 8
        assert sizes["q_size"] == 64  # 8 * 8
        assert sizes["kv_size"] == 32  # 4 * 8
        assert sizes["scaling"] == pytest.approx(8**-0.5)

    def test_explicit_head_dim_overrides_derived(self):
        sizes = compute_attention_sizes(hidden_size=64, total_num_heads=8, total_num_kv_heads=8, head_dim=16, tp_size=1)
        assert sizes["head_dim"] == 16
        assert sizes["q_size"] == 128  # 8 * 16
        assert sizes["kv_size"] == 128

    def test_tp_size_2_halves_local_heads(self):
        sizes = compute_attention_sizes(
            hidden_size=64, total_num_heads=8, total_num_kv_heads=8, head_dim=None, tp_size=2
        )
        assert sizes["num_heads"] == 4
        assert sizes["num_kv_heads"] == 4

    def test_scaling_is_inverse_sqrt_head_dim(self):
        sizes = compute_attention_sizes(
            hidden_size=128, total_num_heads=8, total_num_kv_heads=8, head_dim=None, tp_size=1
        )
        # head_dim = 128 // 8 = 16
        assert sizes["scaling"] == pytest.approx(16**-0.5)

    def test_kv_channels_override(self):
        # When head_dim is explicitly 4 (kv_channels), derive from it
        sizes = compute_attention_sizes(hidden_size=64, total_num_heads=8, total_num_kv_heads=2, head_dim=4, tp_size=1)
        assert sizes["head_dim"] == 4
        assert sizes["kv_size"] == 8  # 2 * 4


# ---------------------------------------------------------------------------
# parse_qkv_lora_targets
# ---------------------------------------------------------------------------


class TestParseQkvLoraTargets:
    def test_all_qkv(self):
        result = parse_qkv_lora_targets(["q_proj", "k_proj", "v_proj", "o_proj"])
        assert result == ["q", "k", "v"]

    def test_only_q(self):
        result = parse_qkv_lora_targets(["q_proj"])
        assert result == ["q"]

    def test_empty_input(self):
        result = parse_qkv_lora_targets([])
        assert result == []

    def test_o_proj_excluded(self):
        result = parse_qkv_lora_targets(["o_proj", "down_proj"])
        assert result == []

    def test_order_preserved(self):
        result = parse_qkv_lora_targets(["v_proj", "q_proj", "k_proj"])
        assert result == ["v", "q", "k"]


# ---------------------------------------------------------------------------
# parse_gate_up_lora_targets
# ---------------------------------------------------------------------------


class TestParseGateUpLoraTargets:
    def test_both_gate_and_up(self):
        result = parse_gate_up_lora_targets(["gate_proj", "up_proj", "down_proj"])
        assert result == [0, 1]

    def test_only_gate(self):
        result = parse_gate_up_lora_targets(["gate_proj"])
        assert result == [0]

    def test_only_up(self):
        result = parse_gate_up_lora_targets(["up_proj"])
        assert result == [1]

    def test_empty_input(self):
        result = parse_gate_up_lora_targets([])
        assert result == []

    def test_down_proj_excluded(self):
        result = parse_gate_up_lora_targets(["down_proj"])
        assert result == []

    def test_order_always_gate_then_up(self):
        # Even if up_proj comes before gate_proj in input, indices should be [0, 1]
        result = parse_gate_up_lora_targets(["up_proj", "gate_proj"])
        assert result == [0, 1]


# ---------------------------------------------------------------------------
# compute_optimized_scale
# ---------------------------------------------------------------------------


class TestComputeOptimizedScale:
    def test_parallel_vectors(self):
        # When positive == negative, dot / squared_norm should be ~1
        v = torch.ones(2, 4)
        scale = compute_optimized_scale(v, v)
        assert scale.shape == (2, 1)
        torch.testing.assert_close(scale, torch.ones(2, 1), rtol=1e-5, atol=1e-5)

    def test_orthogonal_vectors(self):
        # Dot product = 0 → scale = 0
        pos = torch.tensor([[1.0, 0.0]])
        neg = torch.tensor([[0.0, 1.0]])
        scale = compute_optimized_scale(pos, neg)
        torch.testing.assert_close(scale, torch.zeros(1, 1), rtol=1e-5, atol=1e-5)

    def test_output_shape(self):
        bsz = 5
        pos = torch.randn(bsz, 16)
        neg = torch.randn(bsz, 16)
        scale = compute_optimized_scale(pos, neg)
        assert scale.shape == (bsz, 1)

    def test_numerically_stable_zero_neg(self):
        # All-zero negative_flat → denominator = 1e-8, scale ~ 0
        pos = torch.randn(2, 4)
        neg = torch.zeros(2, 4)
        scale = compute_optimized_scale(pos, neg)
        # Should not raise and should be finite
        assert scale.isfinite().all()

    def test_antiparallel_vectors(self):
        v = torch.ones(1, 4)
        scale = compute_optimized_scale(v, -v)
        assert scale.item() < 0


# ---------------------------------------------------------------------------
# build_cfm_t_span
# ---------------------------------------------------------------------------


class TestBuildCfmTSpan:
    def test_length(self):
        t_span = build_cfm_t_span(10)
        assert len(t_span) == 11

    def test_starts_near_one(self):
        t_span = build_cfm_t_span(10)
        # After cosine adjustment, t_span[0] = 1 + cos(pi/2) - 1 + 1 = 1 + 0 - 1 + 1 = 1
        assert t_span[0].item() == pytest.approx(1.0, abs=1e-5)

    def test_ends_near_zero(self):
        t_span = build_cfm_t_span(10)
        # At t=0: cos(0) = 1 → 0 + (1 - 1 + 0) = 0
        assert t_span[-1].item() == pytest.approx(0.0, abs=1e-5)

    def test_monotonically_decreasing(self):
        t_span = build_cfm_t_span(20)
        assert (t_span[:-1] >= t_span[1:]).all()

    def test_single_step(self):
        t_span = build_cfm_t_span(1)
        assert len(t_span) == 2

    def test_device_cpu(self):
        t_span = build_cfm_t_span(5, device=torch.device("cpu"))
        assert t_span.device.type == "cpu"


# ---------------------------------------------------------------------------
# compute_zero_init_steps
# ---------------------------------------------------------------------------


class TestComputeZeroInitSteps:
    def test_minimum_is_one(self):
        # For very short t_span_len, 4% rounds down to 0 but max(1,...) returns 1
        assert compute_zero_init_steps(1) == 1
        assert compute_zero_init_steps(2) == 1

    def test_proportional_for_large_span(self):
        # 100 * 0.04 = 4
        assert compute_zero_init_steps(100) == 4

    def test_proportional_for_25(self):
        # 25 * 0.04 = 1.0 → int = 1
        assert compute_zero_init_steps(25) == 1

    def test_proportional_for_50(self):
        # 50 * 0.04 = 2
        assert compute_zero_init_steps(50) == 2


# ---------------------------------------------------------------------------
# derive_encoder_config_fields
# ---------------------------------------------------------------------------


class TestDeriveEncoderConfigFields:
    def test_correct_fields_returned(self):
        fields = derive_encoder_config_fields(
            lm_config_hidden_size=2048,
            lm_config_intermediate_size=8192,
            lm_config_num_attention_heads=32,
            encoder_hidden_dim=512,
            encoder_ffn_dim=2048,
            encoder_num_heads=8,
            encoder_num_layers=4,
            encoder_kv_channels=64,
        )
        assert fields["hidden_size"] == 512
        assert fields["intermediate_size"] == 2048
        assert fields["num_attention_heads"] == 8
        assert fields["num_hidden_layers"] == 4
        assert fields["kv_channels"] == 64
        assert fields["vocab_size"] == 0

    def test_none_kv_channels_preserved(self):
        fields = derive_encoder_config_fields(
            lm_config_hidden_size=64,
            lm_config_intermediate_size=128,
            lm_config_num_attention_heads=4,
            encoder_hidden_dim=8,
            encoder_ffn_dim=16,
            encoder_num_heads=2,
            encoder_num_layers=1,
            encoder_kv_channels=None,
        )
        assert fields["kv_channels"] is None


# ---------------------------------------------------------------------------
# derive_decoder_config_fields
# ---------------------------------------------------------------------------


class TestDeriveDecoderConfigFields:
    def test_correct_fields_returned(self):
        fields = derive_decoder_config_fields(
            dit_hidden_dim=256,
            dit_ffn_dim=1024,
            dit_num_heads=4,
            dit_num_layers=2,
            dit_kv_channels=32,
        )
        assert fields["hidden_size"] == 256
        assert fields["intermediate_size"] == 1024
        assert fields["num_attention_heads"] == 4
        assert fields["num_hidden_layers"] == 2
        assert fields["kv_channels"] == 32
        assert fields["vocab_size"] == 0

    def test_none_kv_channels_preserved(self):
        fields = derive_decoder_config_fields(
            dit_hidden_dim=8, dit_ffn_dim=16, dit_num_heads=2, dit_num_layers=1, dit_kv_channels=None
        )
        assert fields["kv_channels"] is None


# ---------------------------------------------------------------------------
# Integration: helpers consistent with MiniCPMLongRoPE module
# ---------------------------------------------------------------------------


class TestHelpersConsistentWithModule:
    """Verify that the extracted helpers reproduce what MiniCPMLongRoPE stores."""

    def test_scaling_factor_matches_module(self):
        from nanovllm_voxcpm.models.voxcpm2.model import MiniCPMLongRoPE

        head_dim = 8
        rope = MiniCPMLongRoPE(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=64,
            base=10000.0,
            short_factor=[1.0] * (head_dim // 2),
            long_factor=[1.0] * (head_dim // 2),
            original_max_position_embeddings=32,
        )
        expected = compute_rope_scaling_factor(64, 32)
        assert rope.scaling_factor == pytest.approx(expected)

    def test_inv_freq_matches_module(self):
        from nanovllm_voxcpm.models.voxcpm2.model import MiniCPMLongRoPE

        head_dim = 8
        rope = MiniCPMLongRoPE(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=32,
            base=10000.0,
            original_max_position_embeddings=32,
        )
        expected = build_rope_inv_freq(head_dim, 10000.0)
        torch.testing.assert_close(rope.inv_freq, expected)

    def test_apply_rotary_emb_matches_module_method(self):
        from nanovllm_voxcpm.models.voxcpm2.model import MiniCPMLongRoPE

        head_dim = 8
        rope = MiniCPMLongRoPE(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=32,
            base=10000.0,
            original_max_position_embeddings=32,
        )
        x = torch.randn(4, 2, head_dim)
        cos = torch.randn(4, head_dim)
        sin = torch.randn(4, head_dim)
        module_out = rope._apply_rotary_emb(x, cos, sin)
        helper_out = apply_rotary_emb(x, cos, sin)
        torch.testing.assert_close(module_out, helper_out)

    def test_sinusoidal_pos_emb_matches_module(self):
        from nanovllm_voxcpm.models.voxcpm2.model import SinusoidalPosEmb

        dim = 16
        module = SinusoidalPosEmb(dim)
        x = torch.tensor([0.1, 0.5, 0.9])
        module_out = module(x, scale=1000)
        helper_out = sinusoidal_pos_emb(x, dim, scale=1000.0)
        torch.testing.assert_close(module_out, helper_out)
