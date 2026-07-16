"""Unit tests for nanovllm_voxcpm/models/voxcpm/model_utils.py.

TDD approach:
  Phase 1 — tests import from model.py directly (green baseline before extraction).
  Phase 2 — after extraction, tests import from model_utils.py (still green).

All tests are CPU-only (no @pytest.mark.gpu required).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# rotate_half
# ---------------------------------------------------------------------------


def test_rotate_half_shape_preserved():
    from nanovllm_voxcpm.models.voxcpm.model_utils import rotate_half

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    result = rotate_half(x)
    assert result.shape == x.shape


def test_rotate_half_swaps_and_negates():
    """rotate_half(x) = cat(-x2, x1) where x = cat(x1, x2)."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import rotate_half

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    result = rotate_half(x)
    # x1=[1,2], x2=[3,4]  →  cat(-x2, x1) = [-3, -4, 1, 2]
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.allclose(result, expected)


def test_rotate_half_double_rotation_is_negation():
    """Applying rotate_half twice should negate the original tensor."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import rotate_half

    x = torch.randn(4, 8)
    assert torch.allclose(rotate_half(rotate_half(x)), -x)


def test_rotate_half_zero_tensor():
    from nanovllm_voxcpm.models.voxcpm.model_utils import rotate_half

    x = torch.zeros(3, 6)
    assert torch.allclose(rotate_half(x), torch.zeros(3, 6))


def test_rotate_half_3d_input():
    """rotate_half should work on any tensor where last dim is even."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import rotate_half

    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    result = rotate_half(x)
    assert result.shape == x.shape
    # Verify first slice manually
    # x[0,0] = [0,1,2,3]  →  x1=[0,1], x2=[2,3]  →  [-2,-3,0,1]
    assert torch.allclose(result[0, 0], torch.tensor([-2.0, -3.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# apply_rotary_pos_emb
# ---------------------------------------------------------------------------


def _make_rotary_fixtures(seq_len: int = 4, head_dim: int = 8, num_heads: int = 2, dtype=torch.float32):
    """Return (q, k, cos, sin, position_ids) on CPU."""
    q = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype)
    cos = torch.ones(seq_len, head_dim, dtype=torch.float32)
    sin = torch.zeros(seq_len, head_dim, dtype=torch.float32)
    position_ids = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
    return q, k, cos, sin, position_ids


def test_apply_rotary_pos_emb_output_shapes():
    from nanovllm_voxcpm.models.voxcpm.model_utils import apply_rotary_pos_emb

    q, k, cos, sin, pos_ids = _make_rotary_fixtures()
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_apply_rotary_pos_emb_identity_when_sin_zero():
    """When sin=0 and cos=1, rotary embedding is identity."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import apply_rotary_pos_emb

    q, k, cos, sin, pos_ids = _make_rotary_fixtures()
    # cos=1, sin=0 → (q * 1) + (rotate_half(q) * 0) = q  (in float32)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
    assert torch.allclose(q_out, q.float(), atol=1e-6)
    assert torch.allclose(k_out, k.float(), atol=1e-6)


def test_apply_rotary_pos_emb_dtype_preserved():
    """Output dtype must match the input key tensor dtype."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import apply_rotary_pos_emb

    q, k, cos, sin, pos_ids = _make_rotary_fixtures(dtype=torch.float32)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
    assert q_out.dtype == torch.float32
    assert k_out.dtype == torch.float32


def test_apply_rotary_pos_emb_numerical_sanity():
    """Spot-check: cos=0, sin=1 → output equals rotate_half of input."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import apply_rotary_pos_emb, rotate_half

    seq_len, head_dim, num_heads = 2, 4, 1
    q = torch.randn(1, num_heads, seq_len, head_dim)
    k = torch.randn(1, num_heads, seq_len, head_dim)
    cos = torch.zeros(seq_len, head_dim)
    sin = torch.ones(seq_len, head_dim)
    pos_ids = torch.arange(seq_len).unsqueeze(0)

    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
    # (q_fp32 * 0) + (rotate_half(q_fp32) * 1) = rotate_half(q_fp32)
    expected_q = rotate_half(q.float())
    assert torch.allclose(q_out, expected_q, atol=1e-6)


# ---------------------------------------------------------------------------
# MiniCPMLongRoPE._apply_rotary_emb (extracted as apply_rotary_emb_tokens)
# ---------------------------------------------------------------------------


def test_apply_rotary_emb_tokens_shape():
    from nanovllm_voxcpm.models.voxcpm.model_utils import apply_rotary_emb_tokens

    num_tokens, num_heads, head_dim = 5, 3, 8
    x = torch.randn(num_tokens, num_heads, head_dim)
    cos = torch.ones(num_tokens, head_dim)
    sin = torch.zeros(num_tokens, head_dim)
    result = apply_rotary_emb_tokens(x, cos, sin)
    assert result.shape == x.shape


def test_apply_rotary_emb_tokens_identity_when_sin_zero():
    """cos=1, sin=0 → identity transform."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import apply_rotary_emb_tokens

    num_tokens, num_heads, head_dim = 3, 2, 4
    x = torch.randn(num_tokens, num_heads, head_dim)
    cos = torch.ones(num_tokens, head_dim)
    sin = torch.zeros(num_tokens, head_dim)
    result = apply_rotary_emb_tokens(x, cos, sin)
    assert torch.allclose(result, x, atol=1e-6)


def test_apply_rotary_emb_tokens_dtype_preserved():
    from nanovllm_voxcpm.models.voxcpm.model_utils import apply_rotary_emb_tokens

    x = torch.randn(4, 2, 8, dtype=torch.float32)
    cos = torch.ones(4, 8, dtype=torch.float32)
    sin = torch.zeros(4, 8, dtype=torch.float32)
    result = apply_rotary_emb_tokens(x, cos, sin)
    assert result.dtype == torch.float32


def test_apply_rotary_emb_tokens_numerical_sanity():
    """cos=0, sin=1 → rotate_half of x."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import apply_rotary_emb_tokens

    num_tokens, num_heads, head_dim = 2, 1, 4
    x = torch.randn(num_tokens, num_heads, head_dim)
    cos = torch.zeros(num_tokens, head_dim)
    sin = torch.ones(num_tokens, head_dim)
    result = apply_rotary_emb_tokens(x, cos, sin)
    # Inline rotate_half logic: x1, x2 = x.chunk(2, -1) → cat(-x2, x1)
    x1, x2 = x.float().chunk(2, dim=-1)
    expected = torch.cat((-x2, x1), dim=-1)
    assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_longrope_scaling_factor
# ---------------------------------------------------------------------------


def test_compute_longrope_scaling_factor_no_extension():
    """When max_pos == original_max_pos, scale=1 → factor should be 1.0."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import compute_longrope_scaling_factor

    factor = compute_longrope_scaling_factor(max_position_embeddings=4096, original_max_position_embeddings=4096)
    # log(1)/log(4096) = 0  →  sqrt(1+0) = 1
    assert math.isclose(factor, 1.0, rel_tol=1e-6)


def test_compute_longrope_scaling_factor_extended():
    """Extended context (32768 vs 4096) → factor > 1."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import compute_longrope_scaling_factor

    factor = compute_longrope_scaling_factor(max_position_embeddings=32768, original_max_position_embeddings=4096)
    expected = math.sqrt(1 + math.log(32768 / 4096) / math.log(4096))
    assert math.isclose(factor, expected, rel_tol=1e-6)


def test_compute_longrope_scaling_factor_positive():
    """Scaling factor is always >= 1."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import compute_longrope_scaling_factor

    for ratio in [1, 2, 4, 8, 16]:
        base = 4096
        factor = compute_longrope_scaling_factor(
            max_position_embeddings=base * ratio,
            original_max_position_embeddings=base,
        )
        assert factor >= 1.0


# ---------------------------------------------------------------------------
# compute_longrope_freqs
# ---------------------------------------------------------------------------


def test_compute_longrope_freqs_shape():
    from nanovllm_voxcpm.models.voxcpm.model_utils import compute_longrope_freqs

    seq_len, head_dim = 16, 8
    inv_freq = torch.ones(head_dim // 2)
    ext_factors = torch.ones(head_dim // 2)
    cos, sin = compute_longrope_freqs(seq_len, inv_freq, ext_factors, dtype=torch.float32, device="cpu")
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)


def test_compute_longrope_freqs_dtype():
    from nanovllm_voxcpm.models.voxcpm.model_utils import compute_longrope_freqs

    inv_freq = torch.ones(4)
    ext_factors = torch.ones(4)
    cos, sin = compute_longrope_freqs(8, inv_freq, ext_factors, dtype=torch.float32, device="cpu")
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


def test_compute_longrope_freqs_values_sanity():
    """At position 0, freq=0, so cos=1 and sin=0 for any inv_freq."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import compute_longrope_freqs

    inv_freq = torch.tensor([1.0, 2.0])
    ext_factors = torch.ones(2)
    cos, sin = compute_longrope_freqs(4, inv_freq, ext_factors, dtype=torch.float32, device="cpu")
    # Position 0: freqs = 0 * inv_freq = 0  →  cos(0)=1, sin(0)=0
    # emb at pos 0 is cat([freqs[0]], [freqs[0]]) = [0,0,0,0]
    assert torch.allclose(cos[0], torch.ones(4), atol=1e-6)
    assert torch.allclose(sin[0], torch.zeros(4), atol=1e-6)


# ---------------------------------------------------------------------------
# optimized_scale (UnifiedCFM helper)
# ---------------------------------------------------------------------------


def test_optimized_scale_parallel_vectors():
    """When pos and neg are identical, st_star should be 1.0."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import optimized_scale

    v = torch.tensor([[3.0, 4.0]])  # [1, 2]
    st = optimized_scale(v, v)
    assert torch.allclose(st, torch.tensor([[1.0]]), atol=1e-6)


def test_optimized_scale_orthogonal_vectors():
    """When pos and neg are orthogonal, dot product=0 → st_star=0."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import optimized_scale

    pos = torch.tensor([[1.0, 0.0]])
    neg = torch.tensor([[0.0, 1.0]])
    st = optimized_scale(pos, neg)
    assert torch.allclose(st, torch.tensor([[0.0]]), atol=1e-6)


def test_optimized_scale_batch_shape():
    """Output should be [batch, 1]."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import optimized_scale

    pos = torch.randn(4, 16)
    neg = torch.randn(4, 16)
    st = optimized_scale(pos, neg)
    assert st.shape == (4, 1)


def test_optimized_scale_numerical():
    """Manual dot product check."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import optimized_scale

    pos = torch.tensor([[2.0, 0.0]])
    neg = torch.tensor([[1.0, 0.0]])
    # dot=2, norm²=1 → st_star=2
    st = optimized_scale(pos, neg)
    assert torch.allclose(st, torch.tensor([[2.0]]), atol=1e-6)


# ---------------------------------------------------------------------------
# sway_sampling_schedule
# ---------------------------------------------------------------------------


def test_sway_sampling_schedule_shape():
    from nanovllm_voxcpm.models.voxcpm.model_utils import sway_sampling_schedule

    n = 10
    t_span = sway_sampling_schedule(n, device="cpu", dtype=torch.float32)
    assert t_span.shape == (n + 1,)


def test_sway_sampling_schedule_boundary_values():
    """First value ≈ 1 and last value ≈ 0 after sway shift."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import sway_sampling_schedule

    # The sway transform can slightly shift boundary values
    t_span = sway_sampling_schedule(20, device="cpu", dtype=torch.float32)
    # After sway: t' = t + (cos(pi/2 * t) - 1 + t) = 2t + cos(pi/2 * t) - 1
    # At t=1: 2*1 + cos(pi/2) - 1 = 2 + 0 - 1 = 1
    # At t=0: 2*0 + cos(0) - 1 = 0 + 1 - 1 = 0
    assert torch.allclose(t_span[0], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(t_span[-1], torch.tensor(0.0), atol=1e-5)


def test_sway_sampling_schedule_monotonically_decreasing():
    """t_span should decrease from 1 to 0."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import sway_sampling_schedule

    t_span = sway_sampling_schedule(10, device="cpu", dtype=torch.float32)
    diffs = t_span[:-1] - t_span[1:]
    assert (diffs >= 0).all(), "t_span should be non-increasing"


# ---------------------------------------------------------------------------
# sinusoidal_pos_emb_forward (SinusoidalPosEmb logic)
# ---------------------------------------------------------------------------


def test_sinusoidal_pos_emb_shape():
    from nanovllm_voxcpm.models.voxcpm.model_utils import sinusoidal_pos_emb

    x = torch.tensor([0.0, 1.0, 2.0])
    out = sinusoidal_pos_emb(x, dim=16)
    assert out.shape == (3, 16)


def test_sinusoidal_pos_emb_dim_must_be_even():
    from nanovllm_voxcpm.models.voxcpm.model_utils import sinusoidal_pos_emb

    x = torch.tensor([1.0])
    with pytest.raises((AssertionError, ValueError)):
        sinusoidal_pos_emb(x, dim=7)


def test_sinusoidal_pos_emb_zero_position():
    """At x=0, sin(0)=0 and cos(0)=1 → first half all 0, second half all 1."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import sinusoidal_pos_emb

    x = torch.tensor([0.0])
    out = sinusoidal_pos_emb(x, dim=8, scale=1000)
    # emb = scale * 0 * freqs = 0  → sin(0)=0, cos(0)=1
    half = 4
    assert torch.allclose(out[0, :half], torch.zeros(half), atol=1e-6)
    assert torch.allclose(out[0, half:], torch.ones(half), atol=1e-6)


def test_sinusoidal_pos_emb_scalar_input_squeezed():
    """A 0-d tensor input should be handled (unsqueezed internally)."""
    from nanovllm_voxcpm.models.voxcpm.model_utils import sinusoidal_pos_emb

    x = torch.tensor(1.0)  # 0-d
    out = sinusoidal_pos_emb(x, dim=8)
    assert out.shape == (1, 8)


def test_combine_feature_and_text_embeddings_selects_by_mask():
    from nanovllm_voxcpm.models.voxcpm.model_utils_shapes import combine_feature_and_text_embeddings

    feature_embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    text_embeddings = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    feature_mask = torch.tensor([True, False])

    result = combine_feature_and_text_embeddings(feature_embeddings, text_embeddings, feature_mask)

    assert torch.equal(result, torch.tensor([[1.0, 2.0], [30.0, 40.0]]))


def test_select_prefill_hidden_selects_final_token_from_each_sequence():
    from nanovllm_voxcpm.models.voxcpm.model_utils_shapes import select_prefill_hidden

    hidden_states = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    cu_seqlens_q = torch.tensor([0, 2, 5])

    result = select_prefill_hidden(hidden_states, cu_seqlens_q)

    assert torch.equal(result, hidden_states[torch.tensor([1, 4])])
    assert result.is_contiguous()


def test_prepare_local_encoder_inputs_prepends_special_token_per_batch_item():
    from nanovllm_voxcpm.models.voxcpm.model_utils_shapes import prepare_local_encoder_inputs

    projected_inputs = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    special_token = torch.tensor([[[[9.0, 8.0]]]])

    result = prepare_local_encoder_inputs(projected_inputs, special_token)

    assert torch.equal(result, torch.tensor([[[9.0, 8.0], [1.0, 2.0]], [[9.0, 8.0], [3.0, 4.0]]]))


def test_quantize_scalar_latents_rounds_to_configured_scale():
    from nanovllm_voxcpm.models.voxcpm.model_utils_shapes import quantize_scalar_latents

    hidden_states = torch.tensor([[-2.0, -0.26, 0.24, 2.0]])

    result = quantize_scalar_latents(hidden_states, scale=4)

    expected = torch.round(torch.tanh(hidden_states) * 4) / 4
    assert torch.equal(result, expected)


def test_prepare_dit_decoder_input_prepends_conditioned_prefix():
    from nanovllm_voxcpm.models.voxcpm.model_utils_shapes import prepare_dit_decoder_input

    sample = torch.tensor([[[1.0], [2.0]]])
    condition = torch.tensor([[[3.0]]])
    conditioned_mu = torch.tensor([[[4.0]]])

    result = prepare_dit_decoder_input(sample, condition, conditioned_mu)

    assert torch.equal(result, torch.tensor([[[4.0], [3.0], [1.0], [2.0]]]))


def test_apply_classifier_free_guidance_uses_optimal_negative_scale():
    from nanovllm_voxcpm.models.voxcpm.model_utils_shapes import apply_classifier_free_guidance

    positive = torch.tensor([[[[2.0, 0.0]]]])
    negative = torch.tensor([[[[1.0, 0.0]]]])
    cfg_value = torch.tensor([0.5])

    result = apply_classifier_free_guidance(positive, negative, cfg_value)

    assert torch.allclose(result, torch.tensor([[[[2.0, 0.0]]]]), atol=1e-6)


def test_minicpm_longrope_rotates_cpu_queries_and_keys():
    from nanovllm_voxcpm.models.voxcpm.model import MiniCPMLongRoPE

    rope = MiniCPMLongRoPE(
        head_size=4,
        rotary_dim=4,
        max_position_embeddings=8,
        base=10000.0,
        short_factor=[1.0, 1.0],
        long_factor=[2.0, 2.0],
        original_max_position_embeddings=4,
    )
    positions = torch.tensor([0, 1])
    query = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    key = query + 10

    rotated_query, rotated_key = rope(positions, query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert torch.allclose(rotated_query[0], query[0] * rope.scaling_factor)


def test_cpu_module_wrappers_preserve_expected_shapes():
    from nanovllm_voxcpm.models.voxcpm.model import ScalarQuantizationLayer, SinusoidalPosEmb, TimestepEmbedding

    timestep = TimestepEmbedding(in_channels=4, time_embed_dim=6, out_dim=3)
    quantizer = ScalarQuantizationLayer(in_dim=3, out_dim=2, latent_dim=4, scale=4)
    positions = SinusoidalPosEmb(dim=4)

    assert positions(torch.tensor([0.0, 1.0])).shape == (2, 4)
    assert timestep(torch.ones(2, 4)).shape == (2, 3)
    assert quantizer(torch.ones(2, 3)).shape == (2, 2)


def test_decoder_and_model_wrappers_apply_residuals_without_attention_kernel():
    import torch.nn as nn

    from nanovllm_voxcpm.models.voxcpm.model import Cpm4DecoderLayer, Cpm4Model

    class AddOne(nn.Module):
        def forward(self, positions, hidden_states):
            return hidden_states + 1

    class Double(nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 2

    decoder = Cpm4DecoderLayer.__new__(Cpm4DecoderLayer)
    nn.Module.__init__(decoder)
    decoder.input_layernorm = nn.Identity()
    decoder.self_attn = AddOne()
    decoder.post_attention_layernorm = nn.Identity()
    decoder.mlp = Double()
    hidden_states = torch.ones(2, 3)
    decoder_output, residual = decoder(torch.tensor([0, 1]), hidden_states, None)

    assert torch.equal(decoder_output, torch.full((2, 3), 9.0))
    assert torch.equal(residual, torch.full((2, 3), 3.0))

    model = Cpm4Model.__new__(Cpm4Model)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([decoder])
    model.norm = nn.Identity()

    assert torch.equal(model(hidden_states, torch.tensor([0, 1])), decoder_output)


def test_local_encoder_and_cfm_forward_bookkeeping_runs_on_cpu():
    import torch.nn as nn

    from nanovllm_voxcpm.models.voxcpm.model import UnifiedCFM, VoxCPMLocEnc

    class EncoderIdentity(nn.Module):
        def forward(self, inputs, positions):
            return inputs

    encoder = VoxCPMLocEnc.__new__(VoxCPMLocEnc)
    nn.Module.__init__(encoder)
    encoder.in_proj = nn.Identity()
    encoder.special_token = nn.Parameter(torch.tensor([[[[9.0, 8.0]]]]))
    encoder.encoder = EncoderIdentity()

    assert torch.equal(encoder(torch.tensor([[[1.0, 2.0]]])), torch.tensor([[9.0, 8.0]]))

    cfm = UnifiedCFM.__new__(UnifiedCFM)
    nn.Module.__init__(cfm)
    cfm.patch_size = 2
    cfm.in_channels = 3
    cfm.inference_timesteps = 2
    cfm.solve_euler = lambda z, t_span, mu, cond, cfg_value: z
    noise = torch.ones(1, 3, 2)

    result = cfm(
        mu=torch.ones(1, 2),
        cond=torch.ones(1, 3, 2),
        temperature=torch.tensor([2.0]),
        cfg_value=torch.tensor([1.0]),
        z_noise=noise,
    )

    assert torch.equal(result, torch.full((1, 3, 2), 2.0))


def test_voxcpm_model_forward_uses_cpu_bookkeeping_with_stubbed_submodules(monkeypatch):
    from types import SimpleNamespace

    import torch.nn as nn

    import nanovllm_voxcpm.models.voxcpm.model as model_module

    class FeatureEncoder(nn.Module):
        def forward(self, feature):
            return feature[:, 0, :]

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 2)
            with torch.no_grad():
                self.embed_tokens.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

        def forward(self, embeddings, positions):
            return embeddings + 1

    class FeatureDecoder(nn.Module):
        def forward(self, mu, cond, temperature, cfg_value, z_noise):
            return cond

    model = model_module.VoxCPMModel.__new__(model_module.VoxCPMModel)
    nn.Module.__init__(model)
    model.feat_encoder = FeatureEncoder()
    model.enc_to_lm_proj = nn.Identity()
    model.base_lm = LanguageModel()
    model.fsq_layer = nn.Identity()
    model.residual_lm = LanguageModel()
    model.lm_to_dit_proj = nn.Identity()
    model.res_to_dit_proj = nn.Identity()
    model.feat_decoder = FeatureDecoder()
    model.stop_proj = nn.Identity()
    model.stop_actn = nn.Identity()
    model.stop_head = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.stop_head.weight.copy_(torch.eye(2))
    monkeypatch.setattr(model_module, "get_context", lambda: SimpleNamespace(is_prefill=False))

    result = model(
        positions=torch.tensor([0, 1]),
        text_tokens=torch.tensor([1, 2]),
        feat=torch.tensor([[[10.0, 20.0]], [[30.0, 40.0]]]),
        feat_mask=torch.tensor([True, False]),
        temperature=torch.tensor([1.0]),
        cfg_value=torch.tensor([1.0]),
    )

    assert torch.equal(result["latents"], torch.tensor([[[10.0, 20.0]], [[30.0, 40.0]]]))
    assert torch.equal(result["stop_flag"], torch.tensor([1, 1]))


def _mini_cpm_config(num_hidden_layers=1):
    from nanovllm_voxcpm.models.voxcpm.config import MiniCPM4Config

    return MiniCPM4Config(
        bos_token_id=0,
        eos_token_id=1,
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=16,
        num_attention_heads=2,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_scaling={
            "type": "longrope",
            "long_factor": [2.0, 2.0],
            "short_factor": [1.0, 1.0],
            "original_max_position_embeddings": 8,
        },
        vocab_size=0,
        use_mup=False,
        scale_emb=1.0,
        dim_model_base=8,
        scale_depth=1.0,
        rope_theta=10000.0,
        kv_channels=4,
    )


def test_cpu_constructors_wire_attention_mlp_decoder_and_model(monkeypatch):
    import nanovllm_voxcpm.models.voxcpm.model as model_module

    monkeypatch.setattr(model_module, "get_tp_world_size", lambda: 1)
    config = _mini_cpm_config()

    attention = model_module.Cpm4Attention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        max_position=config.max_position_embeddings,
        head_dim=config.kv_channels,
        rope_scaling=config.rope_scaling,
        apply_qk_norm=True,
    )
    mlp = model_module.Cpm4MLP(config.hidden_size, config.intermediate_size)
    decoder = model_module.Cpm4DecoderLayer(config)
    transformer = model_module.Cpm4Model(config, is_causal=False)

    assert attention.q_size == 8
    assert attention.kv_size == 8
    assert attention.q_norm is not None
    assert mlp.gate_up_proj.output_sizes == [16, 16]
    assert decoder.scale_depth == 1.0
    assert len(transformer.layers) == 1


def test_dit_constructor_builds_cpu_reachable_projection_stack(monkeypatch):
    import nanovllm_voxcpm.models.voxcpm.model as model_module

    monkeypatch.setattr(model_module, "get_tp_world_size", lambda: 1)
    dit = model_module.VoxCPMLocDiT(_mini_cpm_config(), in_channels=3)

    assert dit.in_channels == 3
    assert dit.out_channels == 3
    assert dit.decoder.embed_tokens.__class__.__name__ == "Identity"
