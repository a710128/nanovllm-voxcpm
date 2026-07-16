"""Unit tests for nanovllm_voxcpm/models/voxcpm2/model_utils.py.

All tests run on CPU and exercise the pure-math helpers extracted from model.py.
No GPU, flash-attn, or triton dependencies are required.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanovllm_voxcpm.models.voxcpm2 import model_utils
from nanovllm_voxcpm.models.voxcpm2.model_utils import (
    build_cfm_t_span,
    compute_attention_sizes,
    compute_optimized_scale,
    compute_zero_init_steps,
    parse_gate_up_lora_targets,
    parse_qkv_lora_targets,
)

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


class _ZeroEstimator:
    def __init__(self):
        self.dt_inputs: list[torch.Tensor] = []

    def __call__(self, x, mu, t, cond, dt):
        self.dt_inputs.append(dt.clone())
        return torch.zeros_like(x)


class _EulerOps:
    def __init__(self):
        self.estimator = _ZeroEstimator()

    def optimized_scale(self, positive_flat, negative_flat):
        return compute_optimized_scale(positive_flat, negative_flat)


def test_solve_euler_preserves_input_for_zero_flow_and_zeroes_dt_outside_mean_mode():
    x = torch.randn(2, 4, 3)
    inputs = model_utils.EulerSolverInputs(
        x=x,
        t_span=torch.linspace(1, 0, 7),
        mu=torch.randn(2, 12),
        cond=torch.randn(2, 4, 3),
        cfg_value=torch.ones(2),
    )
    config = model_utils.EulerSolverConfig(in_channels=4, mean_mode=False)
    ops = _EulerOps()

    result = model_utils.solve_euler(inputs, config, ops)

    torch.testing.assert_close(result, x)
    assert ops.estimator.dt_inputs
    for dt_input in ops.estimator.dt_inputs:
        torch.testing.assert_close(dt_input, torch.zeros_like(dt_input))


def test_minicpm_long_rope_forward_matches_token_helper():
    from nanovllm_voxcpm.models.voxcpm2.model import MiniCPMLongRoPE

    rope = MiniCPMLongRoPE(8, 8, 16, 10000.0)
    positions = torch.tensor([1, 3])
    query = torch.randn(2, 16)
    key = torch.randn(2, 8)

    query_out, key_out = rope(positions, query, key)

    expected_query = model_utils.apply_rotary_emb(
        query.view(2, 2, 8), rope.cos_cached[positions], rope.sin_cached[positions]
    )
    expected_key = model_utils.apply_rotary_emb(
        key.view(2, 1, 8), rope.cos_cached[positions], rope.sin_cached[positions]
    )
    torch.testing.assert_close(query_out, expected_query.view_as(query))
    torch.testing.assert_close(key_out, expected_key.view_as(key))


def test_scalar_quantization_layer_rounds_projected_latents():
    from nanovllm_voxcpm.models.voxcpm2.model import ScalarQuantizationLayer

    layer = ScalarQuantizationLayer(2, 2, latent_dim=2, scale=2)
    with torch.no_grad():
        layer.in_proj.weight.copy_(torch.eye(2))
        layer.in_proj.bias.zero_()
        layer.out_proj.weight.copy_(torch.eye(2))
        layer.out_proj.bias.zero_()

    result = layer(torch.tensor([[0.2, 1.0]]))

    expected = torch.round(torch.tanh(torch.tensor([[0.2, 1.0]])) * 2) / 2
    torch.testing.assert_close(result, expected)


class _InertLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, value):
        return value


def test_attention_and_mlp_select_lora_layers_on_cpu(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm2 import model as voxcpm2_model

    monkeypatch.setattr(voxcpm2_model, "get_tp_world_size", lambda: 1)
    for layer_name in (
        "Attention",
        "LoRAQKVParallelLinear",
        "LoRARowParallelLinear",
        "LoRAMergedColumnParallelLinear",
        "RMSNorm",
        "SiluAndMul",
    ):
        monkeypatch.setattr(voxcpm2_model, layer_name, _InertLayer)
    lora_config = SimpleNamespace(
        max_loras=2,
        max_lora_rank=4,
        target_modules_lm=["q_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    attention = voxcpm2_model.Cpm4Attention(
        hidden_size=8,
        num_heads=2,
        num_kv_heads=2,
        head_dim=4,
        use_rope=False,
        apply_qk_norm=True,
        lora_config=lora_config,
    )
    mlp = voxcpm2_model.Cpm4MLP(hidden_size=8, intermediate_size=16, lora_config=lora_config)

    assert isinstance(attention.qkv_proj, _InertLayer)
    assert isinstance(attention.o_proj, _InertLayer)
    assert attention.rotary_emb is None
    assert isinstance(attention.q_norm, _InertLayer)
    assert isinstance(attention.k_norm, _InertLayer)
    assert isinstance(mlp.gate_up_proj, _InertLayer)
    assert isinstance(mlp.down_proj, _InertLayer)
