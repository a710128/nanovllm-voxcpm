import pytest

torch = pytest.importorskip("torch")


def _make_config(residual_lm_no_rope: bool):
    from nanovllm_voxcpm.models.voxcpm2.config import (
        CfmConfig,
        MiniCPM4Config,
        RopeScalingConfig,
        VoxCPM2Config,
        VoxCPM2DitConfig,
        VoxCPM2EncoderConfig,
    )

    lm_config = MiniCPM4Config(
        bos_token_id=0,
        eos_token_id=1,
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=32,
        num_attention_heads=2,
        num_hidden_layers=3,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        rope_scaling=RopeScalingConfig(
            type="longrope",
            long_factor=[1.0, 1.0],
            short_factor=[1.0, 1.0],
            original_max_position_embeddings=32,
        ),
        vocab_size=32,
        use_mup=False,
        scale_emb=1.0,
        dim_model_base=8,
        scale_depth=1.0,
        rope_theta=10000.0,
        kv_channels=4,
    )
    return VoxCPM2Config(
        lm_config=lm_config,
        feat_dim=4,
        patch_size=2,
        residual_lm_num_layers=2,
        residual_lm_no_rope=residual_lm_no_rope,
        scalar_quantization_latent_dim=4,
        scalar_quantization_scale=9,
        encoder_config=VoxCPM2EncoderConfig(hidden_dim=8, ffn_dim=16, num_heads=2, num_layers=1, kv_channels=4),
        dit_config=VoxCPM2DitConfig(
            hidden_dim=8,
            ffn_dim=16,
            num_heads=2,
            num_layers=1,
            kv_channels=4,
            cfm_config=CfmConfig(),
        ),
    )


def test_residual_lm_no_rope_defaults_to_false():
    config = _make_config(residual_lm_no_rope=False)
    assert config.residual_lm_no_rope is False


def test_voxcpm2_model_disables_rope_only_for_residual_lm(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm2 import model as voxcpm2_model

    captured_use_rope = []

    class FakeCpm4Model(torch.nn.Module):
        def __init__(self, config, is_causal=True, lora_config=None, use_rope=True, lora_domain="lm_domain"):
            super().__init__()
            captured_use_rope.append(use_rope)
            self.embed_tokens = torch.nn.Identity()

        def forward(self, input_embeds, positions):
            return input_embeds

    monkeypatch.setattr(voxcpm2_model, "Cpm4Model", FakeCpm4Model)

    model = voxcpm2_model.VoxCPM2Model(_make_config(residual_lm_no_rope=True), inference_timesteps=2)

    assert isinstance(model.base_lm, FakeCpm4Model)
    assert isinstance(model.residual_lm, FakeCpm4Model)
    assert captured_use_rope == [True, False, True, True]
