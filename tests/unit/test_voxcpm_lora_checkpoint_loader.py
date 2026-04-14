import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _write_checkpoint(tmp_path, tensors: dict[str, torch.Tensor], *, rank: int = 2, alpha: float = 8.0):
    checkpoint_dir = tmp_path / "demo_lora"
    checkpoint_dir.mkdir()
    safetensors_torch.save_file(tensors, str(checkpoint_dir / "lora_weights.safetensors"))
    (checkpoint_dir / "lora_config.json").write_text(
        json.dumps({"lora_config": {"r": rank, "alpha": alpha}}),
        encoding="utf-8",
    )
    return checkpoint_dir


def test_load_voxcpm_lora_checkpoint_builds_payload_and_tp_shards(tmp_path):
    from nanovllm_voxcpm.models.voxcpm.lora_loader import load_voxcpm_lora_checkpoint

    checkpoint_dir = _write_checkpoint(
        tmp_path,
        {
            "base_lm.layers.0.self_attn.q_proj.lora_A.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "base_lm.layers.0.self_attn.q_proj.lora_B.weight": torch.tensor(
                [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]]
            ),
            "base_lm.layers.0.self_attn.v_proj.lora_A.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            "base_lm.layers.0.self_attn.v_proj.lora_B.weight": torch.tensor(
                [[20.0, 21.0], [22.0, 23.0], [24.0, 25.0], [26.0, 27.0]]
            ),
            "base_lm.layers.0.self_attn.o_proj.lora_A.weight": torch.tensor(
                [[30.0, 31.0, 32.0, 33.0], [34.0, 35.0, 36.0, 37.0]]
            ),
            "base_lm.layers.0.self_attn.o_proj.lora_B.weight": torch.tensor([[40.0, 41.0], [42.0, 43.0]]),
            "enc_to_lm_proj.lora_A.weight": torch.tensor([[50.0, 51.0], [52.0, 53.0]]),
            "enc_to_lm_proj.lora_B.weight": torch.tensor([[60.0, 61.0], [62.0, 63.0]]),
        },
    )

    payloads = load_voxcpm_lora_checkpoint(str(checkpoint_dir), tp_size=2)

    assert len(payloads) == 2
    rank0 = payloads[0]
    rank1 = payloads[1]
    assert rank0.rank == 2
    assert rank0.alpha == 8.0
    assert rank0.modules["base_lm.layers.0.self_attn.qkv_proj"].lora_a.shape == (2, 2, 2)
    assert [tensor.shape for tensor in rank0.modules["base_lm.layers.0.self_attn.qkv_proj"].lora_b] == [(2, 2), (2, 2)]
    assert torch.equal(
        rank0.modules["base_lm.layers.0.self_attn.qkv_proj"].lora_b[0],
        torch.tensor([[10.0, 11.0], [12.0, 13.0]]),
    )
    assert torch.equal(
        rank1.modules["base_lm.layers.0.self_attn.qkv_proj"].lora_b[0],
        torch.tensor([[14.0, 15.0], [16.0, 17.0]]),
    )
    assert torch.equal(
        rank0.modules["base_lm.layers.0.self_attn.o_proj"].lora_a,
        torch.tensor([[30.0, 31.0], [34.0, 35.0]]),
    )
    assert torch.equal(
        rank1.modules["base_lm.layers.0.self_attn.o_proj"].lora_a,
        torch.tensor([[32.0, 33.0], [36.0, 37.0]]),
    )
    assert torch.equal(rank0.modules["enc_to_lm_proj"].lora_a, rank1.modules["enc_to_lm_proj"].lora_a)


def test_load_voxcpm_lora_checkpoint_requires_lora_config_object(tmp_path):
    from nanovllm_voxcpm.models.voxcpm.lora_loader import load_voxcpm_lora_checkpoint

    checkpoint_dir = tmp_path / "broken_lora"
    checkpoint_dir.mkdir()
    safetensors_torch.save_file({}, str(checkpoint_dir / "lora_weights.safetensors"))
    (checkpoint_dir / "lora_config.json").write_text(json.dumps({"alpha": 8}), encoding="utf-8")

    with pytest.raises(ValueError, match="lora_config"):
        load_voxcpm_lora_checkpoint(str(checkpoint_dir))
