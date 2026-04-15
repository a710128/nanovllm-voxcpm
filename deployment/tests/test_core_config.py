import os

import pytest


def test_get_int_env_default(monkeypatch):
    from app.core.config import _get_int_env

    monkeypatch.delenv("NANOVLLM_X", raising=False)
    assert _get_int_env("NANOVLLM_X", 123) == 123

    monkeypatch.setenv("NANOVLLM_X", "")
    assert _get_int_env("NANOVLLM_X", 123) == 123


def test_get_int_env_invalid_raises(monkeypatch):
    from app.core.config import _get_int_env

    monkeypatch.setenv("NANOVLLM_X", "abc")
    with pytest.raises(RuntimeError, match="Invalid env NANOVLLM_X"):
        _get_int_env("NANOVLLM_X", 1)


def test_get_bool_env_parses_common_values(monkeypatch):
    from app.core.config import _get_bool_env

    monkeypatch.delenv("NANOVLLM_B", raising=False)
    assert _get_bool_env("NANOVLLM_B", True) is True
    assert _get_bool_env("NANOVLLM_B", False) is False

    monkeypatch.setenv("NANOVLLM_B", "true")
    assert _get_bool_env("NANOVLLM_B", False) is True
    monkeypatch.setenv("NANOVLLM_B", "0")
    assert _get_bool_env("NANOVLLM_B", True) is False


def test_get_bool_env_invalid_raises(monkeypatch):
    from app.core.config import _get_bool_env

    monkeypatch.setenv("NANOVLLM_B", "maybe")
    with pytest.raises(RuntimeError, match="Invalid env NANOVLLM_B"):
        _get_bool_env("NANOVLLM_B", False)


def test_get_float_env_invalid_raises(monkeypatch):
    from app.core.config import _get_float_env

    monkeypatch.setenv("NANOVLLM_F", "abc")
    with pytest.raises(RuntimeError, match="Invalid env NANOVLLM_F"):
        _get_float_env("NANOVLLM_F", 0.1)


def test_get_int_list_env_parses(monkeypatch):
    from app.core.config import _get_int_list_env

    monkeypatch.delenv("NANOVLLM_L", raising=False)
    assert _get_int_list_env("NANOVLLM_L", (3,)) == (3,)

    monkeypatch.setenv("NANOVLLM_L", "0,1, 2")
    assert _get_int_list_env("NANOVLLM_L", (3,)) == (0, 1, 2)


def test_get_int_list_env_invalid_raises(monkeypatch):
    from app.core.config import _get_int_list_env

    monkeypatch.setenv("NANOVLLM_L", " , ")
    with pytest.raises(RuntimeError, match="Invalid env NANOVLLM_L"):
        _get_int_list_env("NANOVLLM_L", (0,))

    monkeypatch.setenv("NANOVLLM_L", "0,a")
    with pytest.raises(RuntimeError, match="Invalid env NANOVLLM_L"):
        _get_int_list_env("NANOVLLM_L", (0,))


def test_get_str_list_env_parses_and_validates(monkeypatch):
    from app.core.config import _get_str_list_env

    monkeypatch.delenv("NANOVLLM_S", raising=False)
    assert _get_str_list_env("NANOVLLM_S", ("a",)) == ("a",)

    monkeypatch.setenv("NANOVLLM_S", "q_proj, v_proj")
    assert _get_str_list_env("NANOVLLM_S", ("a",)) == ("q_proj", "v_proj")

    monkeypatch.setenv("NANOVLLM_S", " , ")
    with pytest.raises(RuntimeError, match="Invalid env NANOVLLM_S"):
        _get_str_list_env("NANOVLLM_S", ("a",))


def test_load_config_validates_mp3_ranges(monkeypatch):
    from app.core.config import load_config

    monkeypatch.setenv("NANOVLLM_MP3_BITRATE_KBPS", "0")
    with pytest.raises(RuntimeError, match="NANOVLLM_MP3_BITRATE_KBPS must be > 0"):
        load_config()

    monkeypatch.setenv("NANOVLLM_MP3_BITRATE_KBPS", "192")
    monkeypatch.setenv("NANOVLLM_MP3_QUALITY", "3")
    with pytest.raises(RuntimeError, match=r"NANOVLLM_MP3_QUALITY must be in \[0, 2\]"):
        load_config()


def test_load_config_validates_serverpool(monkeypatch):
    from app.core.config import load_config

    monkeypatch.setenv("NANOVLLM_SERVERPOOL_MAX_NUM_SEQS", "0")
    with pytest.raises(RuntimeError, match="NANOVLLM_SERVERPOOL_MAX_NUM_SEQS must be > 0"):
        load_config()

    monkeypatch.setenv("NANOVLLM_SERVERPOOL_MAX_NUM_SEQS", "16")
    monkeypatch.setenv("NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION", "1.1")
    with pytest.raises(RuntimeError, match=r"NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION must be in \(0, 1\]"):
        load_config()

    monkeypatch.setenv("NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION", "0.95")
    monkeypatch.setenv("NANOVLLM_SERVERPOOL_DEVICES", " , ")
    with pytest.raises(RuntimeError, match="Invalid env NANOVLLM_SERVERPOOL_DEVICES"):
        load_config()

    monkeypatch.setenv("NANOVLLM_SERVERPOOL_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="NANOVLLM_SERVERPOOL_DEVICES entries must be >= 0"):
        load_config()


def test_load_config_rejects_legacy_lora_startup_env(monkeypatch):
    from app.core.config import load_config

    monkeypatch.setenv("NANOVLLM_LORA_URI", "file:///tmp/lora")
    with pytest.raises(RuntimeError, match="LoRA startup preload env vars were removed"):
        load_config()


def test_load_config_runtime_lora(monkeypatch):
    from app.core.config import load_config

    monkeypatch.setenv("NANOVLLM_LORA_ENABLED", "true")
    monkeypatch.setenv("NANOVLLM_LORA_MAX_LORAS", "2")
    monkeypatch.setenv("NANOVLLM_LORA_MAX_LORA_RANK", "16")
    monkeypatch.setenv("NANOVLLM_LORA_ENABLE_PROJ", "true")
    monkeypatch.setenv("NANOVLLM_LORA_TARGET_MODULES_LM", "q_proj,o_proj")

    cfg = load_config()
    assert cfg.lora is not None
    assert cfg.lora.max_loras == 2
    assert cfg.lora.max_lora_rank == 16
    assert cfg.lora.enable_proj is True
    assert cfg.lora.target_modules_lm == ("q_proj", "o_proj")
    assert cfg.lora.target_modules_dit is None
    assert cfg.lora.target_proj_modules is None


def test_materialize_lora_config_defaults_are_architecture_aware():
    from app.core.config import RuntimeLoRAConfig, materialize_lora_config

    raw = RuntimeLoRAConfig(
        enable_lm=None,
        enable_dit=None,
        enable_proj=None,
        max_loras=1,
        max_lora_rank=32,
        target_modules_lm=None,
        target_modules_dit=None,
        target_proj_modules=None,
    )

    voxcpm = materialize_lora_config(raw, "voxcpm")
    assert voxcpm.enable_lm is True
    assert voxcpm.enable_dit is True
    assert voxcpm.enable_proj is True
    assert voxcpm.target_modules_lm == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assert voxcpm.target_proj_modules == ("enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj")

    voxcpm2 = materialize_lora_config(raw, "voxcpm2")
    assert voxcpm2.target_proj_modules == (
        "enc_to_lm_proj",
        "lm_to_dit_proj",
        "res_to_dit_proj",
        "fusion_concat_proj",
    )


def test_materialize_lora_config_preserves_overrides():
    from app.core.config import RuntimeLoRAConfig, materialize_lora_config

    raw = RuntimeLoRAConfig(
        enable_lm=True,
        enable_dit=False,
        enable_proj=False,
        max_loras=2,
        max_lora_rank=16,
        target_modules_lm=("q_proj",),
        target_modules_dit=("k_proj",),
        target_proj_modules=("enc_to_lm_proj",),
    )

    config = materialize_lora_config(raw, "voxcpm2")
    assert config.enable_lm is True
    assert config.enable_dit is False
    assert config.enable_proj is False
    assert config.max_loras == 2
    assert config.max_lora_rank == 16
    assert config.target_modules_lm == ("q_proj",)
    assert config.target_modules_dit == ("k_proj",)
    assert config.target_proj_modules == ("enc_to_lm_proj",)


def test_load_config_runtime_lora_validates_ranges(monkeypatch):
    from app.core.config import load_config

    monkeypatch.setenv("NANOVLLM_LORA_ENABLED", "true")
    monkeypatch.setenv("NANOVLLM_LORA_MAX_LORAS", "0")
    with pytest.raises(RuntimeError, match="NANOVLLM_LORA_MAX_LORAS must be > 0"):
        load_config()

    monkeypatch.setenv("NANOVLLM_LORA_MAX_LORAS", "1")
    monkeypatch.setenv("NANOVLLM_LORA_MAX_LORA_RANK", "0")
    with pytest.raises(RuntimeError, match="NANOVLLM_LORA_MAX_LORA_RANK must be > 0"):
        load_config()

    monkeypatch.setenv("NANOVLLM_LORA_MAX_LORA_RANK", "32")
    monkeypatch.setenv("NANOVLLM_LORA_ENABLE_LM", "false")
    monkeypatch.setenv("NANOVLLM_LORA_ENABLE_DIT", "false")
    monkeypatch.setenv("NANOVLLM_LORA_ENABLE_PROJ", "false")
    with pytest.raises(RuntimeError, match="At least one"):
        load_config()


def test_load_config_expands_user_paths(monkeypatch):
    from app.core.config import load_config

    monkeypatch.setenv("NANOVLLM_MODEL_PATH", "~/VoxCPM1.5")
    cfg = load_config()
    assert cfg.model_path == os.path.expanduser("~/VoxCPM1.5")
