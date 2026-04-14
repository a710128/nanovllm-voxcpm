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


def test_load_config_expands_user_paths(monkeypatch):
    from app.core.config import load_config

    monkeypatch.setenv("NANOVLLM_MODEL_PATH", "~/VoxCPM1.5")
    cfg = load_config()
    assert cfg.model_path == os.path.expanduser("~/VoxCPM1.5")
