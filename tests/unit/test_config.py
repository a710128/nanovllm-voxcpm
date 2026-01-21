import pytest


def test_config_post_init_asserts(tmp_path):
    from nanovllm_voxcpm.config import Config

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    cfg = Config(model=str(model_dir))
    assert cfg.model == str(model_dir)

    with pytest.raises(AssertionError):
        _ = Config(model=str(model_dir), kvcache_block_size=128)

    with pytest.raises(AssertionError):
        _ = Config(model=str(model_dir), tensor_parallel_size=0)

    with pytest.raises(AssertionError):
        _ = Config(model=str(model_dir), max_num_batched_tokens=16, max_model_len=32)
