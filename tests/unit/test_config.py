import pytest


def test_config_post_init_asserts(tmp_path):
    from nanovllm_voxcpm.config import CUDAGraphMode, Config, resolve_cudagraph_mode

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    cfg = Config(model=str(model_dir))
    assert cfg.model == str(model_dir)
    assert cfg.cudagraph_mode is CUDAGraphMode.FULL
    assert resolve_cudagraph_mode(enforce_eager=False, enforce_dit_prefill_eager=True) is CUDAGraphMode.DECODE_ONLY
    assert resolve_cudagraph_mode(enforce_eager=True, enforce_dit_prefill_eager=False) is CUDAGraphMode.DISABLED
    assert resolve_cudagraph_mode(enforce_eager=True, enforce_dit_prefill_eager=True) is CUDAGraphMode.DISABLED

    with pytest.raises(AssertionError):
        _ = Config(model=str(model_dir), kvcache_block_size=128)

    with pytest.raises(AssertionError):
        _ = Config(model=str(model_dir), tensor_parallel_size=0)

    with pytest.raises(AssertionError):
        _ = Config(model=str(model_dir), max_num_batched_tokens=16, max_model_len=32)
