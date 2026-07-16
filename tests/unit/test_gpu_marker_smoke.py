import pytest


@pytest.mark.gpu
def test_gpu_marker_skips_on_cpu():
    import torch

    assert torch.cuda.is_available(), "This test requires a real GPU"
