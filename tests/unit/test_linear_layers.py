import pytest


torch = pytest.importorskip("torch")


def test_column_parallel_linear_weight_loader_shards(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    # Pretend we are in a 2-way tensor parallel group.
    monkeypatch.setattr(linear.dist, "get_world_size", lambda: 2)

    full_weight = torch.arange(6 * 4, dtype=torch.float32).view(6, 4)

    monkeypatch.setattr(linear.dist, "get_rank", lambda: 0)
    m0 = linear.ColumnParallelLinear(input_size=4, output_size=6, bias=False)
    m0.weight_loader(m0.weight, full_weight)

    monkeypatch.setattr(linear.dist, "get_rank", lambda: 1)
    m1 = linear.ColumnParallelLinear(input_size=4, output_size=6, bias=False)
    m1.weight_loader(m1.weight, full_weight)

    # Column parallel splits output rows evenly across ranks.
    assert torch.allclose(m0.weight, full_weight[:3])
    assert torch.allclose(m1.weight, full_weight[3:])


def test_divide_requires_exact_division():
    import nanovllm_voxcpm.layers.linear as linear

    assert linear.divide(8, 2) == 4
    with pytest.raises(AssertionError):
        _ = linear.divide(7, 2)
