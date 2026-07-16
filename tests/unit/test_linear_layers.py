import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_tp1(monkeypatch, linear_mod):
    """Monkeypatch TP helpers to single-GPU mode (rank=0, size=1)."""
    monkeypatch.setattr(linear_mod, "get_tp_rank", lambda: 0)
    monkeypatch.setattr(linear_mod, "get_tp_world_size", lambda: 1)


# ---------------------------------------------------------------------------
# divide()
# ---------------------------------------------------------------------------

def test_divide_requires_exact_division():
    import nanovllm_voxcpm.layers.linear as linear

    assert linear.divide(8, 2) == 4
    with pytest.raises(AssertionError):
        _ = linear.divide(7, 2)


# ---------------------------------------------------------------------------
# LinearBase — not directly instantiable, but we verify the abstract stubs
# ---------------------------------------------------------------------------

def test_linear_base_forward_raises(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    # Must go through a concrete subclass because LinearBase.__init__ calls
    # get_tp_rank / get_tp_world_size via the module-level names we patched.
    m = linear.ReplicatedLinear(4, 8)
    # weight_loader is overridden; calling the base version would raise
    import nanovllm_voxcpm.layers.linear as lin_mod
    base_instance = lin_mod.LinearBase.__new__(lin_mod.LinearBase)
    # Directly invoking the ABC stubs should raise NotImplementedError
    with pytest.raises((NotImplementedError, TypeError)):
        lin_mod.LinearBase.forward(base_instance, torch.zeros(1, 4))
    with pytest.raises((NotImplementedError, TypeError)):
        lin_mod.LinearBase.weight_loader(base_instance, None, None)


# ---------------------------------------------------------------------------
# ReplicatedLinear
# ---------------------------------------------------------------------------

def test_replicated_linear_constructor_no_bias(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.ReplicatedLinear(input_size=4, output_size=8, bias=False)
    assert m.weight.shape == (8, 4)
    assert m.bias is None
    assert m.tp_size == 1
    assert m.tp_rank == 0


def test_replicated_linear_constructor_with_bias(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.ReplicatedLinear(input_size=4, output_size=8, bias=True)
    assert m.weight.shape == (8, 4)
    assert m.bias is not None
    assert m.bias.shape == (8,)


def test_replicated_linear_weight_loader(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.ReplicatedLinear(input_size=4, output_size=8)
    w = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    m.weight_loader(m.weight, w)
    assert torch.allclose(m.weight, w)


def test_replicated_linear_forward(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.ReplicatedLinear(input_size=4, output_size=8)
    torch.nn.init.eye_(m.weight[:4])  # only top 4 rows for clarity
    x = torch.ones(2, 4)
    out = m(x)
    assert out.shape == (2, 8)


def test_replicated_linear_forward_with_bias(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.ReplicatedLinear(input_size=3, output_size=5, bias=True)
    torch.nn.init.zeros_(m.bias)
    x = torch.randn(2, 3)
    out = m(x)
    assert out.shape == (2, 5)


# ---------------------------------------------------------------------------
# ColumnParallelLinear — tp_size=1
# ---------------------------------------------------------------------------

def test_column_parallel_linear_constructor_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.ColumnParallelLinear(input_size=4, output_size=8)
    # With tp_size=1 output stays 8
    assert m.weight.shape == (8, 4)
    assert m.tp_dim == 0


def test_column_parallel_linear_weight_loader_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.ColumnParallelLinear(input_size=4, output_size=8)
    w = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    m.weight_loader(m.weight, w)
    assert torch.allclose(m.weight, w)


def test_column_parallel_linear_forward_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.ColumnParallelLinear(input_size=4, output_size=8)
    torch.nn.init.constant_(m.weight, 0.5)
    x = torch.ones(3, 4)
    out = m(x)
    assert out.shape == (3, 8)


# ---------------------------------------------------------------------------
# ColumnParallelLinear — tp_size=2 weight loader (no distributed calls needed)
# ---------------------------------------------------------------------------

def test_column_parallel_linear_weight_loader_shards(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    monkeypatch.setattr(linear, "get_tp_world_size", lambda: 2)

    full_weight = torch.arange(6 * 4, dtype=torch.float32).view(6, 4)

    monkeypatch.setattr(linear, "get_tp_rank", lambda: 0)
    m0 = linear.ColumnParallelLinear(input_size=4, output_size=6, bias=False)
    m0.weight_loader(m0.weight, full_weight)

    monkeypatch.setattr(linear, "get_tp_rank", lambda: 1)
    m1 = linear.ColumnParallelLinear(input_size=4, output_size=6, bias=False)
    m1.weight_loader(m1.weight, full_weight)

    # Column parallel splits output rows evenly across ranks.
    assert torch.allclose(m0.weight, full_weight[:3])
    assert torch.allclose(m1.weight, full_weight[3:])


# ---------------------------------------------------------------------------
# MergedColumnParallelLinear
# ---------------------------------------------------------------------------

def test_merged_column_parallel_linear_constructor_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    output_sizes = [4, 6]
    m = linear.MergedColumnParallelLinear(input_size=8, output_sizes=output_sizes)
    assert m.output_sizes == output_sizes
    assert m.weight.shape == (sum(output_sizes), 8)  # 10 x 8


def test_merged_column_parallel_linear_weight_loader_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.MergedColumnParallelLinear(input_size=4, output_sizes=[4, 4])
    # Shard 0 — first 4 rows of a (8,4) weight
    full_w0 = torch.zeros(4, 4)
    full_w0[0] = 1.0
    m.weight_loader(m.weight, full_w0, loaded_shard_id=0)
    # Shard 1 — second 4 rows
    full_w1 = torch.zeros(4, 4)
    full_w1[0] = 2.0
    m.weight_loader(m.weight, full_w1, loaded_shard_id=1)
    # Row 0 comes from shard 0, row 4 from shard 1
    assert m.weight[0, 0].item() == pytest.approx(1.0)
    assert m.weight[4, 0].item() == pytest.approx(2.0)


def test_merged_column_parallel_linear_weight_loader_requires_int_shard(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.MergedColumnParallelLinear(input_size=4, output_sizes=[4, 4])
    w = torch.zeros(4, 4)
    with pytest.raises(AssertionError):
        m.weight_loader(m.weight, w, loaded_shard_id=None)


def test_merged_column_parallel_linear_forward_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.MergedColumnParallelLinear(input_size=4, output_sizes=[4, 4])
    x = torch.randn(2, 4)
    out = m(x)
    assert out.shape == (2, 8)


# ---------------------------------------------------------------------------
# QKVParallelLinear
# ---------------------------------------------------------------------------

def test_qkv_parallel_linear_constructor_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    hidden_size = 8
    head_size = 4
    total_num_heads = 2
    m = linear.QKVParallelLinear(
        hidden_size=hidden_size,
        head_size=head_size,
        total_num_heads=total_num_heads,
    )
    # With tp=1: output = (2 + 2*2) * 4 = 24
    expected_output = (total_num_heads + 2 * total_num_heads) * head_size
    assert m.weight.shape == (expected_output, hidden_size)
    assert m.num_heads == total_num_heads
    assert m.num_kv_heads == total_num_heads
    assert m.head_size == head_size


def test_qkv_parallel_linear_constructor_gqa_tp1(monkeypatch):
    """GQA: separate num_heads and num_kv_heads."""
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.QKVParallelLinear(
        hidden_size=16,
        head_size=4,
        total_num_heads=4,
        total_num_kv_heads=2,
    )
    assert m.num_heads == 4
    assert m.num_kv_heads == 2
    expected_output = (4 + 2 * 2) * 4  # 32
    assert m.weight.shape == (expected_output, 16)


def test_qkv_parallel_linear_weight_loader_qkv(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    head_size = 4
    m = linear.QKVParallelLinear(
        hidden_size=8,
        head_size=head_size,
        total_num_heads=2,
        total_num_kv_heads=2,
    )
    # q shard
    q_weight = torch.ones(m.num_heads * head_size, 8)
    m.weight_loader(m.weight, q_weight, loaded_shard_id="q")
    # k shard
    k_weight = torch.ones(m.num_kv_heads * head_size, 8) * 2
    m.weight_loader(m.weight, k_weight, loaded_shard_id="k")
    # v shard
    v_weight = torch.ones(m.num_kv_heads * head_size, 8) * 3
    m.weight_loader(m.weight, v_weight, loaded_shard_id="v")

    q_end = m.num_heads * head_size
    k_end = q_end + m.num_kv_heads * head_size
    assert torch.all(m.weight[:q_end] == 1.0)
    assert torch.all(m.weight[q_end:k_end] == 2.0)
    assert torch.all(m.weight[k_end:] == 3.0)


def test_qkv_parallel_linear_weight_loader_invalid_shard(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.QKVParallelLinear(hidden_size=8, head_size=4, total_num_heads=2)
    w = torch.zeros(8, 8)
    with pytest.raises(AssertionError):
        m.weight_loader(m.weight, w, loaded_shard_id="x")


def test_qkv_parallel_linear_forward_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.QKVParallelLinear(hidden_size=8, head_size=4, total_num_heads=2)
    x = torch.randn(3, 8)
    out = m(x)
    assert out.shape[0] == 3


# ---------------------------------------------------------------------------
# RowParallelLinear — tp_size=1
# ---------------------------------------------------------------------------

def test_row_parallel_linear_constructor_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.RowParallelLinear(input_size=8, output_size=4)
    # tp_size=1 → input stays 8
    assert m.weight.shape == (4, 8)
    assert m.tp_dim == 1


def test_row_parallel_linear_weight_loader_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.RowParallelLinear(input_size=8, output_size=4)
    w = torch.arange(4 * 8, dtype=torch.float32).view(4, 8)
    m.weight_loader(m.weight, w)
    assert torch.allclose(m.weight, w)


def test_row_parallel_linear_forward_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.RowParallelLinear(input_size=8, output_size=4)
    torch.nn.init.constant_(m.weight, 0.1)
    x = torch.ones(2, 8)
    out = m(x)
    assert out.shape == (2, 4)
    # bias is None by default → no bias addition
    assert m.bias is None


def test_row_parallel_linear_forward_with_bias_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    _patch_tp1(monkeypatch, linear)
    m = linear.RowParallelLinear(input_size=8, output_size=4, bias=True)
    torch.nn.init.zeros_(m.bias)
    x = torch.randn(3, 8)
    out = m(x)
    assert out.shape == (3, 4)


def test_row_parallel_linear_weight_loader_tp2_shards(monkeypatch):
    import nanovllm_voxcpm.layers.linear as linear

    full_weight = torch.arange(4 * 8, dtype=torch.float32).view(4, 8)

    monkeypatch.setattr(linear, "get_tp_world_size", lambda: 2)
    monkeypatch.setattr(linear, "get_tp_rank", lambda: 0)
    m0 = linear.RowParallelLinear(input_size=8, output_size=4)
    m0.weight_loader(m0.weight, full_weight)

    monkeypatch.setattr(linear, "get_tp_rank", lambda: 1)
    m1 = linear.RowParallelLinear(input_size=8, output_size=4)
    m1.weight_loader(m1.weight, full_weight)

    # Row parallel splits along input columns (dim=1)
    assert torch.allclose(m0.weight, full_weight[:, :4])
    assert torch.allclose(m1.weight, full_weight[:, 4:])
