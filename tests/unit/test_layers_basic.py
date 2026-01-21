import math

import pytest


torch = pytest.importorskip("torch")


def test_sampler_shapes_and_range():
    from nanovllm_voxcpm.layers.sampler import Sampler

    sampler = Sampler()
    logits = torch.randn(4, 10)
    temps = torch.ones(4)
    out = sampler(logits, temps)
    assert out.shape == (4,)
    assert int(out.min()) >= 0
    assert int(out.max()) < 10


def test_rotary_embedding_preserves_shapes():
    from nanovllm_voxcpm.layers.rotary_embedding import RotaryEmbedding

    rope = RotaryEmbedding(head_size=8, rotary_dim=8, max_position_embeddings=32, base=10000.0)
    positions = torch.tensor([0, 1, 2], dtype=torch.long)
    q = torch.randn(3, 1, 8)
    k = torch.randn(3, 1, 8)
    q2, k2 = rope(positions, q, k)
    assert q2.shape == q.shape
    assert k2.shape == k.shape


def test_rmsnorm_forward_and_residual_path():
    from nanovllm_voxcpm.layers.layernorm import RMSNorm

    norm = RMSNorm(hidden_size=8, eps=1e-6)
    x = torch.randn(2, 8)
    y = norm(x)
    assert y.shape == x.shape

    residual = torch.randn(2, 8)
    y2, r2 = norm(x, residual)
    assert y2.shape == x.shape
    assert r2.shape == x.shape


def test_silu_and_mul():
    from nanovllm_voxcpm.layers.activation import SiluAndMul

    m = SiluAndMul()
    x = torch.randn(2, 6)
    y = m(x)
    assert y.shape == (2, 3)
