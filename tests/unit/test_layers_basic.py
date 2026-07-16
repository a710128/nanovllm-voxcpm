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


def _reference_rmsnorm(x: "torch.Tensor", weight: "torch.Tensor", eps: float) -> "torch.Tensor":
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = xf * torch.rsqrt(var + eps)
    return (normed * weight.float()).to(x.dtype)


def test_rmsnorm_forward_matches_reference():
    from nanovllm_voxcpm.layers.layernorm import RMSNorm

    eps = 1e-6
    norm = RMSNorm(hidden_size=16, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16) * 0.1 + 1.0)

    x = torch.randn(4, 16)
    got = norm(x)
    expected = _reference_rmsnorm(x, norm.weight, eps)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_rmsnorm_add_residual_is_exact_sum():
    from nanovllm_voxcpm.layers.layernorm import RMSNorm

    eps = 1e-6
    norm = RMSNorm(hidden_size=16, eps=eps)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(16) * 0.1 + 1.0)

    x = torch.randn(4, 16)
    residual = torch.randn(4, 16)

    out, new_residual = norm(x, residual)

    # add_rms_forward must return the pre-normalization sum bit-exactly (rtol=atol=0).
    torch.testing.assert_close(new_residual, (x.float() + residual.float()).to(x.dtype), rtol=0, atol=0)
    expected_out = _reference_rmsnorm(new_residual, norm.weight, eps)
    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-5)


def test_silu_and_mul():
    from nanovllm_voxcpm.layers.activation import SiluAndMul

    m = SiluAndMul()
    x = torch.randn(2, 6)
    y = m(x)
    assert y.shape == (2, 3)
