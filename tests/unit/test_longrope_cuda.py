import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 1, 8, 2, 64), (3, 5, 4, 4, 64), (107, 1, 16, 2, 128), (2, 3, 3, 1, 48)])
def test_fused_longrope_matches_eager_and_replays_with_new_positions(dtype, shape):
    from nanovllm_voxcpm.layers.longrope import apply_longrope
    from nanovllm_voxcpm.models.voxcpm.model_utils import apply_rotary_emb_tokens

    batch, patches, query_heads, key_heads, dim = shape
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(123)
    packed = torch.randn(
        batch, patches, (query_heads + 2 * key_heads) * dim, generator=generator, dtype=dtype, device=device
    )
    query, key, _ = packed.split([query_heads * dim, key_heads * dim, key_heads * dim], dim=-1)
    positions = (torch.arange(batch * patches * 2, device=device) % 32)[::2]
    angles = torch.randn(64, dim // 2, generator=generator, device=device)
    angles = torch.cat([angles, angles], dim=-1)
    cos, sin = angles.cos() * 1.2, angles.sin() * 1.2

    def reference(value):
        return apply_rotary_emb_tokens(value.reshape(batch * patches, -1, dim), cos[positions], sin[positions]).reshape(
            value.shape
        )

    with torch.inference_mode():
        actual = apply_longrope(positions, query, key, cos, sin, dim)
        assert torch.equal(actual[0], reference(query))
        assert torch.equal(actual[1], reference(key))
        # Captured kernels must read new positions and noncontiguous QKV values.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = apply_longrope(positions, query, key, cos, sin, dim)
        positions.add_(1)
        packed.mul_(0.5)
        graph.replay()
        assert torch.equal(replayed[0], reference(query))
        assert torch.equal(replayed[1], reference(key))
