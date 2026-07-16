import pytest

torch = pytest.importorskip("torch")


def _patch_tp1(monkeypatch, mod):
    monkeypatch.setattr(mod, "get_tp_rank", lambda: 0)
    monkeypatch.setattr(mod, "get_tp_world_size", lambda: 1)


def test_vocab_parallel_embedding_constructor_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    _patch_tp1(monkeypatch, eh)
    m = eh.VocabParallelEmbedding(num_embeddings=16, embedding_dim=8)
    assert m.tp_rank == 0
    assert m.tp_size == 1
    assert m.num_embeddings == 16
    assert m.num_embeddings_per_partition == 16
    assert m.vocab_start_idx == 0
    assert m.vocab_end_idx == 16
    assert m.weight.shape == (16, 8)


def test_vocab_parallel_embedding_partition_indices_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    _patch_tp1(monkeypatch, eh)
    m = eh.VocabParallelEmbedding(num_embeddings=32, embedding_dim=4)
    assert m.vocab_start_idx == 0
    assert m.vocab_end_idx == 32
    assert m.num_embeddings_per_partition == 32


def test_vocab_parallel_embedding_partition_indices_tp2_rank0(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    monkeypatch.setattr(eh, "get_tp_rank", lambda: 0)
    monkeypatch.setattr(eh, "get_tp_world_size", lambda: 2)
    m = eh.VocabParallelEmbedding(num_embeddings=32, embedding_dim=4)
    assert m.num_embeddings_per_partition == 16
    assert m.vocab_start_idx == 0
    assert m.vocab_end_idx == 16
    assert m.weight.shape == (16, 4)


def test_vocab_parallel_embedding_partition_indices_tp2_rank1(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    monkeypatch.setattr(eh, "get_tp_rank", lambda: 1)
    monkeypatch.setattr(eh, "get_tp_world_size", lambda: 2)
    m = eh.VocabParallelEmbedding(num_embeddings=32, embedding_dim=4)
    assert m.num_embeddings_per_partition == 16
    assert m.vocab_start_idx == 16
    assert m.vocab_end_idx == 32


def test_vocab_parallel_embedding_requires_divisible_num_embeddings(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    monkeypatch.setattr(eh, "get_tp_rank", lambda: 0)
    monkeypatch.setattr(eh, "get_tp_world_size", lambda: 3)
    with pytest.raises(AssertionError):
        eh.VocabParallelEmbedding(num_embeddings=10, embedding_dim=4)


def test_vocab_parallel_embedding_weight_loader_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    _patch_tp1(monkeypatch, eh)
    m = eh.VocabParallelEmbedding(num_embeddings=8, embedding_dim=4)
    loaded = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    m.weight_loader(m.weight, loaded)
    assert torch.allclose(m.weight, loaded)


def test_vocab_parallel_embedding_weight_loader_tp2_rank0(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    monkeypatch.setattr(eh, "get_tp_rank", lambda: 0)
    monkeypatch.setattr(eh, "get_tp_world_size", lambda: 2)
    m = eh.VocabParallelEmbedding(num_embeddings=8, embedding_dim=4)
    loaded = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    m.weight_loader(m.weight, loaded)
    assert torch.allclose(m.weight, loaded[:4])


def test_vocab_parallel_embedding_weight_loader_tp2_rank1(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    monkeypatch.setattr(eh, "get_tp_rank", lambda: 1)
    monkeypatch.setattr(eh, "get_tp_world_size", lambda: 2)
    m = eh.VocabParallelEmbedding(num_embeddings=8, embedding_dim=4)
    loaded = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
    m.weight_loader(m.weight, loaded)
    assert torch.allclose(m.weight, loaded[4:])


def test_vocab_parallel_embedding_forward_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    _patch_tp1(monkeypatch, eh)
    m = eh.VocabParallelEmbedding(num_embeddings=8, embedding_dim=4)
    torch.nn.init.constant_(m.weight, 1.0)
    indices = torch.tensor([0, 3, 7])
    out = m(indices)
    assert out.shape == (3, 4)
    assert torch.all(out == 1.0)


def test_vocab_parallel_embedding_weight_has_loader_attr(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    _patch_tp1(monkeypatch, eh)
    m = eh.VocabParallelEmbedding(num_embeddings=8, embedding_dim=4)
    assert hasattr(m.weight, "weight_loader")
    assert callable(m.weight.weight_loader)


def test_parallel_lm_head_constructor_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    _patch_tp1(monkeypatch, eh)
    m = eh.ParallelLMHead(num_embeddings=16, embedding_dim=8)
    assert m.weight.shape == (16, 8)
    assert m.tp_size == 1
    assert m.tp_rank == 0


def test_parallel_lm_head_rejects_bias(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    _patch_tp1(monkeypatch, eh)
    with pytest.raises(AssertionError):
        eh.ParallelLMHead(num_embeddings=16, embedding_dim=8, bias=True)


def test_parallel_lm_head_forward_decode_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh
    from nanovllm_voxcpm.utils import context as ctx_mod

    _patch_tp1(monkeypatch, eh)
    ctx_mod.set_context(is_prefill=False)
    try:
        m = eh.ParallelLMHead(num_embeddings=16, embedding_dim=8)
        torch.nn.init.constant_(m.weight, 0.5)
        x = torch.randn(3, 8)
        out = m(x)
        assert out.shape == (3, 16)
    finally:
        ctx_mod.reset_context()


def test_parallel_lm_head_forward_prefill_tp1(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh
    from nanovllm_voxcpm.utils import context as ctx_mod

    _patch_tp1(monkeypatch, eh)
    cu_seqlens_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    ctx_mod.set_context(is_prefill=True, cu_seqlens_q=cu_seqlens_q)
    try:
        m = eh.ParallelLMHead(num_embeddings=16, embedding_dim=8)
        torch.nn.init.constant_(m.weight, 0.5)
        x = torch.randn(5, 8)
        out = m(x)
        assert out.shape == (2, 16)
    finally:
        ctx_mod.reset_context()


@pytest.mark.gpu
def test_vocab_parallel_embedding_forward_tp2_all_reduce(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh

    monkeypatch.setattr(eh, "get_tp_rank", lambda: 0)
    monkeypatch.setattr(eh, "get_tp_world_size", lambda: 2)
    m = eh.VocabParallelEmbedding(num_embeddings=8, embedding_dim=4)
    indices = torch.tensor([0, 3])
    m(indices)


@pytest.mark.gpu
def test_parallel_lm_head_forward_tp2_all_gather(monkeypatch):
    import nanovllm_voxcpm.layers.embed_head as eh
    from nanovllm_voxcpm.utils import context as ctx_mod

    monkeypatch.setattr(eh, "get_tp_rank", lambda: 0)
    monkeypatch.setattr(eh, "get_tp_world_size", lambda: 2)
    ctx_mod.set_context(is_prefill=False)
    try:
        m = eh.ParallelLMHead(num_embeddings=8, embedding_dim=4)
        x = torch.randn(2, 4)
        m(x)
    finally:
        ctx_mod.reset_context()
