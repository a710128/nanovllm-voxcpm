import pytest


torch = pytest.importorskip("torch")


def test_context_set_get_reset():
    from nanovllm_voxcpm.utils.context import get_context, set_context, reset_context

    reset_context()
    ctx = get_context()
    assert ctx.is_prefill is False
    assert ctx.cu_seqlens_q is None

    t = torch.tensor([1, 2, 3])
    set_context(True, cu_seqlens_q=t, max_seqlen_q=3)
    ctx = get_context()
    assert ctx.is_prefill is True
    assert ctx.cu_seqlens_q is t
    assert ctx.max_seqlen_q == 3

    reset_context()
    ctx = get_context()
    assert ctx.is_prefill is False
    assert ctx.cu_seqlens_q is None
