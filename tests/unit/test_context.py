import pytest

torch = pytest.importorskip("torch")


def test_context_set_get_reset():
    from nanovllm_voxcpm.utils.context import get_context, reset_context, set_context

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


def test_lora_context_set_get_reset():
    from nanovllm_voxcpm.utils.context import (
        PROJ_LORA_DOMAIN,
        get_lora_context,
        reset_lora_context,
        set_lora_context,
    )

    reset_lora_context()
    ctx = get_lora_context()
    assert ctx.no_lora_flag is True
    assert ctx.token_to_slot is None

    token_to_slot = torch.tensor([0, -1, 1], dtype=torch.int32)
    active_slot_ids = torch.tensor([0, 1], dtype=torch.int32)
    set_lora_context(
        token_to_slot=token_to_slot,
        active_slot_ids=active_slot_ids,
        no_lora_flag=False,
    )
    ctx = get_lora_context()
    assert ctx.no_lora_flag is False
    assert ctx.token_to_slot is token_to_slot
    assert ctx.active_slot_ids is active_slot_ids

    proj_token_to_slot = torch.tensor([1], dtype=torch.int32)
    set_lora_context(token_to_slot=proj_token_to_slot, no_lora_flag=False, domain=PROJ_LORA_DOMAIN)
    proj_ctx = get_lora_context(PROJ_LORA_DOMAIN)
    assert proj_ctx.token_to_slot is proj_token_to_slot
    assert get_lora_context().token_to_slot is token_to_slot

    reset_lora_context()
    ctx = get_lora_context()
    assert ctx.no_lora_flag is True
    assert ctx.token_to_slot is None
    assert get_lora_context(PROJ_LORA_DOMAIN).token_to_slot is None
