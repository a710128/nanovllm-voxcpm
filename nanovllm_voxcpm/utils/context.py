from dataclasses import dataclass, field
import torch


LM_LORA_DOMAIN = "lm_domain"
PROJ_LORA_DOMAIN = "proj_domain"
DIT_LORA_DOMAIN = "dit_domain"


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None


@dataclass
class LoRAContext:
    token_to_slot: torch.Tensor | None = None
    token_indices_sorted_by_slot: torch.Tensor | None = None
    active_slot_ids: torch.Tensor | None = None
    num_tokens_per_slot: torch.Tensor | None = None
    slot_start_offsets: torch.Tensor | None = None
    no_lora_flag: bool = True
    scratch_buffer: torch.Tensor | None = None
    no_lora_flag_cpu: torch.Tensor | None = None
    num_active_loras_cpu: torch.Tensor | None = None


@dataclass
class LoRAContexts:
    domains: dict[str, LoRAContext] = field(default_factory=dict)


_CONTEXT = Context()
_LORA_CONTEXTS = LoRAContexts()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def get_lora_context(domain: str = LM_LORA_DOMAIN):
    return _LORA_CONTEXTS.domains.get(domain, LoRAContext())


def build_lora_context_from_token_to_slot(
    token_to_slot: torch.Tensor | None,
    *,
    max_lora_rank: int = 0,
    dtype: torch.dtype | None = None,
) -> LoRAContext:
    if token_to_slot is None or token_to_slot.numel() == 0:
        return LoRAContext(
            token_to_slot=token_to_slot,
            no_lora_flag=True,
            no_lora_flag_cpu=torch.tensor([True], dtype=torch.bool, device="cpu"),
            num_active_loras_cpu=torch.tensor([0], dtype=torch.int32, device="cpu"),
        )

    token_to_slot = token_to_slot.to(dtype=torch.int32)
    active_slot_ids = torch.unique(token_to_slot[token_to_slot >= 0], sorted=True).to(dtype=torch.int32)
    no_lora_flag = active_slot_ids.numel() == 0
    if no_lora_flag:
        return LoRAContext(
            token_to_slot=token_to_slot,
            no_lora_flag=True,
            scratch_buffer=torch.zeros(token_to_slot.numel(), max_lora_rank, dtype=dtype, device=token_to_slot.device)
            if dtype is not None
            else None,
            no_lora_flag_cpu=torch.tensor([True], dtype=torch.bool, device="cpu"),
            num_active_loras_cpu=torch.tensor([0], dtype=torch.int32, device="cpu"),
        )

    token_indices_by_slot = []
    token_counts = []
    for slot_id in active_slot_ids.tolist():
        indices = torch.nonzero(token_to_slot == slot_id, as_tuple=False).flatten().to(dtype=torch.int32)
        token_indices_by_slot.append(indices)
        token_counts.append(indices.numel())

    num_tokens_per_slot = torch.tensor(token_counts, dtype=torch.int32, device=token_to_slot.device)
    slot_start_offsets = torch.zeros(active_slot_ids.numel() + 1, dtype=torch.int32, device=token_to_slot.device)
    slot_start_offsets[1:] = torch.cumsum(num_tokens_per_slot, dim=0)
    token_indices_sorted_by_slot = torch.cat(token_indices_by_slot).to(device=token_to_slot.device, dtype=torch.int32)
    scratch_buffer = (
        torch.zeros(token_to_slot.numel(), max_lora_rank, dtype=dtype, device=token_to_slot.device)
        if dtype is not None
        else None
    )
    return LoRAContext(
        token_to_slot=token_to_slot,
        token_indices_sorted_by_slot=token_indices_sorted_by_slot,
        active_slot_ids=active_slot_ids,
        num_tokens_per_slot=num_tokens_per_slot,
        slot_start_offsets=slot_start_offsets,
        no_lora_flag=False,
        scratch_buffer=scratch_buffer,
        no_lora_flag_cpu=torch.tensor([False], dtype=torch.bool, device="cpu"),
        num_active_loras_cpu=torch.tensor([active_slot_ids.numel()], dtype=torch.int32, device="cpu"),
    )


def set_lora_context_from_token_to_slot(
    token_to_slot: torch.Tensor | None,
    *,
    domain: str = LM_LORA_DOMAIN,
    max_lora_rank: int = 0,
    dtype: torch.dtype | None = None,
) -> None:
    set_lora_context(
        domain=domain,
        **build_lora_context_from_token_to_slot(
            token_to_slot,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
        ).__dict__,
    )


def set_lora_context(
    token_to_slot=None,
    token_indices_sorted_by_slot=None,
    active_slot_ids=None,
    num_tokens_per_slot=None,
    slot_start_offsets=None,
    no_lora_flag=True,
    scratch_buffer=None,
    no_lora_flag_cpu=None,
    num_active_loras_cpu=None,
    domain: str = LM_LORA_DOMAIN,
):
    _LORA_CONTEXTS.domains[domain] = LoRAContext(
        token_to_slot=token_to_slot,
        token_indices_sorted_by_slot=token_indices_sorted_by_slot,
        active_slot_ids=active_slot_ids,
        num_tokens_per_slot=num_tokens_per_slot,
        slot_start_offsets=slot_start_offsets,
        no_lora_flag=no_lora_flag,
        scratch_buffer=scratch_buffer,
        no_lora_flag_cpu=no_lora_flag_cpu,
        num_active_loras_cpu=num_active_loras_cpu,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


def reset_lora_context(domain: str | None = None):
    global _LORA_CONTEXTS
    if domain is None:
        _LORA_CONTEXTS = LoRAContexts()
    else:
        _LORA_CONTEXTS.domains.pop(domain, None)


def reset_all_contexts():
    reset_context()
    reset_lora_context()
