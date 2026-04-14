from dataclasses import dataclass
import torch


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


_CONTEXT = Context()
_LORA_CONTEXT = LoRAContext()


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


def get_lora_context():
    return _LORA_CONTEXT


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
):
    global _LORA_CONTEXT
    _LORA_CONTEXT = LoRAContext(
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


def reset_lora_context():
    global _LORA_CONTEXT
    _LORA_CONTEXT = LoRAContext()


def reset_all_contexts():
    reset_context()
    reset_lora_context()
