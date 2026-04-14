from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import torch


@dataclass
class LoRAKernelMeta:
    token_lora_mapping: torch.Tensor
    token_indices_sorted_by_lora_ids: torch.Tensor
    active_lora_ids: torch.Tensor
    num_tokens_per_lora: torch.Tensor
    lora_token_start_loc: torch.Tensor
    no_lora_flag_cpu: torch.Tensor
    num_active_loras_cpu: torch.Tensor
    default_num_active_loras_cpu: torch.Tensor
    captured_lora_counts: list[int] = field(default_factory=list)

    @staticmethod
    def make(
        max_loras: int,
        max_num_tokens: int,
        device: torch.device | str,
        captured_lora_counts: list[int] | None = None,
    ) -> "LoRAKernelMeta":
        return LoRAKernelMeta(
            token_lora_mapping=torch.empty(max_num_tokens, dtype=torch.int32, device=device),
            token_indices_sorted_by_lora_ids=torch.empty(max_num_tokens, dtype=torch.int32, device=device),
            active_lora_ids=torch.empty(max_loras + 1, dtype=torch.int32, device=device),
            num_tokens_per_lora=torch.zeros(max_loras + 1, dtype=torch.int32, device=device),
            lora_token_start_loc=torch.zeros(max_loras + 2, dtype=torch.int32, device=device),
            no_lora_flag_cpu=torch.tensor([False], dtype=torch.bool, device="cpu"),
            num_active_loras_cpu=torch.tensor([0], dtype=torch.int32, device="cpu"),
            default_num_active_loras_cpu=torch.tensor([max_loras + 1], dtype=torch.int32, device="cpu"),
            captured_lora_counts=sorted(captured_lora_counts) if captured_lora_counts else [],
        )

    def _reset(self):
        self.active_lora_ids.fill_(-1)
        self.num_tokens_per_lora.fill_(0)
        self.lora_token_start_loc.fill_(0)
        self.no_lora_flag_cpu.fill_(False)
        self.num_active_loras_cpu.fill_(0)

    def prepare_tensors(self, token_lora_mapping: torch.Tensor) -> None:
        self._reset()
        no_lora = torch.all(token_lora_mapping == -1)
        self.no_lora_flag_cpu[0] = no_lora
        if no_lora:
            return
        num_tokens = token_lora_mapping.size(0)
        self.token_lora_mapping[:num_tokens].copy_(token_lora_mapping, non_blocking=True)
        _, token_indices_sorted = torch.sort(token_lora_mapping, stable=True)
        self.token_indices_sorted_by_lora_ids[:num_tokens].copy_(token_indices_sorted, non_blocking=True)
        lora_ids, num_tokens_per_lora = torch.unique(token_lora_mapping, sorted=True, return_counts=True)
        self.active_lora_ids[: lora_ids.size(0)].copy_(lora_ids, non_blocking=True)
        self.num_tokens_per_lora[: num_tokens_per_lora.size(0)].copy_(num_tokens_per_lora, non_blocking=True)
        num_active_loras = lora_ids.size(0)
        if self.captured_lora_counts and num_active_loras > 0:
            idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
            if idx < len(self.captured_lora_counts):
                num_active_loras = self.captured_lora_counts[idx]
        self.num_active_loras_cpu[0] = num_active_loras
        lora_token_start_loc = torch.cumsum(num_tokens_per_lora, dim=0)
        self.lora_token_start_loc[1 : 1 + lora_token_start_loc.size(0)].copy_(lora_token_start_loc, non_blocking=True)

    def meta_args(self, token_nums: int, specialize_active_lora: bool):
        num_active_loras = self.num_active_loras_cpu if specialize_active_lora else self.default_num_active_loras_cpu
        return (
            self.token_lora_mapping[:token_nums],
            self.token_indices_sorted_by_lora_ids[:token_nums],
            self.num_tokens_per_lora,
            self.lora_token_start_loc,
            self.active_lora_ids,
            self.no_lora_flag_cpu,
            num_active_loras,
        )
