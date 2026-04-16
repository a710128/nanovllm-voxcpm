from __future__ import annotations

import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nanovllm_voxcpm.lora import LoRAMetadata, get_backend
from nanovllm_voxcpm.utils.context import get_lora_context
from nanovllm_voxcpm.utils.torch_param import set_weight_loader

ShardId = str | int


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def _get_world_size() -> int:
    if not dist.is_available():
        return 1
    try:
        return dist.get_world_size()
    except Exception:
        return 1


def _get_rank() -> int:
    if not dist.is_available():
        return 0
    try:
        return dist.get_rank()
    except Exception:
        return 0


def _flatten_tokens(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    original_shape = x.shape
    if x.ndim == 2:
        return x, original_shape
    return x.reshape(-1, x.size(-1)), original_shape


def _restore_tokens(x: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
    if len(original_shape) == 2:
        return x
    return x.reshape(*original_shape[:-1], x.size(-1))


def _is_cuda_graph_capture() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class _LoRALayerBase(nn.Module):
    lora_scaling: torch.Tensor
    effective_lora_rank: torch.Tensor

    def __init__(self, max_loras: int, max_lora_rank: int, supports_lora: bool):
        super().__init__()
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.supports_lora = supports_lora
        self.register_buffer("lora_scaling", torch.zeros(max_loras), persistent=False)
        self.register_buffer("lora_base_scaling", torch.zeros(max_loras), persistent=False)
        self.register_buffer("effective_lora_rank", torch.zeros(max_loras, dtype=torch.int32), persistent=False)
        self._lora_scaling_values = [0.0 for _ in range(max_loras)]
        self._lora_base_scaling_values = [0.0 for _ in range(max_loras)]
        self._effective_lora_rank_values = [0 for _ in range(max_loras)]

    def _active_rank(self, slot_id: int) -> int:
        return self._effective_lora_rank_values[slot_id]

    def _slot_scaling(self, slot_id: int) -> float:
        return self._lora_scaling_values[slot_id]

    def _resolve_token_slots(self, x_flat: torch.Tensor) -> torch.Tensor | None:
        if not self.supports_lora:
            return None
        context = get_lora_context()
        if context.token_to_slot is not None:
            token_to_slot = context.token_to_slot
            if context.no_lora_flag:
                return None
            if token_to_slot.numel() == 0:
                return None
            if token_to_slot.device != x_flat.device:
                raise RuntimeError("LoRA token_to_slot must be prepared on the execution device by the model runner")
            if token_to_slot.numel() != x_flat.size(0):
                raise RuntimeError(
                    "LoRA token_to_slot length does not match flattened input rows: "
                    f"token_to_slot={token_to_slot.numel()} rows={x_flat.size(0)}"
                )
            return token_to_slot.to(dtype=torch.int64)
        return None

    def _get_grouped_token_indices(self, token_to_slot: torch.Tensor, slot_id: int, context) -> torch.Tensor:
        if (
            context.active_slot_ids is not None
            and context.slot_start_offsets is not None
            and context.token_indices_sorted_by_slot is not None
            and not context.active_slot_ids.is_cuda
        ):
            active_slot_ids = context.active_slot_ids.to(dtype=torch.int64)
            matches = torch.nonzero(active_slot_ids == slot_id, as_tuple=False).flatten()
            if matches.numel() > 0:
                group_idx = int(matches[0].item())
                start = int(context.slot_start_offsets[group_idx].item())
                end = int(context.slot_start_offsets[group_idx + 1].item())
                return context.token_indices_sorted_by_slot[start:end].to(
                    device=token_to_slot.device, dtype=torch.int64
                )
        if _is_cuda_graph_capture():
            return torch.nonzero(token_to_slot == slot_id, as_tuple=False).flatten()
        return torch.nonzero(token_to_slot == slot_id, as_tuple=False).flatten()

    def _get_active_slot_ids(self, token_to_slot: torch.Tensor, context) -> list[int]:
        if context.active_slot_ids is not None and not context.active_slot_ids.is_cuda:
            return context.active_slot_ids.to(dtype=torch.int64).tolist()
        if _is_cuda_graph_capture():
            return list(range(self.max_loras))
        if context.active_slot_ids is not None:
            return context.active_slot_ids.to(device=token_to_slot.device, dtype=torch.int64).tolist()
        return torch.unique(token_to_slot[token_to_slot >= 0], sorted=True).tolist()

    def _validate_effective_rank(self, effective_rank: int) -> None:
        if effective_rank < 0 or effective_rank > self.max_lora_rank:
            raise ValueError(f"effective_rank={effective_rank} exceeds max_lora_rank={self.max_lora_rank}")

    def _validate_common_slot_payload(self, effective_rank: int, scaling: float) -> None:
        if not self.supports_lora:
            raise ValueError(f"{self.__class__.__name__} does not enable LoRA")
        if effective_rank <= 0:
            raise ValueError(f"effective_rank must be > 0, got {effective_rank}")
        if not math.isfinite(scaling):
            raise ValueError(f"LoRA scaling must be finite, got {scaling}")
        self._validate_effective_rank(effective_rank)

    def _runtime_metadata(self) -> LoRAMetadata | None:
        context = get_lora_context()
        if context.token_to_slot is None:
            return None
        return LoRAMetadata(
            token_to_slot=context.token_to_slot,
            token_indices_sorted_by_slot=context.token_indices_sorted_by_slot,
            active_slot_ids=context.active_slot_ids,
            num_tokens_per_slot=context.num_tokens_per_slot,
            slot_start_offsets=context.slot_start_offsets,
            no_lora_flag=context.no_lora_flag,
            scratch_buffer=context.scratch_buffer,
            no_lora_flag_cpu=context.no_lora_flag_cpu,
            num_active_loras_cpu=context.num_active_loras_cpu,
        )

    @property
    def lora_enabled(self) -> bool:
        return self.supports_lora

    def set_slot_lora(
        self,
        slot_id: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor | list[torch.Tensor],
        effective_rank: int,
        scaling: float,
    ) -> None:
        raise NotImplementedError

    def validate_slot_lora_payload(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor | list[torch.Tensor],
        effective_rank: int,
        scaling: float,
    ) -> None:
        raise NotImplementedError

    def reset_lora_parameters(self):
        raise NotImplementedError

    def _clear_slot_metadata(self, slot_id: int) -> None:
        self.effective_lora_rank[slot_id] = 0
        self.lora_scaling[slot_id] = 0
        self.lora_base_scaling[slot_id] = 0
        self._effective_lora_rank_values[slot_id] = 0
        self._lora_scaling_values[slot_id] = 0.0
        self._lora_base_scaling_values[slot_id] = 0.0

    def clear_slot_lora(self, slot_id: int) -> None:
        raise NotImplementedError


class LoRAQKVParallelLinear(_LoRALayerBase):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = False,
        lora_targets: Optional[list[str]] = None,
        max_loras: int = 1,
        max_lora_rank: int | None = None,
    ):
        resolved_max_lora_rank = max_lora_rank or 0
        resolved_lora_targets = lora_targets or ["q", "k", "v"]
        supports_lora = resolved_max_lora_rank > 0 and len(resolved_lora_targets) > 0
        super().__init__(max_loras=max_loras, max_lora_rank=resolved_max_lora_rank, supports_lora=supports_lora)
        self.tp_size = _get_world_size()
        self.tp_rank = _get_rank()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = divide(total_num_heads, self.tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, self.tp_size)
        self.q_size = self.num_heads * head_size
        self.kv_size = self.num_kv_heads * head_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size

        self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        set_weight_loader(self.weight, self._base_weight_loader)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            set_weight_loader(self.bias, self._base_weight_loader)
        else:
            self.register_parameter("bias", None)

        self.lora_targets = resolved_lora_targets
        self.target_to_index = {target: idx for idx, target in enumerate(self.lora_targets)}
        if self.supports_lora:
            self.lora_A = nn.Parameter(torch.zeros(max_loras, len(self.lora_targets), self.max_lora_rank, hidden_size))
            if "q" in self.lora_targets:
                self.lora_B_q = nn.Parameter(torch.zeros(max_loras, self.q_size, self.max_lora_rank))
                set_weight_loader(self.lora_B_q, self._make_lora_b_weight_loader("q"))
            if "k" in self.lora_targets:
                self.lora_B_k = nn.Parameter(torch.zeros(max_loras, self.kv_size, self.max_lora_rank))
                set_weight_loader(self.lora_B_k, self._make_lora_b_weight_loader("k"))
            if "v" in self.lora_targets:
                self.lora_B_v = nn.Parameter(torch.zeros(max_loras, self.kv_size, self.max_lora_rank))
                set_weight_loader(self.lora_B_v, self._make_lora_b_weight_loader("v"))

    def _base_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: ShardId | None = None,
    ):
        if loaded_shard_id is None:
            param.data.copy_(loaded_weight)
            return
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param.data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight.chunk(self.tp_size, 0)[self.tp_rank])

    def _make_lora_b_weight_loader(self, target: str):
        def loader(param: nn.Parameter, loaded_weight: torch.Tensor):
            if loaded_weight.size(1) > self.max_lora_rank:
                raise ValueError(f"Loaded LoRA rank {loaded_weight.size(1)} exceeds max_lora_rank={self.max_lora_rank}")
            param.data.zero_()
            param.data[0, :, : loaded_weight.size(1)].copy_(loaded_weight.chunk(self.tp_size, 0)[self.tp_rank])

        return loader

    def load_lora_A(self, loaded_weight: torch.Tensor, target: str):
        if target not in self.target_to_index:
            return
        self._validate_effective_rank(loaded_weight.size(0))
        target_idx = self.target_to_index[target]
        self.lora_A.data[0, target_idx].zero_()
        self.lora_A.data[0, target_idx, : loaded_weight.size(0)].copy_(loaded_weight)
        self.effective_lora_rank[0] = loaded_weight.size(0)
        self._effective_lora_rank_values[0] = loaded_weight.size(0)

    def _apply_target(
        self, x_flat: torch.Tensor, output: torch.Tensor, token_to_slot: torch.Tensor, target: str
    ) -> torch.Tensor:
        backend = get_backend()
        metadata = self._runtime_metadata()
        target_idx = self.target_to_index[target]
        if target == "q":
            lora_b = self.lora_B_q
        elif target == "k":
            lora_b = self.lora_B_k
        else:
            lora_b = self.lora_B_v
        return backend.add_lora(
            output,
            x_flat,
            self.lora_A[:, target_idx],
            lora_b,
            indices=token_to_slot,
            metadata=metadata,
            scaling=1.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = F.linear(x, self.weight, self.bias)
        token_to_slot = self._resolve_token_slots(_flatten_tokens(x)[0])
        if token_to_slot is None:
            return qkv
        x_flat, original_shape = _flatten_tokens(x)
        out_flat, _ = _flatten_tokens(qkv)
        q, k, v = out_flat.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if "q" in self.lora_targets:
            q = self._apply_target(x_flat, q, token_to_slot, "q")
        if "k" in self.lora_targets:
            k = self._apply_target(x_flat, k, token_to_slot, "k")
        if "v" in self.lora_targets:
            v = self._apply_target(x_flat, v, token_to_slot, "v")
        out_flat = torch.cat([q, k, v], dim=-1)
        return _restore_tokens(out_flat, original_shape)

    def set_slot_lora(
        self,
        slot_id: int,
        lora_a: torch.Tensor,
        lora_b: list[torch.Tensor],
        effective_rank: int,
        scaling: float,
    ) -> None:
        self._validate_effective_rank(effective_rank)
        self.lora_A.data[slot_id].zero_()
        for target_idx, target_a in enumerate(lora_a):
            self.lora_A.data[slot_id, target_idx, :effective_rank].copy_(target_a[:effective_rank])
        for target, target_b in zip(self.lora_targets, lora_b):
            getattr(self, f"lora_B_{target}").data[slot_id].zero_()
            getattr(self, f"lora_B_{target}").data[slot_id, :, :effective_rank].copy_(
                target_b[:, :effective_rank] * scaling
            )
        self.effective_lora_rank[slot_id] = effective_rank
        self.lora_scaling[slot_id] = scaling
        self.lora_base_scaling[slot_id] = scaling
        self._effective_lora_rank_values[slot_id] = effective_rank
        self._lora_scaling_values[slot_id] = scaling
        self._lora_base_scaling_values[slot_id] = scaling

    def validate_slot_lora_payload(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor | list[torch.Tensor],
        effective_rank: int,
        scaling: float,
    ) -> None:
        self._validate_common_slot_payload(effective_rank, scaling)
        if not isinstance(lora_b, list):
            raise ValueError("LoRAQKVParallelLinear expects lora_b to be a list ordered as layer lora_targets")
        if lora_a.ndim != 3:
            raise ValueError(f"LoRAQKVParallelLinear expects 3D lora_a, got shape={tuple(lora_a.shape)}")
        if lora_a.size(0) != len(self.lora_targets):
            raise ValueError(f"Expected {len(self.lora_targets)} LoRA A targets, got {lora_a.size(0)}")
        if lora_a.size(1) < effective_rank:
            raise ValueError(f"LoRA A rank dim {lora_a.size(1)} is smaller than effective_rank={effective_rank}")
        if lora_a.size(2) != self.hidden_size:
            raise ValueError(f"Expected lora_a input dim {self.hidden_size}, got {lora_a.size(2)}")
        if len(lora_b) != len(self.lora_targets):
            raise ValueError(f"Expected {len(self.lora_targets)} LoRA B tensors ordered as {self.lora_targets}")
        expected_outputs = {"q": self.q_size, "k": self.kv_size, "v": self.kv_size}
        for target, target_b in zip(self.lora_targets, lora_b):
            if target_b.ndim != 2:
                raise ValueError(f"Target '{target}' expects 2D lora_b, got shape={tuple(target_b.shape)}")
            if target_b.size(0) != expected_outputs[target]:
                raise ValueError(
                    f"Target '{target}' expects output dim {expected_outputs[target]}, got {target_b.size(0)}"
                )
            if target_b.size(1) < effective_rank:
                raise ValueError(
                    f"Target '{target}' rank dim {target_b.size(1)} is smaller than effective_rank={effective_rank}"
                )

    def reset_lora_parameters(self):
        if not self.supports_lora:
            return
        self.lora_A.data.zero_()
        for target in self.lora_targets:
            getattr(self, f"lora_B_{target}").data.zero_()

    def clear_slot_lora(self, slot_id: int) -> None:
        if not self.supports_lora:
            return
        self.lora_A.data[slot_id].zero_()
        for target in self.lora_targets:
            getattr(self, f"lora_B_{target}").data[slot_id].zero_()
        self._clear_slot_metadata(slot_id)


class LoRAMergedColumnParallelLinear(_LoRALayerBase):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        lora_targets: Optional[list[int]] = None,
        max_loras: int = 1,
        max_lora_rank: int | None = None,
    ):
        resolved_max_lora_rank = max_lora_rank or 0
        resolved_lora_targets = lora_targets if lora_targets is not None else list(range(len(output_sizes)))
        supports_lora = resolved_max_lora_rank > 0 and len(resolved_lora_targets) > 0
        super().__init__(max_loras=max_loras, max_lora_rank=resolved_max_lora_rank, supports_lora=supports_lora)
        self.tp_size = _get_world_size()
        self.tp_rank = _get_rank()
        self.output_sizes = output_sizes
        self.input_size = input_size
        total_output = sum(output_sizes)
        self.shard_output_sizes = [s // self.tp_size for s in output_sizes]
        shard_total_output = total_output // self.tp_size

        self.weight = nn.Parameter(torch.empty(shard_total_output, input_size))
        set_weight_loader(self.weight, self._base_weight_loader)
        if bias:
            self.bias = nn.Parameter(torch.empty(shard_total_output))
            set_weight_loader(self.bias, self._base_weight_loader)
        else:
            self.register_parameter("bias", None)

        self.lora_targets = resolved_lora_targets
        self.target_to_index = {target: idx for idx, target in enumerate(self.lora_targets)}
        if self.supports_lora:
            self.lora_A = nn.Parameter(torch.zeros(max_loras, len(self.lora_targets), self.max_lora_rank, input_size))
            for target_idx in self.lora_targets:
                lora_b = nn.Parameter(torch.zeros(max_loras, self.shard_output_sizes[target_idx], self.max_lora_rank))
                set_weight_loader(lora_b, self._make_lora_b_weight_loader(target_idx))
                setattr(self, f"lora_B_{target_idx}", lora_b)

    def _base_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: ShardId | None = None,
    ):
        if loaded_shard_id is None:
            param.data.copy_(loaded_weight)
            return
        assert isinstance(loaded_shard_id, int)
        shard_offset = sum(self.shard_output_sizes[:loaded_shard_id])
        shard_size = self.shard_output_sizes[loaded_shard_id]
        param.data.narrow(0, shard_offset, shard_size).copy_(loaded_weight.chunk(self.tp_size, 0)[self.tp_rank])

    def _make_lora_b_weight_loader(self, target_idx: int):
        def loader(param: nn.Parameter, loaded_weight: torch.Tensor):
            if loaded_weight.size(1) > self.max_lora_rank:
                raise ValueError(f"Loaded LoRA rank {loaded_weight.size(1)} exceeds max_lora_rank={self.max_lora_rank}")
            param.data.zero_()
            param.data[0, :, : loaded_weight.size(1)].copy_(loaded_weight.chunk(self.tp_size, 0)[self.tp_rank])

        return loader

    def load_lora_A(self, loaded_weight: torch.Tensor, target_idx: int):
        if target_idx not in self.target_to_index:
            return
        self._validate_effective_rank(loaded_weight.size(0))
        fused_idx = self.target_to_index[target_idx]
        self.lora_A.data[0, fused_idx].zero_()
        self.lora_A.data[0, fused_idx, : loaded_weight.size(0)].copy_(loaded_weight)
        self.effective_lora_rank[0] = loaded_weight.size(0)
        self._effective_lora_rank_values[0] = loaded_weight.size(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        token_to_slot = self._resolve_token_slots(_flatten_tokens(x)[0])
        if token_to_slot is None:
            return result
        x_flat, original_shape = _flatten_tokens(x)
        out_flat, _ = _flatten_tokens(result)
        backend = get_backend()
        metadata = self._runtime_metadata()
        splits = list(out_flat.split(self.shard_output_sizes, dim=-1))
        for target_idx in self.lora_targets:
            fused_idx = self.target_to_index[target_idx]
            splits[target_idx] = backend.add_lora(
                splits[target_idx],
                x_flat,
                self.lora_A[:, fused_idx],
                getattr(self, f"lora_B_{target_idx}"),
                indices=token_to_slot,
                metadata=metadata,
                scaling=1.0,
            )
        out_flat = torch.cat(splits, dim=-1)
        return _restore_tokens(out_flat, original_shape)

    def set_slot_lora(
        self,
        slot_id: int,
        lora_a: torch.Tensor,
        lora_b: list[torch.Tensor],
        effective_rank: int,
        scaling: float,
    ) -> None:
        self._validate_effective_rank(effective_rank)
        self.lora_A.data[slot_id].zero_()
        for fused_idx, target_a in enumerate(lora_a):
            self.lora_A.data[slot_id, fused_idx, :effective_rank].copy_(target_a[:effective_rank])
        for target_idx, target_b in zip(self.lora_targets, lora_b):
            getattr(self, f"lora_B_{target_idx}").data[slot_id].zero_()
            getattr(self, f"lora_B_{target_idx}").data[slot_id, :, :effective_rank].copy_(
                target_b[:, :effective_rank] * scaling
            )
        self.effective_lora_rank[slot_id] = effective_rank
        self.lora_scaling[slot_id] = scaling
        self.lora_base_scaling[slot_id] = scaling
        self._effective_lora_rank_values[slot_id] = effective_rank
        self._lora_scaling_values[slot_id] = scaling
        self._lora_base_scaling_values[slot_id] = scaling

    def reset_lora_parameters(self):
        if not self.supports_lora:
            return
        self.lora_A.data.zero_()
        for target_idx in self.lora_targets:
            getattr(self, f"lora_B_{target_idx}").data.zero_()

    def clear_slot_lora(self, slot_id: int) -> None:
        if not self.supports_lora:
            return
        self.lora_A.data[slot_id].zero_()
        for target_idx in self.lora_targets:
            getattr(self, f"lora_B_{target_idx}").data[slot_id].zero_()
        self._clear_slot_metadata(slot_id)

    def validate_slot_lora_payload(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor | list[torch.Tensor],
        effective_rank: int,
        scaling: float,
    ) -> None:
        self._validate_common_slot_payload(effective_rank, scaling)
        if not isinstance(lora_b, list):
            raise ValueError("LoRAMergedColumnParallelLinear expects lora_b to be a list ordered as layer lora_targets")
        if lora_a.ndim != 3:
            raise ValueError(f"LoRAMergedColumnParallelLinear expects 3D lora_a, got shape={tuple(lora_a.shape)}")
        if lora_a.size(0) != len(self.lora_targets):
            raise ValueError(f"Expected {len(self.lora_targets)} LoRA A targets, got {lora_a.size(0)}")
        if lora_a.size(1) < effective_rank:
            raise ValueError(f"LoRA A rank dim {lora_a.size(1)} is smaller than effective_rank={effective_rank}")
        if lora_a.size(2) != self.input_size:
            raise ValueError(f"Expected lora_a input dim {self.input_size}, got {lora_a.size(2)}")
        if len(lora_b) != len(self.lora_targets):
            raise ValueError(f"Expected {len(self.lora_targets)} LoRA B tensors ordered as {self.lora_targets}")
        for target_idx, target_b in zip(self.lora_targets, lora_b):
            if target_b.ndim != 2:
                raise ValueError(f"Target {target_idx} expects 2D lora_b, got shape={tuple(target_b.shape)}")
            expected_output = self.shard_output_sizes[target_idx]
            if target_b.size(0) != expected_output:
                raise ValueError(f"Target {target_idx} expects output dim {expected_output}, got {target_b.size(0)}")
            if target_b.size(1) < effective_rank:
                raise ValueError(
                    f"Target {target_idx} rank dim {target_b.size(1)} is smaller than effective_rank={effective_rank}"
                )


class LoRARowParallelLinear(_LoRALayerBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        max_loras: int = 1,
        max_lora_rank: int | None = None,
    ):
        resolved_max_lora_rank = max_lora_rank or 0
        supports_lora = resolved_max_lora_rank > 0
        super().__init__(max_loras=max_loras, max_lora_rank=resolved_max_lora_rank, supports_lora=supports_lora)
        self.tp_size = _get_world_size()
        self.tp_rank = _get_rank()
        self.input_size = input_size
        self.output_size = output_size
        self.shard_input_size = divide(input_size, self.tp_size)

        self.weight = nn.Parameter(torch.empty(output_size, self.shard_input_size))
        set_weight_loader(self.weight, self._base_weight_loader)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            set_weight_loader(self.bias, self._base_weight_loader)
        else:
            self.register_parameter("bias", None)

        if self.supports_lora:
            self.lora_A = nn.Parameter(torch.zeros(max_loras, self.max_lora_rank, self.shard_input_size))
            set_weight_loader(self.lora_A, self._lora_a_weight_loader)
            self.lora_B = nn.Parameter(torch.zeros(max_loras, output_size, self.max_lora_rank))

    def _base_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if param.dim() == 2:
            shard_size = self.shard_input_size
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(1, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def _lora_a_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        if loaded_weight.size(0) > self.max_lora_rank:
            raise ValueError(f"Loaded LoRA rank {loaded_weight.size(0)} exceeds max_lora_rank={self.max_lora_rank}")
        shard_size = self.shard_input_size
        start_idx = self.tp_rank * shard_size
        param.data.zero_()
        param.data[0, : loaded_weight.size(0)].copy_(loaded_weight.narrow(1, start_idx, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        token_to_slot = self._resolve_token_slots(_flatten_tokens(x)[0])
        if token_to_slot is not None:
            x_flat, original_shape = _flatten_tokens(x)
            y_flat, _ = _flatten_tokens(y)
            backend = get_backend()
            metadata = self._runtime_metadata()
            y_flat = backend.add_lora(
                y_flat,
                x_flat,
                self.lora_A,
                self.lora_B,
                indices=token_to_slot,
                metadata=metadata,
                scaling=1.0,
            )
            y = _restore_tokens(y_flat, original_shape)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y

    def set_slot_lora(
        self,
        slot_id: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        effective_rank: int,
        scaling: float,
    ) -> None:
        self._validate_effective_rank(effective_rank)
        self.lora_A.data[slot_id].zero_()
        self.lora_B.data[slot_id].zero_()
        self.lora_A.data[slot_id, :effective_rank].copy_(lora_a[:effective_rank])
        self.lora_B.data[slot_id, :, :effective_rank].copy_(lora_b[:, :effective_rank] * scaling)
        self.effective_lora_rank[slot_id] = effective_rank
        self.lora_scaling[slot_id] = scaling
        self.lora_base_scaling[slot_id] = scaling
        self._effective_lora_rank_values[slot_id] = effective_rank
        self._lora_scaling_values[slot_id] = scaling
        self._lora_base_scaling_values[slot_id] = scaling

    def reset_lora_parameters(self):
        if self.supports_lora:
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()

    def clear_slot_lora(self, slot_id: int) -> None:
        if not self.supports_lora:
            return
        self.lora_A.data[slot_id].zero_()
        self.lora_B.data[slot_id].zero_()
        self._clear_slot_metadata(slot_id)

    def validate_slot_lora_payload(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor | list[torch.Tensor],
        effective_rank: int,
        scaling: float,
    ) -> None:
        self._validate_common_slot_payload(effective_rank, scaling)
        if isinstance(lora_b, list):
            raise ValueError("LoRARowParallelLinear expects tensor lora_b")
        if lora_a.ndim != 2:
            raise ValueError(f"LoRARowParallelLinear expects 2D lora_a, got shape={tuple(lora_a.shape)}")
        if lora_b.ndim != 2:
            raise ValueError(f"LoRARowParallelLinear expects 2D lora_b, got shape={tuple(lora_b.shape)}")
        if lora_a.size(0) < effective_rank:
            raise ValueError(f"LoRA A rank dim {lora_a.size(0)} is smaller than effective_rank={effective_rank}")
        if lora_a.size(1) != self.shard_input_size:
            raise ValueError(f"Expected lora_a input dim {self.shard_input_size}, got {lora_a.size(1)}")
        if lora_b.size(0) != self.output_size:
            raise ValueError(f"Expected lora_b output dim {self.output_size}, got {lora_b.size(0)}")
        if lora_b.size(1) < effective_rank:
            raise ValueError(f"LoRA B rank dim {lora_b.size(1)} is smaller than effective_rank={effective_rank}")


class LoRALinear(_LoRALayerBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        max_loras: int = 1,
        max_lora_rank: int | None = None,
    ):
        resolved_max_lora_rank = max_lora_rank or 0
        supports_lora = resolved_max_lora_rank > 0
        super().__init__(max_loras=max_loras, max_lora_rank=resolved_max_lora_rank, supports_lora=supports_lora)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        if self.supports_lora:
            self.lora_A = nn.Parameter(torch.zeros(max_loras, self.max_lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(max_loras, out_features, self.max_lora_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        token_to_slot = self._resolve_token_slots(_flatten_tokens(x)[0])
        if token_to_slot is None:
            return y
        x_flat, original_shape = _flatten_tokens(x)
        y_flat, _ = _flatten_tokens(y)
        backend = get_backend()
        metadata = self._runtime_metadata()
        y_flat = backend.add_lora(
            y_flat,
            x_flat,
            self.lora_A,
            self.lora_B,
            indices=token_to_slot,
            metadata=metadata,
            scaling=1.0,
        )
        return _restore_tokens(y_flat, original_shape)

    def set_slot_lora(
        self,
        slot_id: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        effective_rank: int,
        scaling: float,
    ) -> None:
        self._validate_effective_rank(effective_rank)
        self.lora_A.data[slot_id].zero_()
        self.lora_B.data[slot_id].zero_()
        self.lora_A.data[slot_id, :effective_rank].copy_(lora_a[:effective_rank])
        self.lora_B.data[slot_id, :, :effective_rank].copy_(lora_b[:, :effective_rank] * scaling)
        self.effective_lora_rank[slot_id] = effective_rank
        self.lora_scaling[slot_id] = scaling
        self.lora_base_scaling[slot_id] = scaling
        self._effective_lora_rank_values[slot_id] = effective_rank
        self._lora_scaling_values[slot_id] = scaling
        self._lora_base_scaling_values[slot_id] = scaling

    def reset_lora_parameters(self):
        if self.supports_lora:
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()

    def clear_slot_lora(self, slot_id: int) -> None:
        if not self.supports_lora:
            return
        self.lora_A.data[slot_id].zero_()
        self.lora_B.data[slot_id].zero_()
        self._clear_slot_metadata(slot_id)

    def validate_slot_lora_payload(
        self,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor | list[torch.Tensor],
        effective_rank: int,
        scaling: float,
    ) -> None:
        self._validate_common_slot_payload(effective_rank, scaling)
        if isinstance(lora_b, list):
            raise ValueError("LoRALinear expects tensor lora_b")
        if lora_a.ndim != 2:
            raise ValueError(f"LoRALinear expects 2D lora_a, got shape={tuple(lora_a.shape)}")
        if lora_b.ndim != 2:
            raise ValueError(f"LoRALinear expects 2D lora_b, got shape={tuple(lora_b.shape)}")
        if lora_a.size(0) < effective_rank:
            raise ValueError(f"LoRA A rank dim {lora_a.size(0)} is smaller than effective_rank={effective_rank}")
        if lora_a.size(1) != self.in_features:
            raise ValueError(f"Expected lora_a input dim {self.in_features}, got {lora_a.size(1)}")
        if lora_b.size(0) != self.out_features:
            raise ValueError(f"Expected lora_b output dim {self.out_features}, got {lora_b.size(0)}")
        if lora_b.size(1) < effective_rank:
            raise ValueError(f"LoRA B rank dim {lora_b.size(1)} is smaller than effective_rank={effective_rank}")


def iter_lora_modules(model: nn.Module):
    for module in model.modules():
        if isinstance(
            module, (LoRAQKVParallelLinear, LoRAMergedColumnParallelLinear, LoRARowParallelLinear, LoRALinear)
        ):
            if module.lora_enabled:
                yield module


def get_lora_state_dict(model: nn.Module) -> dict:
    return {name: param.data.clone() for name, param in model.named_parameters() if "lora_" in name}
