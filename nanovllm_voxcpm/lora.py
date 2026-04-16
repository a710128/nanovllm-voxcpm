from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

import torch

from nanovllm_voxcpm.lora_ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta


@dataclass(frozen=True)
class LoRAAvailability:
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class LoRAMetadata:
    token_to_slot: torch.Tensor | None
    token_indices_sorted_by_slot: torch.Tensor | None
    active_slot_ids: torch.Tensor | None
    num_tokens_per_slot: torch.Tensor | None
    slot_start_offsets: torch.Tensor | None
    no_lora_flag: bool
    num_active_loras: int = 0

    def as_kernel_metadata(self, token_count: int):
        return (
            self.token_to_slot[:token_count] if self.token_to_slot is not None else None,
            self.token_indices_sorted_by_slot[:token_count] if self.token_indices_sorted_by_slot is not None else None,
            self.num_tokens_per_slot,
            self.slot_start_offsets,
            self.active_slot_ids,
            self.no_lora_flag,
            self.num_active_loras,
        )


class LoRABackend(Protocol):
    def availability(self) -> LoRAAvailability: ...

    def add_lora(
        self,
        y_slices: list[torch.Tensor],
        x: torch.Tensor,
        lora_a_slices: list[torch.Tensor],
        lora_b_slices: list[torch.Tensor],
        *,
        indices: torch.Tensor,
        metadata: LoRAMetadata | None,
        scaling: float,
    ) -> list[torch.Tensor]: ...

    def shrink(self, x: torch.Tensor, lora_a: torch.Tensor) -> torch.Tensor: ...

    def expand(self, hidden: torch.Tensor, lora_b: torch.Tensor, *, scaling: float) -> torch.Tensor: ...


_BACKEND_OVERRIDE: LoRABackend | None = None
_PROBED_BACKEND: LoRABackend | None = None


class _UnavailableBackend:
    def __init__(self, reason: str):
        self.reason = reason

    def availability(self) -> LoRAAvailability:
        return LoRAAvailability(available=False, reason=self.reason)

    def shrink(self, x: torch.Tensor, lora_a: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(self.reason)

    def expand(self, hidden: torch.Tensor, lora_b: torch.Tensor, *, scaling: float) -> torch.Tensor:
        raise RuntimeError(self.reason)

    def add_lora(
        self,
        y_slices: list[torch.Tensor],
        x: torch.Tensor,
        lora_a_slices: list[torch.Tensor],
        lora_b_slices: list[torch.Tensor],
        *,
        indices: torch.Tensor,
        metadata: LoRAMetadata | None,
        scaling: float,
    ) -> list[torch.Tensor]:
        raise RuntimeError(self.reason)


class _VendoredTritonPunicaBackend:
    def _ops(self):
        shrink_module = import_module("nanovllm_voxcpm.lora_ops.triton_ops.lora_shrink_op")
        expand_module = import_module("nanovllm_voxcpm.lora_ops.triton_ops.lora_expand_op")
        return shrink_module.lora_shrink, expand_module.lora_expand

    def prime_slice_caches(
        self,
        lora_a_slices: list[torch.Tensor],
        lora_b_slices: list[torch.Tensor],
        *,
        offset_start: int = 0,
    ) -> None:
        if not lora_a_slices or not lora_b_slices:
            return
        utils_module = import_module("nanovllm_voxcpm.lora_ops.triton_ops.utils")
        shrink_groups: dict[tuple[int, int], list[torch.Tensor]] = {}
        for lora_a in lora_a_slices:
            shrink_groups.setdefault((lora_a.size(-2), lora_a.size(-1)), []).append(lora_a)
        for group_lora_a in shrink_groups.values():
            utils_module._get_lora_a_ptr(group_lora_a, group_lora_a[0].device)

        expand_groups: dict[tuple[int, int], list[torch.Tensor]] = {}
        for lora_b in lora_b_slices:
            expand_groups.setdefault((lora_b.size(1), lora_b.size(-1)), []).append(lora_b)
        for group_lora_b in expand_groups.values():
            utils_module._get_lora_b_ptr(group_lora_b, offset_start, group_lora_b[0].device)

    def availability(self) -> LoRAAvailability:
        if not torch.cuda.is_available():
            return LoRAAvailability(available=False, reason="CUDA is unavailable")
        try:
            self._ops()
        except Exception as exc:
            return LoRAAvailability(available=False, reason=f"Vendored Triton LoRA ops unavailable: {exc}")
        return LoRAAvailability(available=True, reason=None)

    def _make_metadata(self, num_tokens: int, device: torch.device, indices: torch.Tensor):
        max_loras = int(indices[indices >= 0].max().item()) + 1 if bool((indices >= 0).any().item()) else 0
        kernel_meta = LoRAKernelMeta.make(max_loras=max(max_loras, 1), max_num_tokens=max(num_tokens, 1), device=device)
        kernel_meta.prepare_tensors(indices.to(device=device, dtype=torch.int32))
        return kernel_meta.meta_args(token_nums=num_tokens, specialize_active_lora=True)

    def add_lora(
        self,
        y_slices: list[torch.Tensor],
        x: torch.Tensor,
        lora_a_slices: list[torch.Tensor],
        lora_b_slices: list[torch.Tensor],
        *,
        indices: torch.Tensor,
        metadata: LoRAMetadata | None,
        scaling: float,
    ) -> list[torch.Tensor]:
        if not y_slices:
            return []
        if len(y_slices) != len(lora_a_slices) or len(y_slices) != len(lora_b_slices):
            raise ValueError("add_lora expects aligned y/lora_a/lora_b slice lists")

        outputs = [y_slice.clone() for y_slice in y_slices]
        active_indices = [slice_idx for slice_idx, y_slice in enumerate(y_slices) if y_slice.numel() > 0]
        if not active_indices:
            return outputs

        active_y_slices = [y_slices[idx] for idx in active_indices]
        active_lora_a_slices = [lora_a_slices[idx] for idx in active_indices]
        active_lora_b_slices = [lora_b_slices[idx] for idx in active_indices]

        if metadata is None:
            raise RuntimeError("LoRA metadata must be prepared by the model runner before backend execution")
        meta = metadata.as_kernel_metadata(x.size(0))
        if not all(tensor.is_contiguous() for tensor in active_lora_a_slices):
            raise ValueError("add_lora expects contiguous lora_a slices")
        if not all(tensor.is_contiguous() for tensor in active_lora_b_slices):
            raise ValueError("add_lora expects contiguous lora_b slices")

        lora_shrink, lora_expand = self._ops()
        tmp_slices: list[torch.Tensor | None] = [None for _ in active_y_slices]
        shrink_groups: dict[tuple[int, int], list[int]] = {}
        for slice_idx, lora_a in enumerate(active_lora_a_slices):
            shrink_groups.setdefault((lora_a.size(-2), lora_a.size(-1)), []).append(slice_idx)

        for group_indices in shrink_groups.values():
            rank = active_lora_a_slices[group_indices[0]].size(-2)
            tmp = torch.empty((len(group_indices), x.size(0), rank), dtype=x.dtype, device=x.device)
            lora_shrink(x, [active_lora_a_slices[idx] for idx in group_indices], tmp, *meta, scaling)
            for local_idx, slice_idx in enumerate(group_indices):
                tmp_slices[slice_idx] = tmp[local_idx]

        expand_groups: dict[tuple[int, int], list[int]] = {}
        for slice_idx, (y_slice, lora_b) in enumerate(zip(active_y_slices, active_lora_b_slices)):
            expand_groups.setdefault((y_slice.size(-1), lora_b.size(-1)), []).append(slice_idx)

        for group_indices in expand_groups.values():
            group_tmp_slices = [tmp_slices[idx] for idx in group_indices]
            if any(tmp_slice is None for tmp_slice in group_tmp_slices):
                raise RuntimeError("LoRA shrink did not produce all intermediate slices")
            group_tmp = torch.stack(group_tmp_slices)
            out = torch.cat([active_y_slices[idx].clone() for idx in group_indices], dim=-1)
            lora_expand(
                group_tmp,
                [active_lora_b_slices[idx] for idx in group_indices],
                out,
                *meta,
                offset_start=0,
                add_inputs=True,
            )
            split_sizes = [active_y_slices[idx].size(-1) for idx in group_indices]
            for slice_idx, output in zip(group_indices, out.split(split_sizes, dim=-1)):
                outputs[active_indices[slice_idx]] = output

        return outputs

    def shrink(self, x: torch.Tensor, lora_a: torch.Tensor) -> torch.Tensor:
        lora_shrink, _ = self._ops()
        rank = lora_a.size(-2)
        tmp = torch.empty((1, x.size(0), rank), dtype=x.dtype, device=x.device)
        meta = self._make_metadata(x.size(0), x.device, torch.zeros(x.size(0), dtype=torch.int32, device=x.device))
        lora_shrink(x, [lora_a.contiguous()], tmp, *meta, 1.0)
        return tmp.squeeze(0)

    def expand(self, hidden: torch.Tensor, lora_b: torch.Tensor, *, scaling: float) -> torch.Tensor:
        _, lora_expand = self._ops()
        inputs = hidden.unsqueeze(0)
        out = torch.zeros(hidden.size(0), lora_b.size(0), dtype=hidden.dtype, device=hidden.device)
        meta = self._make_metadata(
            hidden.size(0), hidden.device, torch.zeros(hidden.size(0), dtype=torch.int32, device=hidden.device)
        )
        lora_expand(inputs, [lora_b.contiguous()], out, *meta, offset_start=0, add_inputs=False)
        return out * scaling


def _probe_vendored_backend() -> LoRABackend:
    return _VendoredTritonPunicaBackend()


def _probe_backend() -> LoRABackend:
    vendored_backend = _probe_vendored_backend()
    vendored_availability = vendored_backend.availability()
    if vendored_availability.available:
        return vendored_backend
    return _UnavailableBackend(vendored_availability.reason or "Vendored Triton LoRA ops are unavailable")


def get_backend() -> LoRABackend:
    global _PROBED_BACKEND
    if _BACKEND_OVERRIDE is not None:
        return _BACKEND_OVERRIDE
    if _PROBED_BACKEND is None:
        _PROBED_BACKEND = _probe_backend()
    return _PROBED_BACKEND


def set_backend_for_testing(backend: LoRABackend | None) -> None:
    global _BACKEND_OVERRIDE
    _BACKEND_OVERRIDE = backend


def get_availability() -> LoRAAvailability:
    return get_backend().availability()


def is_available() -> bool:
    return get_availability().available


def assert_available() -> None:
    availability = get_availability()
    if not availability.available:
        raise RuntimeError(availability.reason or "LoRA runtime is unavailable")
