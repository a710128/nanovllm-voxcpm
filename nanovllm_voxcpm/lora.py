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
        y_packed: torch.Tensor | None = None,
    ) -> list[torch.Tensor]: ...

    def shrink(self, x: torch.Tensor, lora_a: torch.Tensor) -> torch.Tensor: ...

    def expand(self, hidden: torch.Tensor, lora_b: torch.Tensor, *, scaling: float) -> torch.Tensor: ...


_BACKEND_OVERRIDE: LoRABackend | None = None
_PROBED_BACKEND: LoRABackend | None = None


@dataclass(frozen=True)
class _AddLoraPlan:
    """Precomputed shrink/expand slice grouping for a fixed list of LoRA weights.

    Layers always pass the same `nn.Parameter` instances, so the grouping by
    `(rank, in_dim)` for shrink and `(rank, hidden_out)` for expand is a pure
    function of those tensors' shapes. We compute it once and reuse, avoiding
    several Python dict allocations per LoRA layer per step.

    Attributes:
        active: indices into the caller's slice lists that have non-empty
            output (TP shards owning no rows are excluded).
        shrink_groups: list of buckets; each bucket is the list of indices
            into ``active`` that share the same A weight shape.
        expand_groups: same idea for B weights.
        shrink_ranks: per-bucket rank, parallel to ``shrink_groups``.
        all_active_same_expand_group: True iff every slice is active and they
            all share one ``(rank, hidden_out)`` bucket — enables the
            zero-copy "expand straight into y_packed" fast path.
    """

    active: tuple[int, ...]
    shrink_groups: tuple[tuple[int, ...], ...]
    shrink_ranks: tuple[int, ...]
    expand_groups: tuple[tuple[int, ...], ...]
    all_active_same_expand_group: bool


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
        y_packed: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        raise RuntimeError(self.reason)


class _VendoredTritonPunicaBackend:
    def __init__(self) -> None:
        # Per-call grouping of shrink/expand slices depends only on weight
        # shapes, which are fixed once modules are constructed. Cache by
        # ``(id(lora_a_slices[0]), id(lora_a_slices[1]), ...)`` so each LoRA
        # layer pays the Python grouping cost exactly once.
        self._add_lora_plan_cache: dict[tuple, _AddLoraPlan] = {}

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

        # Prime the lora_b pointer cache with the SAME offset/grouping
        # combinations that ``add_lora`` will use at runtime, otherwise the
        # first decode step (which runs inside CUDA graph capture) hits a
        # cache miss and calls ``torch.tensor(...)`` mid-capture — illegal.
        #
        # Two patterns matter:
        #   (a) all-active-same-expand-bucket fast path: every slice grouped
        #       together with offset 0;
        #   (b) per-bucket fallback with each bucket starting at its own
        #       column offset (QKV is the canonical case: q at 0, k/v at
        #       q_size).
        # We prime both, plus the grouped offset_start variant the caller
        # explicitly requested.
        col_offsets = [0]
        for lora_b in lora_b_slices:
            col_offsets.append(col_offsets[-1] + lora_b.size(1))

        # Pattern (a): all slices, offset 0.
        utils_module._get_lora_b_ptr(lora_b_slices, offset_start, lora_b_slices[0].device)

        # Pattern (b): grouped by (rank, hidden_out), each group at the column
        # offset of its first slice in the natural packed layout.
        expand_groups: dict[tuple[int, int], list[int]] = {}
        for slice_idx, lora_b in enumerate(lora_b_slices):
            expand_groups.setdefault((lora_b.size(-1), lora_b.size(1)), []).append(slice_idx)
        for group_indices in expand_groups.values():
            group_lora_b = [lora_b_slices[i] for i in group_indices]
            group_offset = col_offsets[group_indices[0]]
            utils_module._get_lora_b_ptr(group_lora_b, group_offset, group_lora_b[0].device)

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

    def _get_plan(
        self,
        y_slices: list[torch.Tensor],
        lora_a_slices: list[torch.Tensor],
        lora_b_slices: list[torch.Tensor],
    ) -> _AddLoraPlan:
        # Key by tensor identity — parameters are never reassigned on a module
        # once constructed, so this is stable across steps. We include the
        # empty-vs-nonempty mask of y_slices because TP shards with no output
        # rows get filtered out.
        key = (
            tuple(id(t) for t in lora_a_slices)
            + tuple(id(t) for t in lora_b_slices)
            + tuple(y.numel() > 0 for y in y_slices)
        )
        cached = self._add_lora_plan_cache.get(key)
        if cached is not None:
            return cached

        active = tuple(i for i, y in enumerate(y_slices) if y.numel() > 0)
        shrink_by_shape: dict[tuple[int, int], list[int]] = {}
        for local_i, slice_idx in enumerate(active):
            lora_a = lora_a_slices[slice_idx]
            shrink_by_shape.setdefault((lora_a.size(-2), lora_a.size(-1)), []).append(local_i)
        shrink_groups = tuple(tuple(g) for g in shrink_by_shape.values())
        shrink_ranks = tuple(lora_a_slices[active[group[0]]].size(-2) for group in shrink_groups)

        expand_by_shape: dict[tuple[int, int], list[int]] = {}
        for local_i, slice_idx in enumerate(active):
            lora_b = lora_b_slices[slice_idx]
            expand_by_shape.setdefault((lora_b.size(-1), lora_b.size(1)), []).append(local_i)
        expand_groups = tuple(tuple(g) for g in expand_by_shape.values())

        all_active_same_expand_group = len(expand_groups) == 1 and len(active) == len(y_slices)

        plan = _AddLoraPlan(
            active=active,
            shrink_groups=shrink_groups,
            shrink_ranks=shrink_ranks,
            expand_groups=expand_groups,
            all_active_same_expand_group=all_active_same_expand_group,
        )
        self._add_lora_plan_cache[key] = plan
        return plan

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
        y_packed: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Apply grouped LoRA shrink+expand and accumulate into y.

        Fast path (``y_packed`` provided): ``y_slices`` must be views of
        ``y_packed`` laid out as contiguous columns in that order (e.g. the
        output of ``y_packed.split(sizes, dim=-1)``). Expand writes directly
        into ``y_packed``; the returned ``y_slices`` are the same views.

        Slow path (``y_packed is None``, legacy): the caller's slices are
        treated as an independent packed buffer; we allocate a contiguous
        staging tensor, expand into it, and return slice-views of that tensor.
        """
        if not y_slices:
            return []
        if len(y_slices) != len(lora_a_slices) or len(y_slices) != len(lora_b_slices):
            raise ValueError("add_lora expects aligned y/lora_a/lora_b slice lists")
        if metadata is None:
            raise RuntimeError("LoRA metadata must be prepared by the model runner before backend execution")

        if not all(tensor.is_contiguous() for tensor in lora_a_slices):
            raise ValueError("add_lora expects contiguous lora_a slices")
        if not all(tensor.is_contiguous() for tensor in lora_b_slices):
            raise ValueError("add_lora expects contiguous lora_b slices")

        # Resolve packed output buffer.
        if y_packed is None:
            if len(y_slices) == 1:
                # Single-slice layers (row-parallel, plain linear) hand us the
                # flat output directly; write in place.
                y_packed = y_slices[0]
            else:
                # Legacy fallback: allocate a contiguous packed buffer.
                split_sizes = [y_slice.size(-1) for y_slice in y_slices]
                y_packed = torch.cat(y_slices, dim=-1)
                y_slices = list(y_packed.split(split_sizes, dim=-1))
        else:
            if y_packed.dim() != 2:
                raise ValueError("y_packed must be 2D [num_tokens, packed_hidden]")

        meta = metadata.as_kernel_metadata(x.size(0))
        lora_shrink, lora_expand = self._ops()

        plan = self._get_plan(y_slices, lora_a_slices, lora_b_slices)
        active = plan.active
        if not active:
            return list(y_slices)

        # Grouped shrink: one kernel call per (rank, in_dim) bucket.
        tmp_slices: list[torch.Tensor | None] = [None] * len(active)
        for group_local, rank in zip(plan.shrink_groups, plan.shrink_ranks):
            tmp = torch.empty((len(group_local), x.size(0), rank), dtype=x.dtype, device=x.device)
            lora_shrink(x, [lora_a_slices[active[i]] for i in group_local], tmp, *meta, scaling)
            for t_idx, local_i in enumerate(group_local):
                tmp_slices[local_i] = tmp[t_idx]

        # Fast path: every slice active, all in one (rank, hidden_out) bucket —
        # single expand call, accumulating straight into y_packed.
        if plan.all_active_same_expand_group:
            group_local = plan.expand_groups[0]
            group_tmp = (
                torch.stack([tmp_slices[i] for i in group_local])
                if len(group_local) > 1
                else tmp_slices[group_local[0]].unsqueeze(0)
            )
            lora_expand(
                group_tmp,
                [lora_b_slices[active[i]] for i in group_local],
                y_packed,
                *meta,
                offset_start=0,
                add_inputs=True,
            )
            return list(y_slices)

        # Fallback: per-group expand. Contiguous-span groups write directly
        # into y_packed with an offset; non-contiguous groups stage through a
        # temporary buffer (rare).
        outputs = list(y_slices)
        col_offsets = [0]
        for y in y_slices:
            col_offsets.append(col_offsets[-1] + y.size(-1))

        for group_local in plan.expand_groups:
            group_tmp_slices = [tmp_slices[i] for i in group_local]
            if any(t is None for t in group_tmp_slices):
                raise RuntimeError("LoRA shrink did not produce all intermediate slices")
            group_tmp = torch.stack(group_tmp_slices) if len(group_tmp_slices) > 1 else group_tmp_slices[0].unsqueeze(0)
            group_slice_ids = [active[i] for i in group_local]

            is_contiguous_span = all(
                group_slice_ids[k] + 1 == group_slice_ids[k + 1] for k in range(len(group_slice_ids) - 1)
            )
            if is_contiguous_span:
                group_offset = col_offsets[group_slice_ids[0]]
                lora_expand(
                    group_tmp,
                    [lora_b_slices[idx] for idx in group_slice_ids],
                    y_packed,
                    *meta,
                    offset_start=group_offset,
                    add_inputs=True,
                )
                continue

            stage = torch.cat([y_slices[idx] for idx in group_slice_ids], dim=-1)
            lora_expand(
                group_tmp,
                [lora_b_slices[idx] for idx in group_slice_ids],
                stage,
                *meta,
                offset_start=0,
                add_inputs=True,
            )
            split_sizes = [y_slices[idx].size(-1) for idx in group_slice_ids]
            for idx, output in zip(group_slice_ids, stage.split(split_sizes, dim=-1)):
                outputs[idx] = output
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
