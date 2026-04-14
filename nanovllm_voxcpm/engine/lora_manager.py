from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch

from nanovllm_voxcpm.lora import assert_available as assert_lora_available


class LoRALifecycleState(str, Enum):
    REGISTERED = "REGISTERED"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    REMOVED = "REMOVED"


class LoRAResidentState(str, Enum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"


@dataclass
class LoRAModulePayload:
    lora_a: torch.Tensor
    lora_b: torch.Tensor | list[torch.Tensor]
    effective_rank: int
    scaling: float


@dataclass
class LoRAModelPayload:
    modules: dict[str, LoRAModulePayload]
    rank: int
    alpha: float


@dataclass
class LoRAControlEntry:
    name: str
    adapter_id: int
    state: LoRALifecycleState
    rank: int
    alpha: float
    scaling: float
    cpu_ref_count: int = 0
    gpu_running_ref_count: int = 0
    slot_id: int | None = None
    last_used_ts: int = 0


@dataclass
class LoRARuntimeEntry(LoRAControlEntry):
    model_payload: LoRAModelPayload | None = None


@dataclass
class LoRASlot:
    slot_id: int
    adapter_id: int | None = None
    resident_state: LoRAResidentState = LoRAResidentState.IDLE
    last_used_ts: int = 0


@dataclass
class LoRABatchPlan:
    adapter_to_slot: dict[int, int]
    token_to_slot: list[int]
    token_indices_sorted_by_slot: list[int]
    active_slot_ids: list[int]
    num_tokens_per_slot: list[int]
    slot_start_offsets: list[int]


class _LoRAStateBase:
    def __init__(self, max_loras: int, max_lora_rank: int | None = None):
        self.max_loras = max(0, max_loras)
        self.max_lora_rank = max_lora_rank
        self._next_adapter_id = 0
        self._clock = 0
        self._name_to_adapter_id: dict[str, int] = {}
        self._entries: dict[int, LoRAControlEntry | LoRARuntimeEntry] = {}
        self._slots = [LoRASlot(slot_id=i) for i in range(self.max_loras)]

    @property
    def enabled(self) -> bool:
        return self.max_loras > 0

    def peek_next_adapter_id(self) -> int:
        return self._next_adapter_id

    def _validate_registration(self, name: str, payload: LoRAModelPayload, adapter_id: int | None) -> int:
        if not self.enabled:
            raise RuntimeError("LoRA runtime is disabled for this engine")
        assert_lora_available()
        if name in self._name_to_adapter_id:
            raise ValueError(f"LoRA '{name}' is already registered")
        if self.max_lora_rank is not None and payload.rank > self.max_lora_rank:
            raise ValueError(f"LoRA rank {payload.rank} exceeds max_lora_rank={self.max_lora_rank}")

        if adapter_id is None:
            adapter_id = self._next_adapter_id
        elif adapter_id in self._entries:
            raise ValueError(f"LoRA adapter id {adapter_id} is already registered")
        self._next_adapter_id = max(self._next_adapter_id, adapter_id + 1)
        return adapter_id

    def resolve_adapter(self, name: str | None) -> int | None:
        if name is None:
            return None
        try:
            adapter_id = self._name_to_adapter_id[name]
        except KeyError as exc:
            raise ValueError(f"LoRA '{name}' is not registered") from exc
        entry = self._entries[adapter_id]
        if entry.state in (LoRALifecycleState.DRAINING, LoRALifecycleState.REMOVED):
            raise ValueError(f"LoRA '{name}' is draining and cannot accept new requests")
        return adapter_id

    def list_loras(self) -> list[LoRAControlEntry | LoRARuntimeEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.adapter_id)

    def get_entry(self, adapter_id: int) -> LoRAControlEntry | LoRARuntimeEntry:
        return self._entries[adapter_id]

    def unregister_lora(self, name: str) -> None:
        adapter_id = self.resolve_known_adapter(name)
        entry = self._entries[adapter_id]
        entry.state = LoRALifecycleState.DRAINING
        if entry.cpu_ref_count == 0:
            self._remove_entry(adapter_id)

    def resolve_known_adapter(self, name: str) -> int:
        try:
            return self._name_to_adapter_id[name]
        except KeyError as exc:
            raise ValueError(f"LoRA '{name}' is not registered") from exc

    def on_sequence_enqueued(self, adapter_id: int | None) -> None:
        if adapter_id is None:
            return
        entry = self._entries[adapter_id]
        entry.cpu_ref_count += 1
        if entry.state == LoRALifecycleState.REGISTERED:
            entry.state = LoRALifecycleState.ACTIVE

    def on_sequence_started(self, adapter_id: int | None) -> None:
        if adapter_id is None:
            return
        entry = self._entries[adapter_id]
        entry.gpu_running_ref_count += 1
        if entry.state == LoRALifecycleState.REGISTERED:
            entry.state = LoRALifecycleState.ACTIVE
        self._refresh_slot_states()

    def on_sequence_preempted(self, adapter_id: int | None) -> None:
        if adapter_id is None:
            return
        entry = self._entries[adapter_id]
        if entry.gpu_running_ref_count <= 0:
            raise RuntimeError(f"LoRA adapter {adapter_id} GPU ref count underflow")
        entry.gpu_running_ref_count -= 1
        self._refresh_slot_states()

    def on_sequence_finished(self, adapter_id: int | None, *, was_running: bool) -> None:
        if adapter_id is None:
            return
        entry = self._entries[adapter_id]
        if was_running:
            if entry.gpu_running_ref_count <= 0:
                raise RuntimeError(f"LoRA adapter {adapter_id} GPU ref count underflow")
            entry.gpu_running_ref_count -= 1
        if entry.cpu_ref_count <= 0:
            raise RuntimeError(f"LoRA adapter {adapter_id} CPU ref count underflow")
        entry.cpu_ref_count -= 1
        self._refresh_slot_states()
        if entry.state == LoRALifecycleState.DRAINING and entry.cpu_ref_count == 0:
            self._remove_entry(adapter_id)
        elif entry.cpu_ref_count == 0 and entry.state != LoRALifecycleState.DRAINING:
            entry.state = LoRALifecycleState.REGISTERED

    def can_schedule(self, running_adapter_ids: set[int], candidate_adapter_id: int | None) -> bool:
        if not self.enabled or candidate_adapter_id is None:
            return True
        required = set(running_adapter_ids)
        required.add(candidate_adapter_id)
        resident = {slot.adapter_id for slot in self._slots if slot.adapter_id is not None}
        missing = required - resident
        reclaimable = sum(1 for slot in self._slots if slot.adapter_id is None)
        reclaimable += sum(
            1
            for slot in self._slots
            if slot.adapter_id is not None
            and slot.adapter_id not in required
            and slot.resident_state == LoRAResidentState.IDLE
        )
        return len(missing) <= reclaimable

    def _remove_entry(self, adapter_id: int) -> None:
        entry = self._entries[adapter_id]
        if entry.slot_id is not None:
            slot = self._slots[entry.slot_id]
            slot.adapter_id = None
            slot.resident_state = LoRAResidentState.IDLE
            entry.slot_id = None
        entry.state = LoRALifecycleState.REMOVED
        self._name_to_adapter_id.pop(entry.name, None)
        self._entries.pop(adapter_id, None)

    def _refresh_slot_states(self) -> None:
        for slot in self._slots:
            if slot.adapter_id is None:
                continue
            entry = self._entries.get(slot.adapter_id)
            if entry is None:
                slot.adapter_id = None
                slot.resident_state = LoRAResidentState.IDLE
                continue
            slot.resident_state = (
                LoRAResidentState.ACTIVE if entry.gpu_running_ref_count > 0 else LoRAResidentState.IDLE
            )

    def _tick(self) -> int:
        self._clock += 1
        return self._clock


class LoRAManager(_LoRAStateBase):
    def register_lora(self, name: str, payload: LoRAModelPayload, *, adapter_id: int | None = None) -> int:
        adapter_id = self._validate_registration(name, payload, adapter_id)
        scaling = payload.alpha / payload.rank if payload.rank > 0 else 0.0
        self._name_to_adapter_id[name] = adapter_id
        self._entries[adapter_id] = LoRAControlEntry(
            name=name,
            adapter_id=adapter_id,
            state=LoRALifecycleState.REGISTERED,
            rank=payload.rank,
            alpha=payload.alpha,
            scaling=scaling,
        )
        return adapter_id


class LoRARuntime(_LoRAStateBase):
    def register_lora(self, name: str, payload: LoRAModelPayload, *, adapter_id: int | None = None) -> int:
        adapter_id = self._validate_registration(name, payload, adapter_id)
        scaling = payload.alpha / payload.rank if payload.rank > 0 else 0.0
        self._name_to_adapter_id[name] = adapter_id
        self._entries[adapter_id] = LoRARuntimeEntry(
            name=name,
            adapter_id=adapter_id,
            state=LoRALifecycleState.REGISTERED,
            rank=payload.rank,
            alpha=payload.alpha,
            scaling=scaling,
            model_payload=payload,
        )
        return adapter_id

    def get_entry(self, adapter_id: int) -> LoRARuntimeEntry:
        return self._entries[adapter_id]  # type: ignore[return-value]

    def build_batch_plan(
        self,
        adapter_ids: list[int | None],
        token_counts: list[int],
        load_lora: Callable[[int, LoRAModelPayload], None],
    ) -> LoRABatchPlan:
        if len(adapter_ids) != len(token_counts):
            raise ValueError("adapter_ids and token_counts must have the same length")
        if not self.enabled:
            return LoRABatchPlan({}, [-1 for count in token_counts for _ in range(count)], [], [], [], [0])

        self._refresh_slot_states()
        distinct_adapter_ids = []
        for adapter_id in adapter_ids:
            if adapter_id is not None and adapter_id not in distinct_adapter_ids:
                distinct_adapter_ids.append(adapter_id)

        adapter_to_slot: dict[int, int] = {}
        for adapter_id in distinct_adapter_ids:
            slot_id = self._ensure_slot(adapter_id, load_lora)
            adapter_to_slot[adapter_id] = slot_id
            entry = self.get_entry(adapter_id)
            entry.last_used_ts = self._tick()
            slot = self._slots[slot_id]
            slot.last_used_ts = entry.last_used_ts
            slot.resident_state = (
                LoRAResidentState.ACTIVE if entry.gpu_running_ref_count > 0 else LoRAResidentState.IDLE
            )

        slot_to_count: dict[int, int] = {}
        token_to_slot: list[int] = []
        for adapter_id, token_count in zip(adapter_ids, token_counts):
            slot_id = adapter_to_slot.get(adapter_id, -1) if adapter_id is not None else -1
            token_to_slot.extend([slot_id] * token_count)
            if slot_id >= 0:
                slot_to_count[slot_id] = slot_to_count.get(slot_id, 0) + token_count

        active_slot_ids = sorted(slot_to_count)
        num_tokens_per_slot = [slot_to_count[slot_id] for slot_id in active_slot_ids]
        token_indices_sorted_by_slot = [
            token_idx
            for slot_id in active_slot_ids
            for token_idx, mapped_slot_id in enumerate(token_to_slot)
            if mapped_slot_id == slot_id
        ]
        slot_start_offsets = [0]
        for count in num_tokens_per_slot:
            slot_start_offsets.append(slot_start_offsets[-1] + count)
        return LoRABatchPlan(
            adapter_to_slot=adapter_to_slot,
            token_to_slot=token_to_slot,
            token_indices_sorted_by_slot=token_indices_sorted_by_slot,
            active_slot_ids=active_slot_ids,
            num_tokens_per_slot=num_tokens_per_slot,
            slot_start_offsets=slot_start_offsets,
        )

    def _ensure_slot(self, adapter_id: int, load_lora: Callable[[int, LoRAModelPayload], None]) -> int:
        entry = self.get_entry(adapter_id)
        if entry.slot_id is not None:
            return entry.slot_id

        empty_slot = next((slot for slot in self._slots if slot.adapter_id is None), None)
        if empty_slot is not None:
            self._assign_slot(empty_slot, entry, load_lora)
            return empty_slot.slot_id

        victim = self._select_victim({adapter_id})
        if victim is None:
            raise RuntimeError("No idle LoRA GPU slot available for admission")
        self._evict_slot(victim)
        self._assign_slot(victim, entry, load_lora)
        return victim.slot_id

    def _select_victim(self, protected_adapter_ids: set[int]) -> LoRASlot | None:
        idle_slots = [
            slot
            for slot in self._slots
            if slot.adapter_id is not None
            and slot.adapter_id not in protected_adapter_ids
            and slot.resident_state == LoRAResidentState.IDLE
        ]
        if not idle_slots:
            return None
        return min(idle_slots, key=lambda slot: (slot.last_used_ts, slot.slot_id))

    def _assign_slot(
        self, slot: LoRASlot, entry: LoRARuntimeEntry, load_lora: Callable[[int, LoRAModelPayload], None]
    ) -> None:
        if entry.model_payload is None:
            raise RuntimeError(f"LoRA runtime entry {entry.adapter_id} is missing model payload")
        load_lora(slot.slot_id, entry.model_payload)
        slot.adapter_id = entry.adapter_id
        slot.last_used_ts = self._tick()
        slot.resident_state = LoRAResidentState.ACTIVE if entry.gpu_running_ref_count > 0 else LoRAResidentState.IDLE
        entry.slot_id = slot.slot_id
        entry.last_used_ts = slot.last_used_ts

    def _evict_slot(self, slot: LoRASlot) -> None:
        if slot.adapter_id is None:
            return
        entry = self.get_entry(slot.adapter_id)
        if entry.gpu_running_ref_count > 0:
            raise RuntimeError("Cannot evict ACTIVE LoRA slot")
        entry.slot_id = None
        slot.adapter_id = None
        slot.resident_state = LoRAResidentState.IDLE
