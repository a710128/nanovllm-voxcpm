import torch


class _AvailableBackend:
    def availability(self):
        from nanovllm_voxcpm.lora import LoRAAvailability

        return LoRAAvailability(available=True, reason=None)


def _payload(scale: float = 1.0):
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload

    return LoRAModelPayload(
        modules={
            "linear": LoRAModulePayload(
                lora_a=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                lora_b=torch.tensor([[scale]], dtype=torch.float32),
                effective_rank=1,
                scaling=scale,
            )
        },
        rank=1,
        alpha=scale,
    )


def test_lora_runtime_builds_batch_plan_and_loads_slots():
    from nanovllm_voxcpm.engine.lora_manager import LoRARuntime
    from nanovllm_voxcpm.lora import set_backend_for_testing

    set_backend_for_testing(_AvailableBackend())
    try:
        runtime = LoRARuntime(max_loras=2)
        adapter_id = runtime.register_lora("demo", _payload())

        runtime.on_sequence_enqueued(adapter_id)
        runtime.on_sequence_started(adapter_id)

        loads = []
        plan = runtime.build_batch_plan(
            [adapter_id, None, adapter_id],
            [2, 1, 1],
            lambda slot_id, payload: loads.append((slot_id, payload.rank)),
        )

        assert loads == [(0, 1)]
        assert plan.token_to_slot == [0, 0, -1, 0]
        assert plan.token_indices_sorted_by_slot == [0, 1, 3]
        assert plan.active_slot_ids == [0]
        assert plan.num_tokens_per_slot == [3]
        assert plan.slot_start_offsets == [0, 3]
    finally:
        set_backend_for_testing(None)


def test_lora_runtime_capacity_and_lru_eviction():
    from nanovllm_voxcpm.engine.lora_manager import LoRARuntime
    from nanovllm_voxcpm.lora import set_backend_for_testing

    set_backend_for_testing(_AvailableBackend())
    try:
        runtime = LoRARuntime(max_loras=2)
        adapter_a = runtime.register_lora("a", _payload(1.0))
        adapter_b = runtime.register_lora("b", _payload(2.0))
        adapter_c = runtime.register_lora("c", _payload(3.0))

        for adapter_id in (adapter_a, adapter_b):
            runtime.on_sequence_enqueued(adapter_id)
            runtime.on_sequence_started(adapter_id)
        runtime.build_batch_plan([adapter_a, adapter_b], [1, 1], lambda slot_id, payload: None)

        assert runtime.can_schedule({adapter_a, adapter_b}, adapter_c) is False

        runtime.on_sequence_preempted(adapter_b)
        assert runtime.can_schedule({adapter_a}, adapter_c) is True

        loads = []
        plan = runtime.build_batch_plan(
            [adapter_a, adapter_c],
            [1, 1],
            lambda slot_id, payload: loads.append((slot_id, payload.alpha)),
        )
        assert loads == [(1, 3.0)]
        assert plan.active_slot_ids == [0, 1]
        assert sorted(plan.adapter_to_slot) == [adapter_a, adapter_c]
    finally:
        set_backend_for_testing(None)
