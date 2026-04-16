import torch


class _AvailableBackend:
    def availability(self):
        from nanovllm_voxcpm.lora import LoRAAvailability

        return LoRAAvailability(available=True, reason=None)

    def shrink(self, x, lora_a):
        return torch.nn.functional.linear(x, lora_a)

    def expand(self, hidden, lora_b, *, scaling):
        return torch.nn.functional.linear(hidden, lora_b) * scaling

    def add_lora(self, y, x, lora_a, lora_b, *, indices, metadata, scaling):
        out = y.clone()
        for token_idx in range(x.size(0)):
            slot_id = int(indices[token_idx].item())
            if slot_id < 0:
                continue
            hidden = self.shrink(x[token_idx : token_idx + 1], lora_a[slot_id])
            out[token_idx : token_idx + 1] = out[token_idx : token_idx + 1] + self.expand(
                hidden,
                lora_b[slot_id],
                scaling=scaling,
            )
        return out


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


def _module_payload(module_name: str, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float = 1.0):
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload

    return LoRAModelPayload(
        modules={
            module_name: LoRAModulePayload(
                lora_a=lora_a,
                lora_b=lora_b,
                effective_rank=1,
                scaling=scaling,
            )
        },
        rank=1,
        alpha=scaling,
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


def test_lora_runtime_slot_reuse_clears_modules_absent_from_new_adapter():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRARuntime
    from nanovllm_voxcpm.layers.lora import LoRALinear
    from nanovllm_voxcpm.lora import set_backend_for_testing
    from nanovllm_voxcpm.utils.context import LoRAContext, reset_lora_context, set_lora_context

    set_backend_for_testing(_AvailableBackend())
    reset_lora_context()
    try:
        runtime = LoRARuntime(max_loras=1)
        first = LoRALinear(in_features=2, out_features=1, bias=False, max_loras=1, max_lora_rank=1)
        second = LoRALinear(in_features=2, out_features=1, bias=False, max_loras=1, max_lora_rank=1)
        with torch.no_grad():
            first.weight.zero_()
            second.weight.zero_()

        modules = {"first": first, "second": second}

        def load_lora(slot_id, payload):
            model_runner._clear_lora_slot_modules(modules, slot_id)
            for module_name, module_payload in payload.modules.items():
                modules[module_name].set_slot_lora(
                    slot_id=slot_id,
                    lora_a=module_payload.lora_a,
                    lora_b=module_payload.lora_b,
                    effective_rank=module_payload.effective_rank,
                    scaling=module_payload.scaling,
                )

        adapter_a = runtime.register_lora(
            "adapter-a",
            _module_payload(
                "first",
                lora_a=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                lora_b=torch.tensor([[2.0]], dtype=torch.float32),
            ),
        )
        runtime.on_sequence_enqueued(adapter_a)
        runtime.on_sequence_started(adapter_a)
        runtime.build_batch_plan([adapter_a], [1], load_lora)
        runtime.on_sequence_preempted(adapter_a)

        adapter_b = runtime.register_lora(
            "adapter-b",
            _module_payload(
                "second",
                lora_a=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
                lora_b=torch.tensor([[5.0]], dtype=torch.float32),
            ),
        )
        runtime.on_sequence_enqueued(adapter_b)
        runtime.on_sequence_started(adapter_b)
        runtime.build_batch_plan([adapter_b], [1], load_lora)

        set_lora_context(
            LoRAContext(
                token_to_slot=torch.tensor([0], dtype=torch.int32),
                token_indices_sorted_by_slot=torch.tensor([0], dtype=torch.int32),
                active_slot_ids=torch.tensor([0], dtype=torch.int32),
                num_tokens_per_slot=torch.tensor([1], dtype=torch.int32),
                slot_start_offsets=torch.tensor([0, 1], dtype=torch.int32),
                no_lora_flag=False,
            )
        )

        x = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
        first_out = first(x)
        second_out = second(x)

        assert torch.allclose(first_out, torch.zeros_like(first_out))
        assert torch.allclose(second_out, torch.tensor([[15.0]], dtype=torch.float32))
    finally:
        reset_lora_context()
        set_backend_for_testing(None)
