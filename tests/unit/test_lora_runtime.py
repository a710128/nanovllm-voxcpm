import torch


class _AvailableBackend:
    def availability(self):
        from nanovllm_voxcpm.lora import LoRAAvailability

        return LoRAAvailability(available=True, reason=None)

    def shrink(self, x, lora_a):
        return torch.nn.functional.linear(x, lora_a)

    def expand(self, hidden, lora_b, *, scaling):
        return torch.nn.functional.linear(hidden, lora_b) * scaling

    def add_lora(self, y_slices, x, lora_a_slices, lora_b_slices, *, indices, metadata, scaling):
        out_slices = [y.clone() for y in y_slices]
        for token_idx in range(x.size(0)):
            slot_id = int(indices[token_idx].item())
            if slot_id < 0:
                continue
            for slice_idx, out in enumerate(out_slices):
                hidden = self.shrink(x[token_idx : token_idx + 1], lora_a_slices[slice_idx][slot_id])
                out[token_idx : token_idx + 1] = out[token_idx : token_idx + 1] + self.expand(
                    hidden,
                    lora_b_slices[slice_idx][slot_id],
                    scaling=scaling,
                )
        return out_slices


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


def test_vendored_backend_groups_shrink_and_expand_independently():
    from nanovllm_voxcpm.lora import LoRAMetadata, _VendoredTritonPunicaBackend

    backend = _VendoredTritonPunicaBackend()
    shrink_calls = []
    expand_calls = []

    def fake_shrink(inputs, lora_a_weights, output_tensor, *meta_and_scaling):
        scaling = meta_and_scaling[-1]
        shrink_calls.append(len(lora_a_weights))
        for slice_idx, lora_a in enumerate(lora_a_weights):
            output_tensor[slice_idx].copy_(torch.nn.functional.linear(inputs, lora_a[0]) * scaling)

    def fake_expand(inputs, lora_b_weights, output_tensor, *meta, offset_start, add_inputs):
        expand_calls.append(len(lora_b_weights))
        slice_start = offset_start
        for slice_idx, lora_b in enumerate(lora_b_weights):
            width = lora_b.size(1)
            update = torch.nn.functional.linear(inputs[slice_idx], lora_b[0])
            if add_inputs:
                output_tensor[:, slice_start : slice_start + width].add_(update)
            else:
                output_tensor[:, slice_start : slice_start + width].copy_(update)
            slice_start += width

    backend._ops = lambda: (fake_shrink, fake_expand)
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_slices = [torch.ones(2, 4), torch.ones(2, 2) * 2, torch.ones(2, 2) * 3]
    lora_a_slices = [
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
    ]
    lora_b_slices = [
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0], [0.0, 2.0]]]),
        torch.tensor([[[0.5, 0.0], [0.0, 0.5]]]),
    ]
    metadata = LoRAMetadata(
        token_to_slot=torch.zeros(2, dtype=torch.int32),
        token_indices_sorted_by_slot=torch.arange(2, dtype=torch.int32),
        active_slot_ids=torch.tensor([0], dtype=torch.int32),
        num_tokens_per_slot=torch.tensor([2], dtype=torch.int32),
        slot_start_offsets=torch.tensor([0, 2], dtype=torch.int32),
        no_lora_flag=False,
        num_active_loras=1,
    )

    outputs = backend.add_lora(
        y_slices,
        x,
        lora_a_slices,
        lora_b_slices,
        indices=torch.zeros(2, dtype=torch.int64),
        metadata=metadata,
        scaling=0.5,
    )

    expected = []
    for y_slice, lora_a, lora_b in zip(y_slices, lora_a_slices, lora_b_slices):
        hidden = torch.nn.functional.linear(x, lora_a[0]) * 0.5
        expected.append(y_slice + torch.nn.functional.linear(hidden, lora_b[0]))
    assert shrink_calls == [3]
    # Slices split into two expand buckets by output width (4 vs 2), since
    # mixing slices of different hidden_out in a single kernel call would let
    # the kernel touch out-of-range columns for the narrower slices.
    assert expand_calls == [1, 2]
    for output, expected_output in zip(outputs, expected):
        assert torch.allclose(output, expected_output)


def test_vendored_backend_keeps_empty_slices_and_splits_shrink_groups():
    from nanovllm_voxcpm.lora import LoRAMetadata, _VendoredTritonPunicaBackend

    backend = _VendoredTritonPunicaBackend()
    shrink_calls = []

    def fake_shrink(inputs, lora_a_weights, output_tensor, *meta_and_scaling):
        scaling = meta_and_scaling[-1]
        shrink_calls.append(len(lora_a_weights))
        for slice_idx, lora_a in enumerate(lora_a_weights):
            output_tensor[slice_idx].copy_(torch.nn.functional.linear(inputs, lora_a[0]) * scaling)

    def fake_expand(inputs, lora_b_weights, output_tensor, *meta, offset_start, add_inputs):
        slice_start = offset_start
        for slice_idx, lora_b in enumerate(lora_b_weights):
            width = lora_b.size(1)
            update = torch.nn.functional.linear(inputs[slice_idx], lora_b[0])
            if add_inputs:
                output_tensor[:, slice_start : slice_start + width].add_(update)
            else:
                output_tensor[:, slice_start : slice_start + width].copy_(update)
            slice_start += width

    backend._ops = lambda: (fake_shrink, fake_expand)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y_slices = [torch.zeros(1, 2), torch.zeros(1, 0), torch.ones(1, 2)]
    lora_a_slices = [
        torch.tensor([[[1.0, 0.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
        torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]),
    ]
    lora_b_slices = [
        torch.tensor([[[1.0], [2.0]]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
    ]
    metadata = LoRAMetadata(
        token_to_slot=torch.zeros(1, dtype=torch.int32),
        token_indices_sorted_by_slot=torch.arange(1, dtype=torch.int32),
        active_slot_ids=torch.tensor([0], dtype=torch.int32),
        num_tokens_per_slot=torch.tensor([1], dtype=torch.int32),
        slot_start_offsets=torch.tensor([0, 1], dtype=torch.int32),
        no_lora_flag=False,
        num_active_loras=1,
    )

    outputs = backend.add_lora(
        y_slices,
        x,
        lora_a_slices,
        lora_b_slices,
        indices=torch.zeros(1, dtype=torch.int64),
        metadata=metadata,
        scaling=1.0,
    )

    assert shrink_calls == [1, 1]
    assert outputs[1].numel() == 0
    assert torch.allclose(outputs[0], torch.tensor([[1.0, 2.0]]))
    assert torch.allclose(outputs[2], torch.tensor([[3.0, 2.0]]))


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
