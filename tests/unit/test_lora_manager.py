import pytest
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


def test_lora_manager_lifecycle_and_draining():
    from nanovllm_voxcpm.engine.lora_manager import LoRALifecycleState, LoRAManager
    from nanovllm_voxcpm.lora import set_backend_for_testing

    set_backend_for_testing(_AvailableBackend())
    try:
        manager = LoRAManager(max_loras=2)
        adapter_id = manager.register_lora("demo", _payload())

        manager.on_sequence_enqueued(adapter_id)
        manager.on_sequence_started(adapter_id)

        manager.on_sequence_finished(adapter_id, was_running=True)
        assert manager.get_entry(adapter_id).state == LoRALifecycleState.REGISTERED

        manager.unregister_lora("demo")
        assert manager.list_loras() == []
    finally:
        set_backend_for_testing(None)


def test_lora_manager_tracks_admission_capacity_without_runtime_payloads():
    from nanovllm_voxcpm.engine.lora_manager import LoRAManager
    from nanovllm_voxcpm.lora import set_backend_for_testing

    set_backend_for_testing(_AvailableBackend())
    try:
        manager = LoRAManager(max_loras=2)
        adapter_a = manager.register_lora("a", _payload(1.0))
        adapter_b = manager.register_lora("b", _payload(2.0))
        adapter_c = manager.register_lora("c", _payload(3.0))

        for adapter_id in (adapter_a, adapter_b):
            manager.on_sequence_enqueued(adapter_id)
            manager.on_sequence_started(adapter_id)

        assert manager.can_schedule({adapter_a, adapter_b}, adapter_c) is False

        manager.on_sequence_preempted(adapter_b)
        assert manager.can_schedule({adapter_a}, adapter_c) is True
    finally:
        set_backend_for_testing(None)


def test_lora_manager_unregister_drains_old_requests_and_rejects_new_ones():
    from nanovllm_voxcpm.engine.lora_manager import LoRALifecycleState, LoRAManager
    from nanovllm_voxcpm.lora import set_backend_for_testing

    set_backend_for_testing(_AvailableBackend())
    try:
        manager = LoRAManager(max_loras=1)
        adapter_id = manager.register_lora("demo", _payload())

        manager.on_sequence_enqueued(adapter_id)
        manager.unregister_lora("demo")

        assert manager.get_entry(adapter_id).state == LoRALifecycleState.DRAINING
        try:
            manager.resolve_adapter("demo")
        except ValueError as exc:
            assert "draining" in str(exc)
        else:
            raise AssertionError("draining LoRA should reject new requests")

        manager.on_sequence_finished(adapter_id, was_running=False)
        assert manager.list_loras() == []
    finally:
        set_backend_for_testing(None)


def test_lora_manager_supports_rank_local_registration_with_fixed_adapter_id():
    from nanovllm_voxcpm.engine.lora_manager import LoRAManager
    from nanovllm_voxcpm.lora import set_backend_for_testing

    set_backend_for_testing(_AvailableBackend())
    try:
        rank0 = LoRAManager(max_loras=1)
        rank1 = LoRAManager(max_loras=1)

        adapter0 = rank0.register_lora("shared", _payload(1.0), adapter_id=7)
        adapter1 = rank1.register_lora("shared", _payload(2.0), adapter_id=7)

        assert adapter0 == 7
        assert adapter1 == 7
        assert rank0.get_entry(7).alpha == 1.0
        assert rank1.get_entry(7).alpha == 2.0
    finally:
        set_backend_for_testing(None)


def test_lora_manager_validates_max_lora_rank_on_register():
    from nanovllm_voxcpm.engine.lora_manager import LoRAManager, LoRAModelPayload, LoRAModulePayload
    from nanovllm_voxcpm.lora import set_backend_for_testing

    set_backend_for_testing(_AvailableBackend())
    try:
        manager = LoRAManager(max_loras=1, max_lora_rank=1)
        oversized_payload = LoRAModelPayload(
            modules={
                "linear": LoRAModulePayload(
                    lora_a=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                    lora_b=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
                    effective_rank=2,
                    scaling=1.0,
                )
            },
            rank=2,
            alpha=2.0,
        )
        with pytest.raises(ValueError, match="max_lora_rank"):
            manager.register_lora("too-big", oversized_payload)
    finally:
        set_backend_for_testing(None)
