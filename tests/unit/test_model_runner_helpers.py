import pytest
import threading

torch = pytest.importorskip("torch")


def test_runner_task_properties():
    from nanovllm_voxcpm.engine.model_runner import RunnerTask

    t = RunnerTask(block_table=[0, 1, 2], seq_length=600, num_cached_tokens=512, block_size=256)
    assert t.num_blocks == 3
    assert t.num_cached_blocks == 2
    assert t.last_block_num_tokens == 88


def test_cut_inputs_and_assign_outputs():
    from nanovllm_voxcpm.engine.model_runner import cut_inputs, assign_outputs

    inputs = {
        "a": torch.arange(10),
        "b": torch.arange(10) + 100,
    }
    cut = cut_inputs(inputs, 3)
    assert cut["a"].tolist() == [0, 1, 2]
    assert cut["b"].tolist() == [100, 101, 102]

    outputs = {"a": torch.empty(10, dtype=torch.long)}
    assign_outputs({"a": torch.tensor([5, 6, 7])}, outputs, 3)
    assert outputs["a"][:3].tolist() == [5, 6, 7]

    with pytest.raises(KeyError):
        assign_outputs({"missing": torch.tensor([1])}, {"a": torch.empty(1)}, 1)


def test_select_lora_payload_for_rank():
    from nanovllm_voxcpm.engine.model_runner import select_lora_payload_for_rank

    payload0 = object()
    payload1 = object()

    assert select_lora_payload_for_rank(payload0, 0) is payload0
    assert select_lora_payload_for_rank([payload0, payload1], 1) is payload1

    with pytest.raises(ValueError):
        select_lora_payload_for_rank([payload0], 1)


def test_base_model_runner_call_synchronizes_tp_broadcast(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    reduce_calls = []

    def _all_reduce(tensor, op=None):
        reduce_calls.append(int(tensor.item()))

    monkeypatch.setattr(model_runner.dist, "all_reduce", _all_reduce)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2
    runner.rank = 0
    writes = []
    runner.write_shm = lambda method_name, *args: writes.append((method_name, args))
    runner.test_method = lambda value: value + 1

    result = runner.call("test_method", 41)

    assert result == 42
    assert writes == [("test_method", (41,))]
    assert reduce_calls == [0]


def test_base_model_runner_call_skips_outer_barrier_for_exit(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    reduce_calls = []
    monkeypatch.setattr(
        model_runner.dist, "all_reduce", lambda tensor, op=None: reduce_calls.append(int(tensor.item()))
    )

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2
    runner.rank = 0
    runner.write_shm = lambda method_name, *args: None
    runner.exit = lambda: "done"

    result = runner.call("exit")

    assert result == "done"
    assert reduce_calls == []


def test_write_read_shm_uses_file_fallback_for_large_payload(tmp_path):
    import pickle
    from multiprocessing.shared_memory import SharedMemory

    import nanovllm_voxcpm.engine.model_runner as model_runner

    rank0 = object.__new__(model_runner.BaseModelRunner)
    rank0.world_size = 2
    rank0.rank = 0
    rank0.event = []
    rank0.shm = SharedMemory(create=True, size=128)

    rank1 = object.__new__(model_runner.BaseModelRunner)
    rank1.world_size = 2
    rank1.rank = 1

    class _Event:
        def wait(self):
            return None

        def clear(self):
            return None

    rank1.event = _Event()
    rank1.shm = SharedMemory(name=rank0.shm.name)
    overflow_path = None
    try:
        large_payload = ["x" * 1024]
        overflow_path = rank0.write_shm("register_lora", 1, "demo", large_payload)
        assert overflow_path is not None

        method_name, args = rank1.read_shm()
        assert method_name == "register_lora"
        assert args[0] == 1
        assert args[1] == "demo"
        assert args[2] == large_payload
        with open(overflow_path, "rb") as f:
            assert pickle.load(f)[0] == "register_lora"
    finally:
        rank1.shm.close()
        rank0.shm.close()
        rank0.shm.unlink()
        if overflow_path is not None:
            import os

            os.remove(overflow_path)


def test_validate_lora_payload_rejects_unknown_module():
    import torch.nn as nn

    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload
    from nanovllm_voxcpm.layers.lora import LoRALinear

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.rank = 0
    runner.model = nn.Module()
    runner.model.add_module("linear", LoRALinear(2, 1, bias=False, max_loras=1, max_lora_rank=1))

    payload = LoRAModelPayload(
        modules={
            "missing": LoRAModulePayload(
                lora_a=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                lora_b=torch.tensor([[1.0]], dtype=torch.float32),
                effective_rank=1,
                scaling=1.0,
            )
        },
        rank=1,
        alpha=1.0,
    )

    with pytest.raises(ValueError, match="Unknown LoRA target module"):
        runner.validate_lora_payload(payload)


def test_validate_lora_payload_rejects_invalid_linear_shape():
    import torch.nn as nn

    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload
    from nanovllm_voxcpm.layers.lora import LoRALinear

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.rank = 0
    runner.model = nn.Module()
    runner.model.add_module("linear", LoRALinear(2, 1, bias=False, max_loras=1, max_lora_rank=1))

    payload = LoRAModelPayload(
        modules={
            "linear": LoRAModulePayload(
                lora_a=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                lora_b=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
                effective_rank=1,
                scaling=1.0,
            )
        },
        rank=1,
        alpha=1.0,
    )

    with pytest.raises(ValueError, match="output dim"):
        runner.validate_lora_payload(payload)


def test_validate_lora_payload_rejects_qkv_target_count_mismatch():
    import torch.nn as nn

    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload
    from nanovllm_voxcpm.layers.lora import LoRAQKVParallelLinear

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.rank = 0
    runner.model = nn.Module()
    runner.model.add_module(
        "qkv",
        LoRAQKVParallelLinear(
            hidden_size=2,
            head_size=1,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            max_loras=1,
            max_lora_rank=1,
        ),
    )

    payload = LoRAModelPayload(
        modules={
            "qkv": LoRAModulePayload(
                lora_a=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.float32),
                lora_b=[torch.tensor([[1.0]], dtype=torch.float32), torch.tensor([[1.0]], dtype=torch.float32)],
                effective_rank=1,
                scaling=1.0,
            )
        },
        rank=1,
        alpha=1.0,
    )

    with pytest.raises(ValueError, match="LoRA A targets"):
        runner.validate_lora_payload(payload)


def test_tp2_register_lora_uses_rank_local_payload_and_runner_manager(monkeypatch):
    from multiprocessing.shared_memory import SharedMemory
    import torch.nn as nn

    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload, LoRARuntime
    from nanovllm_voxcpm.layers.lora import LoRALinear
    from nanovllm_voxcpm.lora import LoRAAvailability, set_backend_for_testing

    class _AvailableBackend:
        def availability(self):
            return LoRAAvailability(available=True, reason=None)

    barrier = threading.Barrier(2)

    def _all_reduce(tensor, op=None):
        barrier.wait()

    monkeypatch.setattr(model_runner.dist, "all_reduce", _all_reduce)

    class _Event:
        def __init__(self):
            self._event = threading.Event()

        def wait(self):
            self._event.wait()

        def clear(self):
            self._event.clear()

        def set(self):
            self._event.set()

    def _payload(alpha):
        return LoRAModelPayload(
            modules={
                "linear": LoRAModulePayload(
                    lora_a=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                    lora_b=torch.tensor([[alpha]], dtype=torch.float32),
                    effective_rank=1,
                    scaling=alpha,
                )
            },
            rank=1,
            alpha=alpha,
        )

    event = _Event()
    rank0 = object.__new__(model_runner.BaseModelRunner)
    rank0.world_size = 2
    rank0.rank = 0
    rank0.event = [event]
    rank0.max_lora_rank = 2
    rank0.max_loras = 1
    rank0.lora_runtime = LoRARuntime(max_loras=1, max_lora_rank=2)
    rank0.model = nn.Module()
    rank0.model.add_module("linear", LoRALinear(2, 1, bias=False, max_loras=1, max_lora_rank=2))
    rank0.exit = lambda: None
    rank0.shm = SharedMemory(create=True, size=1024)

    rank1 = object.__new__(model_runner.BaseModelRunner)
    rank1.world_size = 2
    rank1.rank = 1
    rank1.event = event
    rank1.max_lora_rank = 2
    rank1.max_loras = 1
    rank1.lora_runtime = LoRARuntime(max_loras=1, max_lora_rank=2)
    rank1.model = nn.Module()
    rank1.model.add_module("linear", LoRALinear(2, 1, bias=False, max_loras=1, max_lora_rank=2))
    rank1.exit = lambda: None
    rank1.shm = SharedMemory(name=rank0.shm.name)

    worker = threading.Thread(target=rank1.loop)
    worker.start()
    set_backend_for_testing(_AvailableBackend())
    try:
        rank0.call("register_lora", 3, "demo", [_payload(1.0), _payload(2.0)])
        assert rank0.lora_runtime.get_entry(3).alpha == 1.0
        assert rank1.lora_runtime.get_entry(3).alpha == 2.0

        rank0.call("exit")
        worker.join(timeout=1)
        assert not worker.is_alive()
    finally:
        rank1.shm.close()
        rank0.shm.close()
        rank0.shm.unlink()
        set_backend_for_testing(None)
