from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload
from nanovllm_voxcpm.lora import LoRAAvailability, set_backend_for_testing


class _AvailableBackend:
    def availability(self):
        return LoRAAvailability(available=True, reason=None)


class _DummyRunner:
    fail_register = False
    instances: list["_DummyRunner"] = []

    def __init__(self, config, rank, device_idx, distributed_port, event):
        self.config = config
        self.rank = rank
        self.device_idx = device_idx
        self.distributed_port = distributed_port
        self.event = event
        self.calls: list[tuple[str, tuple]] = []
        type(self).instances.append(self)

    def call(self, method_name, *args):
        self.calls.append((method_name, args))
        if method_name == "run":
            return [{"token": f"token-{index}".encode("utf-8"), "stop": True} for index, _ in enumerate(args[0])]
        if method_name == "register_lora" and type(self).fail_register:
            raise RuntimeError("register failed")
        return None


class _EngineUnderTest:
    pass


def _make_engine_class():
    from nanovllm_voxcpm.engine.llm_engine import LLMEngineBase
    from nanovllm_voxcpm.engine.model_runner import RunnerTask

    class EngineUnderTest(LLMEngineBase):
        def preprocess_seq(self, seq, is_prefill: bool) -> RunnerTask:
            return RunnerTask(
                block_table=list(seq.block_table),
                seq_length=len(seq),
                num_cached_tokens=seq.num_cached_tokens,
                block_size=seq.block_size,
                custom_payload={"seq_id": seq.seq_id, "is_prefill": is_prefill},
                adapter_id=seq.adapter_id,
            )

        def postprocess_seq(self, seq, outputs: dict, is_prefill: bool):
            seq.append_token(outputs["token"])
            seq.stoped = outputs["stop"]

    return EngineUnderTest


def _make_config(tmp_path, *, devices=None, lora_config=None):
    from nanovllm_voxcpm.config import Config

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return Config(
        model=str(model_dir),
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        max_model_len=512,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
        tensor_parallel_size=1,
        devices=devices,
        lora_config=lora_config,
    )


def _make_lora_payload() -> LoRAModelPayload:
    return LoRAModelPayload(
        modules={
            "linear": LoRAModulePayload(
                lora_a=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
                lora_b=torch.tensor([[1.0]], dtype=torch.float32),
                effective_rank=1,
                scaling=1.0,
            )
        },
        rank=1,
        alpha=1.0,
    )


def test_engine_step_finishes_sequence_and_supports_cancel(monkeypatch, tmp_path):
    import nanovllm_voxcpm.engine.llm_engine as llm_engine
    from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus

    EngineUnderTest = _make_engine_class()
    _DummyRunner.instances.clear()

    monkeypatch.setattr(llm_engine.atexit, "register", lambda fn: None)
    monkeypatch.setattr(llm_engine.torch.cuda, "device_count", lambda: 1)

    engine = EngineUnderTest(_DummyRunner, _make_config(tmp_path, devices=None), tensor_parallel_size=1)

    cancelled = Sequence("cancelled", [10, 11], 256)
    engine.add_sequence(cancelled)
    engine.cancel_sequence("cancelled")
    assert engine.scheduler.is_finished()

    seq = Sequence("seq-1", [1, 2, 3], 256)
    engine.add_sequence(seq)

    scheduled = engine.step()

    assert scheduled == [seq]
    assert seq.status == SequenceStatus.FINISHED
    assert seq.is_finished is True
    assert engine.is_finished() is True
    assert engine.resolve_lora(None) is None

    runner_calls = [name for name, _ in _DummyRunner.instances[0].calls]
    assert runner_calls.count("run") == 1
    assert "lora_on_sequence_enqueued" in runner_calls
    assert "lora_on_sequence_started" in runner_calls
    assert "lora_on_sequence_finished" in runner_calls


def test_engine_registers_and_unregisters_lora(monkeypatch, tmp_path):
    import nanovllm_voxcpm.engine.llm_engine as llm_engine
    from nanovllm_voxcpm.engine.sequence import Sequence

    EngineUnderTest = _make_engine_class()
    _DummyRunner.instances.clear()
    _DummyRunner.fail_register = False

    monkeypatch.setattr(llm_engine.atexit, "register", lambda fn: None)

    set_backend_for_testing(_AvailableBackend())
    try:
        engine = EngineUnderTest(
            _DummyRunner,
            _make_config(tmp_path, devices=[0], lora_config=SimpleNamespace(max_loras=2, max_lora_rank=1)),
            tensor_parallel_size=1,
        )

        adapter_id = engine.register_lora("demo", _make_lora_payload())

        assert adapter_id == 0
        assert engine.resolve_lora("demo") == 0
        assert [entry.name for entry in engine.list_loras()] == ["demo"]
        assert engine.can_schedule(set(), Sequence("seq", [1], 256, adapter_id=adapter_id)) is True

        engine.unregister_lora("demo")
        assert engine.list_loras() == []
    finally:
        set_backend_for_testing(None)


def test_engine_register_lora_rolls_back_runner_state_on_failure(monkeypatch, tmp_path):
    import nanovllm_voxcpm.engine.llm_engine as llm_engine

    EngineUnderTest = _make_engine_class()
    _DummyRunner.instances.clear()
    _DummyRunner.fail_register = True

    monkeypatch.setattr(llm_engine.atexit, "register", lambda fn: None)

    set_backend_for_testing(_AvailableBackend())
    try:
        engine = EngineUnderTest(
            _DummyRunner,
            _make_config(tmp_path, devices=[0], lora_config=SimpleNamespace(max_loras=2, max_lora_rank=1)),
            tensor_parallel_size=1,
        )

        with pytest.raises(RuntimeError, match="register failed"):
            engine.register_lora("demo", _make_lora_payload())

        runner_calls = [name for name, _ in _DummyRunner.instances[0].calls]
        assert runner_calls.count("validate_lora_payload") == 1
        assert runner_calls.count("register_lora") == 1
        assert runner_calls.count("unregister_lora") == 1
        assert engine.list_loras() == []
    finally:
        _DummyRunner.fail_register = False
        set_backend_for_testing(None)


def test_engine_validates_device_count(monkeypatch, tmp_path):
    import nanovllm_voxcpm.engine.llm_engine as llm_engine

    EngineUnderTest = _make_engine_class()
    monkeypatch.setattr(llm_engine.atexit, "register", lambda fn: None)
    monkeypatch.setattr(llm_engine.torch.cuda, "device_count", lambda: 1)

    cfg = _make_config(tmp_path, devices=None)

    with pytest.raises(ValueError, match="greater than the number of available devices"):
        EngineUnderTest(_DummyRunner, cfg, tensor_parallel_size=2)


def test_engine_validates_explicit_devices_match_tensor_parallel(monkeypatch, tmp_path):
    import nanovllm_voxcpm.engine.llm_engine as llm_engine

    EngineUnderTest = _make_engine_class()
    monkeypatch.setattr(llm_engine.atexit, "register", lambda fn: None)

    cfg = _make_config(tmp_path, devices=[0])

    with pytest.raises(ValueError, match="Number of devices 1 is not equal to tensor parallel size 2"):
        EngineUnderTest(_DummyRunner, cfg, tensor_parallel_size=2)


def test_engine_exit_joins_spawned_processes(monkeypatch, tmp_path):
    import nanovllm_voxcpm.engine.llm_engine as llm_engine

    EngineUnderTest = _make_engine_class()
    _DummyRunner.instances.clear()

    events = []
    processes = []

    class FakeEvent:
        pass

    class FakeProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.started = False
            self.joined = False
            processes.append(self)

        def start(self):
            self.started = True

        def join(self):
            self.joined = True

    class FakeContext:
        def Event(self):
            event = FakeEvent()
            events.append(event)
            return event

        def Process(self, target, args):
            return FakeProcess(target, args)

    monkeypatch.setattr(llm_engine.atexit, "register", lambda fn: None)
    monkeypatch.setattr(llm_engine.mp, "get_context", lambda method: FakeContext())

    engine = EngineUnderTest(_DummyRunner, _make_config(tmp_path, devices=[0, 1]), tensor_parallel_size=2)
    engine.exit()

    assert len(events) == 1
    assert len(processes) == 1
    assert processes[0].started is True
    assert processes[0].joined is True
    assert _DummyRunner.instances[0].calls[-1][0] == "exit"
