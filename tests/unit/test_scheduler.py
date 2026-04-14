import pytest
import torch

pydantic = pytest.importorskip("pydantic")
xxhash = pytest.importorskip("xxhash")


def test_scheduler_prefill_then_decode_round_robin(tmp_path):
    # Config asserts the model path exists.
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    from nanovllm_voxcpm.config import Config
    from nanovllm_voxcpm.engine.scheduler import Scheduler
    from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus

    cfg = Config(
        model=str(model_dir),
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        max_model_len=512,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
        tensor_parallel_size=1,
    )
    sched = Scheduler(cfg)

    s1 = Sequence("s1", list(range(300)), cfg.kvcache_block_size)
    s2 = Sequence("s2", list(range(200)), cfg.kvcache_block_size)
    sched.add(s1)
    sched.add(s2)

    seqs, is_prefill = sched.schedule()
    assert is_prefill is True
    assert set(seqs) == {s1, s2}
    assert s1.status == SequenceStatus.RUNNING
    assert s2.status == SequenceStatus.RUNNING

    # Next schedule should be decode and return a non-empty batch.
    seqs2, is_prefill2 = sched.schedule()
    assert is_prefill2 is False
    assert seqs2
    for s in seqs2:
        assert s.status == SequenceStatus.RUNNING


def test_scheduler_cancel_removes_and_deallocates(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    from nanovllm_voxcpm.config import Config
    from nanovllm_voxcpm.engine.scheduler import Scheduler
    from nanovllm_voxcpm.engine.sequence import Sequence

    cfg = Config(
        model=str(model_dir),
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        max_model_len=512,
        kvcache_block_size=256,
        num_kvcache_blocks=8,
        tensor_parallel_size=1,
    )
    sched = Scheduler(cfg)

    seq = Sequence("s1", list(range(300)), cfg.kvcache_block_size)
    sched.add(seq)
    _ = sched.schedule()  # allocate + move to running
    assert seq.block_table

    sched.cancel("s1")
    assert not seq.block_table
    assert sched.is_finished()


def test_scheduler_lora_capacity_blocks_second_adapter(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    from nanovllm_voxcpm.config import Config
    from nanovllm_voxcpm.engine.lora_manager import LoRAManager, LoRAModelPayload, LoRAModulePayload
    from nanovllm_voxcpm.engine.scheduler import Scheduler
    from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus
    from nanovllm_voxcpm.lora import LoRAAvailability, set_backend_for_testing

    class _AvailableBackend:
        def availability(self):
            return LoRAAvailability(available=True, reason=None)

    class _Callbacks:
        def __init__(self):
            self.manager = LoRAManager(max_loras=1)

        def _payload(self):
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

        def register(self, name):
            return self.manager.register_lora(name, self._payload())

        def can_schedule(self, running_adapter_ids, seq):
            return self.manager.can_schedule(running_adapter_ids, seq.adapter_id)

        def on_seq_added(self, seq):
            self.manager.on_sequence_enqueued(seq.adapter_id)

        def on_seq_running(self, seq):
            self.manager.on_sequence_started(seq.adapter_id)

        def on_seq_waiting(self, seq):
            self.manager.on_sequence_preempted(seq.adapter_id)

        def on_seq_removed(self, seq, *, was_running):
            self.manager.on_sequence_finished(seq.adapter_id, was_running=was_running)

    cfg = Config(
        model=str(model_dir),
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        max_model_len=512,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
        tensor_parallel_size=1,
    )
    callbacks = _Callbacks()
    set_backend_for_testing(_AvailableBackend())
    try:
        adapter_a = callbacks.register("a")
        adapter_b = callbacks.register("b")
        sched = Scheduler(cfg, callbacks=callbacks)

        s1 = Sequence("s1", list(range(300)), cfg.kvcache_block_size, adapter_id=adapter_a)
        s2 = Sequence("s2", list(range(200)), cfg.kvcache_block_size, adapter_id=adapter_b)
        sched.add(s1)
        sched.add(s2)

        seqs, is_prefill = sched.schedule()
        assert is_prefill is True
        assert seqs == [s1]
        assert s1.status == SequenceStatus.RUNNING
        assert s2.status == SequenceStatus.WAITING
    finally:
        set_backend_for_testing(None)


def test_scheduler_skips_blocked_adapter_and_admits_base_request(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    from nanovllm_voxcpm.config import Config
    from nanovllm_voxcpm.engine.scheduler import Scheduler
    from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus

    class _Callbacks:
        def can_schedule(self, running_adapter_ids, seq):
            return seq.seq_id != "blocked"

        def on_seq_added(self, seq):
            return None

        def on_seq_running(self, seq):
            return None

        def on_seq_waiting(self, seq):
            return None

        def on_seq_removed(self, seq, *, was_running):
            return None

    cfg = Config(
        model=str(model_dir),
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        max_model_len=512,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
        tensor_parallel_size=1,
    )
    callbacks = _Callbacks()
    sched = Scheduler(cfg, callbacks=callbacks)

    blocked = Sequence("blocked", list(range(200)), cfg.kvcache_block_size)
    base = Sequence("base", list(range(1000, 1100)), cfg.kvcache_block_size)
    sched.add(blocked)
    sched.add(base)

    seqs, is_prefill = sched.schedule()
    assert is_prefill is True
    assert seqs == [base]
    assert blocked.status == SequenceStatus.WAITING
    assert base.status == SequenceStatus.RUNNING
