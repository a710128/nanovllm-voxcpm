"""Regression test for the decode-phase self-preempt branch.

When decode runs with a single running sequence and the block manager
reports that it cannot append to that sequence, ``Scheduler.schedule``
takes the ``else: self.preempt(seq); break`` branch. That legitimately
leaves ``scheduled_seqs`` empty. Before the assert was removed from
``Scheduler.schedule``, this crashed the inference process; the parent
server then hung silently on its output queue.

The assert has been removed on main and ``LLMEngineBase.step`` already
short-circuits on an empty schedule, but the self-preempt decode branch
did not have a dedicated regression test covering the specific path
that triggered this in production.
"""

import pytest

pytest.importorskip("pydantic")
pytest.importorskip("xxhash")
pytest.importorskip("torch")


def test_decode_self_preempt_returns_empty_without_crash(tmp_path, monkeypatch):
    from nanovllm_voxcpm.config import Config
    from nanovllm_voxcpm.engine.scheduler import Scheduler
    from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    cfg = Config(
        model=str(model_dir),
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        max_model_len=512,
        kvcache_block_size=256,
        num_kvcache_blocks=4,
        tensor_parallel_size=1,
    )
    sched = Scheduler(cfg)

    only = Sequence("only", list(range(128)), cfg.kvcache_block_size)
    sched.add(only)

    first_seqs, first_is_prefill = sched.schedule()
    assert first_is_prefill is True
    assert first_seqs == [only]
    assert only.status == SequenceStatus.RUNNING

    monkeypatch.setattr(sched.block_manager, "can_append", lambda seq: False)

    seqs, is_prefill = sched.schedule()

    assert seqs == []
    assert is_prefill is False
    assert only.status == SequenceStatus.WAITING
    assert list(s.seq_id for s in sched.waiting) == ["only"]
    assert not sched.running


def test_engine_step_is_noop_when_decode_self_preempts(tmp_path, monkeypatch):
    from nanovllm_voxcpm.config import Config
    from nanovllm_voxcpm.engine.llm_engine import LLMEngineBase
    from nanovllm_voxcpm.engine.scheduler import Scheduler
    from nanovllm_voxcpm.engine.sequence import Sequence

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    cfg = Config(
        model=str(model_dir),
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        max_model_len=512,
        kvcache_block_size=256,
        num_kvcache_blocks=4,
        tensor_parallel_size=1,
    )
    sched = Scheduler(cfg)
    sched.add(Sequence("only", list(range(128)), cfg.kvcache_block_size))
    sched.schedule()

    monkeypatch.setattr(sched.block_manager, "can_append", lambda seq: False)

    class _Runner:
        def call(self, *args, **kwargs):
            raise AssertionError("model_runner.call must not be invoked on empty schedule")

    engine = object.__new__(LLMEngineBase)
    engine.scheduler = sched
    engine.model_runner = _Runner()

    def _fail_preprocess(seq, is_prefill):
        raise AssertionError("preprocess_seq must not be invoked on empty schedule")

    def _fail_postprocess(seq, output, is_prefill):
        raise AssertionError("postprocess_seq must not be invoked on empty schedule")

    engine.preprocess_seq = _fail_preprocess
    engine.postprocess_seq = _fail_postprocess

    assert engine.step() == []
