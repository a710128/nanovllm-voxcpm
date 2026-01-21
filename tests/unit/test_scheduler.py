import pytest


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
