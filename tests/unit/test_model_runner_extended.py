"""Extended CPU-only tests for nanovllm_voxcpm.engine.model_runner.

These tests cover logic not reached by test_model_runner_helpers.py, targeting
>= 70% line coverage across model_runner.py without any GPU dependency.

Patterns used:
- ``object.__new__(BaseModelRunner)`` for bare-metal runner instances
- direct attribute assignment for state setup
- monkeypatching CUDA-adjacent edges (dist, cuda calls)
"""

import os
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _reset_lora_backend():
    """Ensure LoRA backend is in a clean available state for each test.

    test_lora_layers.py sets the backend to 'vendored unavailable' without teardown.
    This autouse fixture restores a clean state before and after every test in this module.
    """
    from nanovllm_voxcpm.lora import LoRAAvailability, set_backend_for_testing

    class _AvailableBackend:
        def availability(self):
            return LoRAAvailability(available=True, reason=None)

    set_backend_for_testing(_AvailableBackend())
    yield
    set_backend_for_testing(None)


def test_env_int_returns_none_for_absent_var():
    from nanovllm_voxcpm.engine.model_runner import _env_int

    os.environ.pop("_TEST_NANOVLLM_ENV_INT", None)
    assert _env_int("_TEST_NANOVLLM_ENV_INT") is None


def test_env_int_returns_none_for_blank_var():
    from nanovllm_voxcpm.engine.model_runner import _env_int

    os.environ["_TEST_NANOVLLM_ENV_INT"] = "  "
    try:
        assert _env_int("_TEST_NANOVLLM_ENV_INT") is None
    finally:
        del os.environ["_TEST_NANOVLLM_ENV_INT"]


def test_env_int_returns_integer():
    from nanovllm_voxcpm.engine.model_runner import _env_int

    os.environ["_TEST_NANOVLLM_ENV_INT"] = "42"
    try:
        assert _env_int("_TEST_NANOVLLM_ENV_INT") == 42
    finally:
        del os.environ["_TEST_NANOVLLM_ENV_INT"]


def test_env_int_raises_for_non_integer():
    from nanovllm_voxcpm.engine.model_runner import _env_int

    os.environ["_TEST_NANOVLLM_ENV_INT"] = "notanint"
    try:
        with pytest.raises(ValueError, match="must be an integer"):
            _env_int("_TEST_NANOVLLM_ENV_INT")
    finally:
        del os.environ["_TEST_NANOVLLM_ENV_INT"]



def test_runner_task_with_adapter_id():
    from nanovllm_voxcpm.engine.model_runner import RunnerTask

    t = RunnerTask(
        block_table=[0, 1],
        seq_length=512,
        num_cached_tokens=256,
        block_size=256,
        custom_payload={"data": "x"},
        adapter_id=7,
    )
    assert t.adapter_id == 7
    assert t.custom_payload == {"data": "x"}
    assert t.num_blocks == 2
    assert t.num_cached_blocks == 1
    assert t.last_block_num_tokens == 256


def test_runner_task_single_block():
    from nanovllm_voxcpm.engine.model_runner import RunnerTask

    t = RunnerTask(block_table=[5], seq_length=10, num_cached_tokens=0, block_size=16)
    assert t.num_blocks == 1
    assert t.num_cached_blocks == 0
    assert t.last_block_num_tokens == 10


def test_runner_task_exact_block_boundary():
    from nanovllm_voxcpm.engine.model_runner import RunnerTask

    t = RunnerTask(block_table=[0, 1], seq_length=512, num_cached_tokens=512, block_size=256)
    assert t.num_blocks == 2
    assert t.num_cached_blocks == 2
    assert t.last_block_num_tokens == 256



def test_clear_lora_slot_modules_with_module_names_filter():
    """module_names= filters which modules get zeroed (coverage for iterable branch)."""
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.layers.lora import LoRALinear

    model = nn.Module()
    model.add_module("first", LoRALinear(2, 1, bias=False, max_loras=1, max_lora_rank=1))
    model.add_module("second", LoRALinear(2, 1, bias=False, max_loras=1, max_lora_rank=1))

    for module in model.children():
        module.set_slot_lora(
            slot_id=0,
            lora_a=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            lora_b=torch.tensor([[2.0]], dtype=torch.float32),
            effective_rank=1,
            scaling=1.0,
        )

    modules_dict = dict(model.named_modules())
    # Only clear "first"
    model_runner._clear_lora_slot_modules(modules_dict, slot_id=0, module_names=["first"])

    first = dict(model.named_modules())["first"]
    second = dict(model.named_modules())["second"]

    # "first" should be zeroed
    assert torch.count_nonzero(first.lora_A[0]) == 0
    # "second" should be untouched
    assert torch.count_nonzero(second.lora_A[0]) != 0


def test_clear_lora_slot_modules_with_missing_module_name_in_filter():
    """module_names containing unknown name should not raise (quietly skipped)."""
    import nanovllm_voxcpm.engine.model_runner as model_runner

    # Empty modules dict, nonexistent name in filter — should be a no-op
    model_runner._clear_lora_slot_modules({}, slot_id=0, module_names=["nonexistent"])



def test_lora_model_modules_caches_and_returns_dict():
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    runner._lora_model_modules_cache = None
    runner.model = nn.Sequential(nn.Linear(2, 2))

    result1 = runner._lora_model_modules()
    result2 = runner._lora_model_modules()

    assert isinstance(result1, dict)
    assert result1 is result2  # same object → cached


def test_lora_model_modules_uses_existing_cache():
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    sentinel = {"cached": object()}
    runner._lora_model_modules_cache = sentinel

    result = runner._lora_model_modules()
    assert result is sentinel



def test_dit_lora_rows_per_sample_no_lora_config():
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    # No lora_config attribute set → should return 0
    assert runner._dit_lora_rows_per_sample() == 0


def test_dit_lora_rows_per_sample_with_lora_config_not_dit():
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)

    class _FakeLoraCfg:
        enable_dit = False

    runner.lora_config = _FakeLoraCfg()
    assert runner._dit_lora_rows_per_sample() == 0


def test_dit_lora_rows_per_sample_with_dit_enabled():
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)

    class _FakeLoraCfg:
        enable_dit = True

    runner.lora_config = _FakeLoraCfg()
    runner.cfg_branches = 2
    runner.dit_lora_seq_len_offset = 4
    runner.patch_size = 8
    # 2 * (4 + 2*8) = 2 * 20 = 40
    assert runner._dit_lora_rows_per_sample() == 40



def test_build_lora_contexts_no_adapters_returns_empty_lm_and_no_lora_flags():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRARuntime
    from nanovllm_voxcpm.utils.context import LM_LORA_DOMAIN, PROJ_LORA_DOMAIN, DIT_LORA_DOMAIN

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.lora_runtime = LoRARuntime(max_loras=2, max_lora_rank=4)
    runner.lora_config = None

    seqs = [
        model_runner.RunnerTask(
            block_table=[0], seq_length=4, num_cached_tokens=0, block_size=4, adapter_id=None
        ),
        model_runner.RunnerTask(
            block_table=[1], seq_length=2, num_cached_tokens=0, block_size=4, adapter_id=None
        ),
    ]
    token_counts = [4, 2]
    contexts = runner._build_lora_contexts(seqs, token_counts)

    assert LM_LORA_DOMAIN in contexts
    assert PROJ_LORA_DOMAIN in contexts
    assert DIT_LORA_DOMAIN in contexts

    proj_ctx = contexts[PROJ_LORA_DOMAIN]
    dit_ctx = contexts[DIT_LORA_DOMAIN]
    assert proj_ctx.no_lora_flag is True
    assert dit_ctx.no_lora_flag is True
    # LM context should have token_to_slot tensor of length 6 (all -1)
    lm_ctx = contexts[LM_LORA_DOMAIN]
    assert lm_ctx.token_to_slot is not None



def test_unregister_lora_calls_lora_runtime():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRARuntime, LoRAModelPayload, LoRAModulePayload

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.lora_runtime = LoRARuntime(max_loras=2, max_lora_rank=4)

    payload = LoRAModelPayload(
        modules={},
        rank=1,
        alpha=1.0,
    )
    # Register then unregister
    runner.lora_runtime.register_lora("adapter_a", payload, adapter_id=10)
    entry = runner.lora_runtime.get_entry(10)
    assert entry.name == "adapter_a"

    runner.unregister_lora(10)
    # After draining-eligible unregister the entry should be removed (cpu_ref_count==0)
    with pytest.raises(KeyError):
        runner.lora_runtime.get_entry(10)



def test_lora_lifecycle_calls_delegated_to_runtime():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRARuntime, LoRAModelPayload

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.lora_runtime = LoRARuntime(max_loras=2, max_lora_rank=4)

    payload = LoRAModelPayload(modules={}, rank=1, alpha=1.0)
    runner.lora_runtime.register_lora("adap", payload, adapter_id=5)

    # enqueued → active
    runner.lora_on_sequence_enqueued(5)
    assert runner.lora_runtime.get_entry(5).cpu_ref_count == 1

    # started → gpu running ref
    runner.lora_on_sequence_started(5)
    assert runner.lora_runtime.get_entry(5).gpu_running_ref_count == 1

    # preempted → gpu running ref goes back
    runner.lora_on_sequence_preempted(5)
    assert runner.lora_runtime.get_entry(5).gpu_running_ref_count == 0

    # finished (not running since we preempted)
    runner.lora_on_sequence_finished(5, was_running=False)
    assert runner.lora_runtime.get_entry(5).cpu_ref_count == 0

    # None adapter_id should be a no-op, not raise
    runner.lora_on_sequence_enqueued(None)
    runner.lora_on_sequence_started(None)
    runner.lora_on_sequence_preempted(None)  # this will raise if it dereferences None
    runner.lora_on_sequence_finished(None, was_running=False)



def test_validate_lora_payload_rejects_zero_rank():
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.rank = 0
    runner.model = nn.Module()
    runner._lora_model_modules_cache = None

    payload = LoRAModelPayload(modules={"x": object()}, rank=0, alpha=1.0)
    with pytest.raises(ValueError, match="rank must be > 0"):
        runner.validate_lora_payload(payload)


def test_validate_lora_payload_rejects_empty_modules():
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.rank = 0
    runner.model = nn.Module()
    runner._lora_model_modules_cache = None

    payload = LoRAModelPayload(modules={}, rank=1, alpha=1.0)
    with pytest.raises(ValueError, match="at least one target module"):
        runner.validate_lora_payload(payload)


def test_validate_lora_payload_rejects_non_lora_module():
    """Module that exists but has no validate_slot_lora_payload should raise."""
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.rank = 0
    runner.model = nn.Module()
    runner.model.add_module("linear", nn.Linear(2, 2))  # regular Linear has no LoRA slot
    runner._lora_model_modules_cache = None

    payload = LoRAModelPayload(
        modules={
            "linear": LoRAModulePayload(
                lora_a=torch.zeros(1, 2),
                lora_b=torch.zeros(2, 1),
                effective_rank=1,
                scaling=1.0,
            )
        },
        rank=1,
        alpha=1.0,
    )
    with pytest.raises(ValueError, match="does not support LoRA slots"):
        runner.validate_lora_payload(payload)



def test_register_lora_mismatch_raises_runtime_error():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRARuntime

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.rank = 0

    runtime = LoRARuntime(max_loras=2, max_lora_rank=4)
    runner.lora_runtime = runtime

    runner.validate_lora_payload = lambda payload: None

    payload = LoRAModelPayload(modules={}, rank=1, alpha=1.0)

    calls = []

    def fake_register(name, payload_arg, *, adapter_id=None):
        calls.append(adapter_id)
        return adapter_id + 1 if adapter_id is not None else 1

    runner.lora_runtime.register_lora = fake_register  # type: ignore

    with pytest.raises(RuntimeError, match="adapter id mismatch"):
        runner.register_lora(77, "test_adapter", payload)

    assert calls == [77]



def test_call_single_rank_no_shm():
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 1
    runner.rank = 0
    runner.double_it = lambda x: x * 2

    result = runner.call("double_it", 21)
    assert result == 42


def test_call_single_rank_propagates_exception():
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 1
    runner.rank = 0
    runner.bad_method = lambda: (_ for _ in ()).throw(ValueError("oops"))

    with pytest.raises(ValueError, match="oops"):
        runner.call("bad_method")



def test_synchronize_rpc_result_single_rank_success():
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 1
    # No error — should not raise
    runner._synchronize_rpc_result("some_method", None)


def test_synchronize_rpc_result_single_rank_reraises_error():
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 1
    exc = RuntimeError("something broke")

    with pytest.raises(RuntimeError, match="something broke"):
        runner._synchronize_rpc_result("some_method", exc)


def test_synchronize_rpc_result_exit_always_skips_barrier():
    """Even world_size > 1, 'exit' method skips the distributed barrier."""
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2  # Multi-rank but "exit" special case

    # No error, method_name == "exit" → should return without touching dist
    runner._synchronize_rpc_result("exit", None)


def test_synchronize_rpc_result_exit_with_error_raises():
    """world_size > 1 + exit + error → should still re-raise the error."""
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2
    exc = ValueError("exit error")

    with pytest.raises(ValueError, match="exit error"):
        runner._synchronize_rpc_result("exit", exc)



def test_prepare_prefill_context_slot_mapping_correctness(monkeypatch):
    """Verify positions, slot_mapping, cu_seqlens calculated for prefill."""
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.utils.context import LM_LORA_DOMAIN, PROJ_LORA_DOMAIN, DIT_LORA_DOMAIN

    contexts_set = {}

    def fake_set_context(is_prefill, cu_q=None, cu_k=None, max_q=None, max_k=None, slot_mapping=None,
                         context_lens=None, block_tables=None):
        contexts_set["is_prefill"] = is_prefill
        contexts_set["cu_q"] = cu_q.tolist() if cu_q is not None else None
        contexts_set["cu_k"] = cu_k.tolist() if cu_k is not None else None
        contexts_set["slot_mapping"] = slot_mapping.tolist() if slot_mapping is not None else []

    def fake_set_lora_context(ctx, domain=None):
        pass

    monkeypatch.setattr(model_runner, "set_context", fake_set_context)
    monkeypatch.setattr(model_runner, "set_lora_context", fake_set_lora_context)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.block_size = 4
    runner.lora_runtime = __import__(
        "nanovllm_voxcpm.engine.lora_manager", fromlist=["LoRARuntime"]
    ).LoRARuntime(max_loras=0, max_lora_rank=1)
    runner.lora_config = None

    # Single seq: length=6, no prefix cache, block_table=[10, 11]
    seq = model_runner.RunnerTask(
        block_table=[10, 11],
        seq_length=6,
        num_cached_tokens=0,
        block_size=4,
    )
    positions = runner.prepare_prefill_context([seq])

    assert positions.tolist() == [0, 1, 2, 3, 4, 5]
    assert contexts_set["is_prefill"] is True
    assert contexts_set["cu_q"] == [0, 6]
    assert contexts_set["cu_k"] == [0, 6]
    # block 10 → slots 40–43; block 11 → slots 44, 45 (last_block=2)
    assert contexts_set["slot_mapping"] == [40, 41, 42, 43, 44, 45]


def test_prepare_prefill_context_with_prefix_cache(monkeypatch):
    """cu_seqlens_k > cu_seqlens_q triggers block_tables prep (prefix cache path)."""
    import nanovllm_voxcpm.engine.model_runner as model_runner

    bt_calls = []

    def fake_prepare_block_tables(seqs):
        bt_calls.append(len(seqs))
        return torch.zeros(len(seqs), 2, dtype=torch.int32)

    def fake_set_context(*args, **kwargs):
        pass

    def fake_set_lora_context(*args, **kwargs):
        pass

    monkeypatch.setattr(model_runner, "set_context", fake_set_context)
    monkeypatch.setattr(model_runner, "set_lora_context", fake_set_lora_context)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.block_size = 4
    runner.lora_runtime = __import__(
        "nanovllm_voxcpm.engine.lora_manager", fromlist=["LoRARuntime"]
    ).LoRARuntime(max_loras=0, max_lora_rank=1)
    runner.lora_config = None
    runner.prepare_block_tables = fake_prepare_block_tables

    # Prefix cached: seq_length=8, num_cached_tokens=4 → seqlen_q=4, seqlen_k=8
    seq = model_runner.RunnerTask(
        block_table=[10, 11],
        seq_length=8,
        num_cached_tokens=4,
        block_size=4,
    )
    runner.prepare_prefill_context([seq])

    # prefix cache branch should have been triggered
    assert bt_calls == [1]


def test_prepare_prefill_context_warmup_empty_block_table(monkeypatch):
    """Empty block_table (warmup) should not append to slot_mapping."""
    import nanovllm_voxcpm.engine.model_runner as model_runner

    slot_mappings_captured = []

    def fake_set_context(is_prefill, cu_q=None, cu_k=None, max_q=None, max_k=None, slot_mapping=None,
                         context_lens=None, block_tables=None):
        if slot_mapping is not None:
            slot_mappings_captured.append(slot_mapping.tolist())

    def fake_set_lora_context(*args, **kwargs):
        pass

    monkeypatch.setattr(model_runner, "set_context", fake_set_context)
    monkeypatch.setattr(model_runner, "set_lora_context", fake_set_lora_context)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.block_size = 4
    runner.lora_runtime = __import__(
        "nanovllm_voxcpm.engine.lora_manager", fromlist=["LoRARuntime"]
    ).LoRARuntime(max_loras=0, max_lora_rank=1)
    runner.lora_config = None

    # warmup: empty block_table
    seq = model_runner.RunnerTask(
        block_table=[],
        seq_length=4,
        num_cached_tokens=0,
        block_size=4,
    )
    runner.prepare_prefill_context([seq])

    # slot_mapping should be empty (warmup path skips block iteration)
    assert slot_mappings_captured == [[]]



def test_prepare_decode_context_positions_and_slot_mapping(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    captured = {}

    def fake_prepare_block_tables(seqs):
        captured["seqs_count"] = len(seqs)
        return torch.zeros(len(seqs), 2, dtype=torch.int32)

    def fake_set_context(is_prefill, slot_mapping=None, context_lens=None, block_tables=None):
        captured["is_prefill"] = is_prefill
        captured["slot_mapping"] = slot_mapping.tolist() if slot_mapping is not None else []
        captured["context_lens"] = context_lens.tolist() if context_lens is not None else []

    def fake_set_lora_context(*args, **kwargs):
        pass

    monkeypatch.setattr(model_runner, "set_context", fake_set_context)
    monkeypatch.setattr(model_runner, "set_lora_context", fake_set_lora_context)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.block_size = 4
    runner.lora_runtime = __import__(
        "nanovllm_voxcpm.engine.lora_manager", fromlist=["LoRARuntime"]
    ).LoRARuntime(max_loras=0, max_lora_rank=1)
    runner.lora_config = None
    runner.prepare_block_tables = fake_prepare_block_tables

    # seq_length=5, block_table=[0,1], block_size=4
    # last_block_num_tokens = 5 - (2-1)*4 = 1
    # slot = block_table[-1] * block_size + last_block_num_tokens - 1
    #       = 1 * 4 + 1 - 1 = 4
    seq = model_runner.RunnerTask(
        block_table=[0, 1],
        seq_length=5,
        num_cached_tokens=0,
        block_size=4,
    )
    positions = runner.prepare_decode_context([seq])

    assert positions.tolist() == [4]  # seq_length - 1
    assert captured["is_prefill"] is False
    assert captured["slot_mapping"] == [4]
    assert captured["context_lens"] == [5]
    assert captured["seqs_count"] == 1


def test_prepare_decode_context_multiple_seqs(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    def fake_prepare_block_tables(seqs):
        return torch.zeros(len(seqs), 2, dtype=torch.int32)

    def fake_set_context(*args, **kwargs):
        pass

    def fake_set_lora_context(*args, **kwargs):
        pass

    monkeypatch.setattr(model_runner, "set_context", fake_set_context)
    monkeypatch.setattr(model_runner, "set_lora_context", fake_set_lora_context)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.block_size = 4
    runner.lora_runtime = __import__(
        "nanovllm_voxcpm.engine.lora_manager", fromlist=["LoRARuntime"]
    ).LoRARuntime(max_loras=0, max_lora_rank=1)
    runner.lora_config = None
    runner.prepare_block_tables = fake_prepare_block_tables

    seqs = [
        model_runner.RunnerTask(block_table=[0, 1], seq_length=5, num_cached_tokens=0, block_size=4),
        model_runner.RunnerTask(block_table=[2, 3], seq_length=9, num_cached_tokens=0, block_size=4),
    ]
    positions = runner.prepare_decode_context(seqs)
    assert positions.tolist() == [4, 8]



def test_prepare_block_tables_pads_to_max_length(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    # Patch .cuda() to be a no-op on CPU tensors
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)

    runner = object.__new__(model_runner.BaseModelRunner)

    seqs = [
        model_runner.RunnerTask(block_table=[1, 2, 3], seq_length=10, num_cached_tokens=0, block_size=4),
        model_runner.RunnerTask(block_table=[5], seq_length=3, num_cached_tokens=0, block_size=4),
    ]
    result = runner.prepare_block_tables(seqs)

    assert result.shape == (2, 3)
    assert result[0].tolist() == [1, 2, 3]
    assert result[1].tolist() == [5, -1, -1]



def test_loop_runs_method_and_exits():
    """Non-GPU test of the loop() dispatch path."""
    import nanovllm_voxcpm.engine.model_runner as model_runner

    calls = []

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2
    runner.rank = 1

    # Simulate: first call "store_value", then "exit"
    read_sequence = [
        ("store_value", [42]),
        ("exit", []),
    ]
    read_idx = [0]

    def fake_read_shm():
        item = read_sequence[read_idx[0]]
        read_idx[0] += 1
        return item

    def fake_synchronize(method_name, error):
        if error is not None:
            raise error

    runner.read_shm = fake_read_shm
    runner._synchronize_rpc_result = fake_synchronize
    runner.store_value = lambda v: calls.append(v)
    runner.exit = lambda: calls.append("exited")

    runner.loop()

    assert calls == [42, "exited"]


def test_loop_continues_on_method_error_then_rethrows_in_synchronize():
    """Errors from method() are passed to _synchronize_rpc_result, which re-raises."""
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2
    runner.rank = 1

    read_idx = [0]
    sequence = [("bad_method", [])]

    def fake_read_shm():
        item = sequence[read_idx[0]]
        read_idx[0] += 1
        return item

    def fake_synchronize(method_name, error):
        if error is not None:
            raise error
        raise RuntimeError("exit loop sentinel")

    runner.read_shm = fake_read_shm
    runner._synchronize_rpc_result = fake_synchronize
    runner.bad_method = lambda: (_ for _ in ()).throw(ValueError("method failed"))

    with pytest.raises(ValueError, match="method failed"):
        runner.loop()



def test_make_graph_domain_buffers_shapes():
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    buffers = runner._make_graph_domain_buffers(max_rows=8, max_lora_buckets=3)

    assert buffers["token_to_slot"].shape == (8,)
    assert buffers["token_indices_sorted_by_slot"].shape == (8,)
    assert buffers["active_slot_ids"].shape == (3,)
    assert buffers["num_tokens_per_slot"].shape == (3,)
    assert buffers["slot_start_offsets"].shape == (4,)  # max_lora_buckets + 1

    assert (buffers["token_to_slot"] == -1).all()
    assert buffers["active_slot_ids"].tolist() == [-1, 0, 1]



def test_write_shm_inline_small_payload():
    """write_shm should write inline (no file) when data fits in shm."""
    import pickle
    from multiprocessing.shared_memory import SharedMemory
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2
    runner.rank = 0

    class _FakeEvent:
        def set(self):
            pass

    runner.event = [_FakeEvent()]
    runner.shm = SharedMemory(create=True, size=4096)

    try:
        overflow_path = runner.write_shm("my_method", 1, "hello")
        assert overflow_path is None

        n = int.from_bytes(runner.shm.buf[0:4], "little")
        data = pickle.loads(runner.shm.buf[4 : n + 4])
        assert data[0] == "my_method"
        assert data[1] == 1
        assert data[2] == "hello"
    finally:
        runner.shm.close()
        runner.shm.unlink()



def test_call_cleans_up_overflow_path_even_when_file_missing():
    import nanovllm_voxcpm.engine.model_runner as model_runner

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 1
    runner.rank = 0
    runner.my_method = lambda: None

    fake_path = "/tmp/nanovllm_nonexistent_overflow.pkl"
    write_calls = []

    def patched_write_shm(method_name, *args):
        write_calls.append(method_name)
        return fake_path

    runner.write_shm = patched_write_shm
    runner._synchronize_rpc_result = lambda method_name, error: None
    runner.world_size = 2

    runner.call("my_method")
    assert not os.path.exists(fake_path)


def test_synchronize_rpc_result_multi_rank_success(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    reduce_calls = []

    def fake_all_reduce(tensor, op=None):
        reduce_calls.append(int(tensor.item()))

    monkeypatch.setattr(model_runner.dist, "all_reduce", fake_all_reduce)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2

    runner._synchronize_rpc_result("some_method", None)

    assert reduce_calls == [0]


def test_synchronize_rpc_result_multi_rank_remote_failure(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    def fake_all_reduce(tensor, op=None):
        tensor[0] = 1

    monkeypatch.setattr(model_runner.dist, "all_reduce", fake_all_reduce)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2

    with pytest.raises(RuntimeError, match="failed on another rank"):
        runner._synchronize_rpc_result("some_method", None)


def test_synchronize_rpc_result_multi_rank_local_error_reraises(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    def fake_all_reduce(tensor, op=None):
        tensor[0] = 1

    monkeypatch.setattr(model_runner.dist, "all_reduce", fake_all_reduce)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2

    exc = ValueError("local error")
    with pytest.raises(ValueError, match="local error"):
        runner._synchronize_rpc_result("some_method", exc)


def test_build_lora_contexts_with_active_adapter():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRARuntime, LoRAModelPayload
    from nanovllm_voxcpm.utils.context import LM_LORA_DOMAIN, PROJ_LORA_DOMAIN, DIT_LORA_DOMAIN

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.lora_config = None
    runtime = LoRARuntime(max_loras=2, max_lora_rank=4)
    payload = LoRAModelPayload(modules={}, rank=1, alpha=1.0)
    runtime.register_lora("adapter_a", payload, adapter_id=3)
    runner.lora_runtime = runtime

    load_calls = []

    def fake_build_batch_plan(adapter_ids, token_counts, load_lora):
        load_calls.append((adapter_ids, token_counts))
        from nanovllm_voxcpm.engine.lora_manager import LoRABatchPlan
        return LoRABatchPlan(
            adapter_to_slot={3: 0},
            token_to_slot=[0, 0, -1],
            token_indices_sorted_by_slot=[0, 1],
            active_slot_ids=[0],
            num_tokens_per_slot=[2],
            slot_start_offsets=[0, 2],
        )

    runner.lora_runtime.build_batch_plan = fake_build_batch_plan

    seqs = [
        model_runner.RunnerTask(block_table=[0], seq_length=2, num_cached_tokens=0, block_size=4, adapter_id=3),
        model_runner.RunnerTask(block_table=[1], seq_length=1, num_cached_tokens=0, block_size=4, adapter_id=None),
    ]
    token_counts = [2, 1]
    contexts = runner._build_lora_contexts(seqs, token_counts)

    assert LM_LORA_DOMAIN in contexts
    assert PROJ_LORA_DOMAIN in contexts
    assert DIT_LORA_DOMAIN in contexts
    assert len(load_calls) == 1
    assert load_calls[0] == ([3, None], [2, 1])


def test_load_lora_slot_unknown_module_raises():
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.model = nn.Module()
    runner._lora_model_modules_cache = None
    runner._lora_slot_modules = {}

    payload = LoRAModelPayload(
        modules={
            "nonexistent": LoRAModulePayload(
                lora_a=torch.zeros(1, 2),
                lora_b=torch.zeros(2, 1),
                effective_rank=1,
                scaling=1.0,
            )
        },
        rank=1,
        alpha=1.0,
    )

    with pytest.raises(ValueError, match="Unknown LoRA target module"):
        runner._load_lora_slot(0, payload)


def test_load_lora_slot_non_lora_module_raises():
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload, LoRAModulePayload

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.model = nn.Module()
    runner.model.add_module("linear", nn.Linear(2, 2))
    runner._lora_model_modules_cache = None
    runner._lora_slot_modules = {}

    payload = LoRAModelPayload(
        modules={
            "linear": LoRAModulePayload(
                lora_a=torch.zeros(1, 2),
                lora_b=torch.zeros(2, 1),
                effective_rank=1,
                scaling=1.0,
            )
        },
        rank=1,
        alpha=1.0,
    )

    with pytest.raises(ValueError, match="does not support LoRA slots"):
        runner._load_lora_slot(0, payload)


def test_load_lora_slot_initializes_slot_modules_dict():
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.engine.lora_manager import LoRAModelPayload

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.model = nn.Module()
    runner._lora_model_modules_cache = None
    runner._lora_slot_modules = None

    payload = LoRAModelPayload(modules={}, rank=1, alpha=1.0)

    runner._load_lora_slot(0, payload)

    assert isinstance(runner._lora_slot_modules, dict)
    assert runner._lora_slot_modules[0] == []


def test_exit_single_rank_enforce_eager_skips_graphs(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    sync_calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 1
    runner.enforce_eager = True

    runner.exit()

    assert sync_calls == [True]


def test_exit_single_rank_non_eager_deletes_graphs(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 1
    runner.enforce_eager = False
    runner.graphs = {"base": {}, "lora": {}}
    runner.graph_pool = object()

    runner.exit()

    assert not hasattr(runner, "graphs")
    assert not hasattr(runner, "graph_pool")


def _make_domain_vars(max_rows: int, max_lora_buckets: int) -> dict:
    return {
        "token_to_slot": torch.full((max_rows,), -1, dtype=torch.int32),
        "token_indices_sorted_by_slot": torch.arange(max_rows, dtype=torch.int32),
        "active_slot_ids": torch.arange(-1, max_lora_buckets - 1, dtype=torch.int32),
        "num_tokens_per_slot": torch.zeros(max_lora_buckets, dtype=torch.int32),
        "slot_start_offsets": torch.zeros(max_lora_buckets + 1, dtype=torch.int32),
    }


def test_copy_lora_domain_no_lora_flag_resets_to_sentinel():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.utils.context import LoRAContext, LM_LORA_DOMAIN

    runner = object.__new__(model_runner.BaseModelRunner)

    domain_vars = _make_domain_vars(max_rows=4, max_lora_buckets=2)
    graph_vars = {"lora_domains": {LM_LORA_DOMAIN: domain_vars}}

    ctx = LoRAContext(no_lora_flag=True, num_active_loras=0)
    runner._copy_lora_domain_to_graph_vars(graph_vars, LM_LORA_DOMAIN, ctx)

    assert (domain_vars["token_to_slot"] == -1).all()
    assert (domain_vars["num_tokens_per_slot"] == 0).all()
    assert (domain_vars["slot_start_offsets"] == 0).all()


def test_copy_lora_domain_none_token_to_slot_resets():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.utils.context import LoRAContext, LM_LORA_DOMAIN

    runner = object.__new__(model_runner.BaseModelRunner)

    domain_vars = _make_domain_vars(max_rows=4, max_lora_buckets=2)
    graph_vars = {"lora_domains": {LM_LORA_DOMAIN: domain_vars}}

    ctx = LoRAContext(no_lora_flag=False, num_active_loras=0, token_to_slot=None)
    runner._copy_lora_domain_to_graph_vars(graph_vars, LM_LORA_DOMAIN, ctx)

    assert (domain_vars["token_to_slot"] == -1).all()


def test_copy_lora_domain_with_active_lora_writes_metadata():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.utils.context import LoRAContext, LM_LORA_DOMAIN

    runner = object.__new__(model_runner.BaseModelRunner)

    domain_vars = _make_domain_vars(max_rows=4, max_lora_buckets=3)
    graph_vars = {"lora_domains": {LM_LORA_DOMAIN: domain_vars}}

    ctx = LoRAContext(
        no_lora_flag=False,
        num_active_loras=1,
        token_to_slot=torch.tensor([0, 0, -1], dtype=torch.int32),
        token_indices_sorted_by_slot=torch.tensor([0, 1], dtype=torch.int32),
        active_slot_ids=torch.tensor([0], dtype=torch.int32),
        num_tokens_per_slot=torch.tensor([2], dtype=torch.int32),
        slot_start_offsets=torch.tensor([0, 2], dtype=torch.int32),
    )
    runner._copy_lora_domain_to_graph_vars(graph_vars, LM_LORA_DOMAIN, ctx)

    assert domain_vars["token_to_slot"][:3].tolist() == [0, 0, -1]
    assert domain_vars["token_to_slot"][3].item() == -1


def test_copy_lora_domain_partial_fill_pads_remainder_with_minus_one():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.utils.context import LoRAContext, LM_LORA_DOMAIN

    runner = object.__new__(model_runner.BaseModelRunner)

    domain_vars = _make_domain_vars(max_rows=8, max_lora_buckets=3)
    graph_vars = {"lora_domains": {LM_LORA_DOMAIN: domain_vars}}

    ctx = LoRAContext(
        no_lora_flag=False,
        num_active_loras=1,
        token_to_slot=torch.tensor([1, 1], dtype=torch.int32),
        token_indices_sorted_by_slot=torch.tensor([0, 1], dtype=torch.int32),
        active_slot_ids=torch.tensor([1], dtype=torch.int32),
        num_tokens_per_slot=torch.tensor([2], dtype=torch.int32),
        slot_start_offsets=torch.tensor([0, 2], dtype=torch.int32),
    )
    runner._copy_lora_domain_to_graph_vars(graph_vars, LM_LORA_DOMAIN, ctx)

    assert (domain_vars["token_to_slot"][2:] == -1).all()


def test_set_graph_lora_contexts_calls_set_lora_context_for_all_domains(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.utils.context import (
        LoRAContext,
        LM_LORA_DOMAIN,
        PROJ_LORA_DOMAIN,
        DIT_LORA_DOMAIN,
    )

    set_lora_calls = []

    def fake_set_lora_context(ctx, domain=None):
        set_lora_calls.append(domain)

    monkeypatch.setattr(model_runner, "set_lora_context", fake_set_lora_context)

    runner = object.__new__(model_runner.BaseModelRunner)

    max_rows = 4
    max_lora_buckets = 2
    lora_domains = {
        LM_LORA_DOMAIN: _make_domain_vars(max_rows, max_lora_buckets),
        PROJ_LORA_DOMAIN: _make_domain_vars(max_rows, max_lora_buckets),
        DIT_LORA_DOMAIN: _make_domain_vars(max_rows, max_lora_buckets),
    }
    graph_vars = {"lora_domains": lora_domains}

    empty_ctx = LoRAContext(no_lora_flag=True, num_active_loras=0)
    contexts = {
        LM_LORA_DOMAIN: empty_ctx,
        PROJ_LORA_DOMAIN: empty_ctx,
        DIT_LORA_DOMAIN: empty_ctx,
    }
    runner._set_graph_lora_contexts(graph_vars, contexts)

    assert set(set_lora_calls) == {LM_LORA_DOMAIN, PROJ_LORA_DOMAIN, DIT_LORA_DOMAIN}
    assert len(set_lora_calls) == 3


def test_exit_multi_rank_rank0_unlinks_shm(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    barrier_calls = []
    monkeypatch.setattr(model_runner.dist, "barrier", lambda: barrier_calls.append(True))
    monkeypatch.setattr(model_runner.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    close_calls = []
    unlink_calls = []

    class _FakeShm:
        def close(self):
            close_calls.append(True)

        def unlink(self):
            unlink_calls.append(True)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2
    runner.rank = 0
    runner.enforce_eager = True
    runner.shm = _FakeShm()

    runner.exit()

    assert close_calls == [True]
    assert unlink_calls == [True]
    assert len(barrier_calls) == 1


def test_exit_multi_rank_rank1_no_unlink(monkeypatch):
    import nanovllm_voxcpm.engine.model_runner as model_runner

    monkeypatch.setattr(model_runner.dist, "barrier", lambda: None)
    monkeypatch.setattr(model_runner.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    close_calls = []
    unlink_calls = []

    class _FakeShm:
        def close(self):
            close_calls.append(True)

        def unlink(self):
            unlink_calls.append(True)

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.world_size = 2
    runner.rank = 1
    runner.enforce_eager = True
    runner.shm = _FakeShm()

    runner.exit()

    assert close_calls == [True]
    assert unlink_calls == []


def test_copy_lora_domain_exact_fill_no_padding():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.utils.context import LoRAContext, LM_LORA_DOMAIN

    runner = object.__new__(model_runner.BaseModelRunner)

    domain_vars = _make_domain_vars(max_rows=3, max_lora_buckets=3)
    graph_vars = {"lora_domains": {LM_LORA_DOMAIN: domain_vars}}

    ctx = LoRAContext(
        no_lora_flag=False,
        num_active_loras=1,
        token_to_slot=torch.tensor([0, 0, 0], dtype=torch.int32),
        token_indices_sorted_by_slot=None,
        active_slot_ids=None,
        num_tokens_per_slot=None,
        slot_start_offsets=torch.tensor([0, 3], dtype=torch.int32),
    )
    runner._copy_lora_domain_to_graph_vars(graph_vars, LM_LORA_DOMAIN, ctx)

    assert domain_vars["token_to_slot"].tolist() == [0, 0, 0]


def test_copy_lora_domain_no_active_slot_ids_skips_scatter():
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.utils.context import LoRAContext, LM_LORA_DOMAIN

    runner = object.__new__(model_runner.BaseModelRunner)

    domain_vars = _make_domain_vars(max_rows=4, max_lora_buckets=2)
    initial_num_tokens = domain_vars["num_tokens_per_slot"].clone()

    graph_vars = {"lora_domains": {LM_LORA_DOMAIN: domain_vars}}

    ctx = LoRAContext(
        no_lora_flag=False,
        num_active_loras=0,
        token_to_slot=torch.tensor([1], dtype=torch.int32),
        token_indices_sorted_by_slot=None,
        active_slot_ids=None,
        num_tokens_per_slot=None,
        slot_start_offsets=None,
    )
    runner._copy_lora_domain_to_graph_vars(graph_vars, LM_LORA_DOMAIN, ctx)

    assert (domain_vars["num_tokens_per_slot"] == 0).all()



def test_allocate_kv_cache_env_override_negative_raises():
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner

    class _Runner(model_runner.BaseModelRunner):
        @property
        def dtype(self):
            return torch.float16

    runner = object.__new__(_Runner)
    runner.model = nn.Module()
    runner.block_size = 16

    class _FakeConfig:
        num_kvcache_blocks = 0
        gpu_memory_utilization = 0.9

    runner._config = _FakeConfig()

    os.environ[model_runner._NUM_KVCACHE_BLOCKS_ENV] = "0"
    try:
        with pytest.raises(ValueError, match="must be greater than 0"):
            runner.allocate_kv_cache()
    finally:
        del os.environ[model_runner._NUM_KVCACHE_BLOCKS_ENV]


def test_allocate_kv_cache_env_override_sets_num_blocks():
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner

    class _Runner(model_runner.BaseModelRunner):
        @property
        def dtype(self):
            return torch.float16

    runner = object.__new__(_Runner)
    runner.model = nn.Module()
    runner.block_size = 16

    class _FakeConfig:
        num_kvcache_blocks = 0
        gpu_memory_utilization = 0.9

    runner._config = _FakeConfig()

    os.environ[model_runner._NUM_KVCACHE_BLOCKS_ENV] = "42"
    try:
        runner.allocate_kv_cache()
    finally:
        del os.environ[model_runner._NUM_KVCACHE_BLOCKS_ENV]

    assert runner._config.num_kvcache_blocks == 42


def test_allocate_kv_cache_auto_sizing_no_budget_raises(monkeypatch):
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner

    class _Runner(model_runner.BaseModelRunner):
        @property
        def dtype(self):
            return torch.float16

    runner = object.__new__(_Runner)
    runner.block_size = 16

    class _FakeConfig:
        num_kvcache_blocks = 0
        gpu_memory_utilization = 0.9

    runner._config = _FakeConfig()

    os.environ.pop(model_runner._NUM_KVCACHE_BLOCKS_ENV, None)

    class _FakeAttn(nn.Module):
        is_causal = True
        num_kv_heads = 2
        head_dim = 64

    monkeypatch.setattr(model_runner, "Attention", _FakeAttn)

    runner.model = nn.Module()
    runner.model.add_module("attn", _FakeAttn())

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (0, 1000))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {"allocated_bytes.all.peak": 1000, "allocated_bytes.all.current": 900},
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 900)

    with pytest.raises(RuntimeError, match="no safe memory budget"):
        runner.allocate_kv_cache()


@pytest.mark.gpu  # requires real CUDA mem_get_info + attention layers to compute block size
def test_allocate_kv_cache_auto_sizing_positive_blocks(monkeypatch):
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner

    class _Runner(model_runner.BaseModelRunner):
        @property
        def dtype(self):
            return torch.float16

    runner = object.__new__(_Runner)
    runner.model = nn.Module()
    runner.block_size = 16

    class _FakeConfig:
        num_kvcache_blocks = 0
        gpu_memory_utilization = 0.9

    runner._config = _FakeConfig()

    os.environ.pop(model_runner._NUM_KVCACHE_BLOCKS_ENV, None)

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (512 * 1024 * 1024, 1024 * 1024 * 1024))
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda: {"allocated_bytes.all.peak": 100 * 1024 * 1024, "allocated_bytes.all.current": 80 * 1024 * 1024},
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 100 * 1024 * 1024)

    runner.allocate_kv_cache()

    assert runner._config.num_kvcache_blocks > 0


def test_allocate_kv_cache_pre_set_num_blocks_skips_auto(monkeypatch):
    import torch.nn as nn
    import nanovllm_voxcpm.engine.model_runner as model_runner

    class _Runner(model_runner.BaseModelRunner):
        @property
        def dtype(self):
            return torch.float16

    runner = object.__new__(_Runner)
    runner.model = nn.Module()
    runner.block_size = 16

    class _FakeConfig:
        num_kvcache_blocks = 100
        gpu_memory_utilization = 0.9

    runner._config = _FakeConfig()

    os.environ.pop(model_runner._NUM_KVCACHE_BLOCKS_ENV, None)

    cuda_called = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: cuda_called.append(True) or (0, 0))

    runner.allocate_kv_cache()

    assert cuda_called == []
    assert runner._config.num_kvcache_blocks == 100
