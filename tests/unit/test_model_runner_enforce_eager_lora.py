import pytest

torch = pytest.importorskip("torch")


def test_run_model_enforce_eager_with_active_lora_does_not_require_graphs():
    """Regression test for issue #71.

    When ``enforce_eager=True``, ``capture_cudagraph()`` is never called and the
    runner never gains a ``graphs`` attribute. ``run_model`` must still be able
    to serve requests with an active LoRA context by falling back to the eager
    ``self.model(...)`` path, instead of crashing with::

        AttributeError: 'VoxCPM2Runner' object has no attribute 'graphs'
    """
    import nanovllm_voxcpm.engine.model_runner as model_runner
    from nanovllm_voxcpm.utils.context import (
        LM_LORA_DOMAIN,
        reset_all_contexts,
        set_context,
        set_lora_context_from_token_to_slot,
    )

    runner = object.__new__(model_runner.BaseModelRunner)
    runner.enforce_eager = True
    # Intentionally do NOT set ``runner.graphs`` — this mirrors the eager-mode
    # runtime state, where ``capture_cudagraph()`` was skipped.
    assert not hasattr(runner, "graphs")

    captured: dict[str, object] = {}

    def _fake_model(**inputs):
        captured["called"] = True
        captured["inputs"] = inputs
        return {"latents": torch.zeros(1), "stop_flag": torch.zeros(1, dtype=torch.int64)}

    runner.model = _fake_model

    inputs = {"positions": torch.zeros(1, dtype=torch.int64)}

    set_context(False, slot_mapping=torch.zeros(1, dtype=torch.int32))
    # Active LoRA context: one decode token mapped to slot 0.
    set_lora_context_from_token_to_slot(torch.tensor([0], dtype=torch.int32), domain=LM_LORA_DOMAIN)
    try:
        output = runner.run_model(inputs, is_prefill=False)
    finally:
        reset_all_contexts()

    assert captured.get("called") is True
    assert "latents" in output and "stop_flag" in output
