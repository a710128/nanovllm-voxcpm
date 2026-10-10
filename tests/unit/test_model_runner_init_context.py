from contextlib import nullcontext
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("caller_device", [None, "cpu", "meta"])
@pytest.mark.parametrize("failure_stage", [None, "model", "warmup", "cache", "capture"])
def test_runner_initialization_restores_torch_context(monkeypatch, caller_device, failure_stage):
    from torch.overrides import _get_current_function_mode_stack

    from nanovllm_voxcpm.engine.model_runner import BaseModelRunner

    selected_devices = []
    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)
    stages = []

    class Runner(BaseModelRunner):
        dtype = torch.bfloat16

        def record(self, stage):
            # Inspect the real mode without allocating a CUDA tensor, so CPU CI
            # catches both a leaked CPU mode and a lost caller-owned context.
            stack = _get_current_function_mode_stack()
            assert stack[-1].device == torch.device("cuda")
            assert torch.get_default_dtype() == self.dtype
            stages.append(stage)
            if stage == failure_stage:
                torch.set_default_dtype(torch.float64)
                raise RuntimeError("initialization failed")

        def init_model(self, model_config, model_path):
            self.record("model")

        def warmup_model(self):
            self.record("warmup")

        def allocate_kv_cache(self):
            self.record("cache")

        def capture_cudagraph(self):
            self.record("capture")

    config = SimpleNamespace(
        kvcache_block_size=256,
        enforce_eager=False,
        tensor_parallel_size=1,
        lora_config=None,
        model_config=None,
        model="unused",
    )
    with torch.device(caller_device) if caller_device else nullcontext():
        previous_modes = _get_current_function_mode_stack()
        previous_dtype = torch.get_default_dtype()
        expected_error = pytest.raises(RuntimeError, match="initialization failed") if failure_stage else nullcontext()
        with expected_error:
            Runner(config, rank=0, device_idx=1, distributed_port=None, event=None)
        assert _get_current_function_mode_stack() == previous_modes
        assert torch.get_default_dtype() == previous_dtype
        assert torch.empty(0).device.type == (caller_device or "cpu")

    assert selected_devices == [1]
    expected_stages = ["model", "warmup", "cache", "capture"]
    if failure_stage:
        expected_stages = expected_stages[: expected_stages.index(failure_stage) + 1]
    assert stages == expected_stages
