"""Unit tests for nanovllm_voxcpm.models.voxcpm.runner and runner_utils.

All tests are CPU-only (no real GPU workers, no model weights).
The __new__ + attribute injection idiom is used (same as test_voxcpm_engine_max_model_len.py).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_payload(T=3, P=4, D=8, seed=None, seed_step=0, padding_decode=None, temperature=1.0, cfg_value=1.5):
    from nanovllm_voxcpm.models.voxcpm.runner import VoxCPMPayload

    return VoxCPMPayload(
        text_tokens=np.zeros(T, dtype=np.int64),
        feats=np.zeros((T, P, D), dtype=np.float32),
        feat_masks=np.zeros(T, dtype=np.bool_),
        temperature=temperature,
        cfg_value=cfg_value,
        padding_decode=padding_decode,
        seed=seed,
        seed_step=seed_step,
    )


class _FakeTask:
    def __init__(self, payload):
        self.custom_payload = payload


def _make_runner(patch_size=4, feat_dim=8):
    from nanovllm_voxcpm.models.voxcpm.runner import VoxCPMRunner

    r = VoxCPMRunner.__new__(VoxCPMRunner)
    r.patch_size = patch_size
    r.feat_dim = feat_dim
    return r


def _install_cpu_tensor_shims(monkeypatch):
    original_randn = torch.randn
    original_tensor = torch.tensor
    original_zeros = torch.zeros

    def cpu_randn(*args, **kwargs):
        kwargs.pop("device", None)
        return original_randn(*args, **kwargs)

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_tensor(*args, **kwargs)

    def cpu_zeros(*args, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=True: self, raising=False)
    monkeypatch.setattr(torch, "randn", cpu_randn)
    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch, "zeros", cpu_zeros)


# ---------------------------------------------------------------------------
# VoxCPMPayload dataclass
# ---------------------------------------------------------------------------


class TestVoxCPMPayload:
    def test_default_values(self):
        from nanovllm_voxcpm.models.voxcpm.runner import VoxCPMPayload

        p = VoxCPMPayload()
        assert p.text_tokens is None
        assert p.feats is None
        assert p.feat_masks is None
        assert p.temperature == 1.0
        assert p.cfg_value == 1.0
        assert p.padding_decode is None
        assert p.seed is None
        assert p.seed_step == 0

    def test_field_assignment(self):
        from nanovllm_voxcpm.models.voxcpm.runner import VoxCPMPayload

        arr = np.arange(5, dtype=np.int64)
        p = VoxCPMPayload(text_tokens=arr, temperature=0.7, cfg_value=2.0, seed=42, seed_step=3)
        assert p.temperature == 0.7
        assert p.cfg_value == 2.0
        assert p.seed == 42
        assert p.seed_step == 3
        np.testing.assert_array_equal(p.text_tokens, arr)


# ---------------------------------------------------------------------------
# VoxCPMRunner shape methods (no GPU)
# ---------------------------------------------------------------------------


class TestVoxCPMRunnerDummyMethods:
    def test_dtype_is_bfloat16(self):
        from nanovllm_voxcpm.models.voxcpm.runner import VoxCPMRunner

        r = VoxCPMRunner.__new__(VoxCPMRunner)
        assert r.dtype is torch.bfloat16

    def test_dit_lora_seq_len_offset(self):
        from nanovllm_voxcpm.models.voxcpm.runner import VoxCPMRunner

        assert VoxCPMRunner.dit_lora_seq_len_offset == 1

    def test_make_dummy_inputs_keys(self):
        r = _make_runner(patch_size=4, feat_dim=8)
        result = r.make_dummy_inputs(batch_size=2, length=3)
        assert set(result.keys()) == {"text_tokens", "feat", "feat_mask", "temperature", "cfg_value", "z_noise"}

    def test_make_dummy_inputs_text_tokens_shape(self):
        r = _make_runner(patch_size=4, feat_dim=8)
        result = r.make_dummy_inputs(batch_size=2, length=5)
        assert result["text_tokens"].shape == (10,)
        assert result["text_tokens"].dtype == torch.int64

    def test_make_dummy_inputs_feat_shape(self):
        r = _make_runner(patch_size=4, feat_dim=8)
        result = r.make_dummy_inputs(batch_size=3, length=2)
        assert result["feat"].shape == (6, 4, 8)

    def test_make_dummy_inputs_feat_mask_shape(self):
        r = _make_runner(patch_size=4, feat_dim=8)
        result = r.make_dummy_inputs(batch_size=3, length=2)
        assert result["feat_mask"].shape == (6,)
        assert result["feat_mask"].dtype == torch.bool

    def test_make_dummy_inputs_temperature_shape(self):
        r = _make_runner(patch_size=4, feat_dim=8)
        result = r.make_dummy_inputs(batch_size=5, length=1)
        assert result["temperature"].shape == (5,)

    def test_make_dummy_inputs_cfg_value_shape(self):
        r = _make_runner(patch_size=4, feat_dim=8)
        result = r.make_dummy_inputs(batch_size=5, length=1)
        assert result["cfg_value"].shape == (5,)

    def test_make_dummy_inputs_z_noise_shape(self):
        r = _make_runner(patch_size=4, feat_dim=8)
        result = r.make_dummy_inputs(batch_size=2, length=3)
        assert result["z_noise"].shape == (2, 8, 4)
        assert result["z_noise"].dtype == torch.bfloat16

    def test_make_dummy_outputs_keys(self):
        r = _make_runner(patch_size=4, feat_dim=8)
        result = r.make_dummy_outputs(batch_size=2)
        assert set(result.keys()) == {"latents", "stop_flag"}

    def test_make_dummy_outputs_latents_shape(self):
        r = _make_runner(patch_size=4, feat_dim=8)
        result = r.make_dummy_outputs(batch_size=3)
        assert result["latents"].shape == (3, 4, 8)
        assert result["latents"].dtype == torch.bfloat16

    def test_make_dummy_outputs_stop_flag_shape(self):
        r = _make_runner(patch_size=4, feat_dim=8)
        result = r.make_dummy_outputs(batch_size=3)
        assert result["stop_flag"].shape == (3,)
        assert result["stop_flag"].dtype == torch.int64

    def test_make_dummy_outputs_all_zeros(self):
        r = _make_runner(patch_size=2, feat_dim=4)
        out = r.make_dummy_outputs(batch_size=2)
        assert out["latents"].sum().item() == 0.0
        assert out["stop_flag"].sum().item() == 0

    @pytest.mark.parametrize("is_prefill", [True, False])
    def test_run_assembles_inputs_and_outputs_on_cpu(self, monkeypatch, is_prefill):
        _install_cpu_tensor_shims(monkeypatch)
        runner = _make_runner(patch_size=2, feat_dim=3)
        prepared = []
        captured = {}
        runner.prepare_prefill_context = lambda seqs: prepared.append("prefill") or torch.tensor([0, 1])
        runner.prepare_decode_context = lambda seqs: prepared.append("decode") or torch.tensor([2, 3])

        latents = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)

        def run_model(inputs, actual_is_prefill):
            captured.update(inputs)
            captured["is_prefill"] = actual_is_prefill
            return {"latents": latents, "stop_flag": torch.tensor([0, 1])}

        runner.run_model = run_model

        class FakeVAE:
            chunk_size = 2

            def decode(self, inputs):
                captured["vae_inputs"] = inputs
                return torch.arange(24, dtype=torch.float32).reshape(2, 1, 12)

        runner.vae = FakeVAE()
        padding = np.full((1, 3), 7.0, dtype=np.float32)
        tasks = [
            _FakeTask(_make_payload(T=1, P=2, D=3, temperature=0.5, cfg_value=1.5)),
            _FakeTask(_make_payload(T=1, P=2, D=3, padding_decode=padding, temperature=0.8, cfg_value=2.0)),
        ]

        outputs = runner.run(tasks, is_prefill=is_prefill)

        assert prepared == ["prefill" if is_prefill else "decode"]
        assert captured["is_prefill"] is is_prefill
        assert captured["text_tokens"].shape == (2,)
        assert captured["feat"].shape == (2, 2, 3)
        assert captured["temperature"].tolist() == pytest.approx([0.5, 0.8], abs=0.01)
        assert captured["cfg_value"].tolist() == pytest.approx([1.5, 2.0], abs=0.01)
        assert captured["z_noise"].shape == (2, 3, 2)
        assert captured["vae_inputs"].shape == (2, 3, 3)
        assert captured["vae_inputs"][1, :, 0].tolist() == [7.0, 7.0, 7.0]
        assert captured["vae_inputs"][1, :, 1].tolist() == [6.0, 7.0, 8.0]
        assert captured["vae_inputs"][1, :, 2].tolist() == [9.0, 10.0, 11.0]
        assert [output["stop_flag"] for output in outputs] == [0, 1]
        np.testing.assert_array_equal(outputs[0]["latents"], latents[0].numpy())
        np.testing.assert_array_equal(outputs[0]["waveforms"], np.arange(4, dtype=np.float32))
        np.testing.assert_array_equal(outputs[1]["waveforms"], np.arange(14, 18, dtype=np.float32))

    def test_run_rejects_mismatched_payload_shapes_before_cuda(self):
        runner = _make_runner(patch_size=2, feat_dim=3)
        runner.prepare_decode_context = lambda seqs: torch.tensor([0])
        payload = _make_payload(T=2, P=2, D=3)
        payload.feats = np.zeros((1, 2, 3), dtype=np.float32)

        with pytest.raises(AssertionError):
            runner.run([_FakeTask(payload)], is_prefill=False)
