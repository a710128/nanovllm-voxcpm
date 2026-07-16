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


# ---------------------------------------------------------------------------
# runner_utils: collect_seeded_rows
# ---------------------------------------------------------------------------


class TestCollectSeededRows:
    def test_empty_list(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        assert collect_seeded_rows([]) == []

    def test_no_seeds(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        tasks = [_FakeTask(_make_payload(seed=None)) for _ in range(3)]
        assert collect_seeded_rows(tasks) == []

    def test_negative_seed_excluded(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        tasks = [_FakeTask(_make_payload(seed=-1, seed_step=0))]
        assert collect_seeded_rows(tasks) == []

    def test_zero_seed_included(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        tasks = [_FakeTask(_make_payload(seed=0, seed_step=5))]
        result = collect_seeded_rows(tasks)
        assert result == [(0, 0, 5)]

    def test_positive_seed_included(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        tasks = [_FakeTask(_make_payload(seed=42, seed_step=3))]
        result = collect_seeded_rows(tasks)
        assert result == [(0, 42, 3)]

    def test_mixed_seeds_correct_indices(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        tasks = [
            _FakeTask(_make_payload(seed=None)),
            _FakeTask(_make_payload(seed=10, seed_step=1)),
            _FakeTask(_make_payload(seed=-5)),
            _FakeTask(_make_payload(seed=99, seed_step=7)),
        ]
        result = collect_seeded_rows(tasks)
        assert result == [(1, 10, 1), (3, 99, 7)]

    def test_all_seeded(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        tasks = [_FakeTask(_make_payload(seed=i, seed_step=i * 2)) for i in range(4)]
        result = collect_seeded_rows(tasks)
        assert len(result) == 4
        for i, (idx, seed_val, step) in enumerate(result):
            assert idx == i
            assert seed_val == i
            assert step == i * 2


# ---------------------------------------------------------------------------
# runner_utils: compute_pad_lengths
# ---------------------------------------------------------------------------


class TestComputePadLengths:
    def test_all_none(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import compute_pad_lengths

        tasks = [_FakeTask(_make_payload(padding_decode=None)) for _ in range(3)]
        assert compute_pad_lengths(tasks) == [0, 0, 0]

    def test_mixed_padding(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import compute_pad_lengths

        pad2 = np.zeros((2, 8), dtype=np.float32)
        pad5 = np.zeros((5, 8), dtype=np.float32)
        tasks = [
            _FakeTask(_make_payload(padding_decode=None)),
            _FakeTask(_make_payload(padding_decode=pad2)),
            _FakeTask(_make_payload(padding_decode=pad5)),
        ]
        assert compute_pad_lengths(tasks) == [0, 2, 5]

    def test_single_with_padding(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import compute_pad_lengths

        pad = np.zeros((7, 8), dtype=np.float32)
        tasks = [_FakeTask(_make_payload(padding_decode=pad))]
        assert compute_pad_lengths(tasks) == [7]


# ---------------------------------------------------------------------------
# runner_utils: assemble_batch_inputs
# ---------------------------------------------------------------------------


class TestAssembleBatchInputs:
    def test_single_seq_shapes(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_batch_inputs

        tasks = [_FakeTask(_make_payload(T=3, P=4, D=8, temperature=0.5, cfg_value=2.0))]
        tt, feats, masks, temps, cfgs = assemble_batch_inputs(tasks)
        assert tt.shape == (3,)
        assert feats.shape == (3, 4, 8)
        assert masks.shape == (3,)
        assert temps == [0.5]
        assert cfgs == [2.0]

    def test_two_seqs_concatenated(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_batch_inputs

        p1 = _make_payload(T=2, P=4, D=8, temperature=1.0, cfg_value=1.0)
        p2 = _make_payload(T=3, P=4, D=8, temperature=0.8, cfg_value=1.5)
        tasks = [_FakeTask(p1), _FakeTask(p2)]
        tt, feats, masks, temps, cfgs = assemble_batch_inputs(tasks)
        assert tt.shape == (5,)
        assert feats.shape == (5, 4, 8)
        assert masks.shape == (5,)
        assert temps == [1.0, 0.8]
        assert cfgs == [1.0, 1.5]

    def test_values_preserved(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_batch_inputs
        from nanovllm_voxcpm.models.voxcpm.runner import VoxCPMPayload

        arr = np.array([7, 8, 9], dtype=np.int64)
        p = VoxCPMPayload(
            text_tokens=arr,
            feats=np.ones((3, 2, 4), dtype=np.float32),
            feat_masks=np.array([True, False, True], dtype=np.bool_),
            temperature=0.3,
            cfg_value=3.0,
        )
        tt, feats, masks, temps, cfgs = assemble_batch_inputs([_FakeTask(p)])
        np.testing.assert_array_equal(tt, arr)
        assert temps == [0.3]
        assert cfgs == [3.0]
        np.testing.assert_array_equal(masks, [True, False, True])


# ---------------------------------------------------------------------------
# runner_utils: slice_waveforms
# ---------------------------------------------------------------------------


class TestSliceWaveforms:
    def test_no_padding(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import slice_waveforms

        patch_size = 2
        chunk_size = 4
        bsz = 2
        total = patch_size * chunk_size
        vae_out = np.arange(bsz * total, dtype=np.float32).reshape(bsz, total)
        result = slice_waveforms(vae_out, [0, 0], patch_size, chunk_size)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], vae_out[0, :total])
        np.testing.assert_array_equal(result[1], vae_out[1, :total])

    def test_with_padding_slices_correctly(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import slice_waveforms

        patch_size = 1
        chunk_size = 3
        pad = 2
        total = (pad + patch_size) * chunk_size
        vae_out = np.arange(total, dtype=np.float32).reshape(1, total)
        result = slice_waveforms(vae_out, [pad], patch_size, chunk_size)
        expected = vae_out[0, pad * chunk_size : (pad + patch_size) * chunk_size]
        np.testing.assert_array_equal(result[0], expected)

    def test_mixed_padding_batch(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import slice_waveforms

        patch_size = 2
        chunk_size = 5
        pad_lengths = [0, 3]
        max_pad = max(pad_lengths)
        total = (max_pad + patch_size) * chunk_size
        bsz = 2
        vae_out = np.ones((bsz, total), dtype=np.float32)
        result = slice_waveforms(vae_out, pad_lengths, patch_size, chunk_size)
        assert len(result) == 2
        assert result[0].shape == (patch_size * chunk_size,)
        assert result[1].shape == (patch_size * chunk_size,)


# ---------------------------------------------------------------------------
# runner_utils: assemble_run_outputs
# ---------------------------------------------------------------------------


class TestAssembleRunOutputs:
    def test_output_structure(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_run_outputs

        latents = np.zeros((2, 4, 8), dtype=np.float32)
        stop_flags = [0, 1]
        waveforms = [np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float32)]
        result = assemble_run_outputs(latents, stop_flags, waveforms)
        assert len(result) == 2
        assert set(result[0].keys()) == {"latents", "stop_flag", "waveforms"}
        assert set(result[1].keys()) == {"latents", "stop_flag", "waveforms"}

    def test_stop_flag_values(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_run_outputs

        latents = np.zeros((2, 4, 8), dtype=np.float32)
        stop_flags = [0, 1]
        waveforms = [np.zeros(5), np.zeros(5)]
        result = assemble_run_outputs(latents, stop_flags, waveforms)
        assert result[0]["stop_flag"] == 0
        assert result[1]["stop_flag"] == 1

    def test_latents_per_sequence(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_run_outputs

        latents = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
        result = assemble_run_outputs(latents, [0, 0], [np.zeros(5), np.zeros(5)])
        np.testing.assert_array_equal(result[0]["latents"], latents[0])
        np.testing.assert_array_equal(result[1]["latents"], latents[1])

    def test_waveforms_per_sequence(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_run_outputs

        wav0 = np.array([1.0, 2.0], dtype=np.float32)
        wav1 = np.array([3.0, 4.0], dtype=np.float32)
        result = assemble_run_outputs(np.zeros((2, 2, 4)), [0, 0], [wav0, wav1])
        np.testing.assert_array_equal(result[0]["waveforms"], wav0)
        np.testing.assert_array_equal(result[1]["waveforms"], wav1)

    def test_single_sequence(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_run_outputs

        latents = np.ones((1, 3, 6), dtype=np.float32)
        result = assemble_run_outputs(latents, [1], [np.zeros(8)])
        assert len(result) == 1
        assert result[0]["stop_flag"] == 1
