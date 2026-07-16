from __future__ import annotations

import numpy as np


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

        assert collect_seeded_rows([_FakeTask(_make_payload(seed=-1))]) == []

    def test_seeded_rows_preserve_indices_and_steps(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        tasks = [
            _FakeTask(_make_payload(seed=0, seed_step=5)),
            _FakeTask(_make_payload(seed=None)),
            _FakeTask(_make_payload(seed=42, seed_step=3)),
        ]
        assert collect_seeded_rows(tasks) == [(0, 0, 5), (2, 42, 3)]

    def test_zero_seed_included(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        assert collect_seeded_rows([_FakeTask(_make_payload(seed=0, seed_step=5))]) == [(0, 0, 5)]

    def test_positive_seed_included(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        assert collect_seeded_rows([_FakeTask(_make_payload(seed=42, seed_step=3))]) == [(0, 42, 3)]

    def test_mixed_seeds_correct_indices(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        tasks = [
            _FakeTask(_make_payload(seed=None)),
            _FakeTask(_make_payload(seed=10, seed_step=1)),
            _FakeTask(_make_payload(seed=-5)),
            _FakeTask(_make_payload(seed=99, seed_step=7)),
        ]
        assert collect_seeded_rows(tasks) == [(1, 10, 1), (3, 99, 7)]

    def test_all_seeded(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import collect_seeded_rows

        tasks = [_FakeTask(_make_payload(seed=i, seed_step=i * 2)) for i in range(4)]
        result = collect_seeded_rows(tasks)
        assert len(result) == 4
        for i, (index, seed, step) in enumerate(result):
            assert index == i
            assert seed == i
            assert step == i * 2


class TestComputePadLengths:
    def test_all_none(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import compute_pad_lengths

        tasks = [_FakeTask(_make_payload(padding_decode=None)) for _ in range(3)]
        assert compute_pad_lengths(tasks) == [0, 0, 0]

    def test_mixed_padding(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import compute_pad_lengths

        tasks = [
            _FakeTask(_make_payload(padding_decode=None)),
            _FakeTask(_make_payload(padding_decode=np.zeros((2, 8), dtype=np.float32))),
            _FakeTask(_make_payload(padding_decode=np.zeros((5, 8), dtype=np.float32))),
        ]
        assert compute_pad_lengths(tasks) == [0, 2, 5]

    def test_single_with_padding(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import compute_pad_lengths

        tasks = [_FakeTask(_make_payload(padding_decode=np.zeros((7, 8), dtype=np.float32)))]
        assert compute_pad_lengths(tasks) == [7]


class TestAssembleBatchInputs:
    def test_single_seq_shapes(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_batch_inputs

        result = assemble_batch_inputs([_FakeTask(_make_payload(T=3, temperature=0.5, cfg_value=2.0))])
        text_tokens, feats, masks, temperatures, cfg_values = result
        assert text_tokens.shape == (3,)
        assert feats.shape == (3, 4, 8)
        assert masks.shape == (3,)
        assert temperatures == [0.5]
        assert cfg_values == [2.0]

    def test_two_seqs_concatenated(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_batch_inputs

        tasks = [
            _FakeTask(_make_payload(T=2, temperature=1.0, cfg_value=1.0)),
            _FakeTask(_make_payload(T=3, temperature=0.8, cfg_value=1.5)),
        ]
        text_tokens, feats, masks, temperatures, cfg_values = assemble_batch_inputs(tasks)
        assert text_tokens.shape == (5,)
        assert feats.shape == (5, 4, 8)
        assert masks.shape == (5,)
        assert temperatures == [1.0, 0.8]
        assert cfg_values == [1.0, 1.5]

    def test_values_preserved(self):
        from nanovllm_voxcpm.models.voxcpm.runner import VoxCPMPayload
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_batch_inputs

        text_tokens = np.array([7, 8, 9], dtype=np.int64)
        payload = VoxCPMPayload(
            text_tokens=text_tokens,
            feats=np.ones((3, 2, 4), dtype=np.float32),
            feat_masks=np.array([True, False, True], dtype=np.bool_),
            temperature=0.3,
            cfg_value=3.0,
        )
        actual_tokens, _, masks, temperatures, cfg_values = assemble_batch_inputs([_FakeTask(payload)])
        np.testing.assert_array_equal(actual_tokens, text_tokens)
        np.testing.assert_array_equal(masks, [True, False, True])
        assert temperatures == [0.3]
        assert cfg_values == [3.0]


class TestSliceWaveforms:
    def test_no_padding(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import slice_waveforms

        vae_outputs = np.arange(16, dtype=np.float32).reshape(2, 8)
        result = slice_waveforms(vae_outputs, [0, 0], patch_size=2, chunk_size=4)
        np.testing.assert_array_equal(result[0], vae_outputs[0])
        np.testing.assert_array_equal(result[1], vae_outputs[1])

    def test_mixed_padding_slices_current_patch(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import slice_waveforms

        vae_outputs = np.arange(50, dtype=np.float32).reshape(2, 25)
        result = slice_waveforms(vae_outputs, [0, 3], patch_size=2, chunk_size=5)
        np.testing.assert_array_equal(result[0], vae_outputs[0, :10])
        np.testing.assert_array_equal(result[1], vae_outputs[1, 15:25])

    def test_with_padding_slices_correctly(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import slice_waveforms

        vae_outputs = np.arange(9, dtype=np.float32).reshape(1, 9)
        result = slice_waveforms(vae_outputs, [2], patch_size=1, chunk_size=3)
        np.testing.assert_array_equal(result[0], vae_outputs[0, 6:9])


class TestAssembleRunOutputs:
    def test_output_values_are_matched_by_sequence(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_run_outputs

        latents = np.arange(16, dtype=np.float32).reshape(2, 2, 4)
        waveforms = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = assemble_run_outputs(latents, [0, 1], waveforms)
        assert [output["stop_flag"] for output in result] == [0, 1]
        np.testing.assert_array_equal(result[0]["latents"], latents[0])
        np.testing.assert_array_equal(result[1]["waveforms"], waveforms[1])

    def test_output_structure(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_run_outputs

        result = assemble_run_outputs(
            np.zeros((2, 4, 8), dtype=np.float32),
            [0, 1],
            [np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float32)],
        )
        assert len(result) == 2
        assert set(result[0]) == {"latents", "stop_flag", "waveforms"}
        assert set(result[1]) == {"latents", "stop_flag", "waveforms"}

    def test_stop_flag_values(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_run_outputs

        result = assemble_run_outputs(np.zeros((2, 4, 8)), [0, 1], [np.zeros(5), np.zeros(5)])
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

        waveforms = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = assemble_run_outputs(np.zeros((2, 2, 4)), [0, 0], waveforms)
        np.testing.assert_array_equal(result[0]["waveforms"], waveforms[0])
        np.testing.assert_array_equal(result[1]["waveforms"], waveforms[1])

    def test_single_sequence(self):
        from nanovllm_voxcpm.models.voxcpm.runner_utils import assemble_run_outputs

        result = assemble_run_outputs(np.ones((1, 3, 6)), [1], [np.zeros(8)])
        assert len(result) == 1
        assert result[0]["stop_flag"] == 1
