from __future__ import annotations

import numpy as np
import pytest


def _make_sequence(*, adapter_id=7, seed=11):
    from nanovllm_voxcpm.engine.sequence import Sequence
    from nanovllm_voxcpm.models.voxcpm.engine import VoxCPMSeqPayload

    payload = VoxCPMSeqPayload(
        feats=[
            np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            np.arange(6, dtype=np.float32).reshape(1, 2, 3),
        ],
        text_tokens=[10, 20, 30],
        feat_masks=[False, False, True],
        generated_waveforms=[],
        temperature=0.7,
        cfg_value=1.8,
        max_generate_length=4,
        seed=seed,
        seed_step=2,
    )
    seq = Sequence("seq", [10, 20, 30], 4, payload, lora_name="voice", adapter_id=adapter_id)
    seq.block_table = [3]
    seq.num_cached_tokens = 1
    return seq


def _make_engine():
    from nanovllm_voxcpm.models.voxcpm.engine import VoxCPMEngine

    engine = VoxCPMEngine.__new__(VoxCPMEngine)
    engine.feat_dim = 3
    engine.patch_size = 2
    engine.n_decode_pad_frames = 2
    return engine


def test_preprocess_prefill_builds_task_from_uncached_tail():
    engine = _make_engine()
    seq = _make_sequence()

    task = engine.preprocess_seq(seq, is_prefill=True)

    assert task.block_table == [3]
    assert task.seq_length == 3
    assert task.num_cached_tokens == 1
    assert task.block_size == 4
    assert task.adapter_id == 7
    assert len(seq.custom_payload.feats) == 1
    np.testing.assert_array_equal(task.custom_payload.text_tokens, [20, 30])
    np.testing.assert_array_equal(task.custom_payload.feats, seq.custom_payload.feats[0][1:])
    np.testing.assert_array_equal(task.custom_payload.feat_masks, [False, True])
    assert task.custom_payload.seed == 11
    assert task.custom_payload.seed_step == 2


def test_preprocess_decode_builds_single_token_task():
    engine = _make_engine()
    seq = _make_sequence()

    task = engine.preprocess_seq(seq, is_prefill=False)

    assert task.seq_length == 3
    assert task.num_cached_tokens == 2
    assert task.adapter_id == 7
    np.testing.assert_array_equal(task.custom_payload.text_tokens, [30])
    np.testing.assert_array_equal(task.custom_payload.feats, seq.custom_payload.feats[-1][-1:])
    np.testing.assert_array_equal(task.custom_payload.feat_masks, [True])


def test_postprocess_updates_sequence_state_and_generation_limit():
    engine = _make_engine()
    seq = _make_sequence()
    seq.custom_payload.feats = [seq.custom_payload.feats[0]]
    seq.custom_payload.decode_pad = np.full((1, 3), -1.0, dtype=np.float32)
    seq.custom_payload.max_generate_length = 1
    latents = np.arange(6, dtype=np.float32).reshape(2, 3)
    waveforms = np.arange(4, dtype=np.float32)

    engine.postprocess_seq(
        seq,
        {"latents": latents, "stop_flag": 0, "waveforms": waveforms},
        is_prefill=False,
    )

    assert seq.token_ids[-1] == latents.tobytes()
    assert seq.custom_payload.text_tokens[-1] == 0
    assert seq.custom_payload.feat_masks[-1] is True
    assert seq.custom_payload.seed_step == 3
    np.testing.assert_array_equal(seq.custom_payload.generated_waveforms, [waveforms])
    np.testing.assert_array_equal(seq.custom_payload.decode_pad, latents)
    assert seq.stoped is True


def test_add_request_resolves_lora_name_to_adapter_id():
    engine = _make_engine()
    engine.audio_start_token = 101
    engine.block_size = 4
    engine.max_model_len = 10
    engine.tokenizer = lambda text: [5, 6]
    engine.lora_manager = type("Resolver", (), {"resolve_adapter": lambda self, name: {"voice": 9}[name]})()
    added = []
    engine.add_sequence = added.append

    engine.add_request("request", "hello", max_generate_length=2, lora_name="voice")

    assert len(added) == 1
    assert added[0].lora_name == "voice"
    assert added[0].adapter_id == 9
    assert added[0].custom_payload.max_generate_length == 2


def test_add_request_rejects_missing_lora():
    from nanovllm_voxcpm.engine.lora_manager import LoRAManager

    engine = _make_engine()
    engine.audio_start_token = 101
    engine.block_size = 4
    engine.max_model_len = 10
    engine.tokenizer = lambda text: [5]
    engine.lora_manager = LoRAManager(max_loras=1)
    engine.add_sequence = lambda seq: pytest.fail("invalid request must not be scheduled")

    with pytest.raises(ValueError, match="LoRA 'missing' is not registered"):
        engine.add_request("request", "hello", max_generate_length=1, lora_name="missing")
