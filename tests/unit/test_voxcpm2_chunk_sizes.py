import numpy as np
import pytest


torch = pytest.importorskip("torch")


def test_audio_vae_v2_exposes_encoder_and_decoder_chunk_sizes():
    from nanovllm_voxcpm.layers.audio_vae_v2 import AudioVAEConfigV2, AudioVAEV2

    vae = AudioVAEV2(
        config=AudioVAEConfigV2(
            encoder_dim=4,
            latent_dim=4,
            decoder_dim=8,
            encoder_rates=[2, 5],
            decoder_rates=[3, 7],
            sr_bin_boundaries=None,
        )
    )

    assert vae.encoder_chunk_size == 10
    assert vae.decoder_chunk_size == 21

    wav = torch.zeros(1, 13)
    padded = vae.preprocess(wav, sample_rate=vae.sample_rate)
    assert padded.shape[-1] == 20


def test_voxcpm2_engine_aligns_prompt_audio_with_encoder_chunk_size():
    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine

    captured = {}

    def _encode_latents(self, wav):
        captured["wav"] = wav
        return np.zeros((0, 0), dtype=np.float32)

    engine = VoxCPM2Engine.__new__(VoxCPM2Engine)
    engine.patch_size = 4
    engine.model_runner = type(
        "_Runner",
        (),
        {
            "vae": type("_VAE", (), {"encoder_chunk_size": 2})(),
            "encode_latents": _encode_latents,
        },
    )()

    wav = torch.zeros(1, 5)
    engine.encode_latents(wav)

    assert captured["wav"].shape[-1] == 8


def test_voxcpm2_runner_slices_decoded_waveform_with_decoder_chunk_size(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Payload, VoxCPM2Runner

    original_zeros = torch.zeros

    def _cpu_zeros(*args, **kwargs):
        kwargs.pop("device", None)
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=True: self, raising=False)
    monkeypatch.setattr(torch, "zeros", _cpu_zeros)

    runner = VoxCPM2Runner.__new__(VoxCPM2Runner)
    runner.patch_size = 4
    runner.feat_dim = 2
    runner.prepare_decode_context = lambda seqs: torch.zeros(1, dtype=torch.int64)
    runner.run_model = lambda inputs, is_prefill: {
        "latents": torch.zeros((1, runner.patch_size, runner.feat_dim), dtype=torch.float32),
        "stop_flag": torch.zeros((1,), dtype=torch.int64),
    }

    class _FakeDecoded:
        def __init__(self, tensor):
            self.tensor = tensor

        def __getitem__(self, item):
            return _FakeDecoded(self.tensor[item])

        def cpu(self):
            return self

        def numpy(self):
            return self.tensor.numpy()

    class _FakeVAE:
        decoder_chunk_size = 3

        def decode(self, z):
            assert tuple(z.shape) == (1, runner.feat_dim, 6)
            return _FakeDecoded(torch.arange(18, dtype=torch.float32).reshape(1, 1, 18))

    runner.vae = _FakeVAE()

    seq = type(
        "_Seq",
        (),
        {
            "custom_payload": VoxCPM2Payload(
                text_tokens=np.array([1], dtype=np.int64),
                feats=np.zeros((1, runner.patch_size, runner.feat_dim), dtype=np.float32),
                feat_masks=np.array([True], dtype=np.bool_),
                padding_decode=np.zeros((2, runner.feat_dim), dtype=np.float32),
            )
        },
    )()

    outputs = runner.run([seq], is_prefill=False)

    assert outputs[0]["waveforms"].tolist() == list(range(6, 18))
