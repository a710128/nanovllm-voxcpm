import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_encode_latents_uses_librosa_resample(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm2.server import VoxCPM2ServerImpl

    captured = {}

    class _FakeLLM:
        patch_size = 1

        def encode_latents(self, wav_tensor):
            captured["wav_tensor"] = wav_tensor
            return np.zeros((2, 4), dtype=np.float32)

    server = VoxCPM2ServerImpl.__new__(VoxCPM2ServerImpl)
    server.encoder_sample_rate = 16000
    server.output_sample_rate = 48000
    server.llm = _FakeLLM()

    def _fake_librosa_load(file_obj, sr, mono):
        assert isinstance(file_obj, io.BytesIO)
        captured["target_sr"] = sr
        captured["mono"] = mono
        return np.array([0.1, 0.2, 0.3], dtype=np.float32), sr

    def _forbid_cuda(self, *args, **kwargs):
        raise AssertionError("server should not move wav to cuda")

    monkeypatch.setattr("nanovllm_voxcpm.models.voxcpm2.server.librosa.load", _fake_librosa_load)
    monkeypatch.setattr(torch.Tensor, "cuda", _forbid_cuda, raising=False)

    out = server.encode_latents(b"fake-wav-bytes", wav_format="wav")

    assert isinstance(out, bytes)
    assert captured["target_sr"] == 16000
    assert captured["mono"] is False
    assert tuple(captured["wav_tensor"].shape) == (1, 3)
    assert captured["wav_tensor"].device.type == "cpu"


def test_get_model_info_uses_output_sample_rate():
    from nanovllm_voxcpm.models.voxcpm2.server import VoxCPM2ServerImpl

    server = VoxCPM2ServerImpl.__new__(VoxCPM2ServerImpl)
    server.encoder_sample_rate = 16000
    server.output_sample_rate = 48000
    server.model_path = "/fake/model"
    server.llm = type("_LLM", (), {"feat_dim": 64, "patch_size": 4})()

    info = server.get_model_info()

    assert info["sample_rate"] == 48000
    assert info["encoder_sample_rate"] == 16000
    assert info["output_sample_rate"] == 48000
