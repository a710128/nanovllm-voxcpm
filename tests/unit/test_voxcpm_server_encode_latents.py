import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_encode_latents_keeps_server_side_processing_on_cpu(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm.server import VoxCPMServerImpl

    captured = {}

    class _FakeLLM:
        patch_size = 1

        def encode_latents(self, wav_tensor):
            captured["wav_tensor"] = wav_tensor
            return np.zeros((2, 4), dtype=np.float32)

    server = VoxCPMServerImpl.__new__(VoxCPMServerImpl)
    server.sample_rate = 16000
    server.llm = _FakeLLM()

    def _fake_load(file_obj, format):
        assert isinstance(file_obj, io.BytesIO)
        assert format == "wav"
        return torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32), 16000

    def _forbid_cuda(self, *args, **kwargs):
        raise AssertionError("server should not move wav to cuda")

    monkeypatch.setattr("nanovllm_voxcpm.models.voxcpm.server.torchaudio.load", _fake_load)
    monkeypatch.setattr(torch.Tensor, "cuda", _forbid_cuda, raising=False)

    out = server.encode_latents(b"fake-wav-bytes", wav_format="wav")

    assert isinstance(out, bytes)
    assert tuple(captured["wav_tensor"].shape) == (1, 3)
    assert captured["wav_tensor"].device.type == "cpu"
