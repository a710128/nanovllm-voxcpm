import base64
import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("prometheus_client")
pytest.importorskip("lameenc")

import numpy as np
from starlette.testclient import TestClient

DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))


class FakeServerPool:
    def __init__(self, *args, **kwargs):
        self._stopped = False
        self._lora_loaded = False

    async def wait_for_ready(self):
        return None

    async def stop(self):
        self._stopped = True

    async def get_model_info(self):
        return {
            "sample_rate": 48000,
            "encoder_sample_rate": 16000,
            "output_sample_rate": 48000,
            "channels": 1,
            "feat_dim": 64,
            "patch_size": 2,
            "model_path": "/fake/model",
        }

    async def encode_latents(self, wav: bytes, wav_format: str):
        # Return deterministic fake float32 bytes.
        arr = np.arange(0, 64, dtype=np.float32)
        return arr.tobytes()

    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 1.5,
    ):
        # Yield a couple of waveform chunks.
        yield np.zeros((160,), dtype=np.float32)
        yield np.ones((160,), dtype=np.float32) * 0.5

    async def load_lora(self, path: str):
        self._lora_loaded = True

    async def set_lora_enabled(self, enabled: bool):
        return None


@pytest.fixture
def app(monkeypatch):
    # Patch the server pool used by lifespan.
    import app.core.lifespan as lifespan

    monkeypatch.setattr(lifespan, "AsyncVoxCPMServerPool", FakeServerPool)

    from app.main import create_app

    return create_app()


def test_health_and_ready(app):
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

        r = client.get("/ready")
        assert r.status_code == 200


def test_info(app):
    with TestClient(app) as client:
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["model"]["sample_rate"] == 48000
        assert body["model"]["channels"] == 1


def test_encode_latents(app):
    wav_b64 = base64.b64encode(b"FAKEWAV").decode("utf-8")
    with TestClient(app) as client:
        r = client.post(
            "/encode_latents",
            json={"wav_base64": wav_b64, "wav_format": "wav"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["feat_dim"] == 64
        assert body["latents_dtype"] == "float32"
        assert body["sample_rate"] == 16000
        assert body["channels"] == 1
        # Ensure it's decodable base64.
        base64.b64decode(body["prompt_latents_base64"])


def test_generate_streams_mp3(app):
    with TestClient(app) as client:
        with client.stream("POST", "/generate", json={"target_text": "hi"}) as resp:
            assert resp.status_code == 200
            assert resp.headers.get("content-type", "").startswith("audio/mpeg")
            data = resp.read()
            assert data
