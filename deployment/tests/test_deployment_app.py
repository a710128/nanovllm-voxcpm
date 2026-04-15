import base64
import json
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
        self.registered_loras = set()
        self.generate_calls = []
        self.kwargs = kwargs

    async def wait_for_ready(self):
        return None

    async def stop(self):
        self._stopped = True

    async def get_model_info(self):
        return {
            "architecture": "voxcpm",
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

    async def register_lora(self, name: str, path: str):
        if name in self.registered_loras:
            raise ValueError(f"LoRA '{name}' is already registered")
        self.registered_loras.add(name)
        return {"name": name}

    async def unregister_lora(self, name: str):
        if name not in self.registered_loras:
            raise ValueError(f"LoRA '{name}' is not registered")
        self.registered_loras.remove(name)
        return {"name": name}

    async def list_loras(self):
        return [{"name": name} for name in sorted(self.registered_loras)]

    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        ref_audio_latents: bytes | None = None,
        lora_name: str | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 1.5,
    ):
        if lora_name is not None and lora_name not in self.registered_loras:
            raise ValueError(f"LoRA '{lora_name}' is not registered")
        self.generate_calls.append({"target_text": target_text, "lora_name": lora_name})
        # Yield a couple of waveform chunks.
        yield np.zeros((160,), dtype=np.float32)
        yield np.ones((160,), dtype=np.float32) * 0.5


@pytest.fixture
def app(monkeypatch, tmp_path):
    # Patch the server pool used by lifespan.
    import app.core.lifespan as lifespan

    monkeypatch.setattr(lifespan, "SERVER_FACTORY", FakeServerPool)
    monkeypatch.setenv("NANOVLLM_LORA_ENABLED", "true")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architecture": "voxcpm"}), encoding="utf-8")
    monkeypatch.setenv("NANOVLLM_MODEL_PATH", str(model_dir))

    from app.main import create_app

    return create_app()


def test_health_and_ready(app):
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

        r = client.get("/ready")
        assert r.status_code == 200


def test_lifespan_materializes_architecture_default_lora_targets(app):
    with TestClient(app) as client:
        lora_config = client.app.state.server.kwargs["lora_config"]
        assert lora_config.enable_lm is True
        assert lora_config.enable_dit is True
        assert lora_config.enable_proj is True
        assert lora_config.target_modules_lm == (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        assert lora_config.target_proj_modules == ("enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj")


def test_lifespan_resolves_repo_id_before_materializing_lora(monkeypatch, tmp_path):
    import app.core.lifespan as lifespan

    model_dir = tmp_path / "downloaded-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architecture": "voxcpm2"}), encoding="utf-8")

    monkeypatch.setattr(lifespan, "SERVER_FACTORY", FakeServerPool)
    monkeypatch.setattr(lifespan, "snapshot_download", lambda repo_id: str(model_dir))
    monkeypatch.setenv("NANOVLLM_LORA_ENABLED", "true")
    monkeypatch.setenv("NANOVLLM_MODEL_PATH", "org/demo-model")

    from app.main import create_app

    with TestClient(create_app()) as client:
        lora_config = client.app.state.server.kwargs["lora_config"]
        assert lora_config.target_proj_modules == (
            "enc_to_lm_proj",
            "lm_to_dit_proj",
            "res_to_dit_proj",
            "fusion_concat_proj",
        )


def test_info(app):
    with TestClient(app) as client:
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["model"]["sample_rate"] == 48000
        assert body["model"]["channels"] == 1
        assert body["lora"] == {
            "enabled": True,
            "enable_lm": True,
            "enable_dit": True,
            "enable_proj": True,
            "max_loras": 1,
            "max_lora_rank": 32,
            "target_modules_lm": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "target_modules_dit": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "target_proj_modules": ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"],
            "registered_names": [],
            "loaded": False,
        }


def test_lora_management_endpoints(app):
    with TestClient(app) as client:
        r = client.get("/loras")
        assert r.status_code == 200
        assert r.json() == []

        r = client.post("/loras", json={"name": "demo", "path": "/tmp/demo"})
        assert r.status_code == 200
        assert r.json() == {"name": "demo"}

        r = client.get("/loras")
        assert r.status_code == 200
        assert r.json() == [{"name": "demo"}]

        r = client.delete("/loras/demo")
        assert r.status_code == 200
        assert r.json() == {"name": "demo"}


def test_register_lora_rejects_when_runtime_lora_disabled(app):
    with TestClient(app) as client:
        client.app.state.cfg = client.app.state.cfg.__class__(
            model_path=client.app.state.cfg.model_path,
            mp3=client.app.state.cfg.mp3,
            server_pool=client.app.state.cfg.server_pool,
            lora=None,
        )
        r = client.post("/loras", json={"name": "demo", "path": "/tmp/demo"})
        assert r.status_code == 400
        assert "Runtime LoRA is disabled" in r.json()["detail"]


def test_generate_forwards_lora_name(app):
    with TestClient(app) as client:
        r = client.post("/loras", json={"name": "demo", "path": "/tmp/demo"})
        assert r.status_code == 200

        with client.stream("POST", "/generate", json={"target_text": "hi", "lora_name": "demo"}) as resp:
            assert resp.status_code == 200
            assert resp.read()

        assert client.app.state.server.generate_calls[-1]["lora_name"] == "demo"


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


def test_generate_with_reference_latents(app):
    ref_latents_b64 = base64.b64encode(np.arange(0, 64, dtype=np.float32).tobytes()).decode("utf-8")
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/generate",
            json={
                "target_text": "hi",
                "ref_audio_latents_base64": ref_latents_b64,
            },
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers.get("content-type", "").startswith("audio/mpeg")
            assert resp.read()
