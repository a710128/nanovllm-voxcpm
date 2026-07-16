import json
import os
import sys
import types
from pathlib import Path

import pytest

# Ensure `import app...` resolves to deployment/app.
DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

# Ensure repo root is on sys.path so `tests._shims` is importable.
_REPO_ROOT = str(DEPLOYMENT_DIR.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._shims import install_gpu_shims  # noqa: E402


# Skip the entire deployment test suite if optional runtime deps are missing.
pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("prometheus_client")


# Deployment tests exercise the HTTP layer; keep imports CPU-safe even if some
# core modules are decorated with `@torch.compile`.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
try:  # pragma: no cover
    import torch._dynamo

    torch._dynamo.config.disable = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Test-time dependency shims
# ---------------------------------------------------------------------------

install_gpu_shims()


class FakeServerPool:
    """CPU-safe fake for AsyncVoxCPMServerPool used by lifespan."""

    def __init__(self, *args, **kwargs):
        self._stopped = False
        self.registered_loras = set()

    async def wait_for_ready(self):
        return None

    async def stop(self):
        self._stopped = True

    async def get_model_info(self):
        return {
            "architecture": "voxcpm",
            "sample_rate": 16000,
            "channels": 1,
            "feat_dim": 64,
            "patch_size": 2,
            "model_path": "/fake/model",
        }

    async def encode_latents(self, wav: bytes, wav_format: str):
        # Deterministic fake float32 bytes (shape doesn't matter for HTTP layer).
        import numpy as np

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
        import numpy as np

        if lora_name is not None and lora_name not in self.registered_loras:
            raise ValueError(f"LoRA '{lora_name}' is not registered")
        yield np.zeros((160,), dtype=np.float32)
        yield np.ones((160,), dtype=np.float32) * 0.5


@pytest.fixture
def app(monkeypatch, tmp_path):
    import app.core.lifespan as lifespan

    monkeypatch.setattr(lifespan, "SERVER_FACTORY", FakeServerPool)
    monkeypatch.setenv("NANOVLLM_LORA_ENABLED", "true")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"architecture": "voxcpm"}), encoding="utf-8")
    monkeypatch.setenv("NANOVLLM_MODEL_PATH", str(model_dir))

    from app.main import create_app

    return create_app()
