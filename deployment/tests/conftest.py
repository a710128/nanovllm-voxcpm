import json
import os
import sys
import types
from pathlib import Path
import importlib
import importlib.util

import pytest

# Ensure `import app...` resolves to deployment/app.
DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))


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


def _ensure_module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


def _module_available(name: str) -> bool:
    """Return True only if the module can be imported.

    Some GPU-centric deps may be present but fail at import time (e.g. missing
    CUDA extensions). For deployment tests we treat those as unavailable and
    install lightweight shims so HTTP-layer tests can still run.
    """

    try:
        if importlib.util.find_spec(name) is None:
            return False
    except (ModuleNotFoundError, AttributeError, ValueError):
        return False

    try:
        importlib.import_module(name)
        return True
    except Exception:
        for mod_name in list(sys.modules.keys()):
            if mod_name == name or mod_name.startswith(name + "."):
                sys.modules.pop(mod_name, None)
        return False


# ---------------------------------------------------------------------------
# Test-time dependency shims
# ---------------------------------------------------------------------------

# nanovllm_voxcpm/__init__.py imports llm, which checks flash-attn on import.
# In CPU-only/local dev envs flash-attn can be partially installed and crash.
if not _module_available("flash_attn"):
    flash_attn = _ensure_module("flash_attn")

    def _unavailable(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("flash_attn is not available in deployment tests")

    setattr(flash_attn, "flash_attn_varlen_func", _unavailable)
    setattr(flash_attn, "flash_attn_with_kvcache", _unavailable)
    setattr(flash_attn, "flash_attn_func", _unavailable)

if not _module_available("triton"):
    triton = _ensure_module("triton")

    def jit(fn=None, **kwargs):  # pragma: no cover
        if fn is None:
            return lambda f: f
        return fn

    setattr(triton, "jit", jit)

if not _module_available("triton.language"):
    tl = _ensure_module("triton.language")
    setattr(tl, "constexpr", object())

    triton = _ensure_module("triton")
    setattr(triton, "language", tl)

if not _module_available("huggingface_hub"):
    hub = _ensure_module("huggingface_hub")

    def snapshot_download(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("huggingface_hub is not available in deployment tests")

    setattr(hub, "snapshot_download", snapshot_download)


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
