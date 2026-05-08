from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env file from deployment directory before importing config.
# Use absolute path and override=False to not clobber explicitly set env vars.
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)
else:
    # Fallback: try relative to cwd (for 'fastapi run deployment/app/main.py' from repo root)
    _alt_path = Path("deployment/.env")
    if _alt_path.exists():
        load_dotenv(_alt_path, override=False)

from app.api.api import api_router
from app.core.config import load_config
from app.core.lifespan import build_lifespan
from app.core.metrics import install_metrics


def create_app() -> FastAPI:
    cfg = load_config()
    app = FastAPI(
        title="nano-vllm VoxCPM Service",
        version="0.1.0",
        description=(
            "Production-oriented FastAPI wrapper for nano-vllm-voxcpm. "
            "See /docs for interactive API docs and /openapi.json for the OpenAPI schema."
        ),
        openapi_tags=[
            {"name": "health", "description": "Liveness and readiness probes."},
            {"name": "info", "description": "Model and instance metadata."},
            {"name": "metrics", "description": "Prometheus metrics."},
            {"name": "lora", "description": "Runtime LoRA adapter management."},
            {
                "name": "latents",
                "description": "Encode prompt audio to prompt latents.",
            },
            {
                "name": "generation",
                "description": "Text-to-speech generation (streaming MP3).",
            },
        ],
        lifespan=build_lifespan(cfg),
    )
    app.state.cfg = cfg
    install_metrics(app)
    app.include_router(api_router)
    return app


app = create_app()
