from __future__ import annotations

from fastapi import FastAPI

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
