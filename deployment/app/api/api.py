from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.encode_latents import router as encode_latents_router
from app.api.routes.generate import router as generate_router
from app.api.routes.health import router as health_router
from app.api.routes.info import router as info_router
from app.api.routes.lora import router as lora_router
from app.api.routes.metrics import router as metrics_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(info_router)
api_router.include_router(lora_router)
api_router.include_router(metrics_router)
api_router.include_router(encode_latents_router)
api_router.include_router(generate_router)
