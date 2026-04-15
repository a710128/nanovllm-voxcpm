from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_server
from app.core.config import materialize_lora_config
from app.core.lifespan import _read_model_architecture
from app.schemas.http import ErrorResponse, InfoResponse, LoRAInfo, ModelInfo, Mp3Info

router = APIRouter(tags=["info"])


@router.get(
    "/info",
    response_model=InfoResponse,
    summary="Get model and service metadata",
    responses={
        503: {
            "description": "Model server not ready",
            "model": ErrorResponse,
        }
    },
)
async def info(request: Request, server: Any = Depends(get_server)) -> InfoResponse:
    """Return model metadata and instance-level configuration."""

    cfg = getattr(request.app.state, "cfg", None)
    model_info = await server.get_model_info()
    registered_loras = [str(item["name"]) for item in await server.list_loras()]
    lora_config = None
    if getattr(cfg, "lora", None) is not None:
        lora_config = materialize_lora_config(cfg.lora, _read_model_architecture(cfg.model_path))
    return InfoResponse(
        model=ModelInfo(
            sample_rate=int(model_info["sample_rate"]),
            channels=int(model_info["channels"]),
            feat_dim=int(model_info["feat_dim"]),
            patch_size=int(model_info["patch_size"]),
            model_path=str(model_info["model_path"]),
        ),
        lora=LoRAInfo(
            enabled=lora_config is not None,
            enable_lm=bool(getattr(lora_config, "enable_lm", False)),
            enable_dit=bool(getattr(lora_config, "enable_dit", False)),
            enable_proj=bool(getattr(lora_config, "enable_proj", False)),
            max_loras=getattr(lora_config, "max_loras", None),
            max_lora_rank=getattr(lora_config, "max_lora_rank", None),
            target_modules_lm=list(getattr(lora_config, "target_modules_lm", ())),
            target_modules_dit=list(getattr(lora_config, "target_modules_dit", ())),
            target_proj_modules=list(getattr(lora_config, "target_proj_modules", ())),
            registered_names=registered_loras,
            loaded=bool(registered_loras),
        ),
        mp3=Mp3Info(
            bitrate_kbps=getattr(getattr(cfg, "mp3", None), "bitrate_kbps", None),
            quality=getattr(getattr(cfg, "mp3", None), "quality", None),
        ),
    )
