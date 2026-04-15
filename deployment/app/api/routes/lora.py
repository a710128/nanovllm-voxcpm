from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.deps import get_server
from app.schemas.http import (
    ErrorResponse,
    RegisterLoRARequest,
    RegisterLoRAResponse,
    RegisteredLoRA,
    UnregisterLoRAResponse,
)

router = APIRouter(tags=["lora"])


@router.get(
    "/loras",
    response_model=list[RegisteredLoRA],
    summary="List registered LoRA adapters",
    responses={503: {"description": "Model server not ready", "model": ErrorResponse}},
)
async def list_loras(server: Any = Depends(get_server)) -> list[RegisteredLoRA]:
    return [RegisteredLoRA(name=str(item["name"])) for item in await server.list_loras()]


@router.post(
    "/loras",
    response_model=RegisterLoRAResponse,
    summary="Register a LoRA adapter",
    responses={
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
    },
)
async def register_lora(
    req: RegisterLoRARequest, request: Request, server: Any = Depends(get_server)
) -> RegisterLoRAResponse:
    cfg = getattr(request.app.state, "cfg", None)
    if getattr(cfg, "lora", None) is None:
        raise HTTPException(status_code=400, detail="Runtime LoRA is disabled; set NANOVLLM_LORA_ENABLED=true")
    try:
        result = await server.register_lora(req.name, req.path)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return RegisterLoRAResponse(name=str(result["name"]))


@router.delete(
    "/loras/{name}",
    response_model=UnregisterLoRAResponse,
    summary="Unregister a LoRA adapter",
    responses={
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
    },
)
async def unregister_lora(name: str, server: Any = Depends(get_server)) -> UnregisterLoRAResponse:
    try:
        result = await server.unregister_lora(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return UnregisterLoRAResponse(name=str(result["name"]))
