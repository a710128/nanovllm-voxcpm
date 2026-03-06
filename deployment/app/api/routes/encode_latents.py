from __future__ import annotations

import base64
import time

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_server
from app.core.metrics import (
    ENCODE_LATENTS_DURATION_SECONDS,
    ENCODE_LATENTS_REQUESTS_TOTAL,
)
from app.schemas.http import EncodeLatentsRequest, EncodeLatentsResponse, ErrorResponse
from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool

router = APIRouter(tags=["latents"])


@router.post(
    "/encode_latents",
    response_model=EncodeLatentsResponse,
    summary="Encode prompt audio to prompt latents",
    responses={
        400: {"description": "Invalid input", "model": ErrorResponse},
        503: {"description": "Model server not ready", "model": ErrorResponse},
        500: {"description": "Internal error", "model": ErrorResponse},
    },
)
async def encode_latents(
    req: EncodeLatentsRequest,
    server: AsyncVoxCPMServerPool = Depends(get_server),
) -> EncodeLatentsResponse:
    """Decode an audio file and return serialized float32 prompt latents."""

    t0 = time.perf_counter()
    try:
        wav = base64.b64decode(req.wav_base64)
    except Exception as e:
        ENCODE_LATENTS_REQUESTS_TOTAL.labels(status="400").inc()
        raise HTTPException(status_code=400, detail=f"Invalid base64 in wav_base64: {e}") from e

    try:
        latents = await server.encode_latents(wav, req.wav_format)
        model_info = await server.get_model_info()
    except HTTPException:
        raise
    except Exception as e:
        ENCODE_LATENTS_REQUESTS_TOTAL.labels(status="500").inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        ENCODE_LATENTS_DURATION_SECONDS.observe(time.perf_counter() - t0)

    ENCODE_LATENTS_REQUESTS_TOTAL.labels(status="200").inc()
    return EncodeLatentsResponse(
        prompt_latents_base64=base64.b64encode(latents).decode("utf-8"),
        feat_dim=int(model_info["feat_dim"]),
        sample_rate=int(model_info.get("encoder_sample_rate", model_info["sample_rate"])),
        channels=int(model_info["channels"]),
    )
