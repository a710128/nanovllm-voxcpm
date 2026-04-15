from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Standard health response for liveness/readiness endpoints."""

    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    """Default error response shape produced by FastAPI for HTTPException.

    Note: FastAPI's validation errors (422) use a different schema.
    """

    detail: str = Field(..., description="Human-readable error message.")


class ModelInfo(BaseModel):
    """Read-only model metadata returned by the engine."""

    sample_rate: int = Field(..., description="Audio sample rate in Hz.", examples=[16000])
    channels: int = Field(..., description="Number of audio channels.", examples=[1])
    feat_dim: int = Field(..., description="Latent feature dimension.", examples=[64])
    patch_size: int = Field(..., description="Model patch size.", examples=[2])
    model_path: str = Field(
        ...,
        description="Resolved model path used by this instance.",
        examples=["/models/VoxCPM1.5"],
    )


class Mp3Info(BaseModel):
    """MP3 encoder configuration used by /generate."""

    bitrate_kbps: int | None = Field(None, description="Constant bitrate used for MP3 encoding.", examples=[192])
    quality: int | None = Field(None, description="LAME quality preset (0 is best, 2 is fast).", examples=[2])


class LoRAInfo(BaseModel):
    """Runtime LoRA registration state."""

    enabled: bool = Field(..., description="Whether runtime LoRA capacity is enabled for this deployment instance.")
    enable_lm: bool = Field(..., description="Whether LM LoRA capacity is enabled.")
    enable_dit: bool = Field(..., description="Whether DiT LoRA capacity is enabled.")
    enable_proj: bool = Field(..., description="Whether projection-layer LoRA capacity is enabled.")
    max_loras: int | None = Field(None, description="Maximum concurrently resident LoRA adapters per layer.")
    max_lora_rank: int | None = Field(None, description="Maximum supported LoRA rank per layer slot.")
    target_modules_lm: list[str] = Field(default_factory=list, description="Enabled LM LoRA target modules.")
    target_modules_dit: list[str] = Field(default_factory=list, description="Enabled DiT LoRA target modules.")
    target_proj_modules: list[str] = Field(default_factory=list, description="Enabled projection LoRA target modules.")
    registered_names: list[str] = Field(default_factory=list, description="Currently registered LoRA adapter names.")
    loaded: bool = Field(
        ...,
        description="Whether at least one LoRA adapter is currently registered.",
        examples=[False],
    )


class RegisteredLoRA(BaseModel):
    """Registered LoRA adapter metadata."""

    name: str = Field(..., description="Logical LoRA adapter name.", examples=["demo-lora"])


class RegisterLoRARequest(BaseModel):
    """Request body for POST /loras."""

    name: str = Field(..., description="Logical LoRA adapter name.", examples=["demo-lora"])
    path: str = Field(..., description="Filesystem path to the LoRA checkpoint directory.")


class RegisterLoRAResponse(BaseModel):
    """Response body for POST /loras."""

    name: str = Field(..., description="Registered LoRA adapter name.")


class UnregisterLoRAResponse(BaseModel):
    """Response body for DELETE /loras/{name}."""

    name: str = Field(..., description="Unregistered LoRA adapter name.")


class InfoResponse(BaseModel):
    """Response for GET /info."""

    model: ModelInfo
    lora: LoRAInfo
    mp3: Mp3Info


class EncodeLatentsRequest(BaseModel):
    """Request body for POST /encode_latents."""

    wav_base64: str = Field(
        ...,
        description="Base64-encoded audio file bytes (entire file contents). Do not include a data URI prefix.",
        examples=["UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."],
    )
    wav_format: str = Field(
        ...,
        description="Audio container format for decoding (e.g. 'wav', 'flac', 'mp3'); passed to torchaudio.",
        examples=["wav"],
    )


class EncodeLatentsResponse(BaseModel):
    """Response body for POST /encode_latents."""

    prompt_latents_base64: str
    feat_dim: int
    latents_dtype: Literal["float32"] = "float32"
    sample_rate: int
    channels: int


class GenerateRequest(BaseModel):
    """Request body for POST /generate.

    Prompt forms (mutually exclusive):

    - Zero-shot: omit all prompt_* fields.
    - WAV prompt: set prompt_wav_base64 + prompt_wav_format + prompt_text.
    - Latents prompt: set prompt_latents_base64 + prompt_text.

    Reference audio (optional, mutually exclusive within the ref_audio_* group):

    - WAV reference: set ref_audio_wav_base64 + ref_audio_wav_format.
    - Latents reference: set ref_audio_latents_base64.
    """

    target_text: str = Field(..., description="Text to synthesize.")

    # Prompt forms (mutually exclusive):
    prompt_wav_base64: str | None = Field(
        None,
        description="(wav prompt) Base64-encoded audio file bytes (entire file contents).",
    )
    prompt_wav_format: str | None = Field(
        None,
        description="(wav prompt) Audio container format for decoding (e.g. 'wav', 'flac', 'mp3').",
    )
    prompt_latents_base64: str | None = Field(
        None,
        description="(latents prompt) Base64-encoded float32 bytes returned by /encode_latents.",
    )
    prompt_text: str | None = Field(
        None,
        description="Prompt transcript text. Required for wav/latents prompt; omitted for zero-shot.",
    )

    ref_audio_wav_base64: str | None = Field(
        None,
        description="(reference audio) Base64-encoded audio file bytes (entire file contents).",
    )
    ref_audio_wav_format: str | None = Field(
        None,
        description="(reference audio) Audio container format for decoding (e.g. 'wav', 'flac', 'mp3').",
    )
    ref_audio_latents_base64: str | None = Field(
        None,
        description="(reference audio) Base64-encoded float32 bytes returned by /encode_latents.",
    )
    lora_name: str | None = Field(None, description="Registered LoRA adapter name to apply for this request.")

    max_generate_length: int = Field(2000, ge=1, description="Maximum number of model generation steps.")
    temperature: float = Field(1.0, ge=0.0, description="Sampling temperature.")
    cfg_value: float = Field(1.5, ge=0.0, description="Classifier-free guidance scale.")
