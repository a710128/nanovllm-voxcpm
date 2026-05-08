from __future__ import annotations

import inspect
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from huggingface_hub import snapshot_download

from nanovllm_voxcpm.llm import VoxCPM

from app.core.config import ServiceConfig, SpeakerEnvConfig, load_speaker_env_configs, materialize_lora_config
from app.services.llm import LLMClient

logger = logging.getLogger(__name__)

SERVER_FACTORY = VoxCPM.from_pretrained


@dataclass(frozen=True)
class SpeakerRuntimeConfig:
    """Resolved per-speaker config with encoded latents ready for TTS."""

    prompt_latents: bytes | None
    prompt_text: str
    ref_audio_latents: bytes | None
    gemini_prompt_template: str  # empty string means use global default


def _infer_audio_format(path: Path, override: str | None) -> str:
    if override is not None and override.strip() != "":
        return override.strip().lower()
    suffix = path.suffix.lstrip(".").lower()
    if suffix == "":
        raise RuntimeError(
            f"Cannot infer audio format from path '{path}'. "
            "Set the corresponding *_FORMAT env var (e.g. NANOVLLM_WS_PROMPT_AUDIO_FORMAT)."
        )
    return suffix


async def _encode_speaker_config(
    speaker_name: str,
    env_cfg: SpeakerEnvConfig,
    server: Any,
    fallback_prompt_latents: bytes | None,
    fallback_prompt_text: str,
    fallback_ref_latents: bytes | None,
    fallback_prompt_template: str,
) -> SpeakerRuntimeConfig:
    """Encode latents for a single speaker, falling back to defaults."""

    prompt_latents: bytes | None = fallback_prompt_latents
    prompt_text: str = fallback_prompt_text
    ref_audio_latents: bytes | None = fallback_ref_latents
    gemini_prompt_template = env_cfg.gemini_prompt_template or fallback_prompt_template

    # Speaker-specific prompt audio
    if env_cfg.prompt_audio_path:
        if not env_cfg.prompt_text_path:
            raise RuntimeError(
                f"Speaker '{speaker_name}': PROMPT_TEXT_PATH must be set when PROMPT_AUDIO_PATH is set"
            )
        prompt_audio_path = Path(env_cfg.prompt_audio_path)
        prompt_text_path = Path(env_cfg.prompt_text_path)
        if not prompt_audio_path.is_file():
            raise RuntimeError(f"Speaker '{speaker_name}': prompt audio not found: {prompt_audio_path}")
        if not prompt_text_path.is_file():
            raise RuntimeError(f"Speaker '{speaker_name}': prompt text not found: {prompt_text_path}")

        prompt_text = prompt_text_path.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise RuntimeError(f"Speaker '{speaker_name}': prompt text file is empty")

        audio_bytes = prompt_audio_path.read_bytes()
        audio_format = _infer_audio_format(prompt_audio_path, None)
        prompt_latents = await server.encode_latents(audio_bytes, audio_format)
        logger.info("Speaker '%s': encoded prompt audio from %s", speaker_name, prompt_audio_path)

    # Speaker-specific ref audio
    if env_cfg.ref_audio_path:
        generate_params = inspect.signature(server.generate).parameters
        if "ref_audio_latents" not in generate_params:
            raise RuntimeError(
                f"Speaker '{speaker_name}': ref audio not supported by loaded model"
            )
        ref_audio_path = Path(env_cfg.ref_audio_path)
        if not ref_audio_path.is_file():
            raise RuntimeError(f"Speaker '{speaker_name}': ref audio not found: {ref_audio_path}")

        ref_bytes = ref_audio_path.read_bytes()
        ref_format = _infer_audio_format(ref_audio_path, env_cfg.ref_audio_format)
        ref_audio_latents = await server.encode_latents(ref_bytes, ref_format)
        logger.info("Speaker '%s': encoded ref audio from %s", speaker_name, ref_audio_path)

    return SpeakerRuntimeConfig(
        prompt_latents=prompt_latents,
        prompt_text=prompt_text,
        ref_audio_latents=ref_audio_latents,
        gemini_prompt_template=gemini_prompt_template,
    )


async def _load_ws_tts_conditioning(app: FastAPI, server: Any) -> None:
    """Load persistent WS TTS conditioning from env vars (default/fallback voice).

    This is intentionally environment-driven (not part of ServiceConfig) because
    it is optional and specific to the WebSocket pipeline.
    """

    prompt_audio_path_s = os.environ.get("NANOVLLM_WS_PROMPT_AUDIO_PATH", "").strip()
    prompt_text_path_s = os.environ.get("NANOVLLM_WS_PROMPT_TEXT_PATH", "").strip()
    prompt_audio_format_override = os.environ.get("NANOVLLM_WS_PROMPT_AUDIO_FORMAT", "").strip() or None

    ref_audio_path_s = os.environ.get("NANOVLLM_WS_REF_AUDIO_PATH", "").strip()
    ref_audio_format_override = os.environ.get("NANOVLLM_WS_REF_AUDIO_FORMAT", "").strip() or None

    prompt_latents: bytes | None = None
    prompt_text: str = ""
    ref_audio_latents: bytes | None = None

    if prompt_audio_path_s:
        if not prompt_text_path_s:
            raise RuntimeError(
                "NANOVLLM_WS_PROMPT_TEXT_PATH must be set when NANOVLLM_WS_PROMPT_AUDIO_PATH is set"
            )

        prompt_audio_path = Path(prompt_audio_path_s)
        prompt_text_path = Path(prompt_text_path_s)
        if not prompt_audio_path.is_file():
            raise RuntimeError(f"NANOVLLM_WS_PROMPT_AUDIO_PATH does not exist or is not a file: {prompt_audio_path}")
        if not prompt_text_path.is_file():
            raise RuntimeError(f"NANOVLLM_WS_PROMPT_TEXT_PATH does not exist or is not a file: {prompt_text_path}")

        prompt_text = prompt_text_path.read_text(encoding="utf-8").strip()
        if prompt_text == "":
            raise RuntimeError("Prompt text file is empty; NANOVLLM_WS_PROMPT_TEXT_PATH must contain non-empty text")

        prompt_audio_bytes = prompt_audio_path.read_bytes()
        prompt_audio_format = _infer_audio_format(prompt_audio_path, prompt_audio_format_override)
        prompt_latents = await server.encode_latents(prompt_audio_bytes, prompt_audio_format)

    if ref_audio_path_s:
        generate_params = inspect.signature(server.generate).parameters
        if "ref_audio_latents" not in generate_params:
            raise RuntimeError(
                "NANOVLLM_WS_REF_AUDIO_PATH was set but the loaded model does not support reference audio latents. "
                "(VoxCPM2 supports ref_audio_latents; VoxCPM v1.5 does not.)"
            )

        ref_audio_path = Path(ref_audio_path_s)
        if not ref_audio_path.is_file():
            raise RuntimeError(f"NANOVLLM_WS_REF_AUDIO_PATH does not exist or is not a file: {ref_audio_path}")

        ref_audio_bytes = ref_audio_path.read_bytes()
        ref_audio_format = _infer_audio_format(ref_audio_path, ref_audio_format_override)
        ref_audio_latents = await server.encode_latents(ref_audio_bytes, ref_audio_format)

    app.state.ws_prompt_latents = prompt_latents
    app.state.ws_prompt_text = prompt_text
    app.state.ws_ref_audio_latents = ref_audio_latents


async def _load_speaker_configs(app: FastAPI, server: Any) -> None:
    """Load per-speaker configurations and encode their latents."""

    speaker_env_configs = load_speaker_env_configs()

    # Fallback defaults from the global WS conditioning
    fallback_prompt_latents = getattr(app.state, "ws_prompt_latents", None)
    fallback_prompt_text = getattr(app.state, "ws_prompt_text", "")
    fallback_ref_latents = getattr(app.state, "ws_ref_audio_latents", None)

    cfg = getattr(app.state, "cfg", None)
    fallback_prompt_template = cfg.llm.prompt_template if cfg else ""

    speakers: dict[str, SpeakerRuntimeConfig] = {}
    for name, env_cfg in speaker_env_configs.items():
        speakers[name] = await _encode_speaker_config(
            speaker_name=name,
            env_cfg=env_cfg,
            server=server,
            fallback_prompt_latents=fallback_prompt_latents,
            fallback_prompt_text=fallback_prompt_text,
            fallback_ref_latents=fallback_ref_latents,
            fallback_prompt_template=fallback_prompt_template,
        )
        logger.info(
            "Speaker '%s' loaded (prompt=%s, ref=%s, template=%s)",
            name,
            "yes" if speakers[name].prompt_latents else "fallback",
            "yes" if speakers[name].ref_audio_latents else "fallback",
            "custom" if env_cfg.gemini_prompt_template else "global",
        )

    app.state.speakers = speakers


def _read_model_architecture(model_path: str) -> str:
    resolved_model_path = os.path.expanduser(model_path)
    if not os.path.isdir(resolved_model_path):
        resolved_model_path = snapshot_download(repo_id=model_path)
    config_file = os.path.join(resolved_model_path, "config.json")
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file `{config_file}` not found")
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    architecture = config.get("architecture")
    if not isinstance(architecture, str) or architecture == "":
        raise RuntimeError(f"Config file `{config_file}` must define architecture")
    return architecture


def build_lifespan(cfg: ServiceConfig):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Create a single shared LLM client for the lifetime of the process.
        # WebSocket connections reuse this instance instead of constructing one per connection.
        app.state.llm_client = LLMClient(cfg.llm)
        # Best-effort warmup of the underlying SDK/model object.
        # This is local initialization (imports/configure/model construction), not a network call.
        app.state.llm_client.warmup()

        model_architecture = None
        lora_config = None
        if cfg.lora is not None:
            model_architecture = _read_model_architecture(cfg.model_path)
            lora_config = materialize_lora_config(cfg.lora, model_architecture)

        server = SERVER_FACTORY(
            model=cfg.model_path,
            max_num_batched_tokens=cfg.server_pool.max_num_batched_tokens,
            max_num_seqs=cfg.server_pool.max_num_seqs,
            max_model_len=cfg.server_pool.max_model_len,
            gpu_memory_utilization=cfg.server_pool.gpu_memory_utilization,
            enforce_eager=cfg.server_pool.enforce_eager,
            devices=list(cfg.server_pool.devices),
            lora_config=lora_config,
        )
        app.state.server = server
        app.state.model_architecture = model_architecture
        app.state.ready = False

        try:
            await server.wait_for_ready()

            # Load default WS TTS conditioning first (used as fallback).
            await _load_ws_tts_conditioning(app, server)

            # Load per-speaker configs (may use defaults as fallback).
            await _load_speaker_configs(app, server)

            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            await server.stop()
            if getattr(app.state, "server", None) is server:
                delattr(app.state, "server")
            if getattr(app.state, "model_architecture", None) is model_architecture:
                delattr(app.state, "model_architecture")
            if getattr(app.state, "llm_client", None) is not None:
                delattr(app.state, "llm_client")
            if getattr(app.state, "speakers", None) is not None:
                delattr(app.state, "speakers")

    return lifespan
