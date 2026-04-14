from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError as e:
        raise RuntimeError(f"Invalid env {name}={v!r}; expected int") from e


def _get_float_env(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError as e:
        raise RuntimeError(f"Invalid env {name}={v!r}; expected float") from e


def _get_bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default

    s = v.strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise RuntimeError(f"Invalid env {name}={v!r}; expected bool")


def _get_int_list_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    v = os.environ.get(name)
    if v is None or v == "":
        return default

    parts = [p.strip() for p in v.split(",")]
    parts = [p for p in parts if p != ""]
    if len(parts) == 0:
        raise RuntimeError(f"Invalid env {name}={v!r}; expected comma-separated ints")

    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as e:
            raise RuntimeError(f"Invalid env {name}={v!r}; expected comma-separated ints") from e
    return tuple(out)


@dataclass(frozen=True)
class Mp3Config:
    bitrate_kbps: int
    quality: int


@dataclass(frozen=True)
class ServerPoolStartupConfig:
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int
    gpu_memory_utilization: float
    enforce_eager: bool
    devices: tuple[int, ...]


@dataclass(frozen=True)
class ServiceConfig:
    model_path: str
    mp3: Mp3Config
    server_pool: ServerPoolStartupConfig


def load_config() -> ServiceConfig:
    model_path = os.path.expanduser(os.environ.get("NANOVLLM_MODEL_PATH", "~/VoxCPM1.5"))

    mp3_bitrate_kbps = _get_int_env("NANOVLLM_MP3_BITRATE_KBPS", 192)
    mp3_quality = _get_int_env("NANOVLLM_MP3_QUALITY", 2)
    if mp3_bitrate_kbps <= 0:
        raise RuntimeError("NANOVLLM_MP3_BITRATE_KBPS must be > 0")
    if mp3_quality < 0 or mp3_quality > 2:
        raise RuntimeError("NANOVLLM_MP3_QUALITY must be in [0, 2]")

    lora_uri = os.environ.get("NANOVLLM_LORA_URI")
    lora_id = os.environ.get("NANOVLLM_LORA_ID")
    lora_sha256 = os.environ.get("NANOVLLM_LORA_SHA256")

    if lora_uri or lora_id or lora_sha256:
        raise RuntimeError(
            "LoRA startup preload env vars were removed; use the new per-request LoRA API when available"
        )

    # Server pool startup config (read at startup).
    pool_max_num_batched_tokens = _get_int_env("NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS", 8192)
    pool_max_num_seqs = _get_int_env("NANOVLLM_SERVERPOOL_MAX_NUM_SEQS", 16)
    pool_max_model_len = _get_int_env("NANOVLLM_SERVERPOOL_MAX_MODEL_LEN", 4096)
    pool_gpu_memory_utilization = _get_float_env("NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION", 0.95)
    pool_enforce_eager = _get_bool_env("NANOVLLM_SERVERPOOL_ENFORCE_EAGER", False)
    pool_devices = _get_int_list_env("NANOVLLM_SERVERPOOL_DEVICES", (0,))

    if pool_max_num_batched_tokens <= 0:
        raise RuntimeError("NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS must be > 0")
    if pool_max_num_seqs <= 0:
        raise RuntimeError("NANOVLLM_SERVERPOOL_MAX_NUM_SEQS must be > 0")
    if pool_max_model_len <= 0:
        raise RuntimeError("NANOVLLM_SERVERPOOL_MAX_MODEL_LEN must be > 0")
    if not (0.0 < pool_gpu_memory_utilization <= 1.0):
        raise RuntimeError("NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION must be in (0, 1]")
    if len(pool_devices) == 0:
        raise RuntimeError("NANOVLLM_SERVERPOOL_DEVICES must be a non-empty list")
    if any(d < 0 for d in pool_devices):
        raise RuntimeError("NANOVLLM_SERVERPOOL_DEVICES entries must be >= 0")

    return ServiceConfig(
        model_path=model_path,
        mp3=Mp3Config(bitrate_kbps=mp3_bitrate_kbps, quality=mp3_quality),
        server_pool=ServerPoolStartupConfig(
            max_num_batched_tokens=pool_max_num_batched_tokens,
            max_num_seqs=pool_max_num_seqs,
            max_model_len=pool_max_model_len,
            gpu_memory_utilization=pool_gpu_memory_utilization,
            enforce_eager=pool_enforce_eager,
            devices=pool_devices,
        ),
    )
