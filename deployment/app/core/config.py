from __future__ import annotations

import os
from dataclasses import dataclass

ALL_LINEAR_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VOXCPM_PROJ_LORA_TARGETS = ("enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj")
VOXCPM2_PROJ_LORA_TARGETS = (*VOXCPM_PROJ_LORA_TARGETS, "fusion_concat_proj")


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


def _get_str_list_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    v = os.environ.get(name)
    if v is None or v == "":
        return default

    parts = tuple(p.strip() for p in v.split(",") if p.strip() != "")
    if len(parts) == 0:
        raise RuntimeError(f"Invalid env {name}={v!r}; expected comma-separated strings")
    return parts


@dataclass(frozen=True)
class Mp3Config:
    bitrate_kbps: int
    quality: int


@dataclass(frozen=True)
class OpusConfig:
    bitrate: int  # bits per second (e.g., 64000 for 64kbps)
    frame_ms: int  # frame duration in milliseconds (5, 10, 20, 40, 60)


@dataclass(frozen=True)
class ASRConfig:
    api_url: str  # Base URL for ASR API (e.g., http://localhost:8000)
    model: str | None  # ASR model name (e.g., Qwen/Qwen3-ASR-1.7B)
    timeout: float
    api_key: str  # API key for ASR service (default "EMPTY" for local vLLM)


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    timeout: float
    prompt_template: str


@dataclass(frozen=True)
class ServerPoolStartupConfig:
    max_num_batched_tokens: int
    max_num_seqs: int
    max_model_len: int
    gpu_memory_utilization: float
    enforce_eager: bool
    devices: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeLoRAConfig:
    enable_lm: bool | None
    enable_dit: bool | None
    enable_proj: bool | None
    max_loras: int
    max_lora_rank: int
    target_modules_lm: tuple[str, ...] | None
    target_modules_dit: tuple[str, ...] | None
    target_proj_modules: tuple[str, ...] | None


@dataclass(frozen=True)
class MaterializedRuntimeLoRAConfig:
    enable_lm: bool
    enable_dit: bool
    enable_proj: bool
    max_loras: int
    max_lora_rank: int
    target_modules_lm: tuple[str, ...]
    target_modules_dit: tuple[str, ...]
    target_proj_modules: tuple[str, ...]


@dataclass(frozen=True)
class ServiceConfig:
    model_path: str
    mp3: Mp3Config
    opus: OpusConfig
    asr: ASRConfig
    llm: LLMConfig
    server_pool: ServerPoolStartupConfig
    lora: RuntimeLoRAConfig | None


def _get_str_env(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _get_required_str_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        raise RuntimeError(f"Required env {name} is not set")
    return v


def load_config() -> ServiceConfig:
    model_path = os.path.expanduser(os.environ.get("NANOVLLM_MODEL_PATH", "/home/ubuntu/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/bffb3df5a29440629464e5e839f4d214c8714c3d"))

    mp3_bitrate_kbps = _get_int_env("NANOVLLM_MP3_BITRATE_KBPS", 192)
    mp3_quality = _get_int_env("NANOVLLM_MP3_QUALITY", 2)
    if mp3_bitrate_kbps <= 0:
        raise RuntimeError("NANOVLLM_MP3_BITRATE_KBPS must be > 0")
    if mp3_quality < 0 or mp3_quality > 2:
        raise RuntimeError("NANOVLLM_MP3_QUALITY must be in [0, 2]")

    # Opus encoding config
    opus_bitrate = _get_int_env("NANOVLLM_OPUS_BITRATE", 64000)
    opus_frame_ms = _get_int_env("NANOVLLM_OPUS_FRAME_MS", 20)
    if opus_bitrate <= 0:
        raise RuntimeError("NANOVLLM_OPUS_BITRATE must be > 0")
    if opus_frame_ms not in (5, 10, 20, 40, 60):
        raise RuntimeError("NANOVLLM_OPUS_FRAME_MS must be one of: 5, 10, 20, 40, 60")

    # ASR API config (vLLM with Qwen3-ASR using chat completions API)
    asr_api_url = _get_str_env("NANOVLLM_ASR_API_URL", "http://localhost:8001")
    asr_model = _get_str_env("NANOVLLM_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
    asr_timeout = _get_float_env("NANOVLLM_ASR_TIMEOUT", 30.0)
    asr_api_key = _get_str_env("NANOVLLM_ASR_API_KEY", "EMPTY")
    if asr_timeout <= 0:
        raise RuntimeError("NANOVLLM_ASR_TIMEOUT must be > 0")

    # LLM API config (Gemini)
    # GEMINI_API_KEY is required only when WebSocket endpoint is used,
    # so we don't make it required at config load time
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_model = _get_str_env("GEMINI_MODEL", "gpt-5.4-mini")
    prompt_template = _get_str_env("GEMINI_PROMPT_TEMPLATE", "")
    llm_timeout = _get_float_env("GEMINI_TIMEOUT", 60.0)
    if llm_timeout <= 0:
        raise RuntimeError("GEMINI_TIMEOUT must be > 0")

    lora_uri = os.environ.get("NANOVLLM_LORA_URI")
    lora_id = os.environ.get("NANOVLLM_LORA_ID")
    lora_sha256 = os.environ.get("NANOVLLM_LORA_SHA256")

    if lora_uri or lora_id or lora_sha256:
        raise RuntimeError(
            "LoRA startup preload env vars were removed; use the runtime LoRA API with NANOVLLM_LORA_ENABLED=true"
        )

    runtime_lora_enabled = _get_bool_env("NANOVLLM_LORA_ENABLED", False)
    runtime_lora_config: RuntimeLoRAConfig | None = None
    if runtime_lora_enabled:
        lora_max_loras = _get_int_env("NANOVLLM_LORA_MAX_LORAS", 128)
        lora_max_lora_rank = _get_int_env("NANOVLLM_LORA_MAX_LORA_RANK", 64)
        lora_enable_lm = _get_optional_bool_env("NANOVLLM_LORA_ENABLE_LM")
        lora_enable_dit = _get_optional_bool_env("NANOVLLM_LORA_ENABLE_DIT")
        lora_enable_proj = _get_optional_bool_env("NANOVLLM_LORA_ENABLE_PROJ")
        target_modules_lm = _get_optional_str_list_env("NANOVLLM_LORA_TARGET_MODULES_LM")
        target_modules_dit = _get_optional_str_list_env("NANOVLLM_LORA_TARGET_MODULES_DIT")
        target_proj_modules = _get_optional_str_list_env("NANOVLLM_LORA_TARGET_PROJ_MODULES")

        if lora_max_loras <= 0:
            raise RuntimeError("NANOVLLM_LORA_MAX_LORAS must be > 0")
        if lora_max_lora_rank <= 0:
            raise RuntimeError("NANOVLLM_LORA_MAX_LORA_RANK must be > 0")
        if lora_enable_lm is False and lora_enable_dit is False and lora_enable_proj is False:
            raise RuntimeError("At least one of NANOVLLM_LORA_ENABLE_LM/DIT/PROJ must be true")

        runtime_lora_config = RuntimeLoRAConfig(
            enable_lm=lora_enable_lm,
            enable_dit=lora_enable_dit,
            enable_proj=lora_enable_proj,
            max_loras=lora_max_loras,
            max_lora_rank=lora_max_lora_rank,
            target_modules_lm=target_modules_lm,
            target_modules_dit=target_modules_dit,
            target_proj_modules=target_proj_modules,
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
        opus=OpusConfig(bitrate=opus_bitrate, frame_ms=opus_frame_ms),
        asr=ASRConfig(api_url=asr_api_url, model=asr_model, timeout=asr_timeout, api_key=asr_api_key),
        llm=LLMConfig(api_key=gemini_api_key, model=gemini_model, timeout=llm_timeout, prompt_template=prompt_template),
        server_pool=ServerPoolStartupConfig(
            max_num_batched_tokens=pool_max_num_batched_tokens,
            max_num_seqs=pool_max_num_seqs,
            max_model_len=pool_max_model_len,
            gpu_memory_utilization=pool_gpu_memory_utilization,
            enforce_eager=pool_enforce_eager,
            devices=pool_devices,
        ),
        lora=runtime_lora_config,
    )


def _get_optional_bool_env(name: str) -> bool | None:
    if os.environ.get(name) in (None, ""):
        return None
    return _get_bool_env(name, False)


def _get_optional_str_list_env(name: str) -> tuple[str, ...] | None:
    if os.environ.get(name) in (None, ""):
        return None
    return _get_str_list_env(name, ())


VALID_SPEAKERS = ("alaa", "hammad", "hanan", "khalil")


@dataclass(frozen=True)
class SpeakerEnvConfig:
    """Raw per-speaker paths and prompt template read from env vars."""

    prompt_audio_path: str  # may be empty
    prompt_text_path: str  # may be empty
    ref_audio_path: str  # may be empty
    ref_audio_format: str | None  # optional override
    gemini_prompt_template: str  # may be empty → fallback to global


def load_speaker_env_configs() -> dict[str, SpeakerEnvConfig]:
    """Read per-speaker env vars for all speakers.

    Env var pattern:
        NANOVLLM_SPEAKER_<NAME>_PROMPT_AUDIO_PATH
        NANOVLLM_SPEAKER_<NAME>_PROMPT_TEXT_PATH
        NANOVLLM_SPEAKER_<NAME>_REF_AUDIO_PATH
        NANOVLLM_SPEAKER_<NAME>_REF_AUDIO_FORMAT
        GEMINI_PROMPT_TEMPLATE_<NAME>
    """
    speakers: dict[str, SpeakerEnvConfig] = {}
    for name in VALID_SPEAKERS:
        key = name.upper()
        speakers[name] = SpeakerEnvConfig(
            prompt_audio_path=os.environ.get(f"NANOVLLM_SPEAKER_{key}_PROMPT_AUDIO_PATH", "").strip(),
            prompt_text_path=os.environ.get(f"NANOVLLM_SPEAKER_{key}_PROMPT_TEXT_PATH", "").strip(),
            ref_audio_path=os.environ.get(f"NANOVLLM_SPEAKER_{key}_REF_AUDIO_PATH", "").strip(),
            ref_audio_format=os.environ.get(f"NANOVLLM_SPEAKER_{key}_REF_AUDIO_FORMAT", "").strip() or None,
            gemini_prompt_template=os.environ.get(f"GEMINI_PROMPT_TEMPLATE_{key}", "").strip(),
        )
    return speakers


def materialize_lora_config(config: RuntimeLoRAConfig, architecture: str) -> MaterializedRuntimeLoRAConfig:
    default_proj_targets: tuple[str, ...]
    if architecture == "voxcpm":
        default_proj_targets = VOXCPM_PROJ_LORA_TARGETS
    elif architecture == "voxcpm2":
        default_proj_targets = VOXCPM2_PROJ_LORA_TARGETS
    else:
        raise RuntimeError(f"Unsupported model architecture for runtime LoRA: {architecture}")

    enable_lm = True if config.enable_lm is None else config.enable_lm
    enable_dit = True if config.enable_dit is None else config.enable_dit
    enable_proj = True if config.enable_proj is None else config.enable_proj

    return MaterializedRuntimeLoRAConfig(
        enable_lm=enable_lm,
        enable_dit=enable_dit,
        enable_proj=enable_proj,
        max_loras=config.max_loras,
        max_lora_rank=config.max_lora_rank,
        target_modules_lm=config.target_modules_lm or ALL_LINEAR_LORA_TARGETS,
        target_modules_dit=config.target_modules_dit or ALL_LINEAR_LORA_TARGETS,
        target_proj_modules=config.target_proj_modules or default_proj_targets,
    )
