from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from typing import Any
from urllib.parse import urlparse, urlunparse

ATTENTION_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
VOXCPM_PROJ_LORA_TARGETS = ("enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj")
VOXCPM2_PROJ_LORA_TARGETS = (*VOXCPM_PROJ_LORA_TARGETS, "fusion_concat_proj")


def add_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-loras",
        type=int,
        default=0,
        help="Enable LoRA and register this many aliases when > 0; 0 keeps the traditional path",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=8,
        help="Maximum LoRA rank when --max-loras > 0",
    )
    parser.add_argument(
        "--lora-path",
        default="models/lora_10pct_ref/latest",
        help="LoRA checkpoint directory to register when --max-loras > 0",
    )
    parser.add_argument(
        "--lora-name-prefix",
        default=None,
        help="Prefix for registered LoRA names; defaults to a process-unique benchmark prefix",
    )
    parser.add_argument(
        "--num-lora-names",
        type=int,
        default=None,
        help="How many LoRA aliases to register; defaults to --max-loras when omitted",
    )
    parser.add_argument(
        "--lora-random-seed",
        type=int,
        default=None,
        help="Seed for random per-request LoRA selection",
    )


def validate_lora_args(args: argparse.Namespace) -> None:
    if args.max_loras < 0:
        raise ValueError("--max-loras must be >= 0")
    if args.max_loras == 0:
        return
    if args.max_lora_rank <= 0:
        raise ValueError("--max-lora-rank must be > 0")
    if not args.lora_path:
        raise ValueError("--lora-path is required when --max-loras > 0")
    if args.num_lora_names is not None and args.num_lora_names <= 0:
        raise ValueError("--num-lora-names must be > 0 when provided")


def make_lora_names(args: argparse.Namespace) -> list[str]:
    if args.max_loras <= 0:
        return []
    num_lora_names = args.num_lora_names if args.num_lora_names is not None else args.max_loras
    prefix = args.lora_name_prefix or f"bench_lora_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    return [f"{prefix}_{i}" for i in range(int(num_lora_names))]


def make_lora_rng(args: argparse.Namespace) -> random.Random:
    return random.Random(args.lora_random_seed)


def choose_lora_name(lora_names: list[str], rng: random.Random) -> str | None:
    if not lora_names:
        return None
    return rng.choice(lora_names)


def add_lora_to_payload(payload: dict[str, Any], lora_name: str | None) -> dict[str, Any]:
    if lora_name is not None:
        payload["lora_name"] = lora_name
    return payload


def build_lora_config(model: str, *, max_loras: int, max_lora_rank: int) -> Any:
    if max_loras <= 0:
        return None

    architecture = _read_model_architecture(model)
    kwargs = {
        "enable_lm": True,
        "enable_dit": True,
        "enable_proj": False,
        "max_loras": max_loras,
        "max_lora_rank": max_lora_rank,
        "target_modules_lm": list(ATTENTION_LORA_TARGETS),
        "target_modules_dit": list(ATTENTION_LORA_TARGETS),
        "target_proj_modules": list(_proj_targets_for_architecture(architecture)),
    }

    if architecture == "voxcpm":
        from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig
    elif architecture == "voxcpm2":
        from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig
    else:
        raise ValueError(f"Unsupported model architecture for LoRA: {architecture}")

    return LoRAConfig(**kwargs)


async def register_loras_in_process(server_pool: Any, lora_names: list[str], lora_path: str | None) -> None:
    if not lora_names:
        return
    assert lora_path is not None
    for name in lora_names:
        await server_pool.register_lora(name, lora_path)


async def unregister_loras_in_process(server_pool: Any, lora_names: list[str]) -> list[str]:
    errors: list[str] = []
    for name in lora_names:
        try:
            await server_pool.unregister_lora(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return errors


async def register_loras_http(
    generate_url: str, lora_names: list[str], lora_path: str | None, *, timeout_s: float
) -> None:
    if not lora_names:
        return
    assert lora_path is not None
    import asyncio

    loras_url = _loras_url_from_generate_url(generate_url)
    for name in lora_names:
        await asyncio.to_thread(_http_json_request, "POST", loras_url, {"name": name, "path": lora_path}, timeout_s)


async def unregister_loras_http(generate_url: str, lora_names: list[str], *, timeout_s: float) -> list[str]:
    if not lora_names:
        return []
    import asyncio

    loras_url = _loras_url_from_generate_url(generate_url)
    errors: list[str] = []
    for name in lora_names:
        try:
            await asyncio.to_thread(_http_json_request, "DELETE", f"{loras_url.rstrip('/')}/{name}", None, timeout_s)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    return errors


def _read_model_architecture(model: str) -> str:
    model_path = os.path.expanduser(model)
    if not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(repo_id=model)

    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file `{config_path}` not found")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    architecture = config.get("architecture")
    if not isinstance(architecture, str):
        raise ValueError(f"Model config `{config_path}` does not define string field `architecture`")
    return architecture


def _proj_targets_for_architecture(architecture: str) -> tuple[str, ...]:
    if architecture == "voxcpm":
        return VOXCPM_PROJ_LORA_TARGETS
    if architecture == "voxcpm2":
        return VOXCPM2_PROJ_LORA_TARGETS
    raise ValueError(f"Unsupported model architecture for LoRA: {architecture}")


def _loras_url_from_generate_url(generate_url: str) -> str:
    parsed = urlparse(generate_url)
    path = parsed.path or "/"
    if path.rstrip("/").endswith("/generate"):
        path = f"{path.rstrip('/')[: -len('/generate')]}/loras"
    else:
        path = "/loras"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _http_json_request(method: str, url: str, payload: dict[str, Any] | None, timeout_s: float) -> None:
    import http.client
    import ssl

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {parsed.scheme}")
    if parsed.hostname is None:
        raise ValueError("invalid URL host")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    body = b"" if payload is None else json.dumps(payload, ensure_ascii=True).encode("utf-8")
    headers = {"Content-Type": "application/json", "Connection": "close"}
    ctx = ssl.create_default_context() if parsed.scheme == "https" else None
    if parsed.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            parsed.hostname,
            port=port,
            timeout=timeout_s,
            context=ctx,
        )
    else:
        conn = http.client.HTTPConnection(parsed.hostname, port=port, timeout=timeout_s)

    try:
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        response_body = resp.read(4096)
        if resp.status < 200 or resp.status >= 300:
            msg = response_body.decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {url} failed with HTTP {resp.status}: {msg}".strip())
    finally:
        conn.close()
