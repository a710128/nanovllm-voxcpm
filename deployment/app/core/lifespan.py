from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from huggingface_hub import snapshot_download

from nanovllm_voxcpm.llm import VoxCPM

from app.core.config import ServiceConfig, materialize_lora_config

SERVER_FACTORY = VoxCPM.from_pretrained


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

            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            await server.stop()
            if getattr(app.state, "server", None) is server:
                delattr(app.state, "server")
            if getattr(app.state, "model_architecture", None) is model_architecture:
                delattr(app.state, "model_architecture")

    return lifespan
