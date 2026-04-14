from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from nanovllm_voxcpm.llm import VoxCPM

from app.core.config import ServiceConfig

SERVER_FACTORY = VoxCPM.from_pretrained


def build_lifespan(cfg: ServiceConfig):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        server = SERVER_FACTORY(
            model=cfg.model_path,
            max_num_batched_tokens=cfg.server_pool.max_num_batched_tokens,
            max_num_seqs=cfg.server_pool.max_num_seqs,
            max_model_len=cfg.server_pool.max_model_len,
            gpu_memory_utilization=cfg.server_pool.gpu_memory_utilization,
            enforce_eager=cfg.server_pool.enforce_eager,
            devices=list(cfg.server_pool.devices),
            lora_config=None,
        )
        app.state.server = server
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

    return lifespan
