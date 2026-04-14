import asyncio

import pytest


class _FakeServer:
    def __init__(self):
        self.registered = []
        self.unregistered = []
        self.generate_calls = []

    async def register_lora(self, name, path):
        self.registered.append((name, path))
        return {"name": name}

    async def unregister_lora(self, name):
        self.unregistered.append(name)
        return {"name": name}

    async def generate(
        self,
        target_text,
        prompt_latents=None,
        prompt_text="",
        max_generate_length=2000,
        temperature=1.0,
        cfg_value=2.0,
        ref_audio_latents=None,
        lora_name=None,
    ):
        self.generate_calls.append({"lora_name": lora_name, "ref_audio_latents": ref_audio_latents})
        yield "chunk"


class _FailingUnregisterServer(_FakeServer):
    async def unregister_lora(self, name):
        raise RuntimeError("boom")


async def _exercise_async_server_pool_register_list_generate_and_unregister():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool

    pool = object.__new__(AsyncVoxCPM2ServerPool)
    pool.servers = [_FakeServer(), _FakeServer()]
    pool.servers_load = __import__("numpy").zeros(2, dtype=__import__("numpy").int32)
    pool._prompt_pool = {}
    pool._registered_loras = set()
    pool._draining_loras = set()

    await pool.register_lora("demo", "/tmp/demo")
    assert await pool.list_loras() == [{"name": "demo"}]

    chunks = []
    async for chunk in pool.generate("hello", lora_name="demo"):
        chunks.append(chunk)
    assert chunks == ["chunk"]
    assert pool.servers[0].generate_calls[0]["lora_name"] == "demo"

    await pool.unregister_lora("demo")
    assert await pool.list_loras() == []


def test_async_server_pool_register_list_generate_and_unregister():
    asyncio.run(_exercise_async_server_pool_register_list_generate_and_unregister())


async def _exercise_async_server_pool_rejects_unknown_lora_name():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool

    pool = object.__new__(AsyncVoxCPM2ServerPool)
    pool.servers = [_FakeServer()]
    pool.servers_load = __import__("numpy").zeros(1, dtype=__import__("numpy").int32)
    pool._prompt_pool = {}
    pool._registered_loras = set()
    pool._draining_loras = set()

    with pytest.raises(ValueError, match="not registered"):
        async for _ in pool.generate("hello", lora_name="missing"):
            pass


def test_async_server_pool_rejects_unknown_lora_name():
    asyncio.run(_exercise_async_server_pool_rejects_unknown_lora_name())


async def _exercise_async_server_pool_unregister_partial_failure_enters_draining():
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool

    first = _FakeServer()
    second = _FailingUnregisterServer()
    pool = object.__new__(AsyncVoxCPM2ServerPool)
    pool.servers = [first, second]
    pool.servers_load = __import__("numpy").zeros(2, dtype=__import__("numpy").int32)
    pool._prompt_pool = {}
    pool._registered_loras = {"demo"}
    pool._draining_loras = set()

    with pytest.raises(RuntimeError, match="boom"):
        await pool.unregister_lora("demo")

    assert first.unregistered == ["demo"]
    assert pool._draining_loras == {"demo"}
    assert await pool.list_loras() == []
    with pytest.raises(ValueError, match="not registered"):
        async for _ in pool.generate("hello", lora_name="demo"):
            pass


def test_async_server_pool_unregister_partial_failure_enters_draining():
    asyncio.run(_exercise_async_server_pool_unregister_partial_failure_enters_draining())
