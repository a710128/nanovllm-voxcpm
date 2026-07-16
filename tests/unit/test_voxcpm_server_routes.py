"""Unit tests for nanovllm_voxcpm.models.voxcpm.server route/validation logic.

All tests run without a real GPU model.  The ``AsyncVoxCPMServerPool`` is
constructed via ``object.__new__`` (bypassing ``__init__``) and its ``servers``
list is replaced with lightweight fakes – the same technique used in
``test_voxcpm_serverpool_lora_api.py``.

Tests that cover ``VoxCPMServerImpl`` instance methods use a synthetic object
created with ``__new__`` and a fake ``llm`` attribute.
"""

from __future__ import annotations

import asyncio
import io

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fake server helpers
# ---------------------------------------------------------------------------


class _FakeServer:
    """Minimal fake for AsyncVoxCPMServer used inside pool tests."""

    def __init__(self):
        self.registered: list[tuple[str, str]] = []
        self.unregistered: list[str] = []
        self.generate_calls: list[dict] = []
        self.encode_calls: list[tuple[bytes, str]] = []

    async def register_lora(self, name: str, path: str):
        self.registered.append((name, path))
        return {"name": name}

    async def unregister_lora(self, name: str):
        self.unregistered.append(name)
        return {"name": name}

    async def get_model_info(self):
        return {
            "sample_rate": 16000,
            "channels": 1,
            "feat_dim": 4,
            "patch_size": 1,
            "model_path": "/fake",
        }

    async def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        self.encode_calls.append((wav, wav_format))
        return np.zeros((4,), dtype=np.float32).tobytes()

    async def generate(
        self,
        target_text,
        prompt_latents=None,
        prompt_text="",
        max_generate_length=2000,
        temperature=1.0,
        cfg_value=2.0,
        lora_name=None,
        seed=42,
    ):
        self.generate_calls.append({"target_text": target_text, "lora_name": lora_name})
        yield "chunk"


class _FailRegisterServer(_FakeServer):
    async def register_lora(self, name: str, path: str):
        raise RuntimeError("register_failed")


class _FailUnregisterServer(_FakeServer):
    async def unregister_lora(self, name: str):
        raise RuntimeError("unregister_failed")


# ---------------------------------------------------------------------------
# Helper: build a bare AsyncVoxCPMServerPool
# ---------------------------------------------------------------------------


def _make_pool(servers):
    from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool

    pool = object.__new__(AsyncVoxCPMServerPool)
    pool.servers = list(servers)
    pool.servers_load = np.zeros(len(servers), dtype=np.int32)
    pool._prompt_pool = {}
    pool._registered_loras = set()
    pool._draining_loras = set()
    return pool


# ---------------------------------------------------------------------------
# gen_uuid
# ---------------------------------------------------------------------------


def test_gen_uuid_returns_hex_string():
    from nanovllm_voxcpm.models.voxcpm.server import gen_uuid

    uid = gen_uuid()
    assert isinstance(uid, str)
    assert len(uid) == 32
    assert uid != gen_uuid()  # unique each call


# ---------------------------------------------------------------------------
# VoxCPMServerImpl – unit-level (no engine)
# ---------------------------------------------------------------------------


class _FakeLLM:
    feat_dim = 4
    patch_size = 1

    def __init__(self):
        self.added_requests: list[dict] = []
        self.registered: list[tuple[str, str]] = []
        self.unregistered: list[str] = []
        self.loras: list = []

    def add_request(self, **kwargs):
        self.added_requests.append(kwargs)

    def register_lora(self, name: str, path: str):
        self.registered.append((name, path))

    def unregister_lora(self, name: str):
        self.unregistered.append(name)

    def list_loras(self):
        return self.loras

    def cancel_sequence(self, seq_id: str):
        pass

    def step(self):
        return []

    def is_finished(self):
        return True


def _make_server_impl(feat_dim: int = 4, patch_size: int = 1):
    from nanovllm_voxcpm.models.voxcpm.server import VoxCPMServerImpl

    srv = VoxCPMServerImpl.__new__(VoxCPMServerImpl)
    srv.sample_rate = 16000
    srv.model_path = "/fake/model"
    llm = _FakeLLM()
    llm.feat_dim = feat_dim
    llm.patch_size = patch_size
    srv.llm = llm
    return srv


def test_server_impl_health():
    srv = _make_server_impl()
    assert srv.health() == {"status": "ok"}


def test_server_impl_get_model_info():
    srv = _make_server_impl(feat_dim=8, patch_size=2)
    info = srv.get_model_info()
    assert info["sample_rate"] == 16000
    assert info["feat_dim"] == 8
    assert info["patch_size"] == 2
    assert info["model_path"] == "/fake/model"
    assert info["channels"] == 1


def test_server_impl_register_and_list_and_unregister_lora():
    from nanovllm_voxcpm.models.voxcpm.server import LoRAInfo

    srv = _make_server_impl()

    class _LLMEntry:
        def __init__(self, name):
            self.name = name

    srv.llm.loras = [_LLMEntry("alpha")]
    srv.register_lora("alpha", "/tmp/alpha")
    assert srv.llm.registered == [("alpha", "/tmp/alpha")]

    loras = srv.list_loras()
    assert loras == [LoRAInfo(name="alpha")]

    srv.unregister_lora("alpha")
    assert "alpha" in srv.llm.unregistered


def test_server_impl_cancel_and_step_and_is_finished():
    srv = _make_server_impl()
    # These are thin wrappers; just assert they don't raise.
    srv.cancel("some-seq-id")
    result = srv.step()
    assert result == []
    assert srv.is_finished() is True


# ---------------------------------------------------------------------------
# VoxCPMServerImpl.add_request – validation paths
# ---------------------------------------------------------------------------


def test_add_request_no_prompt_latents_no_prompt_text_ok():
    srv = _make_server_impl()
    srv.add_request("seq1", "hello")
    assert len(srv.llm.added_requests) == 1
    req = srv.llm.added_requests[0]
    assert req["seq_id"] == "seq1"
    assert req["target_text"] == "hello"
    assert req["prompt_text"] == ""


def test_add_request_no_prompt_latents_with_prompt_text_raises():
    srv = _make_server_impl()
    with pytest.raises(ValueError, match="Prompt text is not allowed"):
        srv.add_request("seq1", "hello", prompt_text="some text")


def test_add_request_with_prompt_latents_and_prompt_text_ok():
    srv = _make_server_impl(feat_dim=4)
    latents_bytes = np.ones((2, 4), dtype=np.float32).tobytes()
    srv.add_request("seq1", "hello", prompt_latents=latents_bytes, prompt_text="world")
    assert len(srv.llm.added_requests) == 1
    req = srv.llm.added_requests[0]
    assert req["prompt_text"] == "world"
    assert req["prompt_latents"].shape == (2, 4)


def test_add_request_with_prompt_latents_but_no_prompt_text_raises():
    srv = _make_server_impl(feat_dim=4)
    latents_bytes = np.ones((2, 4), dtype=np.float32).tobytes()
    with pytest.raises(ValueError, match="Prompt text is required"):
        srv.add_request("seq1", "hello", prompt_latents=latents_bytes)


def test_add_request_passes_lora_name_and_seed():
    srv = _make_server_impl()
    srv.add_request("seq1", "hi", lora_name="my_lora", seed=99)
    req = srv.llm.added_requests[0]
    assert req["lora_name"] == "my_lora"
    assert req["seed"] == 99


# ---------------------------------------------------------------------------
# VoxCPMServerImpl.encode_latents – resampling + stereo paths
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_encode_latents_resamples_when_sr_mismatch(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm.server import VoxCPMServerImpl
    import torchaudio

    srv = VoxCPMServerImpl.__new__(VoxCPMServerImpl)
    srv.sample_rate = 16000

    class _LLM:
        patch_size = 1

        def encode_latents(self, wav_tensor):
            return np.zeros((2, 4), dtype=np.float32)

    srv.llm = _LLM()

    resample_called = {}

    def _fake_load(file_obj, format):
        # Return 8000 Hz so a resample is triggered.
        return torch.zeros((1, 160), dtype=torch.float32), 8000

    def _fake_resample(wav, orig_sr, new_sr):
        resample_called["orig"] = orig_sr
        resample_called["new"] = new_sr
        return torch.zeros((1, 320), dtype=torch.float32)

    monkeypatch.setattr("nanovllm_voxcpm.models.voxcpm.server.torchaudio.load", _fake_load)
    monkeypatch.setattr("nanovllm_voxcpm.models.voxcpm.server.torchaudio.functional.resample", _fake_resample)

    out = srv.encode_latents(b"fake-wav", "wav")

    assert isinstance(out, bytes)
    assert resample_called.get("orig") == 8000
    assert resample_called.get("new") == 16000


def test_encode_latents_converts_stereo_to_mono(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm.server import VoxCPMServerImpl

    srv = VoxCPMServerImpl.__new__(VoxCPMServerImpl)
    srv.sample_rate = 16000

    captured = {}

    class _LLM:
        patch_size = 1

        def encode_latents(self, wav_tensor):
            captured["shape"] = wav_tensor.shape
            return np.zeros((2, 4), dtype=np.float32)

    srv.llm = _LLM()

    # Return 2-channel (stereo) audio at the correct rate so only mono-conversion runs.
    def _fake_load(file_obj, format):
        return torch.zeros((2, 160), dtype=torch.float32), 16000

    monkeypatch.setattr("nanovllm_voxcpm.models.voxcpm.server.torchaudio.load", _fake_load)

    srv.encode_latents(b"fake-wav", "wav")

    # Stereo → mono: first dim should be 1.
    assert captured["shape"][0] == 1


# ---------------------------------------------------------------------------
# AsyncVoxCPMServerPool – get_model_info empty-pool guard
# ---------------------------------------------------------------------------


async def _pool_get_model_info_empty():
    pool = _make_pool([])
    with pytest.raises(RuntimeError, match="empty"):
        await pool.get_model_info()


def test_pool_get_model_info_raises_on_empty_pool():
    asyncio.run(_pool_get_model_info_empty())


async def _pool_get_model_info_single_server():
    pool = _make_pool([_FakeServer()])
    info = await pool.get_model_info()
    assert info["sample_rate"] == 16000


def test_pool_get_model_info_delegates_to_first_server():
    asyncio.run(_pool_get_model_info_single_server())


# ---------------------------------------------------------------------------
# AsyncVoxCPMServerPool – register_lora duplicate guard
# ---------------------------------------------------------------------------


async def _pool_register_lora_duplicate():
    pool = _make_pool([_FakeServer()])
    pool._registered_loras = {"demo"}
    with pytest.raises(ValueError, match="already registered"):
        await pool.register_lora("demo", "/tmp/demo")


def test_pool_register_lora_duplicate_raises():
    asyncio.run(_pool_register_lora_duplicate())


async def _pool_register_lora_draining_guard():
    pool = _make_pool([_FakeServer()])
    pool._draining_loras = {"demo"}
    with pytest.raises(ValueError, match="already registered"):
        await pool.register_lora("demo", "/tmp/demo")


def test_pool_register_lora_draining_guard_raises():
    asyncio.run(_pool_register_lora_draining_guard())


async def _pool_register_lora_rolls_back_on_failure():
    good = _FakeServer()
    bad = _FailRegisterServer()
    pool = _make_pool([good, bad])

    with pytest.raises(RuntimeError, match="register_failed"):
        await pool.register_lora("alpha", "/tmp/alpha")

    # The good server should have been rolled back.
    assert "alpha" not in pool._registered_loras
    assert len(good.unregistered) == 1
    assert good.unregistered[0] == "alpha"


def test_pool_register_lora_rolls_back_on_failure():
    asyncio.run(_pool_register_lora_rolls_back_on_failure())


# ---------------------------------------------------------------------------
# AsyncVoxCPMServerPool – unregister_lora guards
# ---------------------------------------------------------------------------


async def _pool_unregister_not_registered():
    pool = _make_pool([_FakeServer()])
    with pytest.raises(ValueError, match="not registered"):
        await pool.unregister_lora("missing")


def test_pool_unregister_lora_not_registered_raises():
    asyncio.run(_pool_unregister_not_registered())


async def _pool_unregister_already_draining():
    pool = _make_pool([_FakeServer()])
    pool._registered_loras = {"demo"}
    pool._draining_loras = {"demo"}
    with pytest.raises(ValueError, match="already draining"):
        await pool.unregister_lora("demo")


def test_pool_unregister_lora_already_draining_raises():
    asyncio.run(_pool_unregister_already_draining())


async def _pool_unregister_success():
    pool = _make_pool([_FakeServer(), _FakeServer()])
    pool._registered_loras = {"demo"}
    result = await pool.unregister_lora("demo")
    assert result == {"name": "demo"}
    assert "demo" not in pool._registered_loras
    assert "demo" not in pool._draining_loras


def test_pool_unregister_lora_success():
    asyncio.run(_pool_unregister_success())


# ---------------------------------------------------------------------------
# AsyncVoxCPMServerPool – encode_latents delegates to min-load server
# ---------------------------------------------------------------------------


async def _pool_encode_latents_routes_to_min_load():
    s0 = _FakeServer()
    s1 = _FakeServer()
    pool = _make_pool([s0, s1])
    pool.servers_load[0] = 5  # s1 has lower load

    out = await pool.encode_latents(b"data", "wav")
    assert isinstance(out, bytes)
    assert len(s1.encode_calls) == 1
    assert len(s0.encode_calls) == 0


def test_pool_encode_latents_routes_to_min_load_server():
    asyncio.run(_pool_encode_latents_routes_to_min_load())


# ---------------------------------------------------------------------------
# AsyncVoxCPMServerPool – add_prompt and remove_prompt
# ---------------------------------------------------------------------------


async def _pool_add_and_remove_prompt():
    pool = _make_pool([_FakeServer()])

    prompt_id = await pool.add_prompt(b"wav_bytes", "wav", "prompt text")
    assert isinstance(prompt_id, str)
    assert prompt_id in pool._prompt_pool
    assert pool._prompt_pool[prompt_id]["text"] == "prompt text"

    await pool.remove_prompt(prompt_id)
    assert prompt_id not in pool._prompt_pool


def test_pool_add_and_remove_prompt():
    asyncio.run(_pool_add_and_remove_prompt())


# ---------------------------------------------------------------------------
# AsyncVoxCPMServerPool – generate with prompt_id validation
# ---------------------------------------------------------------------------


async def _pool_generate_unknown_prompt_id():
    pool = _make_pool([_FakeServer()])
    with pytest.raises(ValueError, match="not found"):
        async for _ in pool.generate("hello", prompt_id="bad-id"):
            pass


def test_pool_generate_unknown_prompt_id_raises():
    asyncio.run(_pool_generate_unknown_prompt_id())


async def _pool_generate_prompt_id_and_latents_conflict():
    pool = _make_pool([_FakeServer()])
    pool._prompt_pool["pid1"] = {"latents": b"", "text": "t"}
    with pytest.raises(ValueError, match="cannot be provided at the same time"):
        async for _ in pool.generate(
            "hello",
            prompt_id="pid1",
            prompt_latents=np.zeros((1, 4), dtype=np.float32).tobytes(),
        ):
            pass


def test_pool_generate_prompt_id_and_latents_conflict_raises():
    asyncio.run(_pool_generate_prompt_id_and_latents_conflict())


async def _pool_generate_prompt_id_and_text_conflict():
    pool = _make_pool([_FakeServer()])
    pool._prompt_pool["pid1"] = {"latents": b"", "text": "t"}
    with pytest.raises(ValueError, match="cannot be provided at the same time"):
        async for _ in pool.generate("hello", prompt_id="pid1", prompt_text="extra"):
            pass


def test_pool_generate_prompt_id_and_text_conflict_raises():
    asyncio.run(_pool_generate_prompt_id_and_text_conflict())


async def _pool_generate_resolves_prompt_id():
    s = _FakeServer()
    pool = _make_pool([s])
    lat = np.zeros((2, 4), dtype=np.float32).tobytes()
    pool._prompt_pool["pid1"] = {"latents": lat, "text": "my-text"}

    chunks = []
    async for chunk in pool.generate("hello", prompt_id="pid1"):
        chunks.append(chunk)

    assert chunks == ["chunk"]
    # Prompt latents should be forwarded to the underlying server.generate call.
    assert s.generate_calls[0]["target_text"] == "hello"


def test_pool_generate_resolves_prompt_id():
    asyncio.run(_pool_generate_resolves_prompt_id())


async def _pool_generate_lora_draining_rejected():
    pool = _make_pool([_FakeServer()])
    pool._registered_loras = {"lora1"}
    pool._draining_loras = {"lora1"}
    with pytest.raises(ValueError, match="not registered"):
        async for _ in pool.generate("hello", lora_name="lora1"):
            pass


def test_pool_generate_lora_draining_rejected():
    asyncio.run(_pool_generate_lora_draining_rejected())


async def _pool_generate_updates_load_counter():
    s = _FakeServer()
    pool = _make_pool([s])

    chunks = []
    async for chunk in pool.generate("hello"):
        # Load should be 1 while iterating.
        assert pool.servers_load[0] == 1
        chunks.append(chunk)

    # Load should return to 0 after the generator finishes.
    assert pool.servers_load[0] == 0
    assert chunks == ["chunk"]


def test_pool_generate_updates_load_counter():
    asyncio.run(_pool_generate_updates_load_counter())


# ---------------------------------------------------------------------------
# AsyncVoxCPMServerPool – wait_for_ready and stop delegate to servers
# ---------------------------------------------------------------------------


class _FakeServerWithReadyStop(_FakeServer):
    def __init__(self):
        super().__init__()
        self.ready_called = False
        self.stop_called = False

    async def wait_for_ready(self):
        self.ready_called = True

    async def stop(self):
        self.stop_called = True


async def _pool_wait_for_ready_and_stop():
    s0 = _FakeServerWithReadyStop()
    s1 = _FakeServerWithReadyStop()
    pool = _make_pool([s0, s1])

    await pool.wait_for_ready()
    assert s0.ready_called and s1.ready_called

    await pool.stop()
    assert s0.stop_called and s1.stop_called


def test_pool_wait_for_ready_and_stop():
    asyncio.run(_pool_wait_for_ready_and_stop())


def test_async_server_rejects_unknown_kwargs_before_starting_process():
    from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServer

    with pytest.raises(ValueError, match="Unknown kwargs"):
        AsyncVoxCPMServer("/fake/model", unsupported=True)


def test_async_server_pool_rejects_unknown_kwargs_before_starting_servers():
    from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool

    with pytest.raises(ValueError, match="Unknown kwargs"):
        AsyncVoxCPMServerPool("/fake/model", unsupported=True)


async def _async_server_forwards_control_plane_calls():
    from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServer

    server = object.__new__(AsyncVoxCPMServer)
    calls = []

    async def submit(command, *args):
        calls.append((command, args))
        return command

    server.submit = submit

    assert await server.health() == "health"
    assert await server.get_model_info() == "get_model_info"
    assert await server.encode_latents(b"wav", "wav") == "encode_latents"
    assert await server.register_lora("demo", "/tmp/demo") == "register_lora"
    assert await server.unregister_lora("demo") == "unregister_lora"
    assert await server.list_loras() == "list_loras"
    assert [command for command, _ in calls] == [
        "health",
        "get_model_info",
        "encode_latents",
        "register_lora",
        "unregister_lora",
        "list_loras",
    ]


def test_async_server_forwards_control_plane_calls():
    asyncio.run(_async_server_forwards_control_plane_calls())


async def _async_server_generate_completes_and_cleans_stream_state():
    from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServer

    server = object.__new__(AsyncVoxCPMServer)
    server.stream_table = {}
    commands = []

    async def submit(command, *args):
        commands.append(command)
        if command == "add_request":
            stream = server.stream_table[args[0]]
            await stream.put(np.array([1.0], dtype=np.float32))
            await stream.put(None)

    server.submit = submit

    chunks = [chunk async for chunk in server.generate("hello")]

    assert len(chunks) == 1
    assert chunks[0].tolist() == [1.0]
    assert commands == ["add_request"]
    assert server.stream_table == {}


def test_async_server_generate_completes_and_cleans_stream_state():
    asyncio.run(_async_server_generate_completes_and_cleans_stream_state())


async def _async_server_generate_cancels_when_consumer_closes_early():
    from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServer

    server = object.__new__(AsyncVoxCPMServer)
    server.stream_table = {}
    commands = []

    async def submit(command, *args):
        commands.append(command)
        if command == "add_request":
            await server.stream_table[args[0]].put(np.array([1.0], dtype=np.float32))

    server.submit = submit
    stream = server.generate("hello")

    await stream.__anext__()
    await stream.aclose()

    assert commands == ["add_request", "cancel"]
    assert server.stream_table == {}


def test_async_server_generate_cancels_when_consumer_closes_early():
    asyncio.run(_async_server_generate_cancels_when_consumer_closes_early())
