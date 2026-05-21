from __future__ import annotations

import asyncio
import concurrent.futures
import json
import queue
import signal
import threading
from types import SimpleNamespace


class _FakeLLM:
    feat_dim = 4
    patch_size = 2

    def __init__(self, config):
        self.model_runner = SimpleNamespace(vae=SimpleNamespace(sample_rate=16000, out_sample_rate=24000))

    def is_finished(self):
        return True

    def step(self):
        return []

    def list_loras(self):
        return []


def _patch_voxcpm2_fake_llm(monkeypatch):
    import nanovllm_voxcpm.models.voxcpm2.server as server_mod

    monkeypatch.setattr(server_mod, "VoxCPM2Engine", _FakeLLM)
    monkeypatch.setattr(server_mod.VoxCPM2Config, "model_validate_json", lambda raw: SimpleNamespace())
    return server_mod, server_mod.AsyncVoxCPM2Server


def _patch_voxcpm_fake_llm(monkeypatch):
    import nanovllm_voxcpm.models.voxcpm.server as server_mod

    monkeypatch.setattr(server_mod, "VoxCPMEngine", _FakeLLM)
    monkeypatch.setattr(server_mod.VoxCPMConfig, "model_validate_json", lambda raw: SimpleNamespace())
    return server_mod, server_mod.AsyncVoxCPMServer


class _ThreadProcess:
    def __init__(self, target, args, kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon
        self.exitcode = None
        self._thread = threading.Thread(target=self._run, daemon=daemon)

    def _run(self):
        try:
            self._target(*self._args, **self._kwargs)
            self.exitcode = 0
        except BaseException:
            self.exitcode = 1
            raise

    def start(self):
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def join(self, timeout=None):
        self._thread.join(timeout=timeout)

    def terminate(self):
        self.exitcode = -15

    def kill(self):
        self.exitcode = -9


class _ThreadContext:
    Queue = queue.Queue
    Process = _ThreadProcess


async def _exercise_control_plane_while_default_executor_is_saturated(server_mod, async_server_cls, model_path):
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)

    unblock_executor = threading.Event()
    executor_blocker = loop.run_in_executor(executor, unblock_executor.wait)
    await asyncio.sleep(0)

    server = async_server_cls(model_path=str(model_path))

    try:
        await asyncio.wait_for(server.wait_for_ready(), timeout=0.2)
        await asyncio.wait_for(server.get_model_info(), timeout=0.2)
    finally:
        unblock_executor.set()
        await executor_blocker
        await server.stop()
        assert not server.process.is_alive()


def test_voxcpm2_control_plane_survives_default_executor_starvation(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({}), encoding="utf-8")
    server_mod, async_server_cls = _patch_voxcpm2_fake_llm(monkeypatch)
    monkeypatch.setattr(server_mod.mp, "get_context", lambda method: _ThreadContext())
    monkeypatch.setattr(signal, "signal", lambda *args: None)

    asyncio.run(_exercise_control_plane_while_default_executor_is_saturated(server_mod, async_server_cls, model_path))


def test_voxcpm_control_plane_survives_default_executor_starvation(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({}), encoding="utf-8")
    server_mod, async_server_cls = _patch_voxcpm_fake_llm(monkeypatch)
    monkeypatch.setattr(server_mod.mp, "get_context", lambda method: _ThreadContext())
    monkeypatch.setattr(signal, "signal", lambda *args: None)

    asyncio.run(_exercise_control_plane_while_default_executor_is_saturated(server_mod, async_server_cls, model_path))
