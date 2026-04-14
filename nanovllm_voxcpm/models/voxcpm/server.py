from nanovllm_voxcpm.models.voxcpm.engine import (
    VoxCPMEngine,
    VoxCPMRunner,
    VoxCPMConfig,
    Config,
)
from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig
import os
import torch.multiprocessing as mp
from queue import Empty
import traceback
import uuid
import torchaudio
import io
import time
import asyncio
import contextlib
from typing import Any, AsyncGenerator, List, Optional, cast
from typing_extensions import TypedDict, Literal
import numpy as np
from numpy.typing import NDArray

Waveform = NDArray[np.float32]


class HealthResponse(TypedDict):
    status: Literal["ok"]


class ModelInfoResponse(TypedDict):
    sample_rate: int
    channels: int
    feat_dim: int
    patch_size: int
    model_path: str


class LoRAInfo(TypedDict):
    name: str


class RegisterLoRAResponse(TypedDict):
    name: str


class UnregisterLoRAResponse(TypedDict):
    name: str


def gen_uuid() -> str:
    return uuid.uuid4().hex


class VoxCPMServerImpl:
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Optional[LoRAConfig] = None,
    ):
        model_config = VoxCPMConfig.model_validate_json(open(os.path.join(model_path, "config.json")).read())

        model_config.inference_timesteps = inference_timesteps
        self.lora_config = lora_config
        self.model_path = model_path

        engine_config = Config(
            model=model_path,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            model_config=model_config,
            devices=devices,
            lora_config=lora_config,
        )

        self.llm = VoxCPMEngine(engine_config)

        # VoxCPMRunner attaches VAE helpers; the base runner interface doesn't.
        model_runner = cast(VoxCPMRunner, self.llm.model_runner)
        self.sample_rate = model_runner.vae.sample_rate

    def health(self) -> HealthResponse:
        return HealthResponse(status="ok")

    def get_model_info(self) -> ModelInfoResponse:
        # Read-only metadata for HTTP services; avoids parsing config.json in wrappers.
        return ModelInfoResponse(
            sample_rate=int(self.sample_rate),
            channels=1,
            feat_dim=int(self.llm.feat_dim),
            patch_size=int(self.llm.patch_size),
            model_path=str(self.model_path),
        )

    def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        wav_tensor, sr = torchaudio.load(io.BytesIO(wav), format=wav_format)
        if sr != self.sample_rate:
            wav_tensor = torchaudio.functional.resample(wav_tensor, sr, self.sample_rate)

        if wav_tensor.size(0) > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)

        latents = self.llm.encode_latents(wav_tensor)
        assert latents.shape[0] % self.llm.patch_size == 0

        return latents.tobytes()

    def add_request(
        self,
        seq_id: str,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        lora_name: str | None = None,
    ) -> None:
        if prompt_latents is None:
            if len(prompt_text) > 0:
                raise ValueError("Prompt text is not allowed when prompt latents are not provided")
            self.llm.add_request(
                seq_id=seq_id,
                target_text=target_text,
                prompt_text="",
                max_generate_length=max_generate_length,
                temperature=temperature,
                cfg_value=cfg_value,
                lora_name=lora_name,
            )
            return

        if len(prompt_text) == 0:
            raise ValueError("Prompt text is required when prompt latents are provided")

        # Help static type checkers: prompt_latents is non-None here.
        assert prompt_latents is not None
        prompt_latents_buf: bytes = prompt_latents
        prompt_latents_arr: np.ndarray = np.frombuffer(prompt_latents_buf, dtype=np.float32).reshape(
            -1, self.llm.feat_dim
        )
        self.llm.add_request(
            seq_id=seq_id,
            target_text=target_text,
            prompt_text=prompt_text,
            prompt_latents=prompt_latents_arr,
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
            lora_name=lora_name,
        )

    def register_lora(self, name: str, path: str) -> RegisterLoRAResponse:
        self.llm.register_lora(name, path)
        return RegisterLoRAResponse(name=name)

    def unregister_lora(self, name: str) -> UnregisterLoRAResponse:
        self.llm.unregister_lora(name)
        return UnregisterLoRAResponse(name=name)

    def list_loras(self) -> list[LoRAInfo]:
        return [LoRAInfo(name=entry.name) for entry in self.llm.list_loras()]

    def cancel(self, seq_id: str):
        self.llm.cancel_sequence(seq_id)

    def step(self):
        return self.llm.step()

    def is_finished(self):
        return self.llm.is_finished()


def main_loop(queue_in: mp.Queue, queue_out: mp.Queue, args, kwargs):
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Small coalescing window for bursty arrivals.
    # Parse once so env lookup doesn't sit on the hot path.
    try:
        coalesce_ms = float(os.environ.get("NANOVLLM_QUEUE_COALESCE_MS", "2"))
    except ValueError:
        coalesce_ms = 2.0
    if coalesce_ms > 0:
        coalesce_ms = min(coalesce_ms, 50.0)

    try:
        srv = VoxCPMServerImpl(*args, **kwargs)
    except Exception:
        queue_out.put(
            {
                "type": "init_error",
                "error": traceback.format_exc(),
            }
        )
        return
    else:
        queue_out.put({"type": "init_ok"})

    states = {
        "is_stoped": False,
    }

    def method_call(cmd):
        opid = cmd.get("id", "")
        try:
            method_name = cmd["type"]
            args = cmd["args"]
            kwargs = cmd["kwargs"]

            if method_name == "stop":
                states["is_stoped"] = True
                return {
                    "type": "response",
                    "id": opid,
                    "data": None,
                }

            ret = getattr(srv, method_name)(*args, **kwargs)
            return {
                "type": "response",
                "id": opid,
                "data": ret,
            }
        except Exception:
            traceback_str = traceback.format_exc()
            return {
                "type": "error",
                "id": opid,
                "error": traceback_str,
            }

    while not states["is_stoped"]:
        # while llm server is empty
        cmd = queue_in.get()
        queue_out.put(method_call(cmd))

        if coalesce_ms > 0:
            deadline = time.perf_counter() + (coalesce_ms / 1000.0)
            while not states["is_stoped"]:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    cmd = queue_in.get(timeout=remaining)
                except Empty:
                    break
                queue_out.put(method_call(cmd))

        while not srv.is_finished() and not states["is_stoped"]:
            # while llm server is not empty
            while not states["is_stoped"]:
                # get cmd nowait, and handle it first
                try:
                    cmd = queue_in.get_nowait()
                    queue_out.put(method_call(cmd))
                except Empty:
                    break

            if states["is_stoped"]:
                break

            # then do llm step
            output = srv.step()

            # update output
            for seq in output:
                latest_waveform = seq.custom_payload.generated_waveforms[-1]
                queue_out.put(
                    {
                        "type": "stream",
                        "id": seq.seq_id,
                        "data": latest_waveform,
                    }
                )
                if seq.is_finished:
                    queue_out.put(
                        {
                            "type": "stream",
                            "id": seq.seq_id,
                            "data": None,
                        }
                    )


class AsyncVoxCPMServer:
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Optional[LoRAConfig] = None,
        **kwargs,
    ) -> None:
        if len(kwargs) > 0:
            raise ValueError(f"Unknown kwargs: {kwargs}")

        ctx = mp.get_context("spawn")
        self.queue_in = ctx.Queue()
        self.queue_out = ctx.Queue()
        self.process = ctx.Process(
            target=main_loop,
            args=(
                self.queue_in,
                self.queue_out,
                (
                    model_path,
                    inference_timesteps,
                    max_num_batched_tokens,
                    max_num_seqs,
                    max_model_len,
                    gpu_memory_utilization,
                    enforce_eager,
                    devices,
                    lora_config,
                ),
                {},
            ),
            daemon=True,
        )
        self.process.start()

        # Child process sends an explicit init_ok/init_error message.
        # Track it as a Future so wait_for_ready can block without timeouts,
        # but still fail fast if init throws (e.g. CUDA OOM).
        loop = asyncio.get_running_loop()
        self._init_fut: asyncio.Future[None] = loop.create_future()

        self.recv_task: asyncio.Task = asyncio.create_task(self.recv_queue())
        self.op_table: dict[str, asyncio.Future[Any]] = {}
        self.stream_table: dict[str, asyncio.Queue[Waveform | None]] = {}

    async def recv_queue(self) -> None:
        try:
            while True:
                try:
                    res = await asyncio.to_thread(self.queue_out.get, timeout=1)
                except Empty:
                    continue

                # Init handshake (sent once at process startup).
                if res.get("type") == "init_ok":
                    if not self._init_fut.done():
                        self._init_fut.set_result(None)
                    continue
                if res.get("type") == "init_error":
                    if not self._init_fut.done():
                        self._init_fut.set_exception(RuntimeError(res.get("error", "unknown init error")))
                    continue

                if res["type"] == "stream":
                    if res["id"] in self.stream_table:
                        stream_data = res["data"]
                        await self.stream_table[res["id"]].put(stream_data)
                    else:
                        print(f"Unknown stream_id: {res['id']}")
                elif res["id"] in self.op_table:
                    if res["type"] == "response":
                        self.op_table[res["id"]].set_result(res["data"] if "data" in res else None)
                        del self.op_table[res["id"]]
                    else:
                        self.op_table[res["id"]].set_exception(RuntimeError(res["error"]))
                        del self.op_table[res["id"]]
                else:
                    print(f"Unknown op_id: {res['id']}")
        except asyncio.CancelledError:
            # Normal shutdown path: stop() cancels this task.
            return

    async def submit(self, cmd: str, *args: object, **kwargs: object) -> Any:
        op_id = str(uuid.uuid4())

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()

        self.op_table[op_id] = fut

        await asyncio.to_thread(
            self.queue_in.put,
            {
                "id": op_id,
                "type": cmd,
                "args": args,
                "kwargs": kwargs,
            },
        )
        return await fut

    async def health(self) -> HealthResponse:
        return await self.submit("health")

    async def get_model_info(self) -> ModelInfoResponse:
        return await self.submit("get_model_info")

    async def wait_for_ready(self) -> None:
        # Never time out here; instead fail fast if the child process exits.
        while not self._init_fut.done():
            if self.process.exitcode is not None:
                if not self._init_fut.done():
                    self._init_fut.set_exception(
                        RuntimeError(f"server process exited early: exitcode={self.process.exitcode}")
                    )
                break
            await asyncio.sleep(0.05)
        await self._init_fut

    async def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        return await self.submit("encode_latents", wav, wav_format)

    async def stop(self) -> None:
        # Best-effort graceful shutdown. If init failed or the child process
        # already exited, don't block indefinitely.
        graceful_stop = False
        if self.process.exitcode is None and self.process.is_alive():
            try:
                await asyncio.wait_for(self.submit("stop"), timeout=2.0)
                graceful_stop = True
            except Exception:
                # Fall back to terminate/kill below.
                pass

        self.recv_task.cancel()
        # Ensure the background receiver task is actually done before closing queues.
        # Cancellation may race with the underlying to_thread() call.
        # In Python 3.10+, asyncio.CancelledError may not be an Exception.
        with contextlib.suppress(asyncio.CancelledError):
            await self.recv_task

        # If the child acknowledged stop, give it a chance to exit cleanly
        # before we escalate to terminate/kill.
        if graceful_stop and self.process.is_alive():
            await asyncio.to_thread(self.process.join, 5.0)

        if self.process.is_alive():
            self.process.terminate()
            await asyncio.to_thread(self.process.join, 2.0)

        if self.process.is_alive():
            # Python 3.7+ on POSIX.
            kill = getattr(self.process, "kill", None)
            if callable(kill):
                kill()
                await asyncio.to_thread(self.process.join, 2.0)

        # The parent process created these multiprocessing Queues. If we don't
        # close/join them, Python may warn at interpreter shutdown about leaked
        # semaphore objects via multiprocessing.resource_tracker.
        for q in (getattr(self, "queue_in", None), getattr(self, "queue_out", None)):
            if q is None:
                continue
            with contextlib.suppress(Exception):
                q.close()
            with contextlib.suppress(Exception):
                q.join_thread()

    async def register_lora(self, name: str, path: str) -> RegisterLoRAResponse:
        return await self.submit("register_lora", name, path)

    async def unregister_lora(self, name: str) -> UnregisterLoRAResponse:
        return await self.submit("unregister_lora", name)

    async def list_loras(self) -> list[LoRAInfo]:
        return await self.submit("list_loras")

    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
        lora_name: str | None = None,
    ) -> AsyncGenerator[Waveform, None]:
        seq_id = gen_uuid()
        self.stream_table[seq_id] = asyncio.Queue()

        is_normal_exit = False
        try:
            await self.submit(
                "add_request",
                seq_id,
                target_text,
                prompt_latents,
                prompt_text,
                max_generate_length,
                temperature,
                cfg_value,
                lora_name,
            )

            while True:
                data = await self.stream_table[seq_id].get()
                if data is None:
                    is_normal_exit = True
                    break
                yield data
        finally:
            if not is_normal_exit:
                await self.submit("cancel", seq_id)
            del self.stream_table[seq_id]


class AsyncVoxCPMServerPool:
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Optional[LoRAConfig] = None,
        **kwargs,
    ):
        if len(kwargs) > 0:
            raise ValueError(f"Unknown kwargs: {kwargs}")

        self.servers = [
            AsyncVoxCPMServer(
                model_path=model_path,
                inference_timesteps=inference_timesteps,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                devices=[device_idx],
                lora_config=lora_config,
            )
            for device_idx in devices
        ]

        self.servers_load = np.zeros(len(self.servers), dtype=np.int32)

        self._prompt_pool = {}
        self._registered_loras: set[str] = set()
        self._draining_loras: set[str] = set()

    async def wait_for_ready(self):
        await asyncio.gather(*[server.wait_for_ready() for server in self.servers])

    async def stop(self):
        await asyncio.gather(*[server.stop() for server in self.servers])

    async def encode_latents(self, wav: bytes, wav_format: str):
        # send to one
        min_load_server_idx = np.argmin(self.servers_load)
        return await self.servers[min_load_server_idx].encode_latents(wav, wav_format)

    async def get_model_info(self) -> ModelInfoResponse:
        # Assume homogeneous servers inside a pool.
        if len(self.servers) == 0:
            raise RuntimeError("server pool is empty")
        return await self.servers[0].get_model_info()

    async def register_lora(self, name: str, path: str) -> RegisterLoRAResponse:
        if name in self._registered_loras or name in self._draining_loras:
            raise ValueError(f"LoRA '{name}' is already registered")

        registered_servers: list[AsyncVoxCPMServer] = []
        try:
            for server in self.servers:
                await server.register_lora(name, path)
                registered_servers.append(server)
        except Exception:
            for server in reversed(registered_servers):
                with contextlib.suppress(Exception):
                    await server.unregister_lora(name)
            raise

        self._registered_loras.add(name)
        return RegisterLoRAResponse(name=name)

    async def unregister_lora(self, name: str) -> UnregisterLoRAResponse:
        if name not in self._registered_loras:
            raise ValueError(f"LoRA '{name}' is not registered")
        if name in self._draining_loras:
            raise ValueError(f"LoRA '{name}' is already draining")

        self._draining_loras.add(name)
        try:
            for server in self.servers:
                await server.unregister_lora(name)
        except Exception:
            raise
        self._draining_loras.remove(name)
        self._registered_loras.remove(name)
        return UnregisterLoRAResponse(name=name)

    async def list_loras(self) -> list[LoRAInfo]:
        visible_names = sorted(name for name in self._registered_loras if name not in self._draining_loras)
        return [LoRAInfo(name=name) for name in visible_names]

    async def add_prompt(self, wav: bytes, wav_format: str, prompt_text: str):
        prompt_id = gen_uuid()
        prompt_latents = await self.encode_latents(wav, wav_format)
        self._prompt_pool[prompt_id] = {
            "latents": prompt_latents,
            "text": prompt_text,
        }
        return prompt_id

    async def remove_prompt(self, prompt_id: str):
        del self._prompt_pool[prompt_id]

    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        prompt_id: str | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
        lora_name: str | None = None,
    ):
        """Generate audio conditioned on text and optional prompt.

        This is an async generator that yields waveform chunks (one chunk per
        model step) as NumPy arrays.

        Exactly one of the following prompt sources may be used:

        - Provide ``prompt_latents`` + matching ``prompt_text``.
        - Provide ``prompt_id`` (a previously-added prompt via ``add_prompt``).
        - Provide no prompt (zero-shot).

        Args:
            target_text: Text to synthesize.
            prompt_latents: Serialized prompt latents (float32 bytes). If set,
                ``prompt_text`` must be non-empty.
            prompt_text: Text corresponding to ``prompt_latents``.
            prompt_id: ID of a stored prompt from ``add_prompt``. Mutually
                exclusive with ``prompt_latents`` and ``prompt_text``.
            max_generate_length: Maximum number of generated steps.
            temperature: Sampling temperature.
            cfg_value: Classifier-free guidance scale.

        Yields:
            Waveform chunks as ``np.ndarray`` of dtype ``float32``.

        Raises:
            ValueError: If prompt arguments are inconsistent (e.g. unknown
                ``prompt_id``, or both ``prompt_id`` and ``prompt_latents`` are
                provided).
        """
        if prompt_id is not None:
            if prompt_id not in self._prompt_pool:
                raise ValueError(f"Prompt with id {prompt_id} not found")
            if prompt_latents is not None:
                raise ValueError("Prompt latents and prompt id cannot be provided at the same time")
            if len(prompt_text) > 0:
                raise ValueError("Prompt text and prompt id cannot be provided at the same time")

            prompt_info = self._prompt_pool[prompt_id]
            prompt_latents = prompt_info["latents"]
            prompt_text = prompt_info["text"]

        if lora_name is not None and (lora_name not in self._registered_loras or lora_name in self._draining_loras):
            raise ValueError(f"LoRA '{lora_name}' is not registered")

        min_load_server_idx = np.argmin(self.servers_load)
        self.servers_load[min_load_server_idx] += 1

        server = self.servers[min_load_server_idx]

        try:
            async for data in server.generate(
                target_text,
                prompt_latents,
                prompt_text,
                max_generate_length,
                temperature,
                cfg_value,
                lora_name,
            ):
                yield data
        finally:
            self.servers_load[min_load_server_idx] -= 1


class SyncVoxCPMServerPool:
    def __init__(
        self,
        model_path: str,
        inference_timesteps: int = 10,
        max_num_batched_tokens: int = 16384,
        max_num_seqs: int = 512,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
        devices: List[int] = [],
        lora_config: Optional[LoRAConfig] = None,
        **kwargs,
    ):
        async def init_async_server_pool():
            return AsyncVoxCPMServerPool(
                model_path=model_path,
                inference_timesteps=inference_timesteps,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                devices=devices,
                lora_config=lora_config,
                **kwargs,
            )

        self.loop = asyncio.new_event_loop()
        self.server_pool = self.loop.run_until_complete(init_async_server_pool())
        self.loop.run_until_complete(self.server_pool.wait_for_ready())

    def stop(self):
        assert self.loop is not None
        self.loop.run_until_complete(self.server_pool.stop())
        self.loop.close()
        self.loop = None

    def encode_latents(self, wav: bytes, wav_format: str):
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.encode_latents(wav, wav_format))

    def get_model_info(self) -> ModelInfoResponse:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.get_model_info())

    def register_lora(self, name: str, path: str) -> RegisterLoRAResponse:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.register_lora(name, path))

    def unregister_lora(self, name: str) -> UnregisterLoRAResponse:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.unregister_lora(name))

    def list_loras(self) -> list[LoRAInfo]:
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.list_loras())

    def add_prompt(self, wav: bytes, wav_format: str, prompt_text: str):
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.add_prompt(wav, wav_format, prompt_text))

    def remove_prompt(self, prompt_id: str):
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.remove_prompt(prompt_id))

    def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        prompt_id: str | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
        lora_name: str | None = None,
    ):
        """Generate audio conditioned on text and optional prompt.

        This is a synchronous generator wrapper around
        ``AsyncVoxCPMServerPool.generate``.

        Args:
            target_text: Text to synthesize.
            prompt_latents: Serialized prompt latents (float32 bytes). If set,
                ``prompt_text`` must be non-empty.
            prompt_text: Text corresponding to ``prompt_latents``.
            prompt_id: ID of a stored prompt from ``add_prompt``. Mutually
                exclusive with ``prompt_latents`` and ``prompt_text``.
            max_generate_length: Maximum number of generated steps.
            temperature: Sampling temperature.
            cfg_value: Classifier-free guidance scale.

        Yields:
            Waveform chunks as ``np.ndarray`` of dtype ``float32``.

        Raises:
            ValueError: If prompt arguments are inconsistent.
        """
        assert self.loop is not None
        async_gen = self.server_pool.generate(
            target_text,
            prompt_latents,
            prompt_text,
            prompt_id,
            max_generate_length,
            temperature,
            cfg_value,
            lora_name,
        )
        try:
            while True:
                item = self.loop.run_until_complete(async_gen.__anext__())
                yield item
        except StopAsyncIteration:
            return
