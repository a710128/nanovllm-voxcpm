from nanovllm_voxcpm.models.voxcpm.engine import VoxCPMEngine, VoxCPMRunner, VoxCPMConfig, Config
from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig
from nanovllm_voxcpm.utils.loader import load_lora_weights
import os
import torch.multiprocessing as mp
from queue import Empty
import traceback
import uuid
import torchaudio
import io
import asyncio
from typing import Any, AsyncGenerator, List, Optional, cast
from typing_extensions import TypedDict, Literal
import numpy as np
from numpy.typing import NDArray

Waveform = NDArray[np.float32]


class HealthResponse(TypedDict):
    status: Literal["ok"]


class SetLoraEnabledResponse(TypedDict):
    status: Literal["ok"]
    lora_enabled: bool


class LoadLoraResponse(TypedDict):
    status: Literal["ok"]
    loaded_keys: int
    skipped_keys: int


class ResetLoraResponse(TypedDict):
    status: Literal["ok"]


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

    def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        wav_tensor, sr = torchaudio.load(io.BytesIO(wav), format=wav_format)
        wav_tensor = wav_tensor.cuda()
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
    ) -> None:
        prompt_latents_arr: np.ndarray | None
        if prompt_latents is not None:
            if len(prompt_text) == 0:
                raise ValueError("Prompt text is required when prompt latents are provided")

            prompt_latents_arr = np.frombuffer(prompt_latents, dtype=np.float32).reshape(-1, self.llm.feat_dim)
        else:
            prompt_latents_arr = None
            if len(prompt_text) > 0:
                raise ValueError("Prompt text is not allowed when prompt latents are not provided")

        self.llm.add_request(
            seq_id=seq_id,
            target_text=target_text,
            prompt_text=prompt_text,
            prompt_latents=prompt_latents_arr,
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
        )

    def cancel(self, seq_id: str):
        self.llm.cancel_sequence(seq_id)

    def step(self):
        return self.llm.step()

    def is_finished(self):
        return self.llm.is_finished()

    # ------------------------------------------------------------------ #
    # LoRA Management Methods
    # ------------------------------------------------------------------ #

    def set_lora_enabled(self, enabled: bool) -> SetLoraEnabledResponse:
        """Enable/disable LoRA layers."""
        if self.lora_config is None:
            raise RuntimeError("LoRA is not configured for this model")
        model = cast(VoxCPMRunner, self.llm.model_runner).model
        model.set_lora_enabled(enabled)
        return SetLoraEnabledResponse(status="ok", lora_enabled=enabled)

    def load_lora(self, lora_path: str) -> LoadLoraResponse:
        """Load LoRA weights from a path."""
        if self.lora_config is None:
            raise RuntimeError("LoRA is not configured for this model. Initialize with lora_config.")
        model = cast(VoxCPMRunner, self.llm.model_runner).model
        loaded, skipped = load_lora_weights(model, lora_path, device="cuda")
        return LoadLoraResponse(status="ok", loaded_keys=len(loaded), skipped_keys=len(skipped))

    def reset_lora(self) -> ResetLoraResponse:
        """Reset LoRA weights to initial state (effectively unload)."""
        if self.lora_config is None:
            raise RuntimeError("LoRA is not configured for this model")
        model = cast(VoxCPMRunner, self.llm.model_runner).model
        model.reset_lora_parameters()
        return ResetLoraResponse(status="ok")


def main_loop(queue_in: mp.Queue, queue_out: mp.Queue, args, kwargs):
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    srv = VoxCPMServerImpl(*args, **kwargs)

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

        self.recv_task: asyncio.Task = asyncio.create_task(self.recv_queue())
        self.op_table: dict[str, asyncio.Future[Any]] = {}
        self.stream_table: dict[str, asyncio.Queue[Waveform | None]] = {}

    async def recv_queue(self) -> None:
        while True:
            try:
                res = await asyncio.to_thread(self.queue_out.get, timeout=1)
            except Empty:
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

    async def wait_for_ready(self) -> None:
        await self.health()

    async def encode_latents(self, wav: bytes, wav_format: str) -> bytes:
        return await self.submit("encode_latents", wav, wav_format)

    async def stop(self) -> None:
        await self.submit("stop")
        self.recv_task.cancel()
        await asyncio.to_thread(self.process.join)

    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 2.0,
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

    # LoRA management methods
    async def set_lora_enabled(self, enabled: bool) -> SetLoraEnabledResponse:
        return await self.submit("set_lora_enabled", enabled)

    async def load_lora(self, lora_path: str) -> LoadLoraResponse:
        return await self.submit("load_lora", lora_path)

    async def reset_lora(self) -> ResetLoraResponse:
        return await self.submit("reset_lora")


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

    async def wait_for_ready(self):
        await asyncio.gather(*[server.wait_for_ready() for server in self.servers])

    async def stop(self):
        await asyncio.gather(*[server.stop() for server in self.servers])

    async def encode_latents(self, wav: bytes, wav_format: str):
        # send to one
        min_load_server_idx = np.argmin(self.servers_load)
        return await self.servers[min_load_server_idx].encode_latents(wav, wav_format)

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
            ):
                yield data
        finally:
            self.servers_load[min_load_server_idx] -= 1

    # LoRA management methods (apply to all servers)
    async def set_lora_enabled(self, enabled: bool):
        results = await asyncio.gather(*[server.set_lora_enabled(enabled) for server in self.servers])
        return results[0]

    async def load_lora(self, lora_path: str):
        results = await asyncio.gather(*[server.load_lora(lora_path) for server in self.servers])
        return results[0]

    async def reset_lora(self):
        results = await asyncio.gather(*[server.reset_lora() for server in self.servers])
        return results[0]


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
        )
        try:
            while True:
                item = self.loop.run_until_complete(async_gen.__anext__())
                yield item
        except StopAsyncIteration:
            return

    # LoRA management methods
    def set_lora_enabled(self, enabled: bool):
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.set_lora_enabled(enabled))

    def load_lora(self, lora_path: str):
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.load_lora(lora_path))

    def reset_lora(self):
        assert self.loop is not None
        return self.loop.run_until_complete(self.server_pool.reset_lora())
