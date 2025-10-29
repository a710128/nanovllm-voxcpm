from huggingface_hub import snapshot_download
from nanovllm_voxcpm.models.voxcpm.engine import VoxCPMEngine, VoxCPMConfig, Config
import os
import torch.multiprocessing as mp
from queue import Empty
import traceback
import uuid
import torchaudio
import io
import torch
import asyncio

def gen_uuid() -> str:
    return uuid.uuid4().hex

class VoxCPMServerImpl:
    def __init__(self,
        model : str,
        inference_timesteps : int = 10,
        max_num_batched_tokens : int = 16384,
        max_num_seqs : int = 512,
        max_model_len : int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
    ):
        if "~" in model:
            model_path = os.path.expanduser(model)
            if not os.path.isdir(model_path):
                raise ValueError(f"Model path {model_path} does not exist")
        else:
            if not os.path.isdir(model):
                model_path = snapshot_download(repo_id=model)
            else:
                model_path = model

        model_config = VoxCPMConfig.model_validate_json(
            open(os.path.join(model_path, "config.json")).read()
        )

        model_config.inference_timesteps = inference_timesteps

        engine_config = Config(
            model=model_path,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            model_config=model_config,
        )

        self.llm = VoxCPMEngine(engine_config)
        self.sample_rate = self.llm.model_runner.vae.sample_rate
        self._prompt_map = {}

    def health(self):
        return {
            "status": "ok",
        }
    
    def add_prompt(self, wav : bytes, wav_format : str, prompt_text : str):
        wav, sr = torchaudio.load(io.BytesIO(wav), format=wav_format)
        wav = wav.cuda()
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        align_size = self.llm.patch_size * self.llm.chunk_size
        if wav.size(1) % align_size != 0:
            remained = align_size - wav.size(1) % align_size
            wav = torch.nn.functional.pad(wav, (remained, 0))

        latents = self.llm.encode_latents(wav)
        assert latents.shape[0] % self.llm.patch_size == 0
        prompt_id = gen_uuid()

        self._prompt_map[prompt_id] = {
            "latents": latents,
            "prompt_text": prompt_text,
        }

        return prompt_id
    
    def remove_prompt(self, prompt_id : str):
        del self._prompt_map[prompt_id]

    def add_request(self,
        seq_id : str,
        target_text : str,
        prompt_id : str | None = None,
        max_generate_length : int = 2000,
        temperature : float = 1.0,
        cfg_value : float = 1.0
    ):
        if prompt_id is not None:
            if prompt_id not in self._prompt_map:
                raise ValueError(f"Prompt id {prompt_id} not found")
            
            prompt_latents = self._prompt_map[prompt_id]["latents"]
            prompt_text = self._prompt_map[prompt_id]["prompt_text"]
        else:
            prompt_latents = None
            prompt_text = ""
        
        self.llm.add_request(
            seq_id=seq_id,
            target_text=target_text,
            prompt_text=prompt_text,
            prompt_latents=prompt_latents,
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
        )

    def cancel(self, seq_id : str):
        self.llm.cancel_sequence(seq_id)
    
    def step(self):
        return self.llm.step()
    
    def is_finished(self):
        return self.llm.is_finished()


def main_loop(
    queue_in : mp.Queue,
    queue_out : mp.Queue,
    args, kwargs
):
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    srv = VoxCPMServerImpl(*args, **kwargs)

    states = {
        "is_stoped": False,
    }
    def method_call(cmd):
        try:
            opid = cmd["id"]
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
                queue_out.put({
                    "type": "stream",
                    "id": seq.seq_id,
                    "data": latest_waveform,
                })
                if seq.is_finished:
                    queue_out.put({
                        "type": "stream",
                        "id": seq.seq_id,
                        "data": None,
                    })


class AsyncVoxCPMServer:
    def __init__(self,
        model : str,
        inference_timesteps : int = 10,
        max_num_batched_tokens : int = 16384,
        max_num_seqs : int = 512,
        max_model_len : int = 4096,
        gpu_memory_utilization: float = 0.9,
        enforce_eager: bool = False,
    ):
        ctx = mp.get_context("spawn")
        self.queue_in = ctx.Queue()
        self.queue_out = ctx.Queue()
        self.process = ctx.Process(
            target=main_loop, 
            args=(self.queue_in, self.queue_out, (model, inference_timesteps, max_num_batched_tokens, max_num_seqs, max_model_len, gpu_memory_utilization, enforce_eager), {}),
            daemon=True,
        )
        self.process.start()

        self.recv_task = asyncio.create_task(self.recv_queue())
        self.op_table = {}
        self.stream_table : dict[str, asyncio.Queue] = {}
    
    async def recv_queue(self):
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
    
    async def submit(self, cmd : str, *args, **kwargs):
        op_id = str(uuid.uuid4())

        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        self.op_table[op_id] = fut

        await asyncio.to_thread(self.queue_in.put, {
            "id": op_id,
            "type": cmd,
            "args": args,
            "kwargs": kwargs,
        })
        return await fut
    
    async def health(self):
        return await self.submit("health")
    
    async def wait_for_ready(self):
        await self.health()
    
    async def add_prompt(self, wav : bytes, wav_format : str, prompt_text : str):
        return await self.submit("add_prompt", wav, wav_format, prompt_text)
    
    async def remove_prompt(self, prompt_id : str):
        return await self.submit("remove_prompt", prompt_id)
    
    async def stop(self):
        return await self.submit("stop")
    
    async def generate(
        self,
        target_text : str,
        prompt_id : str | None = None,
        max_generate_length : int = 2000,
        temperature : float = 1.0,
        cfg_value : float = 2.0
    ):
        seq_id = gen_uuid()
        self.stream_table[seq_id] = asyncio.Queue()

        is_normal_exit = False
        try:
            await self.submit("add_request", seq_id, target_text, prompt_id, max_generate_length, temperature, cfg_value)

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
