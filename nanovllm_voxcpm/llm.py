import os
import json
from huggingface_hub import snapshot_download
from typing import List, Optional
import asyncio

try:
    # check if flash-attn is installed
    import flash_attn
except ImportError:
    raise ImportError(
        "flash-attn is not installed. Please install it with `pip install flash-attn --no-build-isolation`"
    )

from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig


class VoxCPM:
    @staticmethod
    def from_pretrained(
        model: str,
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
        if "~" in model:
            model_path = os.path.expanduser(model)
            if not os.path.isdir(model_path):
                raise ValueError(f"Model path {model_path} does not exist")
        else:
            if not os.path.isdir(model):
                model_path = snapshot_download(repo_id=model)
            else:
                model_path = model

        config_file = os.path.expanduser(os.path.join(model_path, "config.json"))

        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file `{config_file}` not found")

        config = json.load(open(config_file))

        arch = config["architecture"]

        if len(devices) == 0:
            devices = [0]

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            is_async_mode = False
        else:
            is_async_mode = True

        if arch == "voxcpm":
            from nanovllm_voxcpm.models.voxcpm.server import (
                AsyncVoxCPMServerPool,
                SyncVoxCPMServerPool,
            )

            if is_async_mode:
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
            else:
                return SyncVoxCPMServerPool(
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
        else:
            raise ValueError(f"Unsupported model architecture: {arch}")
