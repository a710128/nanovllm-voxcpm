import os
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel
from typing import Generic, TypeVar, List, Any

T = TypeVar("T", bound=BaseModel)


class CUDAGraphMode(str, Enum):
    FULL = "full"
    DECODE_ONLY = "decode_only"
    DISABLED = "disabled"


def resolve_cudagraph_mode(*, enforce_eager: bool, enforce_dit_prefill_eager: bool) -> CUDAGraphMode:
    if enforce_eager:
        return CUDAGraphMode.DISABLED
    if enforce_dit_prefill_eager:
        return CUDAGraphMode.DECODE_ONLY
    return CUDAGraphMode.FULL


@dataclass
class Config(Generic[T]):
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    model_config: T | None = None
    devices: List[int] | None = None
    lora_config: Any = None  # Optional[LoRAConfig]
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.max_num_batched_tokens >= self.max_model_len
