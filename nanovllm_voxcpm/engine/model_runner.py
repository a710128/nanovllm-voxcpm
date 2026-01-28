"""nanovllm_voxcpm.engine.model_runner

This module defines the GPU execution abstraction used by the engine.

The high-level runtime separates concerns:
- :mod:`nanovllm_voxcpm.engine.scheduler` decides *what to run* (which sequences)
  and manages KV-cache block allocation.
- :mod:`nanovllm_voxcpm.engine.llm_engine` orchestrates the step loop and
  converts between request objects and runner tasks.
- This module executes the model forward pass on GPU(s) given a batch of
  lightweight :class:`RunnerTask` objects.

RunnerTask
----------
:class:`RunnerTask` is a minimal, picklable view of a sequence needed to build
GPU inputs:
- ``block_table``: physical KV-cache block ids for this request.
- ``seq_length``: logical length (prompt + generated tokens so far).
- ``num_cached_tokens``: cached prefix tokens (prefill only).
- ``custom_payload``: model-specific inputs (e.g. token tensors, sampling params).

BaseModelRunner
---------------
:class:`BaseModelRunner` owns the actual ``torch.nn.Module`` and the KV-cache
tensors stored inside causal :class:`~nanovllm_voxcpm.layers.attention.Attention`
modules. Key responsibilities:

- Initialize NCCL process group and set the CUDA device for the current rank.
- Load and warm up the model (used to measure peak memory).
- Allocate the KV-cache block pool based on available GPU memory and
  ``gpu_memory_utilization``.
- Prepare attention metadata ("context") for flash-attn kernels via
  :func:`nanovllm_voxcpm.utils.context.set_context`.
  * Prefill context supports prefix caching by distinguishing query length
    (new tokens) vs key length (full context).
  * Decode context writes one token per sequence into the KV cache.
- Optional CUDA Graph capture for decode to reduce launch overhead
  (disabled with ``enforce_eager``).

Multi-GPU execution model
-------------------------
Tensor-parallel ranks are spawned as separate processes. Rank 0 acts as the
"driver" and broadcasts method calls to other ranks through shared memory +
``multiprocessing.Event``. Non-zero ranks run :meth:`loop`, which blocks on an
event, reads the serialized method call, and executes it.

Model-specific runners
----------------------
Concrete model families subclass :class:`BaseModelRunner` and implement:
- model construction / weight loading (:meth:`init_model`)
- building inputs/outputs for warmup/graph capture (:meth:`make_dummy_inputs`,
  :meth:`make_dummy_outputs`)
- the actual per-step execution logic (:meth:`run`) which typically:
  1) builds tensors from ``RunnerTask.custom_payload``
  2) calls :meth:`prepare_prefill_context` or :meth:`prepare_decode_context`
  3) runs the model via :meth:`run_model`
  4) returns Python-friendly outputs for engine postprocessing.

Concrete example: VoxCPM
------------------------
``nanovllm_voxcpm/models/voxcpm/runner.py`` shows a typical implementation:

- Prefill: the engine slices away ``num_cached_tokens`` and sends the remaining
  prompt segment (text tokens + audio features + masks) to the runner.
- Decode: the engine sends only the last step (length 1) and sets
  ``RunnerTask.num_cached_tokens = seq_length - 1`` so the runner builds a
  decode context (query length 1, key length = full context).
- The runner concatenates per-sequence numpy arrays into a packed token-major
  batch, runs the model, then converts outputs back to numpy.
- Besides model outputs (e.g. ``latents`` and ``stop_flag``), VoxCPMRunner also
  decodes the generated latents into waveform chunks via an AudioVAE and returns
  them to be streamed.
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.layers.attention import Attention
from nanovllm_voxcpm.utils.context import set_context, reset_context, get_context
from typing import Generic, TypeVar

PlayloadType = TypeVar("PlayloadType")


class RunnerTask(Generic[PlayloadType]):
    def __init__(
        self,
        block_table: list[int],
        seq_length: int,
        num_cached_tokens: int,
        block_size: int,
        custom_payload: PlayloadType = None,
    ):
        self.block_table = block_table
        self.seq_length = seq_length
        self.num_cached_tokens = num_cached_tokens
        self.custom_payload = custom_payload
        self.block_size = block_size

    @property
    def num_blocks(self):
        return (self.seq_length + self.block_size - 1) // self.block_size

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.seq_length - (self.num_blocks - 1) * self.block_size


def cut_inputs(inputs, bs):
    return {k: v[:bs] for k, v in inputs.items()}


def assign_outputs(inputs, outputs, bs):
    for k in outputs.keys():
        if k not in inputs:
            raise KeyError(f"Input {k} is required")
        outputs[k][:bs] = inputs[k]


class BaseModelRunner:
    model: torch.nn.Module

    def __init__(
        self,
        config: Config,
        rank: int,
        device_idx: int,
        distributed_port: int,
        event: Event | list[Event],
    ):
        self._config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group(
            "nccl",
            "tcp://localhost:{}".format(distributed_port),
            world_size=self.world_size,
            rank=rank,
        )
        torch.cuda.set_device(device_idx)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        torch.set_default_device("cuda")
        self.init_model(self._config.model_config, self._config.model)
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=f"nanovllm-{distributed_port}", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=f"nanovllm-{distributed_port}")
                self.loop()

    @property
    def dtype(self) -> torch.dtype:
        raise NotImplementedError()

    def init_model(self, model_config, model_path: str):
        raise NotImplementedError()

    def make_dummy_inputs(self, batch_size: int, length: int) -> torch.Tensor:
        raise NotImplementedError()

    def make_dummy_outputs(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def run(self, seqs: list[RunnerTask], is_prefill: bool):
        raise NotImplementedError()

    @torch.inference_mode()
    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self._config.max_num_batched_tokens,
            self._config.max_model_len,
        )
        num_seqs = min(max_num_batched_tokens // max_model_len, self._config.max_num_seqs)
        seqs = [
            RunnerTask(
                block_table=[],
                seq_length=max_model_len,
                num_cached_tokens=0,
                block_size=self.block_size,
                custom_payload=None,
            )
            for _ in range(num_seqs)
        ]
        inputs = {"positions": self.prepare_prefill_context(seqs)}
        inputs.update(self.make_dummy_inputs(num_seqs, max_model_len))
        _ = self.model(**inputs)
        reset_context()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        total_attention_block_size = 0
        for module in self.model.modules():
            if isinstance(module, Attention) and module.is_causal:
                total_attention_block_size += (
                    2 * self.block_size * module.num_kv_heads * module.head_dim * self.dtype.itemsize
                )

        self._config.num_kvcache_blocks = (
            int(total * self._config.gpu_memory_utilization - used - peak + current) // total_attention_block_size
        )
        assert self._config.num_kvcache_blocks > 0

        for module in self.model.modules():
            if isinstance(module, Attention) and module.is_causal:
                module.k_cache = torch.empty(
                    self._config.num_kvcache_blocks,
                    self.block_size,
                    module.num_kv_heads,
                    module.head_dim,
                )
                module.v_cache = torch.empty(
                    self._config.num_kvcache_blocks,
                    self.block_size,
                    module.num_kv_heads,
                    module.head_dim,
                )

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def prepare_block_tables(self, seqs: list[RunnerTask]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables_list: list[list[int]] = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_prefill_context(self, seqs: list[RunnerTask]):
        positions_list: list[int] = []
        cu_seqlens_q_list: list[int] = [0]
        cu_seqlens_k_list: list[int] = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping_list: list[int] = []
        block_tables: torch.Tensor | None = None
        for seq in seqs:
            seq_len = seq.seq_length
            positions_list.extend(list(range(seq.num_cached_tokens, seq_len)))
            seqlen_q = seq_len - seq.num_cached_tokens
            seqlen_k = seq_len
            cu_seqlens_q_list.append(cu_seqlens_q_list[-1] + seqlen_q)
            cu_seqlens_k_list.append(cu_seqlens_k_list[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping_list.extend(list(range(start, end)))
        if cu_seqlens_k_list[-1] > cu_seqlens_q_list[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)

        positions = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return positions

    def prepare_decode_context(self, seqs: list[RunnerTask]):
        positions_list: list[int] = []
        slot_mapping_list: list[int] = []
        context_lens_list: list[int] = []
        for seq in seqs:
            positions_list.append(seq.seq_length - 1)
            context_lens_list.append(seq.seq_length)
            slot_mapping_list.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        positions = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return positions

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self._config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        positions = torch.zeros(max_bs, dtype=torch.int64)
        inputs = {
            "positions": positions,
        }
        inputs.update(self.make_dummy_inputs(max_bs, 1))

        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = self.make_dummy_outputs(max_bs)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )

            if isinstance(outputs, torch.Tensor):
                outputs[:bs] = self.model(**cut_inputs(inputs, bs))  # warmup
            else:
                assign_outputs(self.model(**cut_inputs(inputs, bs)), outputs, bs)

            with torch.cuda.graph(graph, self.graph_pool):
                if isinstance(outputs, torch.Tensor):
                    outputs[:bs] = self.model(**cut_inputs(inputs, bs))  # capture
                else:
                    assign_outputs(self.model(**cut_inputs(inputs, bs)), outputs, bs)

            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            inputs=inputs,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def run_model(self, inputs: dict, is_prefill: bool):
        if is_prefill or self.enforce_eager or inputs["positions"].size(0) > 512:
            ret = self.model(**inputs)
            reset_context()
            return ret
        else:
            bs = inputs["positions"].size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for kw in graph_vars["inputs"].keys():
                if kw not in inputs:
                    raise ValueError(f"Input {kw} is required")
                graph_vars["inputs"][kw][:bs] = inputs[kw]
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = context.block_tables
            graph.replay()
            # ret = graph_vars["outputs"][:bs]
            if isinstance(graph_vars["outputs"], torch.Tensor):
                ret = graph_vars["outputs"][:bs]
            else:
                ret = cut_inputs(graph_vars["outputs"], bs)
            reset_context()
            return ret
