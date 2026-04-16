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

import os
import pickle
import tempfile
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.lora_manager import (
    LoRAModelPayload,
    LoRARuntime,
    build_lora_context_from_batch_plan,
    build_lora_context_from_slot_list,
)
from nanovllm_voxcpm.layers.attention import Attention
from nanovllm_voxcpm.lora import is_available as is_lora_available
from nanovllm_voxcpm.utils.context import (
    DIT_LORA_DOMAIN,
    LM_LORA_DOMAIN,
    PROJ_LORA_DOMAIN,
    LoRAContext,
    get_context,
    get_lora_context,
    reset_all_contexts,
    set_context,
    set_lora_context,
)
from typing import Generic, TypeVar

PlayloadType = TypeVar("PlayloadType")
LORA_DOMAINS = (LM_LORA_DOMAIN, PROJ_LORA_DOMAIN, DIT_LORA_DOMAIN)


def select_lora_payload_for_rank(payload, rank: int):
    if isinstance(payload, (list, tuple)):
        if rank >= len(payload):
            raise ValueError(f"Missing rank-local LoRA payload for rank {rank}")
        return payload[rank]
    return payload


_RPC_FILE_SENTINEL = "__rpc_file__"


class RunnerTask(Generic[PlayloadType]):
    def __init__(
        self,
        block_table: list[int],
        seq_length: int,
        num_cached_tokens: int,
        block_size: int,
        custom_payload: PlayloadType = None,
        adapter_id: int | None = None,
    ):
        self.block_table = block_table
        self.seq_length = seq_length
        self.num_cached_tokens = num_cached_tokens
        self.custom_payload = custom_payload
        self.block_size = block_size
        self.adapter_id = adapter_id

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


def _clear_lora_slot_modules(modules, slot_id: int) -> None:
    for module in modules.values():
        clear_slot_lora = getattr(module, "clear_slot_lora", None)
        if clear_slot_lora is not None:
            clear_slot_lora(slot_id)


class BaseModelRunner:
    dit_lora_seq_len_offset = 0
    cfg_branches = 2

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
        self.max_lora_rank = max(1, getattr(config.lora_config, "max_lora_rank", 1) if config.lora_config else 1)
        self.max_loras = max(0, getattr(config.lora_config, "max_loras", 0) if config.lora_config else 0)
        self.lora_runtime = LoRARuntime(max_loras=self.max_loras, max_lora_rank=self.max_lora_rank)

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

    def _dit_lora_rows_per_sample(self) -> int:
        lora_config = getattr(self, "lora_config", None)
        if not (lora_config and getattr(lora_config, "enable_dit", False)):
            return 0
        return self.cfg_branches * (self.dit_lora_seq_len_offset + 2 * self.patch_size)

    def _build_lora_contexts(self, seqs: list[RunnerTask], token_counts: list[int]) -> dict[str, LoRAContext]:
        adapter_ids = [seq.adapter_id for seq in seqs]
        dit_rows_per_sample = self._dit_lora_rows_per_sample()
        if not any(adapter_id is not None for adapter_id in adapter_ids):
            sample_to_slot = [-1 for _ in adapter_ids]
            return {
                LM_LORA_DOMAIN: build_lora_context_from_slot_list([-1] * sum(token_counts)),
                PROJ_LORA_DOMAIN: build_lora_context_from_slot_list(sample_to_slot),
                DIT_LORA_DOMAIN: build_lora_context_from_slot_list(
                    [slot for slot in sample_to_slot for _ in range(dit_rows_per_sample)]
                ),
            }

        plan = self.lora_runtime.build_batch_plan(adapter_ids, token_counts, self._load_lora_slot)
        sample_to_slot = [
            plan.adapter_to_slot.get(adapter_id, -1) if adapter_id is not None else -1 for adapter_id in adapter_ids
        ]
        return {
            LM_LORA_DOMAIN: build_lora_context_from_batch_plan(plan),
            PROJ_LORA_DOMAIN: build_lora_context_from_slot_list(sample_to_slot),
            DIT_LORA_DOMAIN: build_lora_context_from_slot_list(
                [slot for slot in sample_to_slot for _ in range(dit_rows_per_sample)]
            ),
        }

    def validate_lora_payload(
        self, payload: LoRAModelPayload | list[LoRAModelPayload] | tuple[LoRAModelPayload, ...]
    ) -> None:
        rank_payload = select_lora_payload_for_rank(payload, self.rank)
        if rank_payload.rank <= 0:
            raise ValueError(f"LoRA payload rank must be > 0, got {rank_payload.rank}")
        if not rank_payload.modules:
            raise ValueError("LoRA payload must contain at least one target module")

        modules = dict(self.model.named_modules())
        for module_name, module_payload in rank_payload.modules.items():
            try:
                module = modules[module_name]
            except KeyError as exc:
                raise ValueError(f"Unknown LoRA target module '{module_name}'") from exc
            validate_payload = getattr(module, "validate_slot_lora_payload", None)
            if validate_payload is None:
                raise ValueError(f"Module '{module_name}' does not support LoRA slots")
            validate_payload(
                module_payload.lora_a,
                module_payload.lora_b,
                module_payload.effective_rank,
                module_payload.scaling,
            )

    def register_lora(
        self,
        adapter_id: int,
        name: str,
        payload: LoRAModelPayload | list[LoRAModelPayload] | tuple[LoRAModelPayload, ...],
    ) -> None:
        rank_payload = select_lora_payload_for_rank(payload, self.rank)
        self.validate_lora_payload(rank_payload)
        registered_adapter_id = self.lora_runtime.register_lora(name, rank_payload, adapter_id=adapter_id)
        if registered_adapter_id != adapter_id:
            raise RuntimeError(f"Runner LoRA adapter id mismatch: expected {adapter_id}, got {registered_adapter_id}")

    def unregister_lora(self, adapter_id: int) -> None:
        entry = self.lora_runtime.get_entry(adapter_id)
        self.lora_runtime.unregister_lora(entry.name)

    def lora_on_sequence_enqueued(self, adapter_id: int | None) -> None:
        self.lora_runtime.on_sequence_enqueued(adapter_id)

    def lora_on_sequence_started(self, adapter_id: int | None) -> None:
        self.lora_runtime.on_sequence_started(adapter_id)

    def lora_on_sequence_preempted(self, adapter_id: int | None) -> None:
        self.lora_runtime.on_sequence_preempted(adapter_id)

    def lora_on_sequence_finished(self, adapter_id: int | None, was_running: bool) -> None:
        self.lora_runtime.on_sequence_finished(adapter_id, was_running=was_running)

    def _load_lora_slot(self, slot_id: int, payload: LoRAModelPayload) -> None:
        modules = dict(self.model.named_modules())
        _clear_lora_slot_modules(modules, slot_id)
        for module_name, module_payload in payload.modules.items():
            try:
                module = modules[module_name]
            except KeyError as exc:
                raise ValueError(f"Unknown LoRA target module '{module_name}'") from exc
            if not hasattr(module, "set_slot_lora"):
                raise ValueError(f"Module '{module_name}' does not support LoRA slots")
            module.set_slot_lora(
                slot_id=slot_id,
                lora_a=module_payload.lora_a.to(device="cuda", non_blocking=True),
                lora_b=(
                    [tensor.to(device="cuda", non_blocking=True) for tensor in module_payload.lora_b]
                    if isinstance(module_payload.lora_b, list)
                    else module_payload.lora_b.to(device="cuda", non_blocking=True)
                ),
                effective_rank=module_payload.effective_rank,
                scaling=module_payload.scaling,
            )

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
        reset_all_contexts()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        reserved = torch.cuda.memory_reserved()

        total_attention_block_size = 0
        for module in self.model.modules():
            if isinstance(module, Attention) and module.is_causal:
                total_attention_block_size += (
                    2 * self.block_size * module.num_kv_heads * module.head_dim * self.dtype.itemsize
                )

        available_budget = total * self._config.gpu_memory_utilization - peak
        available_physical = free + (reserved - current) - (peak - current)
        available_for_kv = min(available_budget, available_physical)
        self._config.num_kvcache_blocks = int(available_for_kv) // total_attention_block_size
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
            method = getattr(self, method_name, None)
            error = None
            try:
                method(*args)
            except Exception as exc:
                error = exc
            self._synchronize_rpc_result(method_name, error)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        if method_name == _RPC_FILE_SENTINEL:
            with open(args[0], "rb") as f:
                method_name, *args = pickle.load(f)
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        overflow_path = None
        if len(data) + 4 > self.shm.size:
            fd, overflow_path = tempfile.mkstemp(prefix="nanovllm-rpc-", suffix=".pkl")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            data = pickle.dumps([_RPC_FILE_SENTINEL, overflow_path])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()
        return overflow_path

    def call(self, method_name, *args):
        overflow_path = None
        if self.world_size > 1 and self.rank == 0:
            overflow_path = self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        result = None
        error = None
        try:
            result = method(*args)
        except Exception as exc:
            error = exc
        try:
            self._synchronize_rpc_result(method_name, error)
            return result
        finally:
            if overflow_path is not None:
                try:
                    os.remove(overflow_path)
                except FileNotFoundError:
                    pass

    def _synchronize_rpc_result(self, method_name: str, error: Exception | None) -> None:
        if self.world_size <= 1 or method_name == "exit":
            if error is not None:
                raise error
            return
        failure = torch.tensor(
            [0 if error is None else 1], dtype=torch.int32, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        dist.all_reduce(failure, op=dist.ReduceOp.MAX)
        if error is not None:
            raise error
        if int(failure.item()) != 0:
            raise RuntimeError(f"Distributed RPC '{method_name}' failed on another rank")

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
        token_counts = [seq.seq_length - seq.num_cached_tokens for seq in seqs]
        for domain, lora_context in self._build_lora_contexts(seqs, token_counts).items():
            set_lora_context(lora_context, domain=domain)
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
        for domain, lora_context in self._build_lora_contexts(seqs, [1 for _ in seqs]).items():
            set_lora_context(lora_context, domain=domain)
        return positions

    def _make_graph_domain_buffers(self, max_rows: int, max_lora_buckets: int) -> dict[str, torch.Tensor]:
        return {
            "token_to_slot": torch.full((max_rows,), -1, dtype=torch.int32),
            "token_indices_sorted_by_slot": torch.arange(max_rows, dtype=torch.int32),
            "active_slot_ids": torch.arange(-1, max_lora_buckets - 1, dtype=torch.int32),
            "num_tokens_per_slot": torch.zeros(max_lora_buckets, dtype=torch.int32),
            "slot_start_offsets": torch.zeros(max_lora_buckets + 1, dtype=torch.int32),
        }

    def _copy_lora_domain_to_graph_vars(
        self,
        graph_vars: dict,
        domain: str,
        context: LoRAContext,
    ) -> None:
        domain_vars = graph_vars["lora_domains"][domain]
        token_count = 0 if context.token_to_slot is None else context.token_to_slot.size(0)
        domain_vars["token_to_slot"].fill_(-1)
        if context.token_to_slot is not None:
            domain_vars["token_to_slot"][:token_count] = context.token_to_slot
        domain_vars["token_indices_sorted_by_slot"][: domain_vars["token_indices_sorted_by_slot"].size(0)] = (
            torch.arange(
                domain_vars["token_indices_sorted_by_slot"].size(0),
                dtype=torch.int32,
                device=domain_vars["token_indices_sorted_by_slot"].device,
            )
        )
        if context.token_indices_sorted_by_slot is not None:
            domain_vars["token_indices_sorted_by_slot"][: context.token_indices_sorted_by_slot.size(0)] = (
                context.token_indices_sorted_by_slot.to(domain_vars["token_indices_sorted_by_slot"].device)
            )
        domain_vars["num_tokens_per_slot"].zero_()
        if context.active_slot_ids is not None and context.num_tokens_per_slot is not None:
            bucket_indices = (
                context.active_slot_ids.to(domain_vars["num_tokens_per_slot"].device, dtype=torch.int64) + 1
            )
            domain_vars["num_tokens_per_slot"].scatter_(
                0,
                bucket_indices,
                context.num_tokens_per_slot.to(domain_vars["num_tokens_per_slot"].device),
            )
        domain_vars["slot_start_offsets"].zero_()
        domain_vars["slot_start_offsets"][1:] = torch.cumsum(domain_vars["num_tokens_per_slot"], dim=0)

    def _set_graph_lora_contexts(self, graph_vars: dict, contexts: dict[str, LoRAContext]) -> None:
        for domain in LORA_DOMAINS:
            context = contexts[domain]
            self._copy_lora_domain_to_graph_vars(graph_vars, domain, context)
            domain_vars = graph_vars["lora_domains"][domain]
            token_count = 0 if context.token_to_slot is None else context.token_to_slot.size(0)
            num_lora_buckets = domain_vars["active_slot_ids"].size(0)
            set_lora_context(
                LoRAContext(
                    token_to_slot=domain_vars["token_to_slot"][:token_count],
                    token_indices_sorted_by_slot=domain_vars["token_indices_sorted_by_slot"][:token_count],
                    active_slot_ids=domain_vars["active_slot_ids"],
                    num_tokens_per_slot=domain_vars["num_tokens_per_slot"],
                    slot_start_offsets=domain_vars["slot_start_offsets"],
                    no_lora_flag=context.no_lora_flag,
                    num_active_loras=num_lora_buckets,
                ),
                domain=domain,
            )

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self._config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        max_dit_lora_rows = self._dit_lora_rows_per_sample() * max_bs
        positions = torch.zeros(max_bs, dtype=torch.int64)
        inputs = {
            "positions": positions,
        }
        inputs.update(self.make_dummy_inputs(max_bs, 1))

        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        max_lora_buckets = self.max_loras + 1
        lora_domains = {
            LM_LORA_DOMAIN: self._make_graph_domain_buffers(max_bs, max_lora_buckets),
            PROJ_LORA_DOMAIN: self._make_graph_domain_buffers(max_bs, max_lora_buckets),
            DIT_LORA_DOMAIN: self._make_graph_domain_buffers(max_dit_lora_rows, max_lora_buckets),
        }
        outputs = self.make_dummy_outputs(max_bs)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {"base": {}, "lora": {}}
        self.graph_pool = None
        capture_lora_graphs = bool(config.lora_config is not None and is_lora_available())

        for bs in reversed(self.graph_bs):
            base_graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            self._set_graph_lora_contexts(
                {"lora_domains": lora_domains},
                {
                    LM_LORA_DOMAIN: build_lora_context_from_slot_list([-1] * bs),
                    PROJ_LORA_DOMAIN: build_lora_context_from_slot_list([-1] * bs),
                    DIT_LORA_DOMAIN: build_lora_context_from_slot_list([-1] * (self._dit_lora_rows_per_sample() * bs)),
                },
            )

            if isinstance(outputs, torch.Tensor):
                outputs[:bs] = self.model(**cut_inputs(inputs, bs))  # warmup
            else:
                assign_outputs(self.model(**cut_inputs(inputs, bs)), outputs, bs)

            with torch.cuda.graph(base_graph, self.graph_pool):
                if isinstance(outputs, torch.Tensor):
                    outputs[:bs] = self.model(**cut_inputs(inputs, bs))  # capture
                else:
                    assign_outputs(self.model(**cut_inputs(inputs, bs)), outputs, bs)

            if self.graph_pool is None:
                self.graph_pool = base_graph.pool()
            self.graphs["base"][bs] = base_graph

            if capture_lora_graphs:
                lora_graph = torch.cuda.CUDAGraph()
                dummy_sample_to_slot = [0 for _ in range(bs)]
                dummy_contexts = {
                    LM_LORA_DOMAIN: build_lora_context_from_slot_list([0 for _ in range(bs)]),
                    PROJ_LORA_DOMAIN: build_lora_context_from_slot_list(dummy_sample_to_slot),
                    DIT_LORA_DOMAIN: build_lora_context_from_slot_list(
                        [slot for slot in dummy_sample_to_slot for _ in range(self._dit_lora_rows_per_sample())]
                    ),
                }
                set_context(
                    False,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                )
                self._set_graph_lora_contexts({"lora_domains": lora_domains}, dummy_contexts)
                if isinstance(outputs, torch.Tensor):
                    outputs[:bs] = self.model(**cut_inputs(inputs, bs))
                else:
                    assign_outputs(self.model(**cut_inputs(inputs, bs)), outputs, bs)
                with torch.cuda.graph(lora_graph, self.graph_pool):
                    if isinstance(outputs, torch.Tensor):
                        outputs[:bs] = self.model(**cut_inputs(inputs, bs))
                    else:
                        assign_outputs(self.model(**cut_inputs(inputs, bs)), outputs, bs)
                self.graphs["lora"][bs] = lora_graph
            torch.cuda.synchronize()
            reset_all_contexts()

        self.graph_vars = dict(
            inputs=inputs,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            lora_domains=lora_domains,
            outputs=outputs,
        )

    @torch.inference_mode()
    def run_model(self, inputs: dict, is_prefill: bool):
        lora_contexts = {domain: get_lora_context(domain) for domain in LORA_DOMAINS}
        has_active_lora = any(
            not context.no_lora_flag and context.token_to_slot is not None for context in lora_contexts.values()
        )
        has_lora_graph = has_active_lora and bool(self.graphs.get("lora"))
        try:
            if (
                is_prefill
                or self.enforce_eager
                or inputs["positions"].size(0) > 512
                or (has_active_lora and not has_lora_graph)
            ):
                return self.model(**inputs)

            bs = inputs["positions"].size(0)
            context = get_context()
            graph_kind = "lora" if has_active_lora else "base"
            graph = self.graphs[graph_kind][next(x for x in self.graph_bs if x >= bs)]
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
            self._set_graph_lora_contexts(graph_vars, lora_contexts)
            graph.replay()
            if isinstance(graph_vars["outputs"], torch.Tensor):
                return graph_vars["outputs"][:bs]
            else:
                return cut_inputs(graph_vars["outputs"], bs)
        finally:
            reset_all_contexts()
