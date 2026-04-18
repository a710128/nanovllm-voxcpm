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
from nanovllm_voxcpm.layers.lora import iter_lora_modules
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


def _clear_lora_slot_modules(modules, slot_id: int, module_names: list[str] | None = None) -> None:
    """Zero out LoRA weights for ``slot_id`` across ``modules``.

    ``module_names`` (when provided) restricts the walk to just the modules
    previously written into this slot. This avoids iterating the entire model
    graph — and issuing dozens of tiny ``zero_()`` kernels per slot admission —
    for the common case where each LoRA only populates a small subset of
    modules. Passing ``None`` preserves the legacy "clear everything" behavior
    (used by tests).
    """
    if module_names is None:
        iterable = modules.values()
    else:
        iterable = (modules[name] for name in module_names if name in modules)
    for module in iterable:
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
        # Lazy cache of ``dict(self.model.named_modules())`` — walking the full
        # VoxCPM module tree is surprisingly expensive (~ms per call on a
        # real-sized model) and was called on every LoRA slot admission and
        # validation. Populated by ``_lora_model_modules()`` on first use.
        self._lora_model_modules_cache: dict[str, torch.nn.Module] | None = None
        # Track which module names each GPU slot currently holds LoRA weights
        # for, so evict/clear can skip the no-op "zero already-zero weights"
        # walk across the entire model graph.
        self._lora_slot_modules: dict[int, list[str]] = {}

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
            # No active LoRA anywhere in this batch. Build just the LM
            # ``token_to_slot=[-1,...]`` tensor; the PROJ and DIT domains can
            # share the "all -1" sentinel since every sample gets slot=-1.
            # Layers short-circuit on ``no_lora_flag=True`` before reading any
            # of the other fields, so PROJ/DIT don't even need a device
            # tensor.
            empty_ctx = LoRAContext(no_lora_flag=True, num_active_loras=0)
            return {
                LM_LORA_DOMAIN: build_lora_context_from_slot_list([-1] * sum(token_counts)),
                PROJ_LORA_DOMAIN: empty_ctx,
                DIT_LORA_DOMAIN: empty_ctx,
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

    def _lora_model_modules(self) -> dict[str, torch.nn.Module]:
        """Memoize ``dict(self.model.named_modules())``.

        The dict is only invalidated by topology changes to the model; LoRA
        admission/validation never mutates the module graph, so it's safe to
        cache for the lifetime of the runner.
        """
        cache = getattr(self, "_lora_model_modules_cache", None)
        if cache is None:
            cache = dict(self.model.named_modules())
            self._lora_model_modules_cache = cache
        return cache

    def validate_lora_payload(
        self, payload: LoRAModelPayload | list[LoRAModelPayload] | tuple[LoRAModelPayload, ...]
    ) -> None:
        rank_payload = select_lora_payload_for_rank(payload, self.rank)
        if rank_payload.rank <= 0:
            raise ValueError(f"LoRA payload rank must be > 0, got {rank_payload.rank}")
        if not rank_payload.modules:
            raise ValueError("LoRA payload must contain at least one target module")

        modules = self._lora_model_modules()
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
        modules = self._lora_model_modules()
        # Only clear modules that the previous occupant of this slot actually
        # populated. This avoids issuing one ``zero_()`` kernel per LoRA-capable
        # layer in the entire model on every admission, which dominated the
        # ~0.18s LoRA TTFB regression.
        slot_modules = getattr(self, "_lora_slot_modules", None)
        if slot_modules is None:
            slot_modules = {}
            self._lora_slot_modules = slot_modules
        previously_loaded = slot_modules.get(slot_id)
        _clear_lora_slot_modules(modules, slot_id, module_names=previously_loaded)
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
        slot_modules[slot_id] = list(payload.modules.keys())

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

        # If LoRA is enabled, run additional warmup prefills with a fake
        # active slot so the Triton shrink/expand kernels JIT-compile and
        # autotune for prefill-shaped inputs during startup. Without this,
        # the first real request pays the JIT cost (hundreds of ms) on its
        # critical path and TTFB regresses significantly.
        #
        # Slot 0 weights are zero at this point; the kernel still runs and
        # contributes 0 to the output, which is exactly what we need for a
        # compile-only warmup.
        #
        # We exercise two shapes because `get_lora_op_configs` picks a
        # different shrink config at M<128 vs M>=128 (different split_k and
        # block_k), which instantiates distinct Triton kernels. Warming only
        # one regime still leaves the other to JIT on the first real request.
        if self.max_loras > 0 and is_lora_available():
            short_len = min(64, max_model_len)
            shape_candidates = []
            if max_model_len >= 128:
                shape_candidates.append((num_seqs, max_model_len))
            if short_len < 128:
                shape_candidates.append((1, short_len))
            # Deduplicate while preserving order.
            seen = set()
            shapes = [s for s in shape_candidates if not (s in seen or seen.add(s))]
            for warmup_num_seqs, warmup_len in shapes:
                warmup_seqs = [
                    RunnerTask(
                        block_table=[],
                        seq_length=warmup_len,
                        num_cached_tokens=0,
                        block_size=self.block_size,
                        custom_payload=None,
                    )
                    for _ in range(warmup_num_seqs)
                ]
                warmup_inputs = {"positions": self.prepare_prefill_context(warmup_seqs)}
                warmup_inputs.update(self.make_dummy_inputs(warmup_num_seqs, warmup_len))
                # Override LoRA contexts with "slot 0 active for every row".
                total_rows = warmup_num_seqs * warmup_len
                dit_rows_per_sample = self._dit_lora_rows_per_sample()
                lm_ctx = build_lora_context_from_slot_list([0] * total_rows)
                proj_ctx = build_lora_context_from_slot_list([0] * warmup_num_seqs)
                dit_ctx = build_lora_context_from_slot_list([0] * (warmup_num_seqs * dit_rows_per_sample))
                set_lora_context(lm_ctx, domain=LM_LORA_DOMAIN)
                set_lora_context(proj_ctx, domain=PROJ_LORA_DOMAIN)
                set_lora_context(dit_ctx, domain=DIT_LORA_DOMAIN)
                _ = self.model(**warmup_inputs)

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
        """Update per-domain graph-captured LoRA metadata buffers in place.

        Hot path: this runs 3× per decode step (once per LoRA domain), and
        previously issued ~10 tiny kernel launches per domain (fill_, two
        narrow-slice copies including a fresh ``torch.arange`` allocation,
        zero_, scatter_, zero_, cumsum). That was ~30 launches / step of pure
        metadata shuffling, dominating the LoRA RTF regression.

        We now precompute the final-form slices on CPU (int32), pack them into
        one pinned buffer, and issue a single H2D copy — plus a single cumsum
        on GPU for ``slot_start_offsets`` (we still derive it from
        ``num_tokens_per_slot`` on-device to keep the final tensor consistent
        with what kernels read).

        The ``no_lora_flag`` case is specialised to a tiny ``fill_(-1)`` on
        ``token_to_slot`` — all other buffers were preallocated to the correct
        sentinel state at capture time and kernels short-circuit on
        ``no_lora`` anyway.
        """
        domain_vars = graph_vars["lora_domains"][domain]
        token_to_slot_buf: torch.Tensor = domain_vars["token_to_slot"]
        token_indices_buf: torch.Tensor = domain_vars["token_indices_sorted_by_slot"]
        num_tokens_buf: torch.Tensor = domain_vars["num_tokens_per_slot"]
        slot_start_buf: torch.Tensor = domain_vars["slot_start_offsets"]

        token_count = 0 if context.token_to_slot is None else context.token_to_slot.size(0)

        if context.no_lora_flag or context.token_to_slot is None:
            # Kernels bail out on no_lora; we only need token_to_slot to be
            # all -1 so downstream sanity checks still see a stable state.
            token_to_slot_buf.fill_(-1)
            num_tokens_buf.zero_()
            slot_start_buf.zero_()
            # token_indices buffer already contains arange(...) from capture
            # time; no kernel needed to restore it since no_lora short-circuits
            # before it's read.
            return

        buf_size = token_to_slot_buf.size(0)
        device = token_to_slot_buf.device

        # 1. token_to_slot: prefix from context, rest -1.
        token_to_slot_buf[:token_count].copy_(context.token_to_slot, non_blocking=True)
        if token_count < buf_size:
            token_to_slot_buf[token_count:].fill_(-1)

        # 2. token_indices_sorted_by_slot: prefix from context, rest stays at
        # whatever arange value it had from capture (kernels only read the
        # first ``token_count`` entries via num_tokens_per_slot+slot_start).
        if context.token_indices_sorted_by_slot is not None:
            token_indices_buf[: context.token_indices_sorted_by_slot.size(0)].copy_(
                context.token_indices_sorted_by_slot, non_blocking=True
            )

        # 3. num_tokens_per_slot: zero, then scatter. Single scatter kernel —
        # unavoidable when we need to honor active_slot_ids ordering.
        num_tokens_buf.zero_()
        if context.active_slot_ids is not None and context.num_tokens_per_slot is not None:
            bucket_indices = context.active_slot_ids.to(device=device, dtype=torch.int64) + 1
            num_tokens_buf.scatter_(0, bucket_indices, context.num_tokens_per_slot.to(device=device))

        # 4. slot_start_offsets: cumsum of num_tokens_per_slot, with a leading
        # zero. Done as one cumsum kernel into the [1:] view.
        slot_start_buf[0] = 0
        torch.cumsum(num_tokens_buf, dim=0, out=slot_start_buf[1:])

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
        if capture_lora_graphs:
            for module in iter_lora_modules(self.model):
                prime_lora_cache = getattr(module, "prime_lora_cache", None)
                if prime_lora_cache is not None:
                    prime_lora_cache()

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
