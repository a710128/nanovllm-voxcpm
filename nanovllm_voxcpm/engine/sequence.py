"""nanovllm_voxcpm.engine.sequence

This module defines the *request-level* runtime state used by the engine.

The central type is :class:`Sequence`, which represents one in-flight generation
request (prompt + produced tokens) plus the bookkeeping needed to map the
sequence onto the shared KV-cache.

How it fits into the engine
---------------------------
The engine operates in a tight loop (see :class:`~nanovllm_voxcpm.engine.llm_engine.LLMEngineBase`):

1) Users (or model-specific engines) construct a :class:`Sequence` from an input
   prompt and enqueue it into :class:`~nanovllm_voxcpm.engine.scheduler.Scheduler`.
2) The :class:`~nanovllm_voxcpm.engine.scheduler.Scheduler` transitions sequences
   between ``WAITING`` and ``RUNNING`` and decides which ones to execute next.
3) :class:`~nanovllm_voxcpm.engine.block_manager.BlockManager` allocates/free KV
   blocks and fills ``Sequence.block_table`` (physical block ids used by the
   Attention KV-cache).
4) The model-specific engine converts :class:`Sequence` ->
   :class:`~nanovllm_voxcpm.engine.model_runner.RunnerTask` (runner-friendly
   view) and later merges outputs back into the :class:`Sequence`.

Key fields
----------
- ``token_ids``: The full token history for the request. Tokens may be ``int``
  or ``bytes``; the latter is supported by the KV prefix-cache hashing logic.
- ``num_prompt_tokens`` / ``num_completion_tokens``: Split prompt vs generated.
- ``num_cached_tokens``: Count of prompt tokens whose KV state is reused from
  the prefix cache (see :mod:`nanovllm_voxcpm.engine.block_manager`).
- ``block_table``: List of KV-cache *block ids* assigned to this sequence.
  The runner uses it to translate logical positions to physical cache slots.
- ``stoped``: A model-specific flag (set during postprocess) indicating the
  request should be finalized and deallocated.

Important invariant
-------------------
``block_table`` is the *only* way other modules refer to KV-cache memory for a
sequence. This keeps the runtime model-agnostic: scheduling and memory
management never touches model tensors directly.

Concrete example: VoxCPM
------------------------
The VoxCPM implementation under ``nanovllm_voxcpm/models/voxcpm`` uses
``Sequence.token_ids`` primarily for *prefix caching* rather than for feeding the
model:

- The list is a mixed stream of ``int`` (text token ids) and ``bytes`` (serialized
  latent patches). This is why the runtime allows ``int | bytes`` token items.
- Actual model inputs (text ids, audio features, masks, sampling params) live in
  ``Sequence.custom_payload`` and are converted to tensors in
  ``VoxCPMEngine.preprocess_seq`` / ``VoxCPMRunner.run``.
- Each decode step appends the newly generated latent patch (as ``bytes``) into
  ``Sequence.token_ids`` so subsequent steps can allocate KV slots and take part
  in the block-level prefix cache.
"""

from copy import copy
from enum import Enum, auto

from typing import Generic, TypeVar


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


PlayloadType = TypeVar("PlayloadType")


class Sequence(Generic[PlayloadType]):
    def __init__(
        self,
        seq_id: str,
        token_ids: list[int | bytes],
        block_size: int,
        custom_payload: PlayloadType = None,
    ):
        self.seq_id = seq_id
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.block_size = block_size

        self.custom_payload = custom_payload
        self.stoped = False

    def __len__(self):
        return self.num_tokens

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i) -> list[int | bytes]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int | bytes):
        self.token_ids.append(token_id)
        self.num_tokens += 1
