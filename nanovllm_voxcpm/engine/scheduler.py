"""nanovllm_voxcpm.engine.scheduler

This module implements the batching/scheduling policy for the inference runtime.

The scheduler owns two queues of :class:`~nanovllm_voxcpm.engine.sequence.Sequence`:
- ``waiting``: admitted requests that are not currently executing.
- ``running``: requests that have KV-cache allocated and participate in decode.

It is responsible for choosing *which* sequences run on the next engine step and
for enforcing resource limits:
- ``max_num_seqs``: maximum number of sequences per batch.
- ``max_num_batched_tokens``: maximum number of tokens computed in a prefill.
- KV-cache capacity: enforced via :class:`~nanovllm_voxcpm.engine.block_manager.BlockManager`.

Two-phase scheduling
--------------------
The scheduler operates in two modes (returned as ``is_prefill``):

1) Prefill phase (``is_prefill=True``)
   - Pull from ``waiting`` in FIFO order.
   - Admit a sequence only if:
     * batched tokens would not exceed ``max_num_batched_tokens``
     * :meth:`BlockManager.can_allocate` is true for the full prompt length.
   - Allocate KV blocks (:meth:`BlockManager.allocate`) and move the sequence to
     ``running``.
   - If prefix caching hits, ``Sequence.num_cached_tokens`` may be > 0, and only
     the remaining prompt tokens (uncached portion) count toward the batch.

2) Decode phase (``is_prefill=False``)
   - Round-robin over ``running``.
   - Before decoding one step for a sequence, ensure there is KV space for the
     *current last token* (see :meth:`BlockManager.can_append`).
   - If KV space is insufficient, preempt other sequences: move them back to
     ``waiting`` and free their blocks (:meth:`preempt`).
   - Once capacity is ensured, prepare KV bookkeeping for the step
     (:meth:`BlockManager.may_append`) and include the sequence in the decode batch.

Decode bookkeeping detail
-------------------------
The engine appends newly generated tokens during postprocessing (after the model
returns). Therefore, on the *next* decode step, the sequence already contains a
new "last token" whose KV state has not been written yet. ``may_append`` is
called before executing the decode step to ensure there is a physical KV slot
(possibly a new block) where that last token will be stored.

Concrete example: VoxCPM
------------------------
VoxCPM appends a newly generated latent patch as ``bytes`` into
``Sequence.token_ids`` in ``VoxCPMEngine.postprocess_seq``. On the next step,
the scheduler will:
- potentially allocate a new KV block if the new token starts a new block;
- then batch the sequence for decode so the runner can compute KV for that last
  latent patch and predict the next one.

Interaction with the engine loop
--------------------------------
The engine calls :meth:`Scheduler.schedule` once per step, then executes the
returned sequences via the model runner. Model-specific postprocessing sets
``seq.stoped`` when an EOS/stop condition is met, after which the engine calls
:meth:`Scheduler.finish` to deallocate KV resources and remove the request.
"""

from collections import deque

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.sequence import Sequence, SequenceStatus
from nanovllm_voxcpm.engine.block_manager import BlockManager

import uuid


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        self._id_to_seq: dict[str, Sequence] = {}

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self._id_to_seq[seq.seq_id] = seq

        self.waiting.append(seq)

    def cancel(self, seq_id: str):
        try:
            seq = self._id_to_seq.pop(seq_id)
        except KeyError:
            return

        self.block_manager.deallocate(seq)
        if seq.status == SequenceStatus.RUNNING:
            self.running.remove(seq)
        elif seq.status == SequenceStatus.WAITING:
            self.waiting.remove(seq)
        return

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break

            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            tokens_to_compute = len(seq) - seq.num_cached_tokens
            if tokens_to_compute > 0:
                num_seqs += 1
                scheduled_seqs.append(seq)
                num_batched_tokens += tokens_to_compute

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def finish(self, seq: Sequence):
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.running.remove(seq)
        self._id_to_seq.pop(seq.seq_id)
