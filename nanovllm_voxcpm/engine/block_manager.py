"""nanovllm_voxcpm.engine.block_manager

This module manages the shared KV-cache *block pool*.

In this runtime, Attention KV memory is allocated as a fixed number of blocks
(``num_kvcache_blocks``), each block holding ``block_size`` token positions. A
sequence does not own contiguous KV memory; instead it owns a ``block_table``
(a list of block ids) that maps logical token positions to physical storage.

Responsibilities
----------------
- Admission control: decide whether a new sequence can be allocated
  (:meth:`BlockManager.can_allocate`).
- Allocate/free physical blocks for a sequence (:meth:`allocate`,
  :meth:`deallocate`).
- Prepare for incremental decode by ensuring a writable slot exists for the
  current last token (:meth:`can_append`, :meth:`may_append`).
- Prefix KV cache reuse ("prefix caching"): reuse already-computed KV blocks for
  identical prefixes across requests.

Prefix caching: how it works
----------------------------
For *full* blocks (exactly ``block_size`` tokens), we compute a rolling hash of
token ids. The hash of block i includes the hash of block i-1, so the hash is a
function of the entire prefix up to that block.

This is crucial because transformer KV states depend on the full causal prefix;
two identical blocks with different prefixes must not share KV memory.

Allocation uses the hash chain:
- If we find an existing block with the same rolling hash *and* identical
  token ids, we treat it as a cache hit and increase its reference count.
- If there is any cache miss at block k, all subsequent blocks must be treated
  as misses as well (because their prefixes differ).

Reference counting
------------------
Blocks are reference-counted so multiple sequences can share KV blocks when
prefix caching hits. A block is returned to the free list only when
``ref_count`` drops to zero.

Collaboration with the scheduler
--------------------------------
:class:`~nanovllm_voxcpm.engine.scheduler.Scheduler` is the only policy layer.
It calls into :class:`BlockManager` to:
- allocate blocks when moving a sequence into RUNNING (prefill);
- free blocks when a sequence finishes/cancels/preempts;
- check/prepare capacity before decode steps.

Concrete example: VoxCPM
------------------------
VoxCPM requests encode both text and audio context. In
``nanovllm_voxcpm/models/voxcpm/engine.py``, the sequence's ``token_ids``
(``hash_tokens``) contains:

- ``int`` token ids for (prompt_text + target_text) plus a special
  ``audio_start_token``;
- ``bytes`` chunks for prompt audio latents (each patch serialized via
  ``tobytes()``);
- during generation, each predicted latent patch is appended as ``bytes``.

This mixed stream is sufficient to uniquely identify the causal prefix, which
means prefix caching can safely reuse KV blocks across requests that share the
same text + prompt-audio prefix.
"""

from collections import deque
import xxhash
from nanovllm_voxcpm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int | bytes]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int | bytes], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        for it in token_ids:
            if isinstance(it, int):
                h.update(it.to_bytes(8, "little"))
            else:
                h.update(it)
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            if last_block.hash == -1:
                token_ids = seq.block(seq.num_blocks - 1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
