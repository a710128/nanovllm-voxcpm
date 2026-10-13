import pytest

xxhash = pytest.importorskip("xxhash")


def _make_seq(seq_id: str, token_ids: list[int], block_size: int):
    from nanovllm_voxcpm.engine.sequence import Sequence

    return Sequence(seq_id=seq_id, token_ids=token_ids, block_size=block_size)


def test_block_manager_allocate_cache_reuse_and_deallocate():
    from nanovllm_voxcpm.engine.block_manager import BlockManager

    bm = BlockManager(num_blocks=8, block_size=4)
    seq1 = _make_seq("s1", [1, 2, 3, 4, 5, 6, 7, 8], 4)
    seq2 = _make_seq("s2", [1, 2, 3, 4, 5, 6, 7, 8], 4)

    bm.allocate(seq1)
    assert seq1.num_cached_tokens == 0
    assert len(seq1.block_table) == 2

    bm.allocate(seq2)
    # All blocks should hit the prefix cache.
    assert seq2.num_cached_tokens == 8
    assert seq1.block_table == seq2.block_table
    for block_id in seq1.block_table:
        assert bm.blocks[block_id].ref_count == 2

    bm.deallocate(seq1)
    for block_id in seq2.block_table:
        assert bm.blocks[block_id].ref_count == 1

    bm.deallocate(seq2)
    assert seq2.block_table == []
    assert len(bm.free_block_ids) == 8
    assert bm.used_block_ids == set()


def _assert_hash_consistency(bm):
    """Invariant: a hash entry lives exactly as long as the content it maps to.

    - Every entry points to the block currently holding that content.
    - Every content-bearing block has its entry registered.
    - The dict is bounded by the physical block count.
    """
    assert len(bm.hash_to_block_id) <= len(bm.blocks)
    for h, block_id in bm.hash_to_block_id.items():
        assert bm.blocks[block_id].hash == h
    for block in bm.blocks:
        if block.hash != -1:
            assert bm.hash_to_block_id.get(block.hash) == block.block_id


def test_block_manager_reuse_after_deallocate():
    from nanovllm_voxcpm.engine.block_manager import BlockManager

    bm = BlockManager(num_blocks=8, block_size=4)
    seq1 = _make_seq("s1", [1, 2, 3, 4, 5, 6, 7, 8], 4)
    bm.allocate(seq1)
    block_table_before = list(seq1.block_table)
    bm.deallocate(seq1)

    # A freed block keeps its content until recycled, so a later request with
    # the same prefix must resurrect it instead of recomputing the prefill.
    seq2 = _make_seq("s2", [1, 2, 3, 4, 5, 6, 7, 8], 4)
    bm.allocate(seq2)
    assert seq2.num_cached_tokens == 8
    assert seq2.block_table == block_table_before
    for block_id in seq2.block_table:
        assert block_id in bm.used_block_ids
        assert bm.blocks[block_id].ref_count == 1


def test_block_manager_recycle_evicts_stale_hash_entries():
    from nanovllm_voxcpm.engine.block_manager import BlockManager

    bm = BlockManager(num_blocks=4, block_size=4)
    seq1 = _make_seq("s1", [1, 2, 3, 4, 5, 6, 7, 8], 4)
    bm.allocate(seq1)
    stale_hashes = [bm.blocks[b].hash for b in seq1.block_table]
    bm.deallocate(seq1)

    # Force every free block through recycling with different content.
    for i in range(10):
        seq = _make_seq(f"s{i}", [100 * (i + 1) + j for j in range(4)], 4)
        bm.allocate(seq)
        bm.deallocate(seq)

    for h in stale_hashes:
        assert h not in bm.hash_to_block_id
    _assert_hash_consistency(bm)


def test_block_manager_hash_entries_bounded_by_num_blocks():
    from nanovllm_voxcpm.engine.block_manager import BlockManager

    bm = BlockManager(num_blocks=4, block_size=4)
    for i in range(50):
        seq = _make_seq(f"s{i}", [100 * (i + 1) + j for j in range(8)], 4)
        bm.allocate(seq)
        bm.deallocate(seq)

    assert len(bm.hash_to_block_id) <= len(bm.blocks)
    _assert_hash_consistency(bm)


def test_block_manager_append_block_boundary():
    from nanovllm_voxcpm.engine.block_manager import BlockManager

    bm = BlockManager(num_blocks=4, block_size=4)
    seq = _make_seq("s1", [1, 2, 3], 4)
    bm.allocate(seq)

    # When we append the 4th token, the last block becomes full and gets hashed.
    seq.append_token(4)
    bm.may_append(seq)
    last_block = bm.blocks[seq.block_table[-1]]
    assert last_block.hash != -1

    # Appending one more token starts a new (unhashed) block.
    seq.append_token(5)
    assert bm.can_append(seq)
    bm.may_append(seq)
    assert len(seq.block_table) == 2
    assert bm.blocks[seq.block_table[-1]].hash == -1
