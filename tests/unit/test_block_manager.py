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
