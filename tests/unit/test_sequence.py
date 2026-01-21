import pytest


def test_sequence_properties_and_append():
    from nanovllm_voxcpm.engine.sequence import Sequence

    tokens = [1, 2, 3, 4, 5]
    seq = Sequence(seq_id="s1", token_ids=tokens, block_size=4)

    # Sequence copies input tokens.
    tokens.append(999)
    assert len(seq.token_ids) == 5

    assert len(seq) == 5
    assert seq.num_prompt_tokens == 5
    assert seq.num_completion_tokens == 0
    assert seq.num_blocks == 2
    assert seq.last_block_num_tokens == 1
    assert seq.block(0) == [1, 2, 3, 4]
    assert seq.block(1) == [5]

    seq.append_token(6)
    assert len(seq) == 6
    assert seq.num_blocks == 2
    assert seq.last_block_num_tokens == 2


def test_sequence_block_index_bounds():
    from nanovllm_voxcpm.engine.sequence import Sequence

    seq = Sequence(seq_id="s1", token_ids=[1, 2, 3], block_size=4)
    with pytest.raises(AssertionError):
        _ = seq.block(-1)
    with pytest.raises(AssertionError):
        _ = seq.block(1)
