import pytest

torch = pytest.importorskip("torch")


def _make_engine(max_model_len: int, token_count: int):
    """Create a VoxCPM2Engine instance without heavy init."""

    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine

    e = VoxCPM2Engine.__new__(VoxCPM2Engine)
    e.n_decode_pad_frames = 4
    e.feat_dim = 8
    e.patch_size = 1
    e.audio_start_token = 101
    e.block_size = 256
    e.max_model_len = max_model_len

    e.tokenizer = lambda _s: list(range(token_count))

    e._captured_seq = None
    e.add_sequence = lambda seq: setattr(e, "_captured_seq", seq)
    return e


def test_add_request_rejects_too_long_prompt():
    e = _make_engine(max_model_len=4, token_count=4)
    with pytest.raises(ValueError, match=r"Prompt is too long"):
        e.add_request(seq_id="s", target_text="x", max_generate_length=1)


def test_add_request_rejects_when_total_can_exceed_max_model_len():
    e = _make_engine(max_model_len=10, token_count=4)
    with pytest.raises(ValueError, match=r"may exceed max_model_len"):
        e.add_request(seq_id="s", target_text="x", max_generate_length=6)


def test_add_request_allows_on_boundary_and_enqueues_sequence():
    e = _make_engine(max_model_len=11, token_count=4)
    e.add_request(seq_id="s", target_text="x", max_generate_length=6)
    assert e._captured_seq is not None
    assert len(e._captured_seq) == 5


def test_add_request_requires_positive_max_generate_length():
    e = _make_engine(max_model_len=10, token_count=1)
    with pytest.raises(ValueError, match=r"max_generate_length must be >= 1"):
        e.add_request(seq_id="s", target_text="x", max_generate_length=0)
