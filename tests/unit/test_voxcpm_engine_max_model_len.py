import pytest

torch = pytest.importorskip("torch")


def _make_engine(max_model_len: int, token_count: int):
    """Create a VoxCPMEngine instance without heavy init.

    We bypass VoxCPMEngine.__init__ because it loads tokenizer/model weights and
    initializes GPU runtime. For these unit tests we only need to exercise
    add_request()'s length checks.
    """

    from nanovllm_voxcpm.models.voxcpm.engine import VoxCPMEngine

    e = VoxCPMEngine.__new__(VoxCPMEngine)
    e.n_decode_pad_frames = 4
    e.feat_dim = 8
    e.patch_size = 1
    e.audio_start_token = 101
    e.block_size = 256
    e.max_model_len = max_model_len

    # The real tokenizer returns a list[int] token ids.
    e.tokenizer = lambda _s: list(range(token_count))

    # LLMEngineBase.add_sequence() would require scheduler/runner.
    e._captured_seq = None
    e.add_sequence = lambda seq: setattr(e, "_captured_seq", seq)
    e.resolve_lora = lambda name: None if name is None else 7
    return e


def test_add_request_rejects_too_long_prompt():
    e = _make_engine(max_model_len=4, token_count=4)  # + audio_start_token => prompt_len=5
    with pytest.raises(ValueError, match=r"Prompt is too long"):
        e.add_request(seq_id="s", target_text="x", max_generate_length=1)


def test_add_request_rejects_when_total_can_exceed_max_model_len():
    e = _make_engine(max_model_len=10, token_count=4)  # prompt_len=5
    with pytest.raises(ValueError, match=r"may exceed max_model_len"):
        e.add_request(seq_id="s", target_text="x", max_generate_length=6)  # 5 + 6 = 11


def test_add_request_allows_on_boundary_and_enqueues_sequence():
    e = _make_engine(max_model_len=11, token_count=4)  # prompt_len=5
    e.add_request(seq_id="s", target_text="x", max_generate_length=6)  # 5 + 6 = 11
    assert e._captured_seq is not None
    assert len(e._captured_seq) == 5


def test_add_request_requires_positive_max_generate_length():
    e = _make_engine(max_model_len=10, token_count=1)
    with pytest.raises(ValueError, match=r"max_generate_length must be >= 1"):
        e.add_request(seq_id="s", target_text="x", max_generate_length=0)


def test_add_request_resolves_lora_name_into_adapter_id():
    e = _make_engine(max_model_len=11, token_count=4)
    e.add_request(seq_id="s", target_text="x", max_generate_length=6, lora_name="demo")
    assert e._captured_seq.lora_name == "demo"
    assert e._captured_seq.adapter_id == 7
