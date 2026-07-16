import sys
import types

import pytest


def _ensure_transformers_stub(monkeypatch):
    """Install a minimal transformers stub so imports work without the real package."""
    if "transformers" not in sys.modules:
        m = types.ModuleType("transformers")

        class PreTrainedTokenizer:  # pragma: no cover
            pass

        m.PreTrainedTokenizer = PreTrainedTokenizer
        monkeypatch.setitem(sys.modules, "transformers", m)


class _DummyTokenizer:
    """Minimal tokenizer stub used across multiple tests."""

    def __init__(self):
        self.vocab = {
            "你好": 1,
            "世界": 2,
            "▁A": 3,
            "你": 4,
            "好": 5,
            "世": 6,
            "界": 7,
        }

    def tokenize(self, text: str, **kwargs):
        assert text == "你好世界A"
        return ["你好", "世界", "▁A"]

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for t in tokens:
            if t in self.vocab:
                ids.append(self.vocab[t])
            elif t.replace("▁", "") in self.vocab:
                ids.append(self.vocab[t.replace("▁", "")])
            else:
                raise KeyError(t)
        return ids


def test_mask_multichar_chinese_tokens_splits_tokens(monkeypatch):
    _ensure_transformers_stub(monkeypatch)
    from nanovllm_voxcpm.models.voxcpm.utils import mask_multichar_chinese_tokens

    wrapper = mask_multichar_chinese_tokens(_DummyTokenizer())
    assert wrapper.tokenize("你好世界A") == ["你", "好", "世", "界", "▁A"]
    assert wrapper("你好世界A") == [4, 5, 6, 7, 3]


def test_tokenize_raises_type_error_for_non_string_input(monkeypatch):
    """Line 61: raise TypeError when text is not a str."""
    _ensure_transformers_stub(monkeypatch)
    from nanovllm_voxcpm.models.voxcpm.utils import mask_multichar_chinese_tokens

    wrapper = mask_multichar_chinese_tokens(_DummyTokenizer())
    with pytest.raises(TypeError, match="Expected string input"):
        wrapper.tokenize(123)


def test_call_raises_value_error_when_tokenization_fails(monkeypatch):
    """Lines 100-101: __call__ wraps internal errors in ValueError."""
    _ensure_transformers_stub(monkeypatch)
    from nanovllm_voxcpm.models.voxcpm.utils import mask_multichar_chinese_tokens

    class _BrokenTokenizer:
        """Tokenizer whose tokenize() raises on purpose."""

        vocab = {}  # no multichar tokens so multichar_tokens set is empty

        def tokenize(self, text: str, **kwargs):
            raise RuntimeError("deliberate tokenize failure")

        def convert_tokens_to_ids(self, tokens):  # pragma: no cover
            return []

    wrapper = mask_multichar_chinese_tokens(_BrokenTokenizer())
    with pytest.raises(ValueError, match="Tokenization failed"):
        wrapper("any text")
