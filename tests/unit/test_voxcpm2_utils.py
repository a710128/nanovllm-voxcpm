"""Tests for nanovllm_voxcpm.models.voxcpm2.utils.

All tests run on CPU and do not require GPU kernels or model weights.

The voxcpm2 package __init__.py triggers a heavy import cascade (engine ->
transformers.LlamaTokenizerFast, pydantic.ConfigDict, librosa, ...).  To avoid
that cascade while still exercising the *correct* source file and getting
accurate coverage numbers, we pre-load the module directly by file path and
register it under its canonical dotted name before any test tries an ordinary
``from nanovllm_voxcpm.models.voxcpm2.utils import ...``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# One-time module bootstrap (runs at collection time)
# ---------------------------------------------------------------------------
_UTILS_PATH = Path(__file__).parents[2] / "nanovllm_voxcpm" / "models" / "voxcpm2" / "utils.py"
_CANONICAL_NAME = "nanovllm_voxcpm.models.voxcpm2.utils"


def _bootstrap_utils_module() -> None:
    if _CANONICAL_NAME in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(_CANONICAL_NAME, str(_UTILS_PATH))
    assert spec is not None and spec.loader is not None, f"Cannot locate {_UTILS_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_CANONICAL_NAME] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]


_bootstrap_utils_module()

# Now the canonical import works cleanly in every test.
from nanovllm_voxcpm.models.voxcpm2.utils import mask_multichar_chinese_tokens  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    """Minimal tokenizer stub used across multiple tests."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        token_map: dict[str, list[str]] | None = None,
    ):
        self.vocab: dict[str, int] = vocab or {}
        self._token_map: dict[str, list[str]] = token_map or {}

    def tokenize(self, text: str, **kwargs) -> list[str]:
        return self._token_map.get(text, [])

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab[t] for t in tokens]


# ---------------------------------------------------------------------------
# mask_multichar_chinese_tokens -- tokenize()
# ---------------------------------------------------------------------------


class TestCharTokenizerWrapperTokenize:
    """Tests for the .tokenize() method of the returned wrapper."""

    def test_splits_multichar_chinese_tokens(self):
        """Multi-character Chinese tokens in the vocabulary are split into chars."""
        tok = _DummyTokenizer(
            vocab={"你好": 1, "世界": 2, "▁A": 3, "你": 4, "好": 5, "世": 6, "界": 7},
            token_map={"你好世界A": ["你好", "世界", "▁A"]},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        result = wrapper.tokenize("你好世界A")
        assert result == ["你", "好", "世", "界", "▁A"]

    def test_leaves_single_chinese_chars_intact(self):
        """Single CJK characters that are not in the multichar set stay as-is."""
        tok = _DummyTokenizer(
            vocab={"你": 1, "好": 2},
            token_map={"你好": ["你", "好"]},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        assert wrapper.tokenize("你好") == ["你", "好"]

    def test_leaves_non_chinese_tokens_intact(self):
        """ASCII / mixed tokens that are not in the multichar set pass through."""
        tok = _DummyTokenizer(
            vocab={"hello": 1, "world": 2},
            token_map={"hello world": ["hello", "world"]},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        assert wrapper.tokenize("hello world") == ["hello", "world"]

    def test_empty_string_returns_empty_list(self):
        """Tokenizing an empty string yields an empty list."""
        tok = _DummyTokenizer(vocab={}, token_map={"": []})
        wrapper = mask_multichar_chinese_tokens(tok)
        assert wrapper.tokenize("") == []

    def test_raises_type_error_for_non_string_input(self):
        """TypeError is raised when the input is not a string."""
        tok = _DummyTokenizer(vocab={}, token_map={})
        wrapper = mask_multichar_chinese_tokens(tok)
        with pytest.raises(TypeError):
            wrapper.tokenize(123)  # type: ignore[arg-type]

    def test_strips_prefix_glyph_before_multichar_lookup(self):
        """Tokens with the sentinel prefix are stripped before the multichar check."""
        tok = _DummyTokenizer(
            vocab={"你好": 1, "你": 2, "好": 3},
            token_map={"你好": ["▁你好"]},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        # "▁你好" strips to "你好" which IS in multichar_tokens -> split into chars
        result = wrapper.tokenize("你好")
        assert result == ["你", "好"]

    def test_mixed_chinese_and_ascii_tokens(self):
        """Mixed sequences split only the CJK multi-char tokens."""
        tok = _DummyTokenizer(
            vocab={"你好": 1, "你": 2, "好": 3, "▁hello": 4},
            token_map={"你好hello": ["你好", "▁hello"]},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        result = wrapper.tokenize("你好hello")
        assert result == ["你", "好", "▁hello"]

    def test_multichar_token_with_non_chinese_chars_not_split(self):
        """A vocab token with mixed/non-CJK chars is NOT in the multichar set."""
        # "ab" is length >= 2 but not all CJK -> should NOT be split
        tok = _DummyTokenizer(
            vocab={"ab": 1},
            token_map={"ab": ["ab"]},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        result = wrapper.tokenize("ab")
        assert result == ["ab"]

    def test_three_char_chinese_token_split(self):
        """A 3-character CJK vocab token is split into 3 individual chars."""
        tok = _DummyTokenizer(
            vocab={"你好啊": 1, "你": 2, "好": 3, "啊": 4},
            token_map={"你好啊": ["你好啊"]},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        result = wrapper.tokenize("你好啊")
        assert result == ["你", "好", "啊"]

    def test_single_char_non_split_passthrough(self):
        """A single-char (non-multichar) token passes through unchanged."""
        tok = _DummyTokenizer(
            vocab={"你": 1},
            token_map={"你": ["你"]},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        assert wrapper.tokenize("你") == ["你"]


# ---------------------------------------------------------------------------
# mask_multichar_chinese_tokens -- __call__()
# ---------------------------------------------------------------------------


class TestCharTokenizerWrapperCall:
    """Tests for the __call__ method (tokenize -> convert_tokens_to_ids)."""

    def test_call_returns_ids(self):
        """__call__ converts tokens to IDs via the base tokenizer."""
        tok = _DummyTokenizer(
            vocab={"你好": 1, "世界": 2, "▁A": 3, "你": 4, "好": 5, "世": 6, "界": 7},
            token_map={"你好世界A": ["你好", "世界", "▁A"]},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        result = wrapper("你好世界A")
        assert result == [4, 5, 6, 7, 3]

    def test_call_empty_string_returns_empty_list(self):
        """__call__ on empty string yields an empty ID list."""
        tok = _DummyTokenizer(vocab={}, token_map={"": []})
        wrapper = mask_multichar_chinese_tokens(tok)
        assert wrapper("") == []

    def test_call_raises_value_error_on_tokenize_failure(self):
        """__call__ wraps exceptions from tokenize() into ValueError."""

        class FailingTokenizer:
            vocab: dict[str, int] = {}

            def tokenize(self, text: str, **kwargs):
                raise RuntimeError("tokenizer exploded")

            def convert_tokens_to_ids(self, tokens):  # pragma: no cover
                return []

        wrapper = mask_multichar_chinese_tokens(FailingTokenizer())
        with pytest.raises(ValueError, match="Tokenization failed"):
            wrapper("any text")

    def test_call_raises_value_error_on_convert_failure(self):
        """__call__ wraps KeyError from convert_tokens_to_ids into ValueError."""

        class BadConvertTokenizer:
            vocab: dict[str, int] = {}

            def tokenize(self, text: str, **kwargs):
                return ["unknown_token"]

            def convert_tokens_to_ids(self, tokens):
                raise KeyError("unknown_token")

        wrapper = mask_multichar_chinese_tokens(BadConvertTokenizer())
        with pytest.raises(ValueError, match="Tokenization failed"):
            wrapper("some text")

    def test_call_type_error_non_string_raises_value_error(self):
        """__call__ with non-string raises ValueError (wrapping TypeError)."""
        tok = _DummyTokenizer(vocab={}, token_map={})
        wrapper = mask_multichar_chinese_tokens(tok)
        with pytest.raises(ValueError, match="Tokenization failed"):
            wrapper(42)  # type: ignore[arg-type]

    def test_call_single_chinese_char(self):
        """__call__ for a single CJK char returns its ID unchanged."""
        tok = _DummyTokenizer(
            vocab={"你": 7},
            token_map={"你": ["你"]},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        assert wrapper("你") == [7]


# ---------------------------------------------------------------------------
# mask_multichar_chinese_tokens -- multichar_tokens set construction
# ---------------------------------------------------------------------------


class TestMulticharTokenSetBuilding:
    """Tests that the multichar_tokens set is built correctly from the vocab."""

    def test_only_all_chinese_tokens_are_included(self):
        """The multichar set excludes tokens with any non-CJK character."""
        tok = _DummyTokenizer(
            vocab={
                "你好": 1,    # all CJK, len >= 2 -> IN multichar_tokens
                "a好": 2,     # mixed -> NOT in multichar_tokens
                "你": 3,      # single char -> NOT in multichar_tokens (len < 2)
                "hello": 4,   # ASCII -> NOT in multichar_tokens
                "世界观": 5,  # all CJK, len 3 -> IN multichar_tokens
            },
            token_map={},
        )
        wrapper = mask_multichar_chinese_tokens(tok)
        assert wrapper.multichar_tokens == {"你好", "世界观"}

    def test_empty_vocab_gives_empty_multichar_set(self):
        """An empty vocab produces an empty multichar_tokens set."""
        tok = _DummyTokenizer(vocab={}, token_map={})
        wrapper = mask_multichar_chinese_tokens(tok)
        assert wrapper.multichar_tokens == set()

    def test_vocab_with_only_single_chars_gives_empty_multichar_set(self):
        """All single-char CJK tokens -> multichar set remains empty."""
        tok = _DummyTokenizer(vocab={"你": 1, "好": 2, "a": 3}, token_map={})
        wrapper = mask_multichar_chinese_tokens(tok)
        assert wrapper.multichar_tokens == set()

    def test_wrapper_exposes_base_tokenizer(self):
        """The wrapper holds a reference to the original tokenizer."""
        tok = _DummyTokenizer(vocab={}, token_map={})
        wrapper = mask_multichar_chinese_tokens(tok)
        assert wrapper.tokenizer is tok
