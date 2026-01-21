import sys
import types


def test_mask_multichar_chinese_tokens_splits_tokens(monkeypatch):
    # Provide a tiny transformers stub so the module imports without installing transformers.
    if "transformers" not in sys.modules:
        m = types.ModuleType("transformers")

        class PreTrainedTokenizer:  # pragma: no cover
            pass

        m.PreTrainedTokenizer = PreTrainedTokenizer
        monkeypatch.setitem(sys.modules, "transformers", m)

    from nanovllm_voxcpm.models.voxcpm.utils import mask_multichar_chinese_tokens

    class DummyTokenizer:
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

    wrapper = mask_multichar_chinese_tokens(DummyTokenizer())
    assert wrapper.tokenize("你好世界A") == ["你", "好", "世", "界", "▁A"]
    assert wrapper("你好世界A") == [4, 5, 6, 7, 3]
