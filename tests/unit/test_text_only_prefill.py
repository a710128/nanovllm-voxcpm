import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn


@pytest.mark.parametrize("family", ["voxcpm", "voxcpm2"])
def test_text_only_prefill_matches_masked_path_and_skips_audio_modules(monkeypatch, family):
    module = importlib.import_module(f"nanovllm_voxcpm.models.{family}.model")
    model_class = module.VoxCPMModel if family == "voxcpm" else module.VoxCPM2Model
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)

        def forward(self, embeddings, positions):
            return embeddings + 1

    class FeatureEncoder(nn.Module):
        def forward(self, features):
            return features[:, 0]

    model.base_lm = LanguageModel()
    model.residual_lm = LanguageModel()
    model.feat_encoder = FeatureEncoder()
    model.enc_to_lm_proj = nn.Linear(4, 4)
    model.fsq_layer = nn.Linear(4, 4)
    model.fusion_concat_proj = nn.Linear(8, 4)
    model.lm_to_dit_proj = nn.Linear(4, 4)
    model.res_to_dit_proj = nn.Linear(4, 4)
    model.stop_proj = nn.Linear(4, 4)
    model.stop_actn = nn.SiLU()
    model.stop_head = nn.Linear(4, 2)
    monkeypatch.setattr(
        module, "get_context", lambda: SimpleNamespace(is_prefill=True, cu_seqlens_q=torch.tensor([0, 2, 5]))
    )
    inputs = dict(
        positions=torch.tensor([0, 1, 0, 1, 2]),
        text_tokens=torch.tensor([1, 2, 3, 4, 5]),
        # Nonzero features and projection bias must still be masked out.
        feat=torch.randn(5, 2, 4),
        feat_mask=torch.zeros(5, dtype=torch.bool),
        temperature=torch.ones(2),
        cfg_value=torch.ones(2),
        z_noise=torch.randn(2, 4, 2),
    )
    expected = model.forward_backbone(**inputs)

    def unexpected_call(*args, **kwargs):
        pytest.fail("text-only prefill must not execute the audio encoder or quantizer")

    monkeypatch.setattr(model.feat_encoder, "forward", unexpected_call)
    monkeypatch.setattr(model.fsq_layer, "forward", unexpected_call)
    actual = model.forward_backbone(**inputs, text_only_prefill=True)
    for expected_group, actual_group in zip(expected, actual):
        assert expected_group.keys() == actual_group.keys()
        for key in expected_group:
            assert torch.equal(expected_group[key], actual_group[key]), key


@pytest.mark.parametrize("has_audio", [False, True])
@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("family", ["voxcpm", "voxcpm2"])
def test_runner_only_selects_text_fast_path_for_all_text_prefill(monkeypatch, family, is_prefill, has_audio):
    import numpy as np

    module = importlib.import_module(f"nanovllm_voxcpm.models.{family}.runner")
    runner_class = module.VoxCPMRunner if family == "voxcpm" else module.VoxCPM2Runner
    payload_class = module.VoxCPMPayload if family == "voxcpm" else module.VoxCPM2Payload
    runner = runner_class.__new__(runner_class)
    runner.patch_size = 2
    runner.feat_dim = 4
    runner.prepare_prefill_context = lambda seqs: torch.arange(3)
    runner.prepare_decode_context = lambda seqs: torch.arange(3)
    original_tensor = torch.tensor
    original_randn = torch.randn
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *args, **kwargs: original_tensor(*args, **{k: v for k, v in kwargs.items() if k != "pin_memory"}),
    )
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *args, **kwargs: original_randn(*args, **{k: v for k, v in kwargs.items() if k != "device"}),
    )
    captured = {}

    class ReachedModel(Exception):
        pass

    def run_model(inputs, actual_is_prefill):
        captured.update(inputs)
        raise ReachedModel

    runner.run_model = run_model
    tasks = [
        SimpleNamespace(
            custom_payload=payload_class(
                text_tokens=np.array([1, 2, 3]),
                feats=np.zeros((3, 2, 4), dtype=np.float32),
                feat_masks=np.array([False, has_audio, False]),
            )
        )
    ]
    with pytest.raises(ReachedModel):
        runner.run(tasks, is_prefill)
    if is_prefill:
        assert captured["text_only_prefill"] is (not has_audio)
    else:
        assert "text_only_prefill" not in captured
