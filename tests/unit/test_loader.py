import types

import pytest

torch = pytest.importorskip("torch")


def test_load_model_supports_packed_and_regular_weights(monkeypatch, tmp_path):
    from nanovllm_voxcpm.utils import loader

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights_path = model_dir / "weights.safetensors"
    weights_path.write_bytes(b"")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.regular = torch.nn.Linear(2, 2, bias=False)
            self.combined = torch.nn.Linear(2, 2, bias=False)
            self.packed_modules_mapping = {"packed": ("combined", 3)}

    model = DummyModel()
    packed_loader_calls: list[tuple[torch.Tensor, int]] = []

    def packed_loader(param, loaded_weight, shard_id):
        packed_loader_calls.append((loaded_weight.clone(), shard_id))
        param.data.copy_(loaded_weight)

    model.combined.weight.weight_loader = packed_loader

    tensors = {
        "regular.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "packed.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
    }

    class FakeSafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return tensors.keys()

        def get_tensor(self, name):
            return tensors[name]

    monkeypatch.setattr(loader, "glob", lambda pattern: [str(weights_path)])
    monkeypatch.setattr(loader, "safe_open", lambda *args, **kwargs: FakeSafeOpen())

    loader.load_model(model, str(model_dir))

    assert torch.equal(model.regular.weight, tensors["regular.weight"])
    assert torch.equal(model.combined.weight, tensors["packed.weight"])
    assert len(packed_loader_calls) == 1
    loaded_weight, shard_id = packed_loader_calls[0]
    assert torch.equal(loaded_weight, tensors["packed.weight"])
    assert shard_id == 3


def test_load_model_skips_optional_lora_parameters(monkeypatch, tmp_path):
    from nanovllm_voxcpm.utils import loader

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter("lora_a", torch.nn.Parameter(torch.zeros(1)))

    monkeypatch.setattr(loader, "glob", lambda pattern: [])

    loader.load_model(DummyModel(), str(model_dir))


def test_load_model_raises_for_missing_non_lora_parameters(monkeypatch, tmp_path):
    from nanovllm_voxcpm.utils import loader

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)
            self.register_parameter("lora_b", torch.nn.Parameter(torch.zeros(1)))

    monkeypatch.setattr(loader, "glob", lambda pattern: [])

    with pytest.raises(ValueError, match="Missing parameters"):
        loader.load_model(DummyModel(), str(model_dir))
