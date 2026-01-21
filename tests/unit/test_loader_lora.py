import sys
import types

import pytest


torch = pytest.importorskip("torch")


def _ensure_safetensors_stub(monkeypatch):
    """Allow importing nanovllm_voxcpm.utils.loader without safetensors installed."""

    if "safetensors" in sys.modules:
        return

    safetensors = types.ModuleType("safetensors")

    def safe_open(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("safe_open is not available in this test stub")

    safetensors.safe_open = safe_open

    safetensors_torch = types.ModuleType("safetensors.torch")

    def load_file(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("safetensors.torch.load_file is not available in this test stub")

    safetensors_torch.load_file = load_file

    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)


def test_map_lora_weight_name(monkeypatch):
    _ensure_safetensors_stub(monkeypatch)

    from nanovllm_voxcpm.utils.loader import _map_lora_weight_name

    assert _map_lora_weight_name("q_proj.lora_A")[0] == "qkv_proj.lora_A"
    assert _map_lora_weight_name("q_proj.lora_A")[1] == "q"
    assert _map_lora_weight_name("k_proj.lora_A")[1] == "k"
    assert _map_lora_weight_name("v_proj.lora_A")[1] == "v"
    assert _map_lora_weight_name("q_proj.lora_B")[0] == "qkv_proj.lora_B_q"

    # No mapping should be a no-op.
    assert _map_lora_weight_name("some_other_param")[0] == "some_other_param"


def test_load_lora_weights_ckpt_fused_lora_A(monkeypatch, tmp_path):
    _ensure_safetensors_stub(monkeypatch)

    from nanovllm_voxcpm.utils.loader import load_lora_weights

    class DummyQKVProj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_targets = ["q", "k", "v"]
            self.lora_r = 2
            self.lora_A = torch.nn.Parameter(torch.zeros(6, 4))  # r * 3, hidden
            self.lora_B_q = torch.nn.Parameter(torch.zeros(2, 2))
            self.lora_B_k = torch.nn.Parameter(torch.zeros(2, 2))
            self.lora_B_v = torch.nn.Parameter(torch.zeros(2, 2))
            self.loaded_A = {}

        def load_lora_A(self, t: torch.Tensor, shard_id: str):
            # The loader calls this for fused lora_A shards.
            self.loaded_A[shard_id] = t.detach().cpu().clone()

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = DummyQKVProj()

    model = DummyModel()

    state = {
        "q_proj.lora_A": torch.full((2, 4), 1.0),
        "k_proj.lora_A": torch.full((2, 4), 2.0),
        "v_proj.lora_A": torch.full((2, 4), 3.0),
        "q_proj.lora_B": torch.full((2, 2), 4.0),
        "k_proj.lora_B": torch.full((2, 2), 5.0),
        "v_proj.lora_B": torch.full((2, 2), 6.0),
        "not_lora.weight": torch.zeros(1),
    }
    ckpt_path = tmp_path / "lora_weights.ckpt"
    torch.save(state, ckpt_path)

    loaded, skipped = load_lora_weights(model, str(ckpt_path), device="cpu")
    assert set(loaded) >= {
        "q_proj.lora_A",
        "k_proj.lora_A",
        "v_proj.lora_A",
        "q_proj.lora_B",
        "k_proj.lora_B",
        "v_proj.lora_B",
    }
    assert "not_lora.weight" in skipped

    assert torch.allclose(model.qkv_proj.lora_B_q, torch.full((2, 2), 4.0))
    assert torch.allclose(model.qkv_proj.lora_B_k, torch.full((2, 2), 5.0))
    assert torch.allclose(model.qkv_proj.lora_B_v, torch.full((2, 2), 6.0))

    # Fused lora_A loads go through the module hook.
    assert set(model.qkv_proj.loaded_A.keys()) == {"q", "k", "v"}
    assert torch.allclose(model.qkv_proj.loaded_A["q"], torch.full((2, 4), 1.0))
    assert torch.allclose(model.qkv_proj.loaded_A["k"], torch.full((2, 4), 2.0))
    assert torch.allclose(model.qkv_proj.loaded_A["v"], torch.full((2, 4), 3.0))
