import pytest


torch = pytest.importorskip("torch")


def test_lora_linear_enable_disable_and_reset():
    from nanovllm_voxcpm.layers.lora import LoRALinear

    layer = LoRALinear(in_features=4, out_features=3, bias=False, lora_r=2, lora_alpha=2.0)
    # Deterministic weights.
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.lora_A.fill_(1.0)
        layer.lora_B.fill_(1.0)

    x = torch.ones(2, 4)
    y_enabled = layer(x)
    assert layer.lora_enabled is True

    layer.set_lora_enabled(False)
    assert layer.lora_enabled is False
    y_disabled = layer(x)

    # With LoRA disabled, output should be base linear only.
    # base: sum(x)=4 for each output.
    assert y_disabled.tolist() == [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]
    # Enabled output differs (LoRA adds a positive term).
    assert not torch.allclose(y_enabled, y_disabled)

    layer.reset_lora_parameters()
    layer.set_lora_enabled(True)
    y_after_reset = layer(x)
    assert torch.allclose(y_after_reset, y_disabled)


def test_iter_and_toggle_all_lora_modules():
    from nanovllm_voxcpm.layers.lora import (
        LoRALinear,
        iter_lora_modules,
        set_all_lora_enabled,
    )

    model = torch.nn.Sequential(
        LoRALinear(4, 4, lora_r=2),
        torch.nn.ReLU(),
        LoRALinear(4, 4, lora_r=0),
    )
    lora_modules = list(iter_lora_modules(model))
    assert len(lora_modules) == 1
    assert lora_modules[0].lora_enabled is True

    set_all_lora_enabled(model, False)
    assert lora_modules[0].lora_enabled is False
