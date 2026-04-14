import pytest

torch = pytest.importorskip("torch")


class _FakePunicaBackend:
    def availability(self):
        from nanovllm_voxcpm.lora import LoRAAvailability

        return LoRAAvailability(available=True, reason=None)

    def shrink(self, x, lora_a, *, scratch_buffer=None):
        return torch.nn.functional.linear(x, lora_a)

    def expand(self, hidden, lora_b, *, scaling):
        return torch.nn.functional.linear(hidden, lora_b) * scaling

    def add_lora(self, y, x, lora_a, lora_b, *, indices, metadata, scaling):
        out = y.clone()
        for token_idx in range(x.size(0)):
            slot_id = int(indices[token_idx].item())
            if slot_id < 0:
                continue
            hidden = self.shrink(x[token_idx : token_idx + 1], lora_a[slot_id])
            out[token_idx : token_idx + 1] = out[token_idx : token_idx + 1] + self.expand(
                hidden,
                lora_b[slot_id],
                scaling=scaling,
            )
        return out


@pytest.fixture(autouse=True)
def _install_fake_punica_backend():
    from nanovllm_voxcpm.lora import set_backend_for_testing

    set_backend_for_testing(_FakePunicaBackend())
    yield
    set_backend_for_testing(None)


def test_lora_linear_context_controls_activation_and_reset():
    from nanovllm_voxcpm.layers.lora import LoRALinear
    from nanovllm_voxcpm.utils.context import reset_lora_context, set_lora_context

    layer = LoRALinear(in_features=4, out_features=3, bias=False, lora_r=2, lora_alpha=2.0)
    # Deterministic weights.
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.lora_A.fill_(1.0)
        layer.lora_B.fill_(1.0)

    x = torch.ones(2, 4)
    set_lora_context(
        token_to_slot=torch.zeros(2, dtype=torch.int32),
        token_indices_sorted_by_slot=torch.arange(2, dtype=torch.int32),
        active_slot_ids=torch.tensor([0], dtype=torch.int32),
        num_tokens_per_slot=torch.tensor([2], dtype=torch.int32),
        slot_start_offsets=torch.tensor([0, 2], dtype=torch.int32),
        no_lora_flag=False,
        no_lora_flag_cpu=torch.tensor([False], dtype=torch.bool),
        num_active_loras_cpu=torch.tensor([1], dtype=torch.int32),
    )
    y_enabled = layer(x)
    assert layer.lora_enabled is True

    reset_lora_context()
    y_disabled = layer(x)

    # With LoRA disabled, output should be base linear only.
    # base: sum(x)=4 for each output.
    assert y_disabled.tolist() == [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]
    # Enabled output differs (LoRA adds a positive term).
    assert not torch.allclose(y_enabled, y_disabled)
    reset_lora_context()

    set_lora_context(
        token_to_slot=torch.zeros(2, dtype=torch.int32),
        token_indices_sorted_by_slot=torch.arange(2, dtype=torch.int32),
        active_slot_ids=torch.tensor([0], dtype=torch.int32),
        num_tokens_per_slot=torch.tensor([2], dtype=torch.int32),
        slot_start_offsets=torch.tensor([0, 2], dtype=torch.int32),
        no_lora_flag=False,
        no_lora_flag_cpu=torch.tensor([False], dtype=torch.bool),
        num_active_loras_cpu=torch.tensor([1], dtype=torch.int32),
    )
    layer.reset_lora_parameters()
    y_after_reset = layer(x)
    assert torch.allclose(y_after_reset, y_disabled)
    reset_lora_context()


def test_iter_lora_modules():
    from nanovllm_voxcpm.layers.lora import (
        LoRALinear,
        iter_lora_modules,
    )

    model = torch.nn.Sequential(
        LoRALinear(4, 4, lora_r=2),
        torch.nn.ReLU(),
        LoRALinear(4, 4, lora_r=0),
    )
    lora_modules = list(iter_lora_modules(model))
    assert len(lora_modules) == 1
    assert lora_modules[0].lora_enabled is True


def test_lora_linear_mixed_slots_with_runtime_context():
    from nanovllm_voxcpm.layers.lora import LoRALinear
    from nanovllm_voxcpm.utils.context import reset_lora_context, set_lora_context

    layer = LoRALinear(in_features=3, out_features=2, bias=False, lora_r=2, max_loras=2, max_lora_rank=3)
    with torch.no_grad():
        layer.weight.zero_()
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            lora_b=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            effective_rank=2,
            scaling=1.0,
        )
        layer.set_slot_lora(
            slot_id=1,
            lora_a=torch.tensor([[0.0, 0.0, 1.0]]),
            lora_b=torch.tensor([[2.0], [3.0]]),
            effective_rank=1,
            scaling=0.5,
        )

    x = torch.tensor(
        [
            [2.0, 3.0, 5.0],
            [7.0, 11.0, 13.0],
            [17.0, 19.0, 23.0],
        ]
    )
    set_lora_context(
        token_to_slot=torch.tensor([0, -1, 1], dtype=torch.int32),
        token_indices_sorted_by_slot=torch.tensor([0, 2, 1], dtype=torch.int32),
        active_slot_ids=torch.tensor([0, 1], dtype=torch.int32),
        num_tokens_per_slot=torch.tensor([1, 1], dtype=torch.int32),
        slot_start_offsets=torch.tensor([0, 1, 2], dtype=torch.int32),
        no_lora_flag=False,
    )

    y = layer(x)
    reset_lora_context()

    assert torch.allclose(y[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(y[1], torch.tensor([0.0, 0.0]))
    assert torch.allclose(y[2], torch.tensor([23.0, 34.5]))


def test_lora_availability_reports_missing_backend():
    from nanovllm_voxcpm import lora

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        lora,
        "_probe_vendored_backend",
        lambda: type("_B", (), {"availability": lambda self: lora.LoRAAvailability(False, "vendored unavailable")})(),
    )
    lora.set_backend_for_testing(None)
    lora._PROBED_BACKEND = None
    assert lora.is_available() is False
    patcher.undo()


def test_vendored_metadata_skips_no_lora_tokens():
    from nanovllm_voxcpm.lora import _VendoredTritonPunicaBackend

    backend = _VendoredTritonPunicaBackend()
    (
        token_lora_mapping,
        token_indices_sorted,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag,
        num_active,
    ) = backend._make_metadata(3, torch.device("cpu"), torch.tensor([0, -1, 1], dtype=torch.int32))

    assert no_lora_flag.item() is False
    assert num_active.item() == 3
    assert torch.equal(token_indices_sorted, torch.tensor([1, 0, 2], dtype=torch.int32))
    assert torch.equal(num_tokens_per_lora[:3], torch.tensor([1, 1, 1], dtype=torch.int32))
    assert torch.equal(lora_token_start_loc[:4], torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    assert torch.equal(lora_ids[:3], torch.tensor([-1, 0, 1], dtype=torch.int32))


def test_no_lora_context_disables_all_slots():
    from nanovllm_voxcpm.layers.lora import LoRALinear
    from nanovllm_voxcpm.utils.context import reset_lora_context, set_lora_context

    layer = LoRALinear(in_features=2, out_features=1, bias=False, lora_r=1, max_loras=2, max_lora_rank=1)
    with torch.no_grad():
        layer.weight.zero_()
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.tensor([[1.0, 0.0]]),
            lora_b=torch.tensor([[2.0]]),
            effective_rank=1,
            scaling=1.0,
        )
        layer.set_slot_lora(
            slot_id=1,
            lora_a=torch.tensor([[0.0, 1.0]]),
            lora_b=torch.tensor([[3.0]]),
            effective_rank=1,
            scaling=1.0,
        )

    set_lora_context(
        token_to_slot=torch.tensor([0, 1], dtype=torch.int32),
        active_slot_ids=torch.tensor([0, 1], dtype=torch.int32),
        no_lora_flag=False,
    )
    y_enabled = layer(torch.tensor([[5.0, 7.0], [11.0, 13.0]])).flatten()

    reset_lora_context()
    y_disabled = layer(torch.tensor([[5.0, 7.0], [11.0, 13.0]])).flatten()

    assert torch.allclose(y_enabled, torch.tensor([10.0, 39.0]))
    assert torch.allclose(y_disabled, torch.zeros(2))


def test_set_slot_lora_validates_max_rank():
    from nanovllm_voxcpm.layers.lora import LoRALinear

    layer = LoRALinear(in_features=2, out_features=1, bias=False, lora_r=1, max_loras=1, max_lora_rank=1)
    with pytest.raises(ValueError):
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            lora_b=torch.tensor([[1.0, 1.0]]),
            effective_rank=2,
            scaling=1.0,
        )
