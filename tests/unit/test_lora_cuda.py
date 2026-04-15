import pytest

torch = pytest.importorskip("torch")


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


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


@pytest.fixture(autouse=True)
def _reset_lora_context():
    from nanovllm_voxcpm.utils.context import reset_lora_context

    reset_lora_context()
    yield
    reset_lora_context()


def test_lora_linear_cuda_modes_and_rank_alpha():
    from nanovllm_voxcpm.layers.lora import LoRALinear
    from nanovllm_voxcpm.utils.context import set_lora_context

    layer = LoRALinear(in_features=3, out_features=2, bias=False, max_loras=2, max_lora_rank=4).cuda()
    with torch.no_grad():
        layer.weight.zero_()
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda"),
            lora_b=torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda"),
            effective_rank=2,
            scaling=1.0,
        )
        layer.set_slot_lora(
            slot_id=1,
            lora_a=torch.tensor([[0.0, 0.0, 1.0]], device="cuda"),
            lora_b=torch.tensor([[2.0], [4.0]], device="cuda"),
            effective_rank=1,
            scaling=0.25,
        )

    x = torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0], [17.0, 19.0, 23.0]], device="cuda")
    y_no_lora = layer(x).cpu()

    set_lora_context(
        token_to_slot=torch.tensor([0, 0, 0], dtype=torch.int32, device="cuda"),
        token_indices_sorted_by_slot=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        active_slot_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
        num_tokens_per_slot=torch.tensor([3], dtype=torch.int32, device="cuda"),
        slot_start_offsets=torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
        no_lora_flag=False,
        scratch_buffer=torch.zeros(3, 4, device="cuda"),
    )
    y_single = layer(x).cpu()

    set_lora_context(
        token_to_slot=torch.tensor([0, -1, 1], dtype=torch.int32, device="cuda"),
        token_indices_sorted_by_slot=torch.tensor([0, 2, 1], dtype=torch.int32, device="cuda"),
        active_slot_ids=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        num_tokens_per_slot=torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
        slot_start_offsets=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        no_lora_flag=False,
        scratch_buffer=torch.zeros(3, 4, device="cuda"),
    )
    y_mixed = layer(x).cpu()

    assert torch.allclose(y_no_lora, torch.zeros_like(y_no_lora))
    assert torch.allclose(y_single[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(y_single[1], torch.tensor([7.0, 11.0]))
    assert torch.allclose(y_mixed[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(y_mixed[1], torch.tensor([0.0, 0.0]))
    assert torch.allclose(y_mixed[2], torch.tensor([11.5, 23.0]))


def test_lora_linear_cuda_graph_replay():
    from nanovllm_voxcpm.layers.lora import LoRALinear
    from nanovllm_voxcpm.lora import _VendoredTritonPunicaBackend, set_backend_for_testing
    from nanovllm_voxcpm.utils.context import set_lora_context

    set_backend_for_testing(_VendoredTritonPunicaBackend())
    layer = LoRALinear(in_features=2, out_features=1, bias=False, max_loras=2, max_lora_rank=1).cuda()
    with torch.no_grad():
        layer.weight.zero_()
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.tensor([[1.0, 0.0]], device="cuda"),
            lora_b=torch.tensor([[2.0]], device="cuda"),
            effective_rank=1,
            scaling=1.0,
        )
        layer.set_slot_lora(
            slot_id=1,
            lora_a=torch.tensor([[0.0, 1.0]], device="cuda"),
            lora_b=torch.tensor([[3.0]], device="cuda"),
            effective_rank=1,
            scaling=1.0,
        )

    x_buffer = torch.zeros(2, 2, device="cuda")
    token_to_slot = torch.zeros(2, dtype=torch.int32, device="cuda")
    sorted_indices = torch.arange(2, dtype=torch.int32, device="cuda")
    active_slot_ids = torch.zeros(2, dtype=torch.int32, device="cuda")
    num_tokens_per_slot = torch.zeros(2, dtype=torch.int32, device="cuda")
    slot_start_offsets = torch.zeros(3, dtype=torch.int32, device="cuda")
    scratch_buffer = torch.zeros(2, 1, device="cuda")
    out_buffer = torch.zeros(2, 1, device="cuda")

    token_to_slot.copy_(torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    active_slot_ids[:2].copy_(torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    num_tokens_per_slot[:2].copy_(torch.tensor([1, 1], dtype=torch.int32, device="cuda"))
    slot_start_offsets.copy_(torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"))
    set_lora_context(
        token_to_slot=token_to_slot,
        token_indices_sorted_by_slot=sorted_indices,
        active_slot_ids=active_slot_ids[:2],
        num_tokens_per_slot=num_tokens_per_slot[:2],
        slot_start_offsets=slot_start_offsets,
        no_lora_flag=False,
        scratch_buffer=scratch_buffer,
        no_lora_flag_cpu=torch.tensor([False], dtype=torch.bool),
        num_active_loras_cpu=torch.tensor([2], dtype=torch.int32),
    )

    graph = torch.cuda.CUDAGraph()
    x_buffer.copy_(torch.tensor([[5.0, 7.0], [11.0, 13.0]], device="cuda"))
    out_buffer.copy_(layer(x_buffer))
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out_buffer.copy_(layer(x_buffer))

    x_buffer.copy_(torch.tensor([[2.0, 3.0], [17.0, 19.0]], device="cuda"))
    graph.replay()
    assert torch.allclose(out_buffer.cpu().flatten(), torch.tensor([4.0, 57.0]))
    set_backend_for_testing(_FakePunicaBackend())


def test_lora_linear_triton_lora_b_pointer_cache_stays_bounded():
    from nanovllm_voxcpm.layers.lora import LoRALinear
    from nanovllm_voxcpm.lora import _VendoredTritonPunicaBackend, set_backend_for_testing
    from nanovllm_voxcpm.lora_ops.triton_ops import utils as lora_utils
    from nanovllm_voxcpm.utils.context import set_lora_context

    set_backend_for_testing(_VendoredTritonPunicaBackend())
    lora_utils._LORA_A_PTR_DICT.clear()
    lora_utils._LORA_B_PTR_DICT.clear()
    layer = LoRALinear(in_features=2, out_features=1, bias=False, max_loras=1, max_lora_rank=1).cuda().half()
    with torch.no_grad():
        layer.weight.zero_()
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.tensor([[1.0, 0.0]], device="cuda", dtype=torch.float16),
            lora_b=torch.tensor([[2.0]], device="cuda", dtype=torch.float16),
            effective_rank=1,
            scaling=0.5,
        )

    x = torch.tensor([[4.0, 8.0]], device="cuda", dtype=torch.float16)
    set_lora_context(
        token_to_slot=torch.tensor([0], dtype=torch.int32, device="cuda"),
        token_indices_sorted_by_slot=torch.tensor([0], dtype=torch.int32, device="cuda"),
        active_slot_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
        num_tokens_per_slot=torch.tensor([1], dtype=torch.int32, device="cuda"),
        slot_start_offsets=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        no_lora_flag=False,
        scratch_buffer=torch.zeros(1, 1, device="cuda", dtype=torch.float16),
        no_lora_flag_cpu=torch.tensor([False], dtype=torch.bool),
        num_active_loras_cpu=torch.tensor([1], dtype=torch.int32),
    )

    for _ in range(10):
        out = layer(x)

    assert torch.allclose(out.cpu().flatten(), torch.tensor([4.0], dtype=torch.float16))
    assert len(lora_utils._LORA_B_PTR_DICT) == 1
    set_backend_for_testing(_FakePunicaBackend())


def test_lora_merged_column_set_slot_applies_scaling_once():
    from nanovllm_voxcpm.layers.lora import LoRAMergedColumnParallelLinear
    from nanovllm_voxcpm.utils.context import set_lora_context

    layer = (
        LoRAMergedColumnParallelLinear(
            input_size=2,
            output_sizes=[1, 1],
            bias=False,
            lora_targets=[0],
            max_loras=1,
            max_lora_rank=1,
        )
        .cuda()
        .half()
    )
    with torch.no_grad():
        layer.weight.zero_()
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.tensor([[[1.0, 0.0]]], device="cuda", dtype=torch.float16),
            lora_b=[torch.tensor([[2.0]], device="cuda", dtype=torch.float16)],
            effective_rank=1,
            scaling=0.5,
        )

    x = torch.tensor([[4.0, 8.0]], device="cuda", dtype=torch.float16)
    set_lora_context(
        token_to_slot=torch.tensor([0], dtype=torch.int32, device="cuda"),
        token_indices_sorted_by_slot=torch.tensor([0], dtype=torch.int32, device="cuda"),
        active_slot_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
        num_tokens_per_slot=torch.tensor([1], dtype=torch.int32, device="cuda"),
        slot_start_offsets=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        no_lora_flag=False,
        scratch_buffer=torch.zeros(1, 1, device="cuda", dtype=torch.float16),
    )

    out = layer(x).cpu()
    assert torch.allclose(out, torch.tensor([[4.0, 0.0]], dtype=torch.float16))


def test_lora_qkv_cuda_graph_replay_after_runtime_slot_update():
    from nanovllm_voxcpm.lora import _VendoredTritonPunicaBackend, set_backend_for_testing
    from nanovllm_voxcpm.utils.context import set_lora_context

    set_backend_for_testing(_VendoredTritonPunicaBackend())
    layer = _make_tp2_qkv_layer(0)
    with torch.no_grad():
        layer.lora_A.zero_()
        layer.lora_B_q.zero_()
        layer.lora_B_k.zero_()
        layer.lora_B_v.zero_()
        layer.effective_lora_rank.zero_()

    x_buffer = torch.zeros(1, 2, device="cuda")
    out_buffer = torch.zeros(1, 3, device="cuda")
    set_lora_context(
        token_to_slot=torch.tensor([0], dtype=torch.int32, device="cuda"),
        token_indices_sorted_by_slot=torch.tensor([0], dtype=torch.int32, device="cuda"),
        active_slot_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
        num_tokens_per_slot=torch.tensor([1], dtype=torch.int32, device="cuda"),
        slot_start_offsets=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        no_lora_flag=False,
        scratch_buffer=torch.zeros(1, 1, device="cuda"),
        no_lora_flag_cpu=torch.tensor([False], dtype=torch.bool),
        num_active_loras_cpu=torch.tensor([1], dtype=torch.int32),
    )

    graph = torch.cuda.CUDAGraph()
    x_buffer.copy_(torch.tensor([[5.0, 7.0]], device="cuda"))
    out_buffer.copy_(layer(x_buffer))
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out_buffer.copy_(layer(x_buffer))

    with torch.no_grad():
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]], device="cuda"),
            lora_b=[
                torch.tensor([[1.0]], device="cuda"),
                torch.tensor([[2.0]], device="cuda"),
                torch.tensor([[3.0]], device="cuda"),
            ],
            effective_rank=1,
            scaling=1.0,
        )

    graph.replay()
    assert torch.allclose(out_buffer.cpu(), torch.tensor([[10.0, 22.0, 25.0]]))
    set_backend_for_testing(_FakePunicaBackend())


def _make_tp2_qkv_layer(rank: int):
    from nanovllm_voxcpm.layers.lora import LoRAQKVParallelLinear
    import nanovllm_voxcpm.layers.lora as lora_layers

    lora_layers.dist.get_world_size = lambda: 2
    lora_layers.dist.get_rank = lambda: rank
    layer = LoRAQKVParallelLinear(
        hidden_size=2,
        head_size=1,
        total_num_heads=2,
        total_num_kv_heads=2,
        bias=False,
        max_loras=2,
        max_lora_rank=1,
    ).cuda()
    with torch.no_grad():
        layer._base_weight_loader(layer.weight, torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda"), "q")
        layer._base_weight_loader(layer.weight, torch.tensor([[1.0, 1.0], [0.0, 1.0]], device="cuda"), "k")
        layer._base_weight_loader(layer.weight, torch.tensor([[2.0, 0.0], [0.0, 2.0]], device="cuda"), "v")
        layer.set_slot_lora(
            slot_id=0,
            lora_a=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]], device="cuda"),
            lora_b=[
                torch.tensor([[1.0]], device="cuda"),
                torch.tensor([[2.0]], device="cuda"),
                torch.tensor([[3.0]], device="cuda"),
            ],
            effective_rank=1,
            scaling=1.0,
        )
        layer.set_slot_lora(
            slot_id=1,
            lora_a=torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]]], device="cuda"),
            lora_b=[
                torch.tensor([[4.0]], device="cuda"),
                torch.tensor([[5.0]], device="cuda"),
                torch.tensor([[6.0]], device="cuda"),
            ],
            effective_rank=1,
            scaling=1.0,
        )
    return layer


@pytest.mark.parametrize(
    ("token_to_slot", "x_rows", "expected_rank0", "expected_rank1"),
    [
        (None, [[2.0, 3.0]], [[2.0, 5.0, 4.0]], [[3.0, 3.0, 6.0]]),
        ([0], [[2.0, 3.0]], [[4.0, 9.0, 10.0]], [[5.0, 7.0, 12.0]]),
        (
            [0, 1],
            [[2.0, 3.0], [3.0, 4.0]],
            [[4.0, 9.0, 10.0], [19.0, 27.0, 30.0]],
            [[5.0, 7.0, 12.0], [20.0, 24.0, 32.0]],
        ),
    ],
)
def test_lora_qkv_parallel_cuda_tp2_modes(token_to_slot, x_rows, expected_rank0, expected_rank1):
    from nanovllm_voxcpm.utils.context import set_lora_context

    layer0 = _make_tp2_qkv_layer(0)
    layer1 = _make_tp2_qkv_layer(1)
    x = torch.tensor(x_rows, device="cuda")
    if token_to_slot is None:
        set_lora_context()
        y0 = layer0(x).cpu()
        y1 = layer1(x).cpu()
    else:
        token_to_slot_tensor = torch.tensor(token_to_slot, dtype=torch.int32, device="cuda")
        active_slot_ids = torch.unique(token_to_slot_tensor[token_to_slot_tensor >= 0]).to(
            device="cuda", dtype=torch.int32
        )
        slot_counts = torch.tensor(
            [(token_to_slot_tensor == slot_id).sum().item() for slot_id in active_slot_ids.tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        slot_offsets = torch.zeros(active_slot_ids.numel() + 1, dtype=torch.int32, device="cuda")
        if slot_counts.numel() > 0:
            slot_offsets[1:] = torch.cumsum(slot_counts, dim=0)
        set_lora_context(
            token_to_slot=token_to_slot_tensor,
            token_indices_sorted_by_slot=torch.arange(len(token_to_slot), dtype=torch.int32, device="cuda"),
            active_slot_ids=active_slot_ids,
            num_tokens_per_slot=slot_counts,
            slot_start_offsets=slot_offsets,
            no_lora_flag=False,
            scratch_buffer=torch.zeros(x.size(0), 1, device="cuda"),
        )
        y0 = layer0(x).cpu()
        y1 = layer1(x).cpu()

    assert torch.allclose(y0, torch.tensor(expected_rank0))
    assert torch.allclose(y1, torch.tensor(expected_rank1))


def test_lora_qkv_parallel_cuda_tp2_graph_replay():
    from nanovllm_voxcpm.lora import _VendoredTritonPunicaBackend, set_backend_for_testing
    from nanovllm_voxcpm.utils.context import set_lora_context

    set_backend_for_testing(_VendoredTritonPunicaBackend())
    layer = _make_tp2_qkv_layer(0)
    x_buffer = torch.zeros(2, 2, device="cuda")
    token_to_slot = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    sorted_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    active_slot_ids = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    num_tokens_per_slot = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    slot_start_offsets = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    scratch_buffer = torch.zeros(2, 1, device="cuda")
    out_buffer = torch.zeros(2, 3, device="cuda")
    set_lora_context(
        token_to_slot=token_to_slot,
        token_indices_sorted_by_slot=sorted_indices,
        active_slot_ids=active_slot_ids,
        num_tokens_per_slot=num_tokens_per_slot,
        slot_start_offsets=slot_start_offsets,
        no_lora_flag=False,
        scratch_buffer=scratch_buffer,
        no_lora_flag_cpu=torch.tensor([False], dtype=torch.bool),
        num_active_loras_cpu=torch.tensor([2], dtype=torch.int32),
    )
    graph = torch.cuda.CUDAGraph()
    x_buffer.copy_(torch.tensor([[2.0, 3.0], [3.0, 4.0]], device="cuda"))
    out_buffer.copy_(layer(x_buffer))
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out_buffer.copy_(layer(x_buffer))

    x_buffer.copy_(torch.tensor([[5.0, 7.0], [11.0, 13.0]], device="cuda"))
    graph.replay()
    assert torch.allclose(out_buffer.cpu(), torch.tensor([[10.0, 22.0, 25.0], [63.0, 89.0, 100.0]]))
    set_backend_for_testing(_FakePunicaBackend())


def test_vendored_triton_backend_add_lora_cuda():
    from nanovllm_voxcpm.lora import LoRAMetadata, _VendoredTritonPunicaBackend, set_backend_for_testing

    set_backend_for_testing(None)
    backend = _VendoredTritonPunicaBackend()
    y = torch.zeros(2, 1, device="cuda", dtype=torch.float16)
    x = torch.tensor([[2.0, 3.0], [5.0, 7.0]], device="cuda", dtype=torch.float16)
    lora_a = torch.tensor([[[1.0, 0.0]]], device="cuda", dtype=torch.float16)
    lora_b = torch.tensor([[[4.0]]], device="cuda", dtype=torch.float16)
    metadata = LoRAMetadata(
        token_to_slot=torch.zeros(2, dtype=torch.int32, device="cuda"),
        token_indices_sorted_by_slot=torch.arange(2, dtype=torch.int32, device="cuda"),
        active_slot_ids=torch.tensor([0], dtype=torch.int32, device="cuda"),
        num_tokens_per_slot=torch.tensor([2], dtype=torch.int32, device="cuda"),
        slot_start_offsets=torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
        no_lora_flag=False,
        no_lora_flag_cpu=torch.tensor([False], dtype=torch.bool),
        num_active_loras_cpu=torch.tensor([1], dtype=torch.int32),
    )

    out = backend.add_lora(
        y,
        x,
        lora_a,
        lora_b,
        indices=torch.zeros(2, dtype=torch.long, device="cuda"),
        metadata=metadata,
        scaling=0.5,
    )

    assert torch.allclose(out.cpu().flatten(), torch.tensor([4.0, 10.0], dtype=torch.float16))
