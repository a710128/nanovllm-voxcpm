import pytest


torch = pytest.importorskip("torch")


def test_runner_task_properties():
    from nanovllm_voxcpm.engine.model_runner import RunnerTask

    t = RunnerTask(block_table=[0, 1, 2], seq_length=600, num_cached_tokens=512, block_size=256)
    assert t.num_blocks == 3
    assert t.num_cached_blocks == 2
    assert t.last_block_num_tokens == 88


def test_cut_inputs_and_assign_outputs():
    from nanovllm_voxcpm.engine.model_runner import cut_inputs, assign_outputs

    inputs = {
        "a": torch.arange(10),
        "b": torch.arange(10) + 100,
    }
    cut = cut_inputs(inputs, 3)
    assert cut["a"].tolist() == [0, 1, 2]
    assert cut["b"].tolist() == [100, 101, 102]

    outputs = {"a": torch.empty(10, dtype=torch.long)}
    assign_outputs({"a": torch.tensor([5, 6, 7])}, outputs, 3)
    assert outputs["a"][:3].tolist() == [5, 6, 7]

    with pytest.raises(KeyError):
        assign_outputs({"missing": torch.tensor([1])}, {"a": torch.empty(1)}, 1)
