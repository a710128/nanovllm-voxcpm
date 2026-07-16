from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist

from nanovllm_voxcpm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm_voxcpm.utils.context import reset_context, set_context

pytestmark = pytest.mark.gpu

_NUM_EMBEDDINGS = 8
_EMBEDDING_DIM = 4


def _preflight_torchrun() -> None:
    if "WORLD_SIZE" not in os.environ:
        pytest.skip("TP GPU smoke requires torchrun --nproc-per-node=2", allow_module_level=True)
    if torch.cuda.device_count() != 2:
        pytest.exit("GPU smoke requires 2 visible CUDA devices", returncode=1)


_preflight_torchrun()


def _assert_rank_agreement(tensor: torch.Tensor) -> None:
    gathered_tensors = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensors, tensor)
    for gathered_tensor in gathered_tensors:
        torch.testing.assert_close(gathered_tensor, tensor)


@pytest.fixture(scope="module", autouse=True)
def initialized_nccl_process_group() -> Iterator[None]:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(backend="nccl")
        assert dist.get_world_size() == 2
        assert dist.get_rank() == local_rank
        yield
    finally:
        if dist.is_initialized():
            try:
                torch.cuda.synchronize()
            finally:
                dist.destroy_process_group()
            print(f"TP_SMOKE_OK rank={local_rank}", flush=True)


def test_tp_embedding_and_head_collectives_agree_on_cuda() -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()
    global_weight = torch.arange(_NUM_EMBEDDINGS * _EMBEDDING_DIM, device=device, dtype=torch.float32).reshape(
        _NUM_EMBEDDINGS, _EMBEDDING_DIM
    )
    local_weight = global_weight.chunk(dist.get_world_size(), dim=0)[rank]

    embedding = VocabParallelEmbedding(_NUM_EMBEDDINGS, _EMBEDDING_DIM).to(device)
    with torch.no_grad():
        embedding.weight.copy_(local_weight)
    token_ids = torch.tensor([0, 3, 4, 7], device=device, dtype=torch.long)
    embedding_output = embedding(token_ids)
    expected_embedding = global_weight[token_ids]
    torch.cuda.synchronize()
    torch.testing.assert_close(embedding_output, expected_embedding)
    _assert_rank_agreement(embedding_output)

    set_context(is_prefill=False)
    try:
        head = ParallelLMHead(_NUM_EMBEDDINGS, _EMBEDDING_DIM).to(device)
        with torch.no_grad():
            head.weight.copy_(local_weight)
        hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]], device=device)
        logits = head(hidden_states)
        expected_logits = hidden_states @ global_weight.transpose(0, 1)
        torch.cuda.synchronize()
        torch.testing.assert_close(logits, expected_logits)
        _assert_rank_agreement(logits)
    finally:
        reset_context()
