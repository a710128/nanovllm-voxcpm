import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm_voxcpm.utils.context import get_context
from nanovllm_voxcpm.utils.distributed import get_tp_rank, get_tp_world_size
from nanovllm_voxcpm.utils.torch_param import set_weight_loader


class VocabParallelEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        set_weight_loader(self.weight, self.weight_loader)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = torch.where(mask, x - self.vocab_start_idx, 0)
            y = F.embedding(x, self.weight)
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
            return y

        return F.embedding(x, self.weight)


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            assert context.cu_seqlens_q is not None
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, -1)

        return logits
