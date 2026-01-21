"""
LoRA layers for nanovllm-voxcpm with optimized fused input projection.

Supports:
- Hot loading/swapping LoRA weights
- Enable/disable without recompiling CUDA Graph
- Tensor parallel compatible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LoRAQKVParallelLinear(nn.Module):
    """
    QKV fused layer + optimized LoRA implementation

    - Base QKV keeps fused computation (1 large matrix multiplication)
    - LoRA input projection (lora_A) fused into single matrix (1 small matrix multiplication)
    - LoRA output projection (lora_B) separated to support different dimensions (3 small matrix multiplications)

    Total: 5 matrix multiplications (vs 1 without LoRA)
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        bias: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
        lora_targets: list = None,  # ["q", "k", "v"]
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # Base parameters
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = divide(total_num_heads, self.tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, self.tp_size)
        self.q_size = self.num_heads * head_size
        self.kv_size = self.num_kv_heads * head_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size

        # Base weights
        self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.weight.weight_loader = self._base_weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self._base_weight_loader
        else:
            self.register_parameter("bias", None)

        # LoRA parameters
        self.lora_r = lora_r
        self.lora_targets = lora_targets or ["q", "k", "v"]
        self._base_lora_alpha = lora_alpha

        if lora_r > 0 and len(self.lora_targets) > 0:
            n_targets = len(self.lora_targets)

            # Fused input projection: all targets share one lora_A
            self.lora_A = nn.Parameter(torch.zeros(lora_r * n_targets, hidden_size))

            # Separated output projections (need to be sharded by TP)
            if "q" in self.lora_targets:
                self.lora_B_q = nn.Parameter(torch.zeros(self.q_size, lora_r))
                self.lora_B_q.weight_loader = self._lora_B_weight_loader
            if "k" in self.lora_targets:
                self.lora_B_k = nn.Parameter(torch.zeros(self.kv_size, lora_r))
                self.lora_B_k.weight_loader = self._lora_B_weight_loader
            if "v" in self.lora_targets:
                self.lora_B_v = nn.Parameter(torch.zeros(self.kv_size, lora_r))
                self.lora_B_v.weight_loader = self._lora_B_weight_loader

            # Use buffer to store scaling (CUDA Graph compatible)
            self._base_scaling = lora_alpha / lora_r
            self.register_buffer("lora_scaling", torch.tensor(self._base_scaling), persistent=False)
        else:
            self.lora_r = 0

    def _base_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str = None,
    ):
        """Base weight loader (supports Q/K/V sharding)"""
        if loaded_shard_id is None:
            param.data.copy_(loaded_weight)
            return

        param_data = param.data
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # v
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size

        param_data = param_data.narrow(0, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def _lora_B_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """LoRA B weight loader (sharded by output dimension)"""
        loaded_weight = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        param.data.copy_(loaded_weight)

    def load_lora_A(self, loaded_weight: torch.Tensor, target: str):
        """Load single target's lora_A to the corresponding position in fused matrix"""
        if target not in self.lora_targets:
            return
        target_idx = self.lora_targets.index(target)
        start = target_idx * self.lora_r
        self.lora_A.data[start : start + self.lora_r].copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base QKV computation
        qkv = F.linear(x, self.weight, self.bias)

        if self.lora_r <= 0:
            return qkv

        # Fused LoRA input projection
        lora_hidden = F.linear(x, self.lora_A)  # [batch, seq, r * n_targets]

        # Separate outputs and compute respective lora_B
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        idx = 0
        if "q" in self.lora_targets:
            lora_q_h = lora_hidden[..., idx * self.lora_r : (idx + 1) * self.lora_r]
            q = q + F.linear(lora_q_h, self.lora_B_q) * self.lora_scaling
            idx += 1
        if "k" in self.lora_targets:
            lora_k_h = lora_hidden[..., idx * self.lora_r : (idx + 1) * self.lora_r]
            k = k + F.linear(lora_k_h, self.lora_B_k) * self.lora_scaling
            idx += 1
        if "v" in self.lora_targets:
            lora_v_h = lora_hidden[..., idx * self.lora_r : (idx + 1) * self.lora_r]
            v = v + F.linear(lora_v_h, self.lora_B_v) * self.lora_scaling

        return torch.cat([q, k, v], dim=-1)

    def set_lora_enabled(self, enabled: bool):
        """Enable/disable LoRA (controlled via scaling, CUDA Graph compatible)"""
        if self.lora_r > 0:
            self.lora_scaling.fill_(self._base_scaling if enabled else 0.0)

    @property
    def lora_enabled(self) -> bool:
        return self.lora_r > 0 and self.lora_scaling.item() != 0.0

    def reset_lora_parameters(self):
        """Reset LoRA parameters (set lora_B to zero to disable LoRA output)"""
        if self.lora_r > 0:
            if hasattr(self, "lora_B_q"):
                self.lora_B_q.data.zero_()
            if hasattr(self, "lora_B_k"):
                self.lora_B_k.data.zero_()
            if hasattr(self, "lora_B_v"):
                self.lora_B_v.data.zero_()


class LoRAMergedColumnParallelLinear(nn.Module):
    """
    MergedColumnParallelLinear (gate_up_proj) + optimized LoRA

    Used for fused gate_proj and up_proj layers in MLP
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list,
        bias: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
        lora_targets: list = None,  # [0, 1] for gate and up
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        self.output_sizes = output_sizes
        self.input_size = input_size
        total_output = sum(output_sizes)
        self.shard_output_sizes = [s // self.tp_size for s in output_sizes]
        shard_total_output = total_output // self.tp_size

        # Base weights
        self.weight = nn.Parameter(torch.empty(shard_total_output, input_size))
        self.weight.weight_loader = self._base_weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(shard_total_output))
            self.bias.weight_loader = self._base_weight_loader
        else:
            self.register_parameter("bias", None)

        # LoRA parameters
        self.lora_r = lora_r
        self.lora_targets = lora_targets if lora_targets is not None else list(range(len(output_sizes)))
        self._base_lora_alpha = lora_alpha

        if lora_r > 0 and len(self.lora_targets) > 0:
            n_targets = len(self.lora_targets)

            # Fused input projection
            self.lora_A = nn.Parameter(torch.zeros(lora_r * n_targets, input_size))

            # Separated output projections
            for i, target_idx in enumerate(self.lora_targets):
                shard_size = self.shard_output_sizes[target_idx]
                lora_B = nn.Parameter(torch.zeros(shard_size, lora_r))
                lora_B.weight_loader = self._lora_B_weight_loader
                setattr(self, f"lora_B_{target_idx}", lora_B)

            self._base_scaling = lora_alpha / lora_r
            self.register_buffer("lora_scaling", torch.tensor(self._base_scaling), persistent=False)
        else:
            self.lora_r = 0

    def _base_weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int = None,
    ):
        """Base weight loader"""
        if loaded_shard_id is None:
            param.data.copy_(loaded_weight)
            return

        param_data = param.data
        shard_offset = sum(self.shard_output_sizes[:loaded_shard_id])
        shard_size = self.shard_output_sizes[loaded_shard_id]
        param_data = param_data.narrow(0, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        param_data.copy_(loaded_weight)

    def _lora_B_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """LoRA B weight loader"""
        loaded_weight = loaded_weight.chunk(self.tp_size, 0)[self.tp_rank]
        param.data.copy_(loaded_weight)

    def load_lora_A(self, loaded_weight: torch.Tensor, target_idx: int):
        """Load single target's lora_A"""
        if target_idx not in self.lora_targets:
            return
        idx = self.lora_targets.index(target_idx)
        start = idx * self.lora_r
        self.lora_A.data[start : start + self.lora_r].copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)

        if self.lora_r <= 0:
            return result

        # Fused LoRA input projection
        lora_hidden = F.linear(x, self.lora_A)

        # Split result and apply respective lora_B
        splits = list(result.split(self.shard_output_sizes, dim=-1))

        for i, target_idx in enumerate(self.lora_targets):
            lora_h = lora_hidden[..., i * self.lora_r : (i + 1) * self.lora_r]
            lora_B = getattr(self, f"lora_B_{target_idx}")
            splits[target_idx] = splits[target_idx] + F.linear(lora_h, lora_B) * self.lora_scaling

        return torch.cat(splits, dim=-1)

    def set_lora_enabled(self, enabled: bool):
        if self.lora_r > 0:
            self.lora_scaling.fill_(self._base_scaling if enabled else 0.0)

    @property
    def lora_enabled(self) -> bool:
        return self.lora_r > 0 and self.lora_scaling.item() != 0.0

    def reset_lora_parameters(self):
        """Reset LoRA parameters (set lora_B to zero to disable LoRA output)"""
        if self.lora_r > 0:
            for target_idx in self.lora_targets:
                getattr(self, f"lora_B_{target_idx}").data.zero_()


class LoRARowParallelLinear(nn.Module):
    """
    RowParallelLinear (o_proj, down_proj) + LoRA

    Input dimension sharded by TP, output dimension not sharded
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        self.input_size = input_size
        self.output_size = output_size
        self.shard_input_size = divide(input_size, self.tp_size)

        # Base weights (input dimension sharded)
        self.weight = nn.Parameter(torch.empty(output_size, self.shard_input_size))
        self.weight.weight_loader = self._base_weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self._base_weight_loader
        else:
            self.register_parameter("bias", None)

        # LoRA parameters
        self.lora_r = lora_r
        self._base_lora_alpha = lora_alpha

        if lora_r > 0:
            # lora_A input dimension needs to be sharded
            self.lora_A = nn.Parameter(torch.zeros(lora_r, self.shard_input_size))
            self.lora_A.weight_loader = self._lora_A_weight_loader

            # lora_B output dimension not sharded
            self.lora_B = nn.Parameter(torch.zeros(output_size, lora_r))

            self._base_scaling = lora_alpha / lora_r
            self.register_buffer("lora_scaling", torch.tensor(self._base_scaling), persistent=False)
        else:
            self.lora_r = 0

    def _base_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Base weight loader (input dimension sharded)"""
        if param.dim() == 2:  # weight
            shard_size = self.shard_input_size
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(1, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def _lora_A_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """LoRA A weight loader (input dimension sharded)"""
        shard_size = self.shard_input_size
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(1, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        if self.lora_r > 0:
            lora_out = F.linear(x, self.lora_A)
            lora_out = F.linear(lora_out, self.lora_B) * self.lora_scaling
            y = y + lora_out

        if self.tp_size > 1:
            dist.all_reduce(y)
        return y

    def set_lora_enabled(self, enabled: bool):
        if self.lora_r > 0:
            self.lora_scaling.fill_(self._base_scaling if enabled else 0.0)

    @property
    def lora_enabled(self) -> bool:
        return self.lora_r > 0 and self.lora_scaling.item() != 0.0

    def reset_lora_parameters(self):
        """Reset LoRA parameters (set lora_B to zero to disable LoRA output)"""
        if self.lora_r > 0:
            self.lora_B.data.zero_()


class LoRALinear(nn.Module):
    """
    Simple LoRA Linear layer for projection layers (no tensor parallel)

    Used for enc_to_lm_proj, lm_to_dit_proj, res_to_dit_proj, etc.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Base weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # LoRA parameters
        self.lora_r = lora_r
        self._base_lora_alpha = lora_alpha

        if lora_r > 0:
            self.lora_A = nn.Parameter(torch.zeros(lora_r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_r))

            self._base_scaling = lora_alpha / lora_r
            self.register_buffer("lora_scaling", torch.tensor(self._base_scaling), persistent=False)
        else:
            self.lora_r = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)

        if self.lora_r > 0:
            lora_out = F.linear(x, self.lora_A)
            lora_out = F.linear(lora_out, self.lora_B) * self.lora_scaling
            y = y + lora_out

        return y

    def set_lora_enabled(self, enabled: bool):
        if self.lora_r > 0:
            self.lora_scaling.fill_(self._base_scaling if enabled else 0.0)

    @property
    def lora_enabled(self) -> bool:
        return self.lora_r > 0 and self.lora_scaling.item() != 0.0

    def reset_lora_parameters(self):
        """Reset LoRA parameters (set lora_B to zero to disable LoRA output)"""
        if self.lora_r > 0:
            self.lora_B.data.zero_()


# ============================================================================
# LoRA Utility Functions
# ============================================================================


def iter_lora_modules(model: nn.Module):
    """Iterate over all LoRA modules in the model"""
    for module in model.modules():
        if isinstance(
            module,
            (
                LoRAQKVParallelLinear,
                LoRAMergedColumnParallelLinear,
                LoRARowParallelLinear,
                LoRALinear,
            ),
        ):
            if module.lora_r > 0:
                yield module


def set_all_lora_enabled(model: nn.Module, enabled: bool):
    """Enable/disable all LoRA layers in the model"""
    for module in iter_lora_modules(model):
        module.set_lora_enabled(enabled)


def reset_all_lora_parameters(model: nn.Module):
    """Reset all LoRA parameters (set lora_B to zero for hot-swapping)"""
    for module in iter_lora_modules(model):
        module.reset_lora_parameters()


def get_lora_state_dict(model: nn.Module) -> dict:
    """Get all LoRA parameters"""
    return {name: param.data.clone() for name, param in model.named_parameters() if "lora_" in name}
