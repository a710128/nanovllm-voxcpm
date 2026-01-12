import torch
from torch import nn
import torch.distributed as dist
from typing import Optional

from nanovllm_voxcpm.layers.activation import SiluAndMul
from nanovllm_voxcpm.layers.attention import Attention
from nanovllm_voxcpm.layers.layernorm import RMSNorm
from nanovllm_voxcpm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm_voxcpm.layers.lora import (
    LoRAQKVParallelLinear, 
    LoRAMergedColumnParallelLinear, 
    LoRARowParallelLinear,
    LoRALinear,
    iter_lora_modules,
    set_all_lora_enabled,
    reset_all_lora_parameters,
    get_lora_state_dict,
)
from nanovllm_voxcpm.layers.embed_head import VocabParallelEmbedding
import math

from nanovllm_voxcpm.models.voxcpm.config import MiniCPM4Config, CfmConfig, VoxCPMConfig, LoRAConfig
from nanovllm_voxcpm.utils.context import get_context

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    
    This is equivalent to the MiniCPM modeling implementation.
    """
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


class MiniCPMLongRoPE(nn.Module):
    """MiniCPM LongRoPE implementation equivalent to modeling_minicpm.py"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=None,
    ) -> None:
        super().__init__()
        self.dim = head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.short_factor = short_factor or [1.0] * (head_size // 2)
        self.long_factor = long_factor or [1.0] * (head_size // 2)
        self.original_max_position_embeddings = original_max_position_embeddings or max_position_embeddings
        
        # Calculate scaling factor (kept for compatibility, but not used to scale cos/sin amplitude)
        scale = (max_position_embeddings / self.original_max_position_embeddings)
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        
        # Create base inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Pre-compute cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=device)
        else:
            ext_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=device)

        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype)
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        # Do NOT scale cos/sin amplitude; only frequency is scaled by ext_factors
        self.register_buffer('cos_cached', emb.cos().to(dtype)* self.scaling_factor, persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype)* self.scaling_factor, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position: [t]
        # query: [t, h, d]
        # key: [t, h, d]
        num_tokens = positions.size(0)
        
        # Get cos/sin for the positions
        cos = self.cos_cached[positions]  # [num_tokens, head_dim]
        sin = self.sin_cached[positions]  # [num_tokens, head_dim]
        
        # Apply rotary embedding using the original nano-vllm method but with corrected math
        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.dim)
        query = self._apply_rotary_emb(query, cos, sin).view(query_shape)
        
        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.dim)
        key = self._apply_rotary_emb(key, cos, sin).view(key_shape)
        
        return query, key
    
    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding with corrected math matching modeling_minicpm.py"""
        # x: [num_tokens, num_heads, head_dim]
        # cos/sin: [num_tokens, head_dim] from _set_cos_sin_cache (already repeated)
        
        cos = cos.unsqueeze(1)  # [num_tokens, 1, head_dim] to broadcast over heads
        sin = sin.unsqueeze(1)  # [num_tokens, 1, head_dim] to broadcast over heads
        
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)
        
        # Apply standard RoPE: (x * cos) + (rotate_half(x) * sin)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        rotate_half_x = torch.cat((-x2, x1), dim=-1)
        
        result = x * cos + rotate_half_x * sin
        return result.to(orig_dtype)


def get_cpm4_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """Get CPM4 LongRoPE implementation"""
    rotary_emb = MiniCPMLongRoPE(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
        short_factor=rope_scaling.short_factor if rope_scaling else None,
        long_factor=rope_scaling.long_factor if rope_scaling else None,
        original_max_position_embeddings=rope_scaling.original_max_position_embeddings if rope_scaling else None,
    )
    return rotary_emb


class Cpm4Attention(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 32768,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        is_causal: bool = True,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        apply_qk_norm: bool = False,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position = max_position
        self.apply_qk_norm = apply_qk_norm
        self.is_causal = is_causal
        
        # Determine LoRA parameters for attention projections
        lora_r = lora_config.r if lora_config else 0
        lora_alpha = lora_config.alpha if lora_config else 16.0
        lora_dropout = lora_config.dropout if lora_config else 0.0
        lora_targets = lora_config.target_modules_lm if lora_config else []
        
        # QKV projection with optional LoRA
        qkv_lora_targets = [t.replace("_proj", "") for t in lora_targets if t in ["q_proj", "k_proj", "v_proj"]]
        if lora_r > 0 and qkv_lora_targets:
            self.qkv_proj = LoRAQKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=qkv_bias,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_targets=qkv_lora_targets,
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=qkv_bias,
            )
        
        # O projection with optional LoRA
        if lora_r > 0 and "o_proj" in lora_targets:
            self.o_proj = LoRARowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=qkv_bias,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        else:
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=qkv_bias,
            )
        self.rotary_emb = get_cpm4_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            is_causal=self.is_causal,
        )
        if self.apply_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.is_causal:
            # Apply Q/K normalization only if enabled
            assert q.ndim == 2 and k.ndim == 2 and v.ndim == 2, "q, k, v must be 2D tensors"
            if self.q_norm is not None:
                q_by_head = q.view(-1, self.num_heads, self.head_dim)
                q_by_head = self.q_norm(q_by_head)
                q = q_by_head.view(q.shape)
                
                k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
                k = k_by_head.view(k.shape)
            
            # Apply rotary embedding using nano-vllm interface
            q, k = self.rotary_emb(positions, q, k)

            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3, "q, k, v must be 3D tensors"
            B = q.size(0)

            if self.q_norm is not None:
                q_by_head = q.view(-1, self.num_heads, self.head_dim)
                q_by_head = self.q_norm(q_by_head)
                q = q_by_head.view(q.shape)
                
                k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
                k = k_by_head.view(k.shape)
            
            # Apply rotary embedding using nano-vllm interface
            q, k = self.rotary_emb(positions.repeat(B), q, k)
            q = q.view(B, -1, self.num_heads, self.head_dim)
            k = k.view(B, -1, self.num_kv_heads, self.head_dim)
            v = v.view(B, -1, self.num_kv_heads, self.head_dim)

        o = self.attn(q, k, v)

        if self.is_causal:
            o = o.view(-1, self.num_heads * self.head_dim)
        else:
            o = o.view(B, -1, self.num_heads * self.head_dim)

        output = self.o_proj(o)
        return output


class Cpm4MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        
        # Determine LoRA parameters for MLP projections
        lora_r = lora_config.r if lora_config else 0
        lora_alpha = lora_config.alpha if lora_config else 16.0
        lora_dropout = lora_config.dropout if lora_config else 0.0
        lora_targets = lora_config.target_modules_lm if lora_config else []
        
        # gate_up_proj with optional LoRA
        gate_up_lora_targets = []
        if "gate_proj" in lora_targets:
            gate_up_lora_targets.append(0)
        if "up_proj" in lora_targets:
            gate_up_lora_targets.append(1)
        
        if lora_r > 0 and gate_up_lora_targets:
            self.gate_up_proj = LoRAMergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_targets=gate_up_lora_targets,
            )
        else:
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
            )
        
        # down_proj with optional LoRA
        if lora_r > 0 and "down_proj" in lora_targets:
            self.down_proj = LoRARowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        else:
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
            )
        
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Cpm4DecoderLayer(nn.Module):

    def __init__(
        self,
        config : MiniCPM4Config,
        is_causal: bool = True,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.self_attn = Cpm4Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            is_causal=is_causal,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            apply_qk_norm=getattr(config, 'apply_qk_norm', False),
            lora_config=lora_config,
        )
        self.mlp = Cpm4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            lora_config=lora_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # depth scaling like MiniCPM
        self.scale_depth = getattr(config, 'scale_depth', 1.0)
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # PreNorm + Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(positions, hidden_states)
        hidden_states = residual + attn_out

        # PreNorm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out
        return hidden_states, residual


class Cpm4Model(nn.Module):

    def __init__(
        self,
        config: MiniCPM4Config,
        is_causal: bool = True,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config

        if config.vocab_size > 0:
            self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.embed_tokens = nn.Identity()
        
        self.layers = nn.ModuleList([
            Cpm4DecoderLayer(config, is_causal, lora_config) 
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=device) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int = None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class VoxCPMLocDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        config: MiniCPM4Config,
        in_channels: int = 64,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.config = config

        self.in_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.cond_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, self.out_channels, bias=True)

        self.time_embeddings = SinusoidalPosEmb(config.hidden_size)
        self.time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )
        self.delta_time_mlp = TimestepEmbedding(
            in_channels=config.hidden_size,
            time_embed_dim=config.hidden_size,
        )

        assert config.vocab_size == 0, "vocab_size must be 0 for local DiT"
        # Create DiT-specific LoRA config if provided
        dit_lora_config = None
        if lora_config and lora_config.enable_dit:
            dit_lora_config = LoRAConfig(
                enable_lm=True,  # Use the same mechanism
                enable_dit=False,
                r=lora_config.r,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                target_modules_lm=lora_config.target_modules_dit,  # Use DiT targets
                target_modules_dit=[],
            )
        self.decoder = Cpm4Model(config, is_causal=False, lora_config=dit_lora_config)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of inputs
        mu: (N, C) tensor of hidden embedding
        t: (N,) tensor of diffusion timesteps
        cond: (N, C, T') tensor of prefix conditions
        dt: (N,) used for mean velocity (may be supported in the future...)
        """
        x = self.in_proj(x.transpose(1, 2).contiguous())

        cond = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix = cond.size(1)

        t = self.time_embeddings(t).to(x.dtype)
        t = self.time_mlp(t)
        dt = self.time_embeddings(dt).to(x.dtype)
        dt = self.delta_time_mlp(dt)
        t = t + dt

        x = torch.cat([(mu + t).unsqueeze(1), cond, x], dim=1)

        position_ids = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        hidden = self.decoder(x, position_ids)
        hidden = hidden[:, prefix + 1 :, :]
        hidden = self.out_proj(hidden)

        return hidden.transpose(1, 2).contiguous()

class UnifiedCFM(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size : int,
        inference_timesteps : int,
        cfm_params: CfmConfig,
        estimator: VoxCPMLocDiT,
        mean_mode: bool = False,
    ):
        super().__init__()
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.in_channels = in_channels
        self.mean_mode = mean_mode
        self.patch_size = patch_size
        self.inference_timesteps = inference_timesteps

        # Just change the architecture of the estimator here
        self.estimator = estimator

    def forward(
        self,
        mu: torch.Tensor,
        cond: torch.Tensor,
        temperature: torch.Tensor,
        cfg_value: torch.Tensor,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats)
            n_timesteps (int): number of diffusion steps
            cond: Not used but kept for future purposes
            temperature (torch.Tensor): temperature for scaling noise. (batch_size,)
            cfg_value (torch.Tensor): cfg value for guidance. (batch_size,)

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, c = mu.shape
        t = self.patch_size
        z = torch.randn((b, self.in_channels, t), device=mu.device, dtype=mu.dtype) * temperature[:, None, None]

        t_span = torch.linspace(1, 0, self.inference_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # Sway sampling strategy
        t_span = t_span + (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

        return self.solve_euler(z, t_span=t_span, mu=mu, cond=cond, cfg_value=cfg_value)

    def optimized_scale(self, positive_flat, negative_flat):
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        
        st_star = dot_product / squared_norm
        return st_star

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: float = 1.0,
    ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats)
            cond: condition -- prefix prompt
            cfg_value (float, optional): cfg value for guidance. Defaults to 1.0.
        """
        t, _, dt = t_span[0], t_span[-1], t_span[0] - t_span[1]

        zero_init_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if step <= zero_init_steps:
                dphi_dt = 0.
            else:
                # Classifier-Free Guidance inference introduced in VoiceBox
                b = x.size(0)
                x_in = torch.zeros([2 * b, self.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
                mu_in = torch.zeros([2 * b, mu.size(1)], device=x.device, dtype=x.dtype)
                t_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
                dt_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
                cond_in = torch.zeros([2 * b, self.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
                x_in[:b], x_in[b:] = x, x
                mu_in[:b] = mu
                t_in[:b], t_in[b:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:b], dt_in[b:] = dt.unsqueeze(0), dt.unsqueeze(0)
                # not used now
                if not self.mean_mode:
                    dt_in = torch.zeros_like(dt_in)
                cond_in[:b], cond_in[b:] = cond, cond

                dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
                
                positive_flat = dphi_dt.view(b, -1)
                negative_flat = cfg_dphi_dt.view(b, -1)
                st_star = self.optimized_scale(positive_flat, negative_flat)
                st_star = st_star.view(b, *([1] * (len(dphi_dt.shape) - 1)))
                
                dphi_dt = cfg_dphi_dt * st_star + cfg_value[:, None, None] * (dphi_dt - cfg_dphi_dt * st_star)

            x = x - dt * dphi_dt
            t = t - dt
            sol = x
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]

        return sol


class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        super().__init__()
        self.config = config
        self.special_token = nn.Parameter(torch.empty(1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)

        assert config.vocab_size == 0, "vocab_size must be 0 for local encoder"
        self.encoder = Cpm4Model(config, is_causal=False)

    def forward(self, x):
        """
        x: [T, P, D]
        """
        T, P, D = x.size()

        x = self.in_proj(x)
        special_tokens = self.special_token[0].expand(T, 1, -1)
        x = torch.cat([special_tokens, x], dim=1)
        position_ids = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        outputs = self.encoder(x, position_ids)
        cls_output = outputs[:, 0, :]

        return cls_output.view(T, -1)


class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim: int = 64, scale: int = 9):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.scale = scale

        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)
    
    def forward(self, hidden):
        hidden = self.in_proj(hidden)
        hidden = torch.tanh(hidden)
        hidden = torch.round(hidden * self.scale) / self.scale

        return self.out_proj(hidden)


class VoxCPMModel(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: VoxCPMConfig,
        inference_timesteps: int,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size

        assert not self.config.lm_config.use_mup, "mup inference is not supported now"

        # Determine LoRA config for LM layers
        lm_lora_config = lora_config if (lora_config and lora_config.enable_lm) else None

        # Text-Semantic LM
        self.base_lm = Cpm4Model(config.lm_config, lora_config=lm_lora_config)

        # Residual Acoustic LM
        residual_lm_config = config.lm_config.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0
        self.residual_lm = Cpm4Model(residual_lm_config, lora_config=lm_lora_config)

        # Local Encoder (no LoRA for encoder)
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        decoder_config = config.lm_config.model_copy(deep=True)
        decoder_config.hidden_size = config.dit_config.hidden_dim
        decoder_config.intermediate_size = config.dit_config.ffn_dim
        decoder_config.num_attention_heads = config.dit_config.num_heads
        decoder_config.num_hidden_layers = config.dit_config.num_layers
        decoder_config.kv_channels = config.dit_config.kv_channels
        decoder_config.vocab_size = 0
        self.feat_decoder = UnifiedCFM(
            in_channels=config.feat_dim,
            patch_size=config.patch_size,
            inference_timesteps=inference_timesteps,
            cfm_params=config.dit_config.cfm_config,
            estimator=VoxCPMLocDiT(decoder_config, in_channels=config.feat_dim, lora_config=lora_config),
        )

        # Projection layers
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size, 
            config.lm_config.hidden_size, 
            config.scalar_quantization_latent_dim, 
            config.scalar_quantization_scale
        )
        
        # Determine LoRA config for projection layers
        proj_lora_r = lora_config.r if (lora_config and lora_config.enable_proj) else 0
        proj_lora_alpha = lora_config.alpha if lora_config else 16.0
        proj_lora_dropout = lora_config.dropout if lora_config else 0.0
        proj_targets = lora_config.target_proj_modules if lora_config else []
        
        # enc_to_lm_proj
        if proj_lora_r > 0 and "enc_to_lm_proj" in proj_targets:
            self.enc_to_lm_proj = LoRALinear(
                config.encoder_config.hidden_dim, config.lm_config.hidden_size,
                lora_r=proj_lora_r, lora_alpha=proj_lora_alpha, lora_dropout=proj_lora_dropout
            )
        else:
            self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        
        # lm_to_dit_proj
        if proj_lora_r > 0 and "lm_to_dit_proj" in proj_targets:
            self.lm_to_dit_proj = LoRALinear(
                config.lm_config.hidden_size, config.dit_config.hidden_dim,
                lora_r=proj_lora_r, lora_alpha=proj_lora_alpha, lora_dropout=proj_lora_dropout
            )
        else:
            self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        
        # res_to_dit_proj
        if proj_lora_r > 0 and "res_to_dit_proj" in proj_targets:
            self.res_to_dit_proj = LoRALinear(
                config.lm_config.hidden_size, config.dit_config.hidden_dim,
                lora_r=proj_lora_r, lora_alpha=proj_lora_alpha, lora_dropout=proj_lora_dropout
            )
        else:
            self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)

        # Stop Predictor
        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)
    
    def forward(
            self,
            positions : torch.Tensor,
            text_tokens : torch.Tensor,
            feat : torch.Tensor,
            feat_mask : torch.Tensor,
            temperature : torch.Tensor,
            cfg_value : torch.Tensor,
        ):
        """
        text_tokens: [T]
        feat: [T, P, D]
        feat_mask: [T]
        temperature: [B]
        cfg_value: [B]
        """
        feat_embeds = self.feat_encoder(feat)
        feat_embeds = self.enc_to_lm_proj(feat_embeds)
        feat_embeds = torch.masked_fill(feat_embeds, feat_mask.unsqueeze(-1).logical_not(), 0)

        text_embeds = self.base_lm.embed_tokens(text_tokens)
        combined_embeds = torch.where(
            feat_mask.unsqueeze(-1),
            feat_embeds,
            text_embeds,
        )

        enc_outputs = self.base_lm(combined_embeds, positions)
        enc_outputs = torch.where(
            feat_mask.unsqueeze(-1),
            self.fsq_layer(enc_outputs),
            enc_outputs,
        )

        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            lm_hidden = enc_outputs[last_indices].contiguous()
        else:
            lm_hidden = enc_outputs
        

        ralm_outputs = self.residual_lm(
            enc_outputs + feat_embeds, 
            positions,
        )

        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            ralm_hidden = ralm_outputs[last_indices].contiguous()
            # (b, P, D)
            prefix_feat_cond = feat[last_indices].contiguous()
        else:
            ralm_hidden = ralm_outputs
            # (b, P, D)
            prefix_feat_cond = feat
        
        dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
        dit_hidden_2 = self.res_to_dit_proj(ralm_hidden)  # [b, h_dit]
        dit_hidden = dit_hidden_1 + dit_hidden_2  # [b, h_dit]

        # (b, P, D)
        pred_feat = self.feat_decoder(
            mu=dit_hidden,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            temperature=temperature,
            cfg_value=cfg_value,
        ).transpose(1, 2)

        stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)

        return {
            "latents":  pred_feat, 
            "stop_flag": stop_flag,
        }
    
    # ------------------------------------------------------------------ #
    # LoRA Management Methods
    # ------------------------------------------------------------------ #
    
    def set_lora_enabled(self, enabled: bool):
        """Enable/disable all LoRA layers (without unloading weights)."""
        set_all_lora_enabled(self, enabled)
    
    def reset_lora_parameters(self):
        """Reset all LoRA parameters to initial state (effectively unloading LoRA)."""
        reset_all_lora_parameters(self)
    
    def get_lora_state_dict(self) -> dict:
        """Get all LoRA parameters (lora_A/lora_B)."""
        return get_lora_state_dict(self)
    
    def iter_lora_modules(self):
        """Iterate over all LoRA modules in the model."""
        return iter_lora_modules(self)
