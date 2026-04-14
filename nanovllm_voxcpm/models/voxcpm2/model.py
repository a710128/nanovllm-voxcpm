import math
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from nanovllm_voxcpm.layers.activation import SiluAndMul
from nanovllm_voxcpm.layers.attention import Attention
from nanovllm_voxcpm.layers.embed_head import VocabParallelEmbedding
from nanovllm_voxcpm.layers.layernorm import RMSNorm
from nanovllm_voxcpm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from nanovllm_voxcpm.layers.lora import (
    LoRALinear,
    LoRAMergedColumnParallelLinear,
    LoRAQKVParallelLinear,
    LoRARowParallelLinear,
)
from nanovllm_voxcpm.models.voxcpm2.config import CfmConfig, LoRAConfig, MiniCPM4Config, VoxCPM2Config
from nanovllm_voxcpm.utils.context import get_context


class MiniCPMLongRoPE(nn.Module):
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
        scale = max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        ext_factors = (
            torch.tensor(self.long_factor, dtype=torch.float32, device=device)
            if seq_len > self.original_max_position_embeddings
            else torch.tensor(self.short_factor, dtype=torch.float32, device=device)
        )
        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device), self.inv_freq.to(device=device).to(dtype)
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype) * self.scaling_factor, persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype) * self.scaling_factor, persistent=False)

    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        rotate_half_x = torch.cat((-x2, x1), dim=-1)
        result = x * cos.to(torch.float32) + rotate_half_x * sin.to(torch.float32)
        return result.to(orig_dtype)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        query_shape = query.shape
        key_shape = key.shape
        query = self._apply_rotary_emb(query.reshape(num_tokens, -1, self.dim), cos, sin).view(query_shape)
        key = self._apply_rotary_emb(key.reshape(num_tokens, -1, self.dim), cos, sin).view(key_shape)
        return query, key


def get_cpm4_rope(head_size: int, rotary_dim: int, max_position: int, base: float, rope_scaling=None):
    return MiniCPMLongRoPE(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position,
        base=base,
        short_factor=rope_scaling.short_factor if rope_scaling else None,
        long_factor=rope_scaling.long_factor if rope_scaling else None,
        original_max_position_embeddings=(rope_scaling.original_max_position_embeddings if rope_scaling else None),
    )


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
        rope_scaling=None,
        use_rope: bool = True,
        apply_qk_norm: bool = False,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position = max_position
        self.use_rope = use_rope
        self.apply_qk_norm = apply_qk_norm
        self.is_causal = is_causal

        lora_r = lora_config.r if lora_config else 0
        lora_alpha = lora_config.alpha if lora_config else 16.0
        max_loras = lora_config.max_loras if lora_config else 1
        max_lora_rank = lora_config.max_lora_rank if lora_config else lora_r
        lora_targets = lora_config.target_modules_lm if lora_config else []
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
                lora_targets=qkv_lora_targets,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=qkv_bias,
            )

        if lora_r > 0 and "o_proj" in lora_targets:
            self.o_proj = LoRARowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=qkv_bias,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
            )
        else:
            self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=qkv_bias)

        self.rotary_emb = (
            get_cpm4_rope(self.head_dim, self.head_dim, self.max_position, rope_theta, rope_scaling)
            if self.use_rope
            else None
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads, is_causal=self.is_causal)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps) if self.apply_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps) if self.apply_qk_norm else None

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        rotary_emb = self.rotary_emb
        if self.is_causal:
            if self.q_norm is not None:
                q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
                k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
            if rotary_emb is not None:
                q, k = rotary_emb(positions, q, k)
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            bsz = q.size(0)
            if self.q_norm is not None:
                q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(q.shape)
                k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(k.shape)
            if rotary_emb is not None:
                q, k = rotary_emb(positions.repeat(bsz), q, k)
            q = q.view(bsz, -1, self.num_heads, self.head_dim)
            k = k.view(bsz, -1, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, -1, self.num_kv_heads, self.head_dim)
        out = self.attn(q, k, v)
        out = out.view(
            (-1, self.num_heads * self.head_dim) if self.is_causal else (bsz, -1, self.num_heads * self.head_dim)
        )
        return self.o_proj(out)


class Cpm4MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, lora_config: Optional[LoRAConfig] = None) -> None:
        super().__init__()
        lora_r = lora_config.r if lora_config else 0
        lora_alpha = lora_config.alpha if lora_config else 16.0
        max_loras = lora_config.max_loras if lora_config else 1
        max_lora_rank = lora_config.max_lora_rank if lora_config else lora_r
        lora_targets = lora_config.target_modules_lm if lora_config else []
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
                lora_targets=gate_up_lora_targets,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
            )
        else:
            self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = (
            LoRARowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
            )
            if lora_r > 0 and "down_proj" in lora_targets
            else RowParallelLinear(intermediate_size, hidden_size, bias=False)
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Cpm4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: MiniCPM4Config,
        is_causal: bool = True,
        lora_config: Optional[LoRAConfig] = None,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = Cpm4Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            is_causal=is_causal,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "kv_channels", None),
            rope_theta=getattr(config, "rope_theta", 10000),
            rope_scaling=getattr(config, "rope_scaling", None),
            use_rope=use_rope,
            apply_qk_norm=getattr(config, "apply_qk_norm", False),
            lora_config=lora_config,
        )
        self.mlp = Cpm4MLP(config.hidden_size, config.intermediate_size, lora_config=lora_config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor | None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(positions, hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, residual


class Cpm4Model(nn.Module):
    def __init__(
        self,
        config: MiniCPM4Config,
        is_causal: bool = True,
        lora_config: Optional[LoRAConfig] = None,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = (
            VocabParallelEmbedding(config.vocab_size, config.hidden_size) if config.vocab_size > 0 else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                Cpm4DecoderLayer(config, is_causal, lora_config, use_rope=use_rope)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        return self.norm(hidden_states)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=x.dtype, device=x.device) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: int | None = None):
        super().__init__()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, bias=True)

    def forward(self, sample):
        return self.linear_2(self.act(self.linear_1(sample)))


class VoxCPM2LocDiT(nn.Module):
    def __init__(self, config: MiniCPM4Config, in_channels: int = 64, lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.in_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.cond_proj = nn.Linear(in_channels, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, self.out_channels, bias=True)
        self.time_embeddings = SinusoidalPosEmb(config.hidden_size)
        self.time_mlp = TimestepEmbedding(config.hidden_size, config.hidden_size)
        self.delta_time_mlp = TimestepEmbedding(config.hidden_size, config.hidden_size)
        dit_lora_config = None
        if lora_config and lora_config.enable_dit:
            dit_lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=False,
                r=lora_config.r,
                alpha=lora_config.alpha,
                max_loras=lora_config.max_loras,
                max_lora_rank=lora_config.max_lora_rank,
                target_modules_lm=lora_config.target_modules_dit,
                target_modules_dit=[],
                target_proj_modules=[],
            )
        self.decoder = Cpm4Model(config, is_causal=False, lora_config=dit_lora_config)

    def forward(self, x: torch.Tensor, mu: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, dt: torch.Tensor):
        x = self.in_proj(x.transpose(1, 2).contiguous())
        cond = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix = cond.size(1)
        t = self.time_mlp(self.time_embeddings(t).to(x.dtype))
        dt = self.delta_time_mlp(self.time_embeddings(dt).to(x.dtype))
        t = t + dt
        mu = mu.view(x.size(0), -1, x.size(-1))
        hidden = torch.cat([mu, t.unsqueeze(1), cond, x], dim=1)
        position_ids = torch.arange(0, hidden.size(1), dtype=torch.long, device=hidden.device)
        hidden = self.decoder(hidden, position_ids)
        hidden = self.out_proj(hidden[:, prefix + mu.size(1) + 1 :, :])
        return hidden.transpose(1, 2).contiguous()


class UnifiedCFM(nn.Module):
    def __init__(
        self,
        in_channels,
        patch_size: int,
        inference_timesteps: int,
        cfm_params: CfmConfig,
        estimator: VoxCPM2LocDiT,
        mean_mode: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.inference_timesteps = inference_timesteps
        self.mean_mode = mean_mode
        self.estimator = estimator

    def forward(self, mu: torch.Tensor, cond: torch.Tensor, temperature: torch.Tensor, cfg_value: torch.Tensor):
        bsz = mu.shape[0]
        z = torch.randn((bsz, self.in_channels, self.patch_size), device=mu.device, dtype=mu.dtype)
        z = z * temperature[:, None, None]
        t_span = torch.linspace(1, 0, self.inference_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = t_span + (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(z, t_span=t_span, mu=mu, cond=cond, cfg_value=cfg_value)

    def optimized_scale(self, positive_flat, negative_flat):
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        return dot_product / squared_norm

    def solve_euler(
        self, x: torch.Tensor, t_span: torch.Tensor, mu: torch.Tensor, cond: torch.Tensor, cfg_value: torch.Tensor
    ):
        t, dt = t_span[0], t_span[0] - t_span[1]
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if step <= zero_init_steps:
                dphi_dt = 0.0
            else:
                bsz = x.size(0)
                x_in = torch.zeros([2 * bsz, self.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
                mu_in = torch.zeros([2 * bsz, mu.size(1)], device=x.device, dtype=x.dtype)
                t_in = torch.zeros([2 * bsz], device=x.device, dtype=x.dtype)
                dt_in = torch.zeros([2 * bsz], device=x.device, dtype=x.dtype)
                cond_in = torch.zeros([2 * bsz, self.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
                x_in[:bsz], x_in[bsz:] = x, x
                mu_in[:bsz] = mu
                t_in[:bsz], t_in[bsz:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:bsz], dt_in[bsz:] = dt.unsqueeze(0), dt.unsqueeze(0)
                if not self.mean_mode:
                    dt_in = torch.zeros_like(dt_in)
                cond_in[:bsz], cond_in[bsz:] = cond, cond
                dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
                st_star = self.optimized_scale(dphi_dt.view(bsz, -1), cfg_dphi_dt.view(bsz, -1))
                st_star = st_star.view(bsz, *([1] * (len(dphi_dt.shape) - 1)))
                dphi_dt = cfg_dphi_dt * st_star + cfg_value[:, None, None] * (dphi_dt - cfg_dphi_dt * st_star)
            x = x - dt * dphi_dt
            t = t - dt
            sol = x
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]
        return sol


class VoxCPM2LocEnc(nn.Module):
    def __init__(self, config: MiniCPM4Config, input_dim: int = 64):
        super().__init__()
        self.special_token = nn.Parameter(torch.empty(1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)
        self.encoder = Cpm4Model(config, is_causal=False)

    def forward(self, x):
        t, _, _ = x.size()
        x = self.in_proj(x)
        special_tokens = self.special_token[0].expand(t, 1, -1)
        x = torch.cat([special_tokens, x], dim=1)
        position_ids = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        outputs = self.encoder(x, position_ids)
        return outputs[:, 0, :].view(t, -1)


class ScalarQuantizationLayer(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim: int = 64, scale: int = 9):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, out_dim)
        self.scale = scale

    def forward(self, hidden):
        hidden = torch.tanh(self.in_proj(hidden))
        hidden = torch.round(hidden * self.scale) / self.scale
        return self.out_proj(hidden)


class VoxCPM2Model(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: VoxCPM2Config, inference_timesteps: int, lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        assert not self.config.lm_config.use_mup, "mup inference is not supported now"

        lm_lora_config = lora_config if (lora_config and lora_config.enable_lm) else None
        self.base_lm = Cpm4Model(config.lm_config, lora_config=lm_lora_config)

        residual_lm_config = config.lm_config.model_copy(deep=True)
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        residual_lm_config.vocab_size = 0
        self.residual_lm = Cpm4Model(
            residual_lm_config,
            lora_config=lm_lora_config,
            use_rope=not config.residual_lm_no_rope,
        )

        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPM2LocEnc(encoder_config, input_dim=config.feat_dim)

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
            estimator=VoxCPM2LocDiT(decoder_config, in_channels=config.feat_dim, lora_config=lora_config),
            mean_mode=config.dit_mean_mode,
        )

        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
        )

        proj_lora_r = lora_config.r if (lora_config and lora_config.enable_proj) else 0
        proj_lora_alpha = lora_config.alpha if lora_config else 16.0
        proj_max_loras = lora_config.max_loras if lora_config else 1
        proj_max_lora_rank = lora_config.max_lora_rank if lora_config else proj_lora_r
        proj_targets = lora_config.target_proj_modules if lora_config else []
        self.enc_to_lm_proj = (
            LoRALinear(
                config.encoder_config.hidden_dim,
                config.lm_config.hidden_size,
                lora_r=proj_lora_r,
                lora_alpha=proj_lora_alpha,
                max_loras=proj_max_loras,
                max_lora_rank=proj_max_lora_rank,
            )
            if proj_lora_r > 0 and "enc_to_lm_proj" in proj_targets
            else nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        )
        self.lm_to_dit_proj = (
            LoRALinear(
                config.lm_config.hidden_size,
                config.dit_config.hidden_dim,
                lora_r=proj_lora_r,
                lora_alpha=proj_lora_alpha,
                max_loras=proj_max_loras,
                max_lora_rank=proj_max_lora_rank,
            )
            if proj_lora_r > 0 and "lm_to_dit_proj" in proj_targets
            else nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        )
        self.res_to_dit_proj = (
            LoRALinear(
                config.lm_config.hidden_size,
                config.dit_config.hidden_dim,
                lora_r=proj_lora_r,
                lora_alpha=proj_lora_alpha,
                max_loras=proj_max_loras,
                max_lora_rank=proj_max_lora_rank,
            )
            if proj_lora_r > 0 and "res_to_dit_proj" in proj_targets
            else nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        )
        self.fusion_concat_proj = (
            LoRALinear(
                config.lm_config.hidden_size * 2,
                config.lm_config.hidden_size,
                lora_r=proj_lora_r,
                lora_alpha=proj_lora_alpha,
                max_loras=proj_max_loras,
                max_lora_rank=proj_max_lora_rank,
            )
            if proj_lora_r > 0 and "fusion_concat_proj" in proj_targets
            else nn.Linear(config.lm_config.hidden_size * 2, config.lm_config.hidden_size)
        )

        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)

    def forward(
        self,
        positions: torch.Tensor,
        text_tokens: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        temperature: torch.Tensor,
        cfg_value: torch.Tensor,
    ):
        feat_embeds = self.enc_to_lm_proj(self.feat_encoder(feat))
        feat_embeds = torch.masked_fill(feat_embeds, feat_mask.unsqueeze(-1).logical_not(), 0)
        text_embeds = self.base_lm.embed_tokens(text_tokens)
        combined_embeds = torch.where(feat_mask.unsqueeze(-1), feat_embeds, text_embeds)
        enc_outputs = self.base_lm(combined_embeds, positions)
        enc_outputs = torch.where(feat_mask.unsqueeze(-1), self.fsq_layer(enc_outputs), enc_outputs)

        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            lm_hidden = enc_outputs[last_indices].contiguous()
        else:
            lm_hidden = enc_outputs

        residual_inputs = self.fusion_concat_proj(
            torch.cat([enc_outputs, torch.where(feat_mask.unsqueeze(-1), feat_embeds, 0)], dim=-1)
        )
        ralm_outputs = self.residual_lm(residual_inputs, positions)
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            ralm_hidden = ralm_outputs[last_indices].contiguous()
            prefix_feat_cond = feat[last_indices].contiguous()
        else:
            ralm_hidden = ralm_outputs
            prefix_feat_cond = feat

        dit_hidden = torch.cat([self.lm_to_dit_proj(lm_hidden), self.res_to_dit_proj(ralm_hidden)], dim=-1)
        pred_feat = self.feat_decoder(
            mu=dit_hidden,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            temperature=temperature,
            cfg_value=cfg_value,
        ).transpose(1, 2)
        stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)
        return {"latents": pred_feat, "stop_flag": stop_flag}
