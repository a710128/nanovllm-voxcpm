import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        x_pad = F.pad(x, (self.__padding * 2 - self.__output_padding, 0))
        return super().forward(x_pad)


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        return super().forward(x)[..., : -(self.__padding * 2 - self.__output_padding)]


def WNCausalConv1d(*args, **kwargs):
    return weight_norm(CausalConv1d(*args, **kwargs))


def WNCausalTransposeConv1d(*args, **kwargs):
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class CausalResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, kernel: int = 7, groups: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNCausalConv1d(
                dim,
                dim,
                kernel_size=kernel,
                dilation=dilation,
                padding=pad,
                groups=groups,
            ),
            Snake1d(dim),
            WNCausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        assert pad == 0
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class CausalEncoderBlock(nn.Module):
    def __init__(self, output_dim: int = 16, input_dim=None, stride: int = 1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNCausalConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        )

    def forward(self, x):
        return self.block(x)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        latent_dim: int = 32,
        strides: list[int] = [2, 4, 8, 8],
        depthwise: bool = False,
    ):
        super().__init__()
        self.block = [WNCausalConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            self.block += [CausalEncoderBlock(output_dim=d_model, stride=stride, groups=groups)]

        self.fc_mu = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        hidden_state = self.block(x)
        return {
            "hidden_state": hidden_state,
            "mu": self.fc_mu(hidden_state),
            "logvar": self.fc_logvar(hidden_state),
        }


class NoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = WNCausalConv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size, _, length = x.shape
        noise = torch.randn((batch_size, 1, length), device=x.device, dtype=x.dtype)
        return x + noise * self.linear(x)


class CausalDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        groups=1,
        use_noise_block: bool = False,
    ):
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNCausalTransposeConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if use_noise_block:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                CausalResidualUnit(output_dim, dilation=1, groups=groups),
                CausalResidualUnit(output_dim, dilation=3, groups=groups),
                CausalResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.input_channels = input_dim

    def forward(self, x):
        return self.block(x)


class SampleRateConditionLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sr_bin_buckets: int | None = None,
        cond_type: str = "scale_bias",
        cond_dim: int = 128,
        out_layer: bool = False,
    ):
        super().__init__()
        self.cond_type = cond_type
        out_layer_in_dim = input_dim

        if cond_type == "scale_bias":
            self.scale_embed = nn.Embedding(sr_bin_buckets, input_dim)
            self.bias_embed = nn.Embedding(sr_bin_buckets, input_dim)
            nn.init.ones_(self.scale_embed.weight)
            nn.init.zeros_(self.bias_embed.weight)
        elif cond_type == "scale_bias_init":
            self.scale_embed = nn.Embedding(sr_bin_buckets, input_dim)
            self.bias_embed = nn.Embedding(sr_bin_buckets, input_dim)
            nn.init.normal_(self.scale_embed.weight, mean=1)
            nn.init.normal_(self.bias_embed.weight)
        elif cond_type == "add":
            self.cond_embed = nn.Embedding(sr_bin_buckets, input_dim)
            nn.init.normal_(self.cond_embed.weight)
        elif cond_type == "concat":
            self.cond_embed = nn.Embedding(sr_bin_buckets, cond_dim)
            if not out_layer:
                raise ValueError("out_layer must be True for concat cond_type")
            out_layer_in_dim = input_dim + cond_dim
        else:
            raise ValueError(f"Invalid cond_type: {cond_type}")

        self.out_layer = (
            nn.Sequential(
                Snake1d(out_layer_in_dim),
                WNCausalConv1d(out_layer_in_dim, input_dim, kernel_size=1),
            )
            if out_layer
            else nn.Identity()
        )

    def forward(self, x, sr_cond):
        if self.cond_type in {"scale_bias", "scale_bias_init"}:
            x = x * self.scale_embed(sr_cond).unsqueeze(-1) + self.bias_embed(sr_cond).unsqueeze(-1)
        elif self.cond_type == "add":
            x = x + self.cond_embed(sr_cond).unsqueeze(-1)
        elif self.cond_type == "concat":
            x = torch.cat([x, self.cond_embed(sr_cond).unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        return self.out_layer(x)


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        depthwise: bool = False,
        d_out: int = 1,
        use_noise_block: bool = False,
        sr_bin_boundaries: List[int] | None = None,
        cond_type: str = "scale_bias",
        cond_dim: int = 128,
        cond_out_layer: bool = False,
    ):
        super().__init__()
        if depthwise:
            layers = [
                WNCausalConv1d(input_channel, input_channel, kernel_size=7, padding=3, groups=input_channel),
                WNCausalConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [WNCausalConv1d(input_channel, channels, kernel_size=7, padding=3)]

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            groups = output_dim if depthwise else 1
            layers += [
                CausalDecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    groups=groups,
                    use_noise_block=use_noise_block,
                )
            ]

        layers += [
            Snake1d(output_dim),
            WNCausalConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        if sr_bin_boundaries is None:
            self.model = nn.Sequential(*layers)
            self.sr_bin_boundaries = None
        else:
            self.model = nn.ModuleList(layers)
            self.register_buffer("sr_bin_boundaries", torch.tensor(sr_bin_boundaries, dtype=torch.int32))
            self.sr_bin_buckets = len(sr_bin_boundaries) + 1
            cond_layers = []
            for layer in self.model:
                if layer.__class__.__name__ == "CausalDecoderBlock":
                    cond_layers.append(
                        SampleRateConditionLayer(
                            input_dim=layer.input_channels,
                            sr_bin_buckets=self.sr_bin_buckets,
                            cond_type=cond_type,
                            cond_dim=cond_dim,
                            out_layer=cond_out_layer,
                        )
                    )
                else:
                    cond_layers.append(None)
            self.sr_cond_model = nn.ModuleList(cond_layers)

    def get_sr_idx(self, sample_rate):
        return torch.bucketize(sample_rate, self.sr_bin_boundaries)

    def forward(self, x, sr_cond=None):
        if self.sr_bin_boundaries is None:
            return self.model(x)

        sr_cond = self.get_sr_idx(sr_cond)
        for layer, sr_cond_layer in zip(self.model, self.sr_cond_model):
            if sr_cond_layer is not None:
                x = sr_cond_layer(x, sr_cond)
            x = layer(x)
        return x


class AudioVAEConfigV2(BaseModel):
    encoder_dim: int = 128
    encoder_rates: List[int] = [2, 5, 8, 8]
    latent_dim: int = 64
    decoder_dim: int = 2048
    decoder_rates: List[int] = [8, 6, 5, 2, 2, 2]
    depthwise: bool = True
    sample_rate: int = 16000
    out_sample_rate: int = 48000
    use_noise_block: bool = False
    sr_bin_boundaries: Optional[List[int]] = [20000, 30000, 40000]
    cond_type: str = "scale_bias"
    cond_dim: int = 128
    cond_out_layer: bool = False


class AudioVAEV2(nn.Module):
    def __init__(self, config: Optional[AudioVAEConfigV2] = None, **kwargs):
        if config is None:
            config = AudioVAEConfigV2(**kwargs) if kwargs else AudioVAEConfigV2()
        elif kwargs:
            raise ValueError("Pass either config or keyword args to AudioVAEV2, not both")

        print("AudioVAEV2 config: ", config)

        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.encoder_rates = config.encoder_rates
        self.decoder_dim = config.decoder_dim
        self.decoder_rates = config.decoder_rates
        self.depthwise = config.depthwise
        self.use_noise_block = config.use_noise_block
        self.sample_rate = config.sample_rate
        self.out_sample_rate = config.out_sample_rate
        self.sr_bin_boundaries = config.sr_bin_boundaries

        latent_dim = config.latent_dim
        if latent_dim is None:
            latent_dim = config.encoder_dim * (2 ** len(config.encoder_rates))

        self.latent_dim = latent_dim
        self.encoder_chunk_size = math.prod(config.encoder_rates)
        self.decoder_chunk_size = math.prod(config.decoder_rates)
        self.encoder = CausalEncoder(
            config.encoder_dim,
            latent_dim,
            config.encoder_rates,
            depthwise=config.depthwise,
        )
        self.decoder = CausalDecoder(
            latent_dim,
            config.decoder_dim,
            config.decoder_rates,
            depthwise=config.depthwise,
            use_noise_block=config.use_noise_block,
            sr_bin_boundaries=config.sr_bin_boundaries,
            cond_type=config.cond_type,
            cond_dim=config.cond_dim,
            cond_out_layer=config.cond_out_layer,
        )

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        if sample_rate != self.sample_rate:
            raise AssertionError(f"Expected sample_rate={self.sample_rate}, got {sample_rate}")
        pad_to = self.encoder_chunk_size
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        return nn.functional.pad(audio_data, (0, right_pad))

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, sr_cond: torch.Tensor | None = None):
        if self.sr_bin_boundaries is not None and sr_cond is None:
            sr_cond = torch.tensor([self.out_sample_rate], device=z.device, dtype=torch.int32)
        return self.decoder(z, sr_cond)

    @torch.inference_mode()
    def encode(self, audio_data: torch.Tensor, sample_rate: int):
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)
        audio_data = self.preprocess(audio_data, sample_rate)
        return self.encoder(audio_data)["mu"]


__all__ = ["AudioVAEConfigV2", "AudioVAEV2"]
