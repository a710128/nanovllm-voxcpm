from dataclasses import dataclass
import torch
from multiprocessing.synchronize import Event

from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.engine.model_runner import RunnerTask, BaseModelRunner
from nanovllm_voxcpm.utils.loader import load_model
from nanovllm_voxcpm.models.voxcpm.model import VoxCPMModel, VoxCPMConfig
from nanovllm_voxcpm.layers.audio_vae import AudioVAE
import numpy as np
import os

@dataclass
class VoxCPMPayload:
    # (T)
    text_tokens : np.ndarray | None = None
    # (T, P, D)
    feats : np.ndarray | None = None
    # (T)
    feat_masks : np.ndarray | None = None
    
    temperature : float = 1.0
    cfg_value : float = 1.0

    # (T, D)
    padding_decode : np.ndarray | None = None


class VoxCPMRunner(BaseModelRunner):
    def __init__(self, config: Config[VoxCPMConfig], rank: int, device_idx : int, distributed_port: int, event: Event | list[Event]):
        self.inference_timesteps = config.model_config.inference_timesteps
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.lora_config = config.lora_config
        super().__init__(config, rank, device_idx, distributed_port, event)
    
    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16
    
    def init_model(self, model_config : VoxCPMConfig, model_path : str):
        self.model = VoxCPMModel(model_config, self.inference_timesteps, lora_config=self.lora_config)
        load_model(self.model, model_path)

        torch.set_default_dtype(torch.float32)
        self.vae = AudioVAE() if model_config.audio_vae_config is None else AudioVAE(**model_config.audio_vae_config.model_dump(mode="dict"))

        vae_state_dict = torch.load(os.path.join(model_path, "audiovae.pth"))["state_dict"]
        self.vae.load_state_dict(vae_state_dict)
        torch.set_default_dtype(torch.bfloat16)
    
    def make_dummy_inputs(self, batch_size: int, length: int) -> torch.Tensor:
        return {
            "text_tokens": torch.zeros(batch_size * length, dtype=torch.int64),
            "feat": torch.zeros(batch_size * length, self.patch_size, self.feat_dim),
            "feat_mask": torch.zeros(batch_size * length, dtype=torch.bool),
            "temperature": torch.zeros(batch_size),
            "cfg_value": torch.zeros(batch_size),
        }

    def make_dummy_outputs(self, batch_size: int) -> torch.Tensor:
        latents = torch.zeros(
            batch_size,
            self.patch_size,
            self.feat_dim,
            dtype=self.dtype,
        )
        stop_flag = torch.zeros(
            batch_size,
            dtype=torch.int64,
        )
        return {
            "latents": latents,
            "stop_flag": stop_flag,
        }
    
    def encode_latents(self, wav : torch.Tensor) -> np.ndarray:
        assert wav.ndim == 2, "Invalid shape of wav"
        return self.vae.encode(wav, self.vae.sample_rate).permute(0, 2, 1).view(-1, self.feat_dim).to(torch.float32).cpu().numpy()
    
    def run(self, seqs: list[RunnerTask[VoxCPMPayload]], is_prefill: bool):
        positions = self.prepare_prefill_context(seqs) if is_prefill else self.prepare_decode_context(seqs)
        inputs = {
            "positions": positions,
        }

        text_tokens = []
        feats = []
        feat_masks = []
        temperatures = []
        cfg_values = []

        ret_new_latents = []
        for seq in seqs:
            payload : VoxCPMPayload = seq.custom_payload
            assert payload.text_tokens.shape[0] == payload.feats.shape[0]
            assert payload.text_tokens.shape[0] == payload.feat_masks.shape[0]

            text_tokens.append(payload.text_tokens)
            feats.append(payload.feats)
            feat_masks.append(payload.feat_masks)

            temperatures.append(payload.temperature)
            cfg_values.append(payload.cfg_value)
        
        inputs["text_tokens"] = torch.from_numpy(np.concatenate(text_tokens, axis=0)).cuda(non_blocking=True)
        inputs["feat"] = torch.from_numpy(np.concatenate(feats, axis=0)).cuda(non_blocking=True).to(self.dtype)
        inputs["feat_mask"] = torch.from_numpy(np.concatenate(feat_masks, axis=0)).cuda(non_blocking=True)
        inputs["temperature"] = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        inputs["cfg_value"] = torch.tensor(cfg_values, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        
        outputs = self.run_model(inputs, is_prefill)

        latents = outputs["latents"]

        pad_lengths = []
        for i in range(len(seqs)):
            if seqs[i].custom_payload.padding_decode is not None:
                pad_lengths.append(seqs[i].custom_payload.padding_decode.shape[0])
            else:
                pad_lengths.append(0)
        max_pad_decode = max(pad_lengths) + self.patch_size

        vae_decoder_inputs = torch.zeros(len(seqs), max_pad_decode, self.feat_dim, dtype=torch.float32, device="cuda")
        for i in range(len(seqs)):
            pad_len = pad_lengths[i]
            if pad_len > 0:
                vae_decoder_inputs[i, :pad_len] = torch.from_numpy(seqs[i].custom_payload.padding_decode).cuda(non_blocking=True)
            vae_decoder_inputs[i, pad_len:pad_len+self.patch_size] = latents[i].to(torch.float32)
        
        vae_decoder_outputs = self.vae.decode(vae_decoder_inputs.permute(0, 2, 1))[:, 0, :].cpu().numpy()
        stop_flag = outputs["stop_flag"].cpu().tolist()

        ret_waveforms = []
        for i in range(len(seqs)):
            pad_len = pad_lengths[i]
            ret_waveforms.append(
                vae_decoder_outputs[
                    i, 
                    pad_len * self.vae.chunk_size: 
                    (pad_len + self.patch_size) * self.vae.chunk_size
                ]
            )

        ret = []
        np_latents = latents.to(torch.float32).cpu().numpy()
        for i in range(len(seqs)):
            ret.append({
                "latents": np_latents[i],
                "stop_flag": stop_flag[i],
                "waveforms": ret_waveforms[i],
            })

        return ret


                

