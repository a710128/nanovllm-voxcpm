from nanovllm_voxcpm.engine.llm_engine import LLMEngineBase
from nanovllm_voxcpm.models.voxcpm.runner import VoxCPMRunner, RunnerTask, VoxCPMPayload
from nanovllm_voxcpm.config import Config
from nanovllm_voxcpm.models.voxcpm.config import VoxCPMConfig
from nanovllm_voxcpm.engine.sequence import Sequence
from dataclasses import dataclass, field
import numpy as np
from transformers import LlamaTokenizerFast
from nanovllm_voxcpm.models.voxcpm.utils import mask_multichar_chinese_tokens
import torch

# 流式累积解码常量（参考 VoxCPMANE）
STREAMING_ACCUMULATE_COUNT = 12  # 累积 12 个 patches 后解码
STREAMING_CONTEXT_COUNT = 2      # 保留 2 个 patches 作为上下文
SAMPLES_PER_FRAME = 640          # 每帧 640 样本
PATCH_SIZE = 2                   # 每个 patch 2 帧


@dataclass
class VoxCPMSeqPayload:
    # [(T, P, D)]
    feats : list[np.ndarray]

    text_tokens : list[int]
    feat_masks : list[bool]

    generated_waveforms : list[np.ndarray]

    temperature : float
    cfg_value : float

    decode_pad : np.ndarray | None = None
    max_generate_length : int | None = None

    # 累积解码相关
    pending_latents: list[np.ndarray] = field(default_factory=list)
    is_first_chunk: bool = True
    generated_patch_count: int = 0


class VoxCPMEngine(LLMEngineBase):
    def __init__(self, config: Config[VoxCPMConfig]):
        self.n_decode_pad_frames = 4
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.chunk_size = 640
        self.audio_start_token = 101
        self.block_size = config.kvcache_block_size

        self.tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(config.model))


        super().__init__(VoxCPMRunner, config, config.tensor_parallel_size)

    def preprocess_seq(self, seq: Sequence[VoxCPMSeqPayload], is_prefill: bool) -> RunnerTask[VoxCPMPayload]:
        if is_prefill:
            if len(seq.custom_payload.feats) > 1:
                feats = np.concatenate(seq.custom_payload.feats, axis=0)
                seq.custom_payload.feats = [feats]

            return RunnerTask(
                seq.block_table,
                len(seq),
                seq.num_cached_tokens,
                seq.block_size,
                VoxCPMPayload(
                    text_tokens=np.array(seq.custom_payload.text_tokens[seq.num_cached_tokens:], dtype=np.int64),
                    feats=seq.custom_payload.feats[-1][seq.num_cached_tokens:],
                    feat_masks=np.array(seq.custom_payload.feat_masks[seq.num_cached_tokens:], dtype=np.bool_),
                    temperature=seq.custom_payload.temperature,
                    cfg_value=seq.custom_payload.cfg_value,
                    padding_decode=seq.custom_payload.decode_pad,
                )
            )
        else:
            return RunnerTask(
                seq.block_table,
                len(seq),
                len(seq) - 1,
                seq.block_size,
                VoxCPMPayload(
                    text_tokens=np.array(seq.custom_payload.text_tokens[-1:], dtype=np.int64),
                    feats=seq.custom_payload.feats[-1][-1:],
                    feat_masks=np.array(seq.custom_payload.feat_masks[-1:], dtype=np.bool_),
                    temperature=seq.custom_payload.temperature,
                    cfg_value=seq.custom_payload.cfg_value,
                    padding_decode=seq.custom_payload.decode_pad,
                )
            )


    def postprocess_seq(self, seq: Sequence[VoxCPMSeqPayload], outputs: dict, is_prefill: bool):
        stop_flag = outputs["stop_flag"]
        latents = outputs["latents"]
        # waveforms 不再使用，改用累积解码

        seq.append_token(latents.tobytes())

        seq.custom_payload.feats.append(latents[None])
        seq.custom_payload.text_tokens.append(0)
        seq.custom_payload.feat_masks.append(True)

        # 累积 latents
        seq.custom_payload.pending_latents.append(latents.copy())
        seq.custom_payload.generated_patch_count += 1

        # 更新 decode_pad（用于 LLM 推理的上下文）
        latents_reshaped = latents.reshape(-1, self.feat_dim)
        if seq.custom_payload.decode_pad is not None:
            seq.custom_payload.decode_pad = np.concatenate(
                [seq.custom_payload.decode_pad, latents_reshaped], axis=0
            )[-self.n_decode_pad_frames:]
        else:
            seq.custom_payload.decode_pad = latents_reshaped[-self.n_decode_pad_frames:]

        # 检查是否需要累积解码
        pending_count = len(seq.custom_payload.pending_latents)
        should_decode = pending_count >= STREAMING_ACCUMULATE_COUNT or stop_flag == 1

        if should_decode and pending_count > 0:
            # 累积解码
            all_latents = np.stack(seq.custom_payload.pending_latents, axis=0)  # (N, P, D)
            all_latents = all_latents.reshape(-1, self.feat_dim)  # (N*P, D)

            # 调用 VAE 解码
            decoded_waveform = self.model_runner.decode_latents(all_latents)

            # 跳过重叠部分（非首次 chunk 跳过 2 patches = 4 帧 = 2560 样本）
            if not seq.custom_payload.is_first_chunk:
                skip_samples = STREAMING_CONTEXT_COUNT * PATCH_SIZE * SAMPLES_PER_FRAME
                decoded_waveform = decoded_waveform[skip_samples:]
            seq.custom_payload.is_first_chunk = False

            # 如果停止，去掉最后 1280 样本（避免尾部杂音）
            if stop_flag == 1:
                decoded_waveform = decoded_waveform[:-PATCH_SIZE * SAMPLES_PER_FRAME]

            # 只有有效数据时才添加
            if len(decoded_waveform) > 0:
                seq.custom_payload.generated_waveforms.append(decoded_waveform)

            # 保留最后 2 个 patches 作为上下文
            seq.custom_payload.pending_latents = seq.custom_payload.pending_latents[-STREAMING_CONTEXT_COUNT:]

        # 检查停止条件
        if stop_flag == 1:
            seq.stoped = True
        elif seq.custom_payload.max_generate_length is not None and \
             seq.custom_payload.generated_patch_count >= seq.custom_payload.max_generate_length:
            # 强制解码剩余的 pending_latents
            pending_count = len(seq.custom_payload.pending_latents)
            if pending_count > STREAMING_CONTEXT_COUNT:
                all_latents = np.stack(seq.custom_payload.pending_latents, axis=0)
                all_latents = all_latents.reshape(-1, self.feat_dim)
                decoded_waveform = self.model_runner.decode_latents(all_latents)

                if not seq.custom_payload.is_first_chunk:
                    skip_samples = STREAMING_CONTEXT_COUNT * PATCH_SIZE * SAMPLES_PER_FRAME
                    decoded_waveform = decoded_waveform[skip_samples:]

                # 去掉尾部
                decoded_waveform = decoded_waveform[:-PATCH_SIZE * SAMPLES_PER_FRAME]

                if len(decoded_waveform) > 0:
                    seq.custom_payload.generated_waveforms.append(decoded_waveform)

            seq.stoped = True

    def add_request(
            self,
            seq_id : str,
            target_text : str,
            prompt_text : str = "",
            prompt_latents : np.ndarray = None,
            max_generate_length : int = 2000,
            temperature : float = 1.0,
            cfg_value : float = 1.0,
        ):
        text_tokens = self.tokenizer(prompt_text + target_text) + [self.audio_start_token]
        audio_feat = np.zeros((len(text_tokens), self.patch_size, self.feat_dim), dtype=np.float32)
        feat_masks = [False for _ in range(len(text_tokens))]
        hash_tokens = []
        for t in text_tokens:
            hash_tokens.append(t)

        decode_pad = None

        if prompt_latents is not None:
            wav_latents = prompt_latents
            decode_pad = wav_latents[-self.n_decode_pad_frames:]

            wav_latents = wav_latents.reshape(-1, self.patch_size, self.feat_dim)
            audio_feat = np.concatenate([audio_feat, wav_latents], axis=0)
            text_tokens.extend([0 for _ in range(wav_latents.shape[0])])
            feat_masks.extend([True for _ in range(wav_latents.shape[0])])

            for i in range(wav_latents.shape[0]):
                hash_tokens.append(wav_latents[i].tobytes())

        seq = Sequence(
            seq_id,
            hash_tokens,
            self.block_size,
            VoxCPMSeqPayload(
                feats=[audio_feat],
                text_tokens=text_tokens,
                feat_masks=feat_masks,
                decode_pad=decode_pad,
                temperature=temperature,
                cfg_value=cfg_value,
                max_generate_length=max_generate_length,
                generated_waveforms=[],
                pending_latents=[],
                is_first_chunk=True,
                generated_patch_count=0,
            )
        )

        self.add_sequence(seq)

    def encode_latents(self, wav : torch.Tensor) -> np.ndarray:
        return self.model_runner.encode_latents(wav)
