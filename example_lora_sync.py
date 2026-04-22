from nanovllm_voxcpm import VoxCPM
import numpy as np
import soundfile as sf
from tqdm.asyncio import tqdm
import time
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig
from nanovllm_voxcpm.models.voxcpm2.server import SyncVoxCPM2ServerPool

MODEL_NAME = "openbmb/VoxCPM2"
LORA_NAME = "demo"
LORA_PATH = "/path/to/lora/checkpoint"  # directory containing lora_weights.safetensors (+ optional lora_config.json)
OUTPUT_WAV = "test_lora_sync.wav"
ALL_LINEAR_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
VOXCPM2_PROJ_LORA_TARGETS = ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj", "fusion_concat_proj"]


def main():
    print("Loading...")
    server: SyncVoxCPM2ServerPool = VoxCPM.from_pretrained(
        MODEL_NAME,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        devices=[0],
        lora_config=LoRAConfig(
            enable_lm=True,
            enable_dit=True,
            enable_proj=True,
            max_loras=1,
            max_lora_rank=32,
            target_modules_lm=ALL_LINEAR_LORA_TARGETS,
            target_modules_dit=ALL_LINEAR_LORA_TARGETS,
            target_proj_modules=VOXCPM2_PROJ_LORA_TARGETS,
        ),
    )
    print("Ready")

    try:
        server.register_lora(LORA_NAME, LORA_PATH)
        print(f"Registered LoRA: {LORA_NAME}")

        model_info = server.get_model_info()
        sample_rate = int(model_info["sample_rate"])

        buf = []
        start_time = time.time()
        for data in tqdm(
            server.generate(
                target_text="This example shows how to generate speech with a runtime LoRA adapter.",
                cfg_value=1.5,
                lora_name=LORA_NAME,
            )
        ):
            buf.append(data)
        wav = np.concatenate(buf, axis=0)
        end_time = time.time()

        time_used = end_time - start_time
        wav_duration = wav.shape[0] / sample_rate
        sf.write(OUTPUT_WAV, wav, sample_rate)

        print(f"Output: {OUTPUT_WAV}")
        print(f"Time: {time_used}s")
        print(f"RTF: {time_used / wav_duration}")
    finally:
        try:
            server.unregister_lora(LORA_NAME)
        finally:
            server.stop()


if __name__ == "__main__":
    main()
