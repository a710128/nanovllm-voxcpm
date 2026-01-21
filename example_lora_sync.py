"""
Synchronous LoRA Test Script
"""

from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm.server import LoRAConfig, SyncVoxCPMServerPool
import numpy as np
import soundfile as sf
from tqdm import tqdm
import time


def main():
    # ==================== Configuration ====================
    MODEL_PATH = "~/VoxCPM1.5"  # Base model path
    LORA_PATH = "/path/to/lora_weights.ckpt"  # LoRA weights path, None means not loading

    LORA_R = 32  # LoRA rank
    LORA_ALPHA = 16.0  # LoRA alpha

    TEXT = "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character. I have a dream today! I have a dream that one day, down in Alabama, with its vicious racists, with its governor having his lips dripping with the words of interposition and nullification; one day right down in Alabama little black boys and black girls will be able to join hands with little white boys and white girls as sisters and brothers. I have a dream today! I have a dream that one day every valley shall be exalted, and every hill and mountain shall be made low, the rough places will be made plain, and the crooked places will be made straight, and the glory of the Lord shall be revealed and all flesh shall see it together."
    OUTPUT_FILE = "test_lora.wav"
    CFG_VALUE = 1.5
    DEVICE = 0
    # ================================================

    # 1. Configure LoRA
    lora_config = LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        enable_proj=False,  # Projection layer LoRA
        r=LORA_R,
        alpha=LORA_ALPHA,
        target_modules_lm=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_modules_dit=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_proj_modules=["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"],
    )

    print("Loading model with LoRA...")
    server: SyncVoxCPMServerPool = VoxCPM.from_pretrained(
        MODEL_PATH,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        devices=[DEVICE],
        lora_config=lora_config,
    )
    print("Ready!")

    # 2. Load LoRA weights
    if LORA_PATH:
        print(f"Loading LoRA weights: {LORA_PATH}")
        result = server.load_lora(LORA_PATH)
        print(f"Result: {result}")
        server.set_lora_enabled(True)
    else:
        print("No LoRA weights, using base model")
        server.set_lora_enabled(False)

    # 3. Generate
    buf = []
    start_time = time.time()
    for data in tqdm(server.generate(target_text=TEXT, cfg_value=CFG_VALUE)):
        buf.append(data)
    wav = np.concatenate(buf, axis=0)
    end_time = time.time()

    # 4. Save
    sf.write(OUTPUT_FILE, wav, 44100)

    time_used = end_time - start_time
    wav_duration = wav.shape[0] / 44100
    print(f"Output: {OUTPUT_FILE}")
    print(f"Duration: {wav_duration:.2f}s, Time: {time_used:.2f}s, RTF: {time_used/wav_duration:.4f}")

    server.stop()


if __name__ == "__main__":
    main()
