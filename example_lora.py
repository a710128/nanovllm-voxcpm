"""
Asynchronous LoRA Test Script
"""

from nanovllm_voxcpm import VoxCPM
from nanovllm_voxcpm.models.voxcpm.server import LoRAConfig, AsyncVoxCPMServerPool
import numpy as np
import soundfile as sf
from tqdm.asyncio import tqdm
import time
import asyncio


async def main():
    # ==================== Configuration ====================
    MODEL_PATH = "~/VoxCPM1.5"  # Base model path
    LORA_PATH = "/path/to/lora_weights.ckpt"  # LoRA weights path, None means not loading

    LORA_R = 32  # LoRA rank
    LORA_ALPHA = 16.0  # LoRA alpha

    TEXT = "有这么一个人呐，一个字都不认识，连他自己的名字都不会写，他上京赶考去了。哎，到那儿还就中了，不但中了，而且升来升去呀，还入阁拜相，你说这不是瞎说吗？哪有这个事啊。当然现在是没有这个事，现在你不能替人民办事，人民也不选举你呀！我说这个事情啊，是明朝的这么一段事情。因为在那个社会啊，甭管你有才学没才学，有学问没学问，你有钱没有？有钱，就能做官，捐个官做。说有势力，也能做官。也没钱也没势力，碰上啦，用上这假势力，也能做官，什么叫“假势力”呀，它因为在那个社会呀，那些个做官的人，都怀着一肚子鬼胎，都是这个拍上欺下，疑神疑鬼，你害怕我，我害怕你，互相害怕，这里头就有矛盾啦。由打这个呢，造成很多可笑的事情。今天我说的这段就这么回事。"
    OUTPUT_FILE = "test_lora.wav"
    CFG_VALUE = 1.5
    DEVICE = 0
    # ================================================

    # 1. Configure LoRA
    lora_config = LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=LORA_R,
        alpha=LORA_ALPHA,
        target_modules_lm=["q_proj", "k_proj", "v_proj", "o_proj"],
        target_modules_dit=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    print("Loading model with LoRA...")
    server: AsyncVoxCPMServerPool = VoxCPM.from_pretrained(
        model=MODEL_PATH,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        devices=[DEVICE],
        lora_config=lora_config,
    )
    await server.wait_for_ready()
    print("Ready!")

    # 2. Load LoRA weights
    if LORA_PATH:
        print(f"Loading LoRA weights: {LORA_PATH}")
        result = await server.load_lora(LORA_PATH)
        print(f"Result: {result}")
        await server.set_lora_enabled(True)
    else:
        print("No LoRA weights, using base model")
        await server.set_lora_enabled(False)

    # 3. Generate
    buf = []
    start_time = time.time()
    async for data in tqdm(server.generate(target_text=TEXT, cfg_value=CFG_VALUE)):
        buf.append(data)
    wav = np.concatenate(buf, axis=0)
    end_time = time.time()

    # 4. Save
    sf.write(OUTPUT_FILE, wav, 44100)

    time_used = end_time - start_time
    wav_duration = wav.shape[0] / 44100
    print(f"Output: {OUTPUT_FILE}")
    print(f"Duration: {wav_duration:.2f}s, Time: {time_used:.2f}s, RTF: {time_used/wav_duration:.4f}")

    await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
