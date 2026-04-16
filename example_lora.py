from nanovllm_voxcpm import VoxCPM
import numpy as np
import soundfile as sf
from tqdm.asyncio import tqdm
import time
from nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig
from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool


MODEL_NAME = "openbmb/VoxCPM2"
LORA_NAME = "demo"
LORA_PATH = "/path/to/lora/checkpoint"  # directory containing lora_weights.safetensors (+ optional lora_config.json)
OUTPUT_WAV = "test_lora_async.wav"
ATTENTION_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


async def main():
    print("Loading...")
    server: AsyncVoxCPM2ServerPool = VoxCPM.from_pretrained(
        model=MODEL_NAME,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        devices=[0],
        lora_config=LoRAConfig(
            enable_lm=True,
            enable_dit=True,
            enable_proj=False,
            max_loras=1,
            max_lora_rank=8,
            target_modules_lm=ATTENTION_LORA_TARGETS,
            target_modules_dit=ATTENTION_LORA_TARGETS,
            target_proj_modules=[],
        ),
    )
    await server.wait_for_ready()
    print("Ready")

    try:
        await server.register_lora(LORA_NAME, LORA_PATH)
        print(f"Registered LoRA: {LORA_NAME}")

        model_info = await server.get_model_info()
        sample_rate = int(model_info["sample_rate"])

        buf = []
        start_time = time.time()
        async for data in tqdm(
            server.generate(
                target_text="这是一个使用 runtime LoRA 的语音生成示例。请把这里替换成你想测试的文本。",
                cfg_value=2,
                lora_name=LORA_NAME,
            )
        ):
            buf.append(data)
        wav = np.concatenate(buf, axis=0)
        end_time = time.time()

        time_used = end_time - start_time
        wav_duration = wav.shape[0] / sample_rate
        print(f"Sample rate: {sample_rate}")
        sf.write(OUTPUT_WAV, wav, sample_rate)

        print(f"Output: {OUTPUT_WAV}")
        print(f"Time: {time_used}s")
        print(f"RTF: {time_used / wav_duration}")
    finally:
        try:
            await server.unregister_lora(LORA_NAME)
        finally:
            await server.stop()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
