from nanovllm_voxcpm import VoxCPM
import numpy as np
import soundfile as sf
from tqdm.asyncio import tqdm
import time
from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool


async def main():
    print("Loading...")
    server: AsyncVoxCPMServerPool = VoxCPM.from_pretrained(
        model="models/VoxCPM2-0.8B",
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        devices=[0],
    )
    await server.wait_for_ready()
    print("Ready")
    model_info = await server.get_model_info()
    sample_rate = int(model_info["sample_rate"])

    buf = []
    start_time = time.time()
    async for data in tqdm(
        server.generate(
            target_text="有这么一个人呐，一个字都不认识，连他自己的名字都不会写，他上京赶考去了。",
            cfg_value=2,
        )
    ):
        buf.append(data)
    wav = np.concatenate(buf, axis=0)
    end_time = time.time()

    time_used = end_time - start_time
    wav_duration = wav.shape[0] / sample_rate
    print(f"Sample rate: {sample_rate}")
    sf.write("test.wav", wav, sample_rate)

    print(f"Time: {end_time - start_time}s")
    print(f"RTF: {time_used / wav_duration}")

    await server.stop()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
