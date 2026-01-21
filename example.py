from nanovllm_voxcpm import VoxCPM
import numpy as np
import soundfile as sf
from tqdm.asyncio import tqdm
import time
from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool


async def main():
    print("Loading...")
    server: AsyncVoxCPMServerPool = VoxCPM.from_pretrained(
        model="~/VoxCPM1.5",
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        devices=[0],
    )
    await server.wait_for_ready()
    print("Ready")

    buf = []
    start_time = time.time()
    async for data in tqdm(
        server.generate(
            target_text="有这么一个人呐，一个字都不认识，连他自己的名字都不会写，他上京赶考去了。哎，到那儿还就中了，不但中了，而且升来升去呀，还入阁拜相，你说这不是瞎说吗？哪有这个事啊。当然现在是没有这个事，现在你不能替人民办事，人民也不选举你呀！我说这个事情啊，是明朝的这么一段事情。因为在那个社会啊，甭管你有才学没才学，有学问没学问，你有钱没有？有钱，就能做官，捐个官做。说有势力，也能做官。也没钱也没势力，碰上啦，用上这假势力，也能做官，什么叫“假势力”呀，它因为在那个社会呀，那些个做官的人，都怀着一肚子鬼胎，都是这个拍上欺下，疑神疑鬼，你害怕我，我害怕你，互相害怕，这里头就有矛盾啦。由打这个呢，造成很多可笑的事情。今天我说的这段就这么回事。",
            cfg_value=1.5,
        )
    ):
        buf.append(data)
    wav = np.concatenate(buf, axis=0)
    end_time = time.time()

    time_used = end_time - start_time
    wav_duration = wav.shape[0] / 44100
    sf.write("test.wav", wav, 44100)

    print(f"Time: {end_time - start_time}s")
    print(f"RTF: {time_used / wav_duration}")

    await server.stop()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
