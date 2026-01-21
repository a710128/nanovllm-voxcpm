from nanovllm_voxcpm import VoxCPM
import numpy as np
import soundfile as sf
from tqdm.asyncio import tqdm
import time
from nanovllm_voxcpm.models.voxcpm.server import SyncVoxCPMServerPool


def main():
    print("Loading...")
    server: SyncVoxCPMServerPool = VoxCPM.from_pretrained(
        "~/VoxCPM1.5",
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        devices=[0],
    )
    print("Ready")

    buf = []
    start_time = time.time()
    for data in tqdm(
        server.generate(
            target_text="I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character. I have a dream today! I have a dream that one day, down in Alabama, with its vicious racists, with its governor having his lips dripping with the words of interposition and nullification; one day right down in Alabama little black boys and black girls will be able to join hands with little white boys and white girls as sisters and brothers. I have a dream today! I have a dream that one day every valley shall be exalted, and every hill and mountain shall be made low, the rough places will be made plain, and the crooked places will be made straight, and the glory of the Lord shall be revealed and all flesh shall see it together.",
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

    server.stop()


if __name__ == "__main__":
    main()
