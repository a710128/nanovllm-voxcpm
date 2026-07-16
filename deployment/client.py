import asyncio
import base64
from pathlib import Path

import aiohttp

API_BASE = "http://localhost:8760"


async def encode_latents(session: aiohttp.ClientSession, wav_path: Path, wav_format: str) -> dict:
    wav_b64 = base64.b64encode(wav_path.read_bytes()).decode("utf-8")
    async with session.post(
        f"{API_BASE}/encode_latents",
        json={
            "wav_base64": wav_b64,
            "wav_format": wav_format,
        },
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def generate_mp3(session: aiohttp.ClientSession, payload: dict, out_path: Path) -> None:
    async with session.post(f"{API_BASE}/generate", json=payload) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                f.write(chunk)


async def main() -> None:
    prompt_wav = Path(__file__).with_name("prompt_audio.wav")
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Missing prompt wav: {prompt_wav}")

    wav_format = prompt_wav.suffix.lstrip(".") or "wav"

    async with aiohttp.ClientSession() as session:
        # Optional: precompute prompt latents so you can reuse them across requests.
        prompt = await encode_latents(session, prompt_wav, wav_format=wav_format)
        prompt_latents_b64 = prompt["prompt_latents_base64"]

        jobs = [
            (
                {
                    "target_text": "Hello world.",
                    "cfg_value": 2,
                },
                Path("out_zero_shot.mp3"),
            ),
            (
                {
                    "target_text": "Hello world.",
                    "prompt_latents_base64": prompt_latents_b64,
                    "prompt_text": "你好，很高兴见到你。",
                    "cfg_value": 2,
                },
                Path("out_prompted.mp3"),
            ),
        ]

        jobs.extend(
            [
                (
                    {
                        "target_text": "Hello world.",
                        "ref_audio_latents_base64": prompt_latents_b64,
                        "cfg_value": 2,
                    },
                    Path("out_zero_shot_with_ref.mp3"),
                ),
            ]
        )

        await asyncio.gather(*[generate_mp3(session, payload, out_path) for payload, out_path in jobs])


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
