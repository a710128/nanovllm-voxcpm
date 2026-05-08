import argparse
import asyncio
import base64
from pathlib import Path

import aiohttp


def _infer_format(path: Path, override: str | None) -> str:
    if override is not None and override.strip() != "":
        return override.strip().lower()
    suffix = path.suffix.lstrip(".").lower()
    return suffix or "wav"


async def encode_latents(
    session: aiohttp.ClientSession,
    api_base: str,
    wav_path: Path,
    wav_format: str,
) -> dict:
    wav_b64 = base64.b64encode(wav_path.read_bytes()).decode("utf-8")
    async with session.post(
        f"{api_base.rstrip('/')}/encode_latents",
        json={
            "wav_base64": wav_b64,
            "wav_format": wav_format,
        },
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def generate_mp3(session: aiohttp.ClientSession, api_base: str, payload: dict, out_path: Path) -> None:
    async with session.post(f"{api_base.rstrip('/')}/generate", json=payload) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                f.write(chunk)


async def main() -> None:
    parser = argparse.ArgumentParser(description="VoxCPM REST client example (encode_latents + generate)")
    parser.add_argument("--api-base", default="http://localhost:8000", help="Base URL for FastAPI service")
    parser.add_argument("--target-text", default="Hello world.")
    parser.add_argument("--cfg-value", type=float, default=2.0)

    parser.add_argument("--prompt-audio", type=Path, default=None, help="Prompt audio file path")
    parser.add_argument(
        "--prompt-text-path",
        type=Path,
        default=None,
        help="Path to a text file containing the prompt transcription",
    )
    parser.add_argument("--prompt-format", default=None, help="Optional audio format override for prompt")

    parser.add_argument("--ref-audio", type=Path, default=None, help="Reference audio file path")
    parser.add_argument("--ref-format", default=None, help="Optional audio format override for reference")

    parser.add_argument("--out", type=Path, default=Path("out.mp3"), help="Output mp3 path")
    args = parser.parse_args()

    prompt_latents_b64: str | None = None
    ref_latents_b64: str | None = None
    prompt_text: str | None = None

    async with aiohttp.ClientSession() as session:
        if args.prompt_audio is not None:
            if args.prompt_text_path is None:
                raise SystemExit("--prompt-text-path is required when --prompt-audio is set")
            if not args.prompt_audio.exists():
                raise SystemExit(f"Prompt audio not found: {args.prompt_audio}")
            if not args.prompt_text_path.exists():
                raise SystemExit(f"Prompt text file not found: {args.prompt_text_path}")

            prompt_text = args.prompt_text_path.read_text(encoding="utf-8").strip()
            if prompt_text == "":
                raise SystemExit("Prompt text file is empty")

            prompt_fmt = _infer_format(args.prompt_audio, args.prompt_format)
            prompt = await encode_latents(session, args.api_base, args.prompt_audio, wav_format=prompt_fmt)
            prompt_latents_b64 = prompt["prompt_latents_base64"]

        if args.ref_audio is not None:
            if not args.ref_audio.exists():
                raise SystemExit(f"Ref audio not found: {args.ref_audio}")
            ref_fmt = _infer_format(args.ref_audio, args.ref_format)
            ref = await encode_latents(session, args.api_base, args.ref_audio, wav_format=ref_fmt)
            ref_latents_b64 = ref["prompt_latents_base64"]

        payload: dict = {
            "target_text": args.target_text,
            "cfg_value": args.cfg_value,
        }
        if prompt_latents_b64 is not None:
            payload["prompt_latents_base64"] = prompt_latents_b64
            payload["prompt_text"] = prompt_text
        if ref_latents_b64 is not None:
            payload["ref_audio_latents_base64"] = ref_latents_b64

        await generate_mp3(session, args.api_base, payload, args.out)


if __name__ == "__main__":
    asyncio.run(main())
