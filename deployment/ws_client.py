#!/usr/bin/env python3
"""WebSocket client for the VoxCPM voice-to-voice pipeline.

Sends an audio file to the server and saves the synthesized response.

Pipeline flow:
    Input Audio -> ASR -> LLM -> TTS -> Output Audio

Usage:
    python ws_client.py -i input.wav -o response.wav --speaker alaa
    python ws_client.py -i input.wav -o response.wav --speaker hammad --server ws://localhost:8001/ws

Configuration can also be set via .env file:
    WS_CLIENT_SERVER_URL   - WebSocket server URL (default: ws://localhost:8001/ws)
    WS_CLIENT_INPUT_FILE   - Default input file path
    WS_CLIENT_OUTPUT_FILE  - Default output file path (default: response.wav)
    WS_CLIENT_SPEAKER      - Default speaker (default: alaa)
    WS_CLIENT_TIMEOUT      - Response timeout in seconds (default: 120)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import wave
from pathlib import Path

import websockets
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)


# Must match server NANOVLLM_OPUS_* config
OPUS_SAMPLE_RATE = 48000
OPUS_CHANNELS = 1
OPUS_FRAME_MS = int(os.environ.get("NANOVLLM_OPUS_FRAME_MS", "20"))
OPUS_FRAME_SAMPLES = int(OPUS_SAMPLE_RATE * OPUS_FRAME_MS / 1000)  # 960 samples @ 20ms

DEFAULT_SERVER_URL = os.environ.get("WS_CLIENT_SERVER_URL", "ws://localhost:8001/ws")
DEFAULT_INPUT = os.environ.get("WS_CLIENT_INPUT_FILE", "")
DEFAULT_OUTPUT = os.environ.get("WS_CLIENT_OUTPUT_FILE", "response.wav")
DEFAULT_SPEAKER = os.environ.get("WS_CLIENT_SPEAKER", "alaa")
DEFAULT_TIMEOUT = float(os.environ.get("WS_CLIENT_TIMEOUT", "120"))


async def run_client(
    server_url: str,
    input_path: Path,
    output_path: Path,
    speaker: str,
    timeout: float,
    verbose: bool,
) -> None:
    """Send audio file to server and save synthesized response."""

    audio_bytes = input_path.read_bytes()
    audio_format = input_path.suffix.lstrip(".").lower() or "wav"

    if verbose:
        print(f"Connecting to {server_url}")
        print(f"Speaker: {speaker}")
        print(f"Input: {input_path} ({len(audio_bytes):,} bytes, format={audio_format})")

    # Allow large audio payloads (base64 + JSON overhead ~1.5x)
    max_size = max(16 * 1024 * 1024, len(audio_bytes) * 2)

    async with websockets.connect(server_url, max_size=max_size) as ws:
        # Step 1: Send init message with speaker selection
        await ws.send(json.dumps({
            "type": "init",
            "speaker": speaker,
        }))

        if verbose:
            print(f"Sent init message (speaker={speaker})")

        # Step 2: Send audio as a single chunk
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        await ws.send(json.dumps({
            "type": "audio_chunk",
            "data": audio_b64,
            "format": audio_format,
        }))
        await ws.send(json.dumps({"type": "audio_end"}))

        if verbose:
            print("Audio sent, waiting for response...")

        # Collect Opus frames from server
        pcm_samples: list[bytes] = []
        opus_frames: list[bytes] = []
        decoder = None
        try:
            import opuslib

            decoder = opuslib.Decoder(OPUS_SAMPLE_RATE, OPUS_CHANNELS)
        except ImportError:
            print("Warning: opuslib not installed; saving raw Opus frames only")

        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                print(f"Error: timeout ({timeout}s) waiting for server")
                break

            if isinstance(msg, bytes):
                opus_frames.append(msg)
                if decoder is not None:
                    try:
                        pcm = decoder.decode(msg, OPUS_FRAME_SAMPLES)
                        pcm_samples.append(pcm)
                    except Exception as e:
                        if verbose:
                            print(f"  Opus decode error: {e}")
                if verbose:
                    print(f"  Received audio frame: {len(msg)} bytes")
            else:
                data = json.loads(msg)
                msg_type = data.get("type")
                if msg_type == "complete":
                    if verbose:
                        print("Received: complete")
                    break
                elif msg_type == "error":
                    print(f"Server error: {data.get('error', 'unknown')}")
                    break
                elif verbose:
                    print(f"  Received message: {data}")

        try:
            await ws.send(json.dumps({"type": "terminate"}))
        except Exception:
            pass

    # Save output
    if pcm_samples:
        _save_wav(output_path, b"".join(pcm_samples))
        total_pcm = sum(len(s) for s in pcm_samples)
        duration = total_pcm / (OPUS_SAMPLE_RATE * OPUS_CHANNELS * 2)
        print(f"Saved {output_path} ({duration:.2f}s, {len(opus_frames)} Opus frames)")
    elif opus_frames:
        # Fallback: save raw concatenated frames
        out = output_path.with_suffix(".opus-raw")
        out.write_bytes(b"".join(opus_frames))
        print(f"Saved raw Opus frames to {out} (install opuslib to decode)")
    else:
        print("No audio received from server")


def _save_wav(path: Path, pcm_data: bytes) -> None:
    """Save raw 16-bit PCM data as a WAV file."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(OPUS_CHANNELS)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(OPUS_SAMPLE_RATE)
        wav.writeframes(pcm_data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WebSocket client for VoxCPM voice-to-voice pipeline",
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path(DEFAULT_INPUT) if DEFAULT_INPUT else None,
        required=not DEFAULT_INPUT,
        help="Input audio file (wav/mp3/flac/ogg)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output WAV file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-s", "--server",
        default=DEFAULT_SERVER_URL,
        help=f"WebSocket URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--speaker",
        default=DEFAULT_SPEAKER,
        choices=["alaa", "hammad", "hanan", "khalil"],
        help=f"Speaker voice to use (default: {DEFAULT_SPEAKER})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Response timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_client(
        server_url=args.server,
        input_path=args.input,
        output_path=args.output,
        speaker=args.speaker,
        timeout=args.timeout,
        verbose=args.verbose,
    ))


if __name__ == "__main__":
    main()
