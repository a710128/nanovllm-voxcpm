from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import AsyncIterator

import numpy as np

from app.core.config import OpusConfig
from app.core.metrics import AUDIO_ENCODE_FAILURES_TOTAL, AUDIO_ENCODE_SECONDS


def float32_to_s16le_bytes(wav: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio to signed 16-bit little-endian PCM bytes."""
    wav_f32 = wav.astype(np.float32, copy=False)
    wav_f32 = np.clip(wav_f32, -1.0, 1.0)
    wav_i16 = (wav_f32 * 32767.0).astype(np.int16, copy=False)
    return wav_i16.tobytes(order="C")


async def stream_opus(
    *,
    wav_chunks: AsyncIterator[np.ndarray],
    sample_rate: int,
    opus_config: OpusConfig,
) -> AsyncIterator[bytes]:
    """Encode float32 mono waveform chunks to Opus and stream bytes.

    Encoding is done in a background thread to avoid blocking the event loop.
    Each yielded item is a raw Opus frame. WebSocket handles message framing,
    so one binary WebSocket message = one Opus frame.

    Args:
        wav_chunks: Async iterator yielding float32 mono waveform chunks.
        sample_rate: Sample rate of input audio (will be resampled to 48kHz for Opus).
        opus_config: Opus encoding configuration.

    Yields:
        Raw Opus-encoded audio frames (no framing prefix).
    """
    pcm_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=8)
    opus_q: queue.Queue[bytes | None] = queue.Queue(maxsize=8)
    stop_evt = threading.Event()
    thread_exc: list[BaseException] = []

    def encoder_thread() -> None:
        try:
            import opuslib

            # Opus supports 8, 12, 16, 24, 48 kHz - use 48kHz for best quality
            opus_sample_rate = 48000
            encoder = opuslib.Encoder(opus_sample_rate, 1, opuslib.APPLICATION_AUDIO)
            encoder.bitrate = opus_config.bitrate

            # Frame size in samples at 48kHz for the given frame duration
            frame_size = int(opus_sample_rate * opus_config.frame_ms / 1000)

            # Buffer for accumulating samples when input rate differs from Opus rate
            resample_buffer: list[np.ndarray] = []
            encoded_any = False

            # Simple linear resampler for rate conversion
            def resample_chunk(chunk: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
                if in_rate == out_rate:
                    return chunk
                duration = len(chunk) / in_rate
                out_samples = int(duration * out_rate)
                indices = np.linspace(0, len(chunk) - 1, out_samples)
                return np.interp(indices, np.arange(len(chunk)), chunk).astype(np.float32)

            while True:
                item = pcm_q.get()
                if item is None or stop_evt.is_set():
                    break

                # Resample if needed
                if sample_rate != opus_sample_rate:
                    item = resample_chunk(item, sample_rate, opus_sample_rate)

                resample_buffer.append(item)
                all_samples = np.concatenate(resample_buffer)

                # Process complete frames
                pos = 0
                while pos + frame_size <= len(all_samples):
                    frame = all_samples[pos : pos + frame_size]
                    pcm_bytes = float32_to_s16le_bytes(frame)

                    t0 = time.perf_counter()
                    opus_data = encoder.encode(pcm_bytes, frame_size)
                    encoded_any = True
                    AUDIO_ENCODE_SECONDS.observe(time.perf_counter() - t0)

                    if opus_data:
                        opus_q.put(opus_data)

                    pos += frame_size

                # Keep remaining samples for next iteration
                if pos < len(all_samples):
                    resample_buffer = [all_samples[pos:]]
                else:
                    resample_buffer = []

            # Flush remaining samples (pad with zeros if needed)
            if resample_buffer and not stop_evt.is_set():
                remaining = np.concatenate(resample_buffer)
                if len(remaining) > 0:
                    # Pad to frame size
                    if len(remaining) < frame_size:
                        remaining = np.pad(remaining, (0, frame_size - len(remaining)))
                    pcm_bytes = float32_to_s16le_bytes(remaining[:frame_size])

                    t0 = time.perf_counter()
                    opus_data = encoder.encode(pcm_bytes, frame_size)
                    encoded_any = True
                    AUDIO_ENCODE_SECONDS.observe(time.perf_counter() - t0)

                    if opus_data:
                        opus_q.put(opus_data)

            opus_q.put(None)

            # opuslib encoder doesn't have a flush method like lameenc
            _ = encoded_any  # Silence unused variable warning

        except BaseException as e:
            AUDIO_ENCODE_FAILURES_TOTAL.inc()
            thread_exc.append(e)
            stop_evt.set()
            try:
                opus_q.put(None)
            except Exception:
                pass

    enc_thread = threading.Thread(target=encoder_thread, name="opus-encoder", daemon=True)
    enc_thread.start()

    async def pcm_producer() -> None:
        try:
            async for chunk in wav_chunks:
                if stop_evt.is_set():
                    break
                while not stop_evt.is_set():
                    try:
                        pcm_q.put_nowait(chunk)
                        break
                    except queue.Full:
                        await asyncio.sleep(0.01)
        finally:
            try:
                while not stop_evt.is_set():
                    try:
                        pcm_q.put_nowait(None)
                        break
                    except queue.Full:
                        await asyncio.sleep(0.01)
            except Exception:
                pass

    producer_task = asyncio.create_task(pcm_producer())

    try:
        while True:
            try:
                item = opus_q.get_nowait()
                if item is None:
                    break
                yield item
            except queue.Empty:
                await asyncio.sleep(0.005)
        if thread_exc:
            raise RuntimeError(f"Opus encoder failed: {thread_exc[0]}") from thread_exc[0]
    finally:
        stop_evt.set()
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        try:
            await asyncio.to_thread(pcm_q.put, None)
        except Exception:
            pass
        await asyncio.to_thread(enc_thread.join, 2.0)


async def encode_opus_single(
    wav: np.ndarray,
    sample_rate: int,
    opus_config: OpusConfig,
) -> bytes:
    """Encode a complete waveform to Opus in one call.

    Args:
        wav: Float32 mono waveform array.
        sample_rate: Sample rate of input audio.
        opus_config: Opus encoding configuration.

    Returns:
        Concatenated Opus frames.
    """
    chunks: list[bytes] = []

    async def single_chunk() -> AsyncIterator[np.ndarray]:
        yield wav

    async for frame in stream_opus(
        wav_chunks=single_chunk(),
        sample_rate=sample_rate,
        opus_config=opus_config,
    ):
        chunks.append(frame)

    return b"".join(chunks)
