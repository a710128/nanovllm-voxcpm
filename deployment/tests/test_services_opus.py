import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")

# Try to import opuslib; skip tests if not available
opuslib = pytest.importorskip("opuslib")

import numpy as np

DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))


class _DummyHistogram:
    def __init__(self):
        self.observations: list[float] = []

    def observe(self, v: float) -> None:
        self.observations.append(float(v))


class _DummyCounter:
    def __init__(self):
        self.count = 0

    def inc(self) -> None:
        self.count += 1


async def _agen(chunks: list[np.ndarray]):
    for c in chunks:
        yield c


def test_float32_to_s16le_bytes_clips_and_converts():
    from app.services.opus import float32_to_s16le_bytes

    wav = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    b = float32_to_s16le_bytes(wav)
    arr = np.frombuffer(b, dtype=np.int16)
    assert arr.tolist() == [-32767, -32767, -16383, 0, 16383, 32767, 32767]


def test_stream_opus_happy_path_encodes(monkeypatch):
    from app.core.config import OpusConfig
    import app.services.opus as opus

    dummy_hist = _DummyHistogram()
    dummy_ctr = _DummyCounter()
    monkeypatch.setattr(opus, "AUDIO_ENCODE_SECONDS", dummy_hist)
    monkeypatch.setattr(opus, "AUDIO_ENCODE_FAILURES_TOTAL", dummy_ctr)

    # Ensure enough PCM data for Opus encoder to output frames
    chunks = [
        np.zeros((16000,), dtype=np.float32),  # 1 second
        np.ones((16000,), dtype=np.float32) * 0.25,  # 1 second
    ]

    async def run() -> list[bytes]:
        out: list[bytes] = []
        async for frame in opus.stream_opus(
            wav_chunks=_agen(chunks),
            sample_rate=16000,
            opus_config=OpusConfig(bitrate=64000, frame_ms=20),
        ):
            out.append(frame)
        return out

    frames = asyncio.run(run())
    # Each yielded item is a raw Opus frame (no length prefix)
    assert len(frames) > 10
    # Verify each frame is decodable Opus data
    decoder = opuslib.Decoder(48000, 1)
    for frame in frames[:3]:
        pcm = decoder.decode(frame, 960)
        assert len(pcm) == 960 * 2  # 960 samples * 2 bytes (16-bit)
    assert dummy_ctr.count == 0
    assert dummy_hist.observations  # observed per frame


def test_stream_opus_different_frame_sizes(monkeypatch):
    from app.core.config import OpusConfig
    import app.services.opus as opus

    monkeypatch.setattr(opus, "AUDIO_ENCODE_SECONDS", _DummyHistogram())
    monkeypatch.setattr(opus, "AUDIO_ENCODE_FAILURES_TOTAL", _DummyCounter())

    chunks = [np.random.randn(24000).astype(np.float32)]  # 1 second at 24kHz

    for frame_ms in [10, 20, 40]:
        async def run() -> list[bytes]:
            out: list[bytes] = []
            async for b in opus.stream_opus(
                wav_chunks=_agen(chunks),
                sample_rate=24000,
                opus_config=OpusConfig(bitrate=48000, frame_ms=frame_ms),
            ):
                out.append(b)
            return out

        frames = asyncio.run(run())
        assert frames, f"No output for frame_ms={frame_ms}"


def test_stream_opus_resamples_from_different_rates(monkeypatch):
    from app.core.config import OpusConfig
    import app.services.opus as opus

    monkeypatch.setattr(opus, "AUDIO_ENCODE_SECONDS", _DummyHistogram())
    monkeypatch.setattr(opus, "AUDIO_ENCODE_FAILURES_TOTAL", _DummyCounter())

    # Test resampling from various sample rates
    for rate in [8000, 16000, 22050, 44100, 48000]:
        chunks = [np.random.randn(rate).astype(np.float32)]  # 1 second

        async def run() -> list[bytes]:
            out: list[bytes] = []
            async for b in opus.stream_opus(
                wav_chunks=_agen(chunks),
                sample_rate=rate,
                opus_config=OpusConfig(bitrate=64000, frame_ms=20),
            ):
                out.append(b)
            return out

        frames = asyncio.run(run())
        assert frames, f"No output for sample_rate={rate}"


def test_encode_opus_single_convenience_function(monkeypatch):
    from app.core.config import OpusConfig
    import app.services.opus as opus

    monkeypatch.setattr(opus, "AUDIO_ENCODE_SECONDS", _DummyHistogram())
    monkeypatch.setattr(opus, "AUDIO_ENCODE_FAILURES_TOTAL", _DummyCounter())

    wav = np.random.randn(16000).astype(np.float32)

    async def run() -> bytes:
        return await opus.encode_opus_single(
            wav=wav,
            sample_rate=16000,
            opus_config=OpusConfig(bitrate=64000, frame_ms=20),
        )

    data = asyncio.run(run())
    assert data
    assert len(data) > 10
