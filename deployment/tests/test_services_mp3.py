import asyncio
import concurrent.futures
import sys
import threading
import types
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("lameenc")

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


class _FakeRequest:
    def __init__(self, disconnect_after_checks: int | None = None):
        self._checks = 0
        self._disconnect_after = disconnect_after_checks

    async def is_disconnected(self) -> bool:
        self._checks += 1
        if self._disconnect_after is None:
            return False
        return self._checks > self._disconnect_after


async def _agen(chunks: list[np.ndarray]):
    for c in chunks:
        yield c


def _looks_like_mp3(data: bytes) -> bool:
    # Some encoders may emit an ID3 header; others start directly with MPEG frames.
    if data.startswith(b"ID3"):
        return True
    for i in range(len(data) - 1):
        b0 = data[i]
        b1 = data[i + 1]
        if b0 == 0xFF and (b1 & 0xE0) == 0xE0:
            return True
    return False


def test_float32_to_s16le_bytes_clips_and_converts():
    from app.services.mp3 import float32_to_s16le_bytes

    wav = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    b = float32_to_s16le_bytes(wav)
    arr = np.frombuffer(b, dtype=np.int16)
    assert arr.tolist() == [-32767, -32767, -16383, 0, 16383, 32767, 32767]


def test_stream_pcm_emits_raw_s16le():
    import app.services.mp3 as mp3

    chunks = [
        np.array([0.0, 0.5, -0.5], dtype=np.float32),
        np.array([1.0, -1.0], dtype=np.float32),
    ]

    async def run() -> bytes:
        out: list[bytes] = []
        async for b in mp3.stream_pcm(request=_FakeRequest(), wav_chunks=_agen(chunks)):
            out.append(b)
        return b"".join(out)

    data = asyncio.run(run())
    arr = np.frombuffer(data, dtype=np.int16)
    # Same s16le conversion stream_mp3 feeds into lameenc, emitted directly.
    assert arr.tolist() == [0, 16383, -16383, 32767, -32767]


def test_stream_pcm_stops_on_disconnect():
    import app.services.mp3 as mp3

    chunks = [np.zeros((4,), dtype=np.float32) for _ in range(4)]

    async def run() -> bytes:
        out: list[bytes] = []
        async for b in mp3.stream_pcm(
            request=_FakeRequest(disconnect_after_checks=0),
            wav_chunks=_agen(chunks),
        ):
            out.append(b)
        return b"".join(out)

    data = asyncio.run(asyncio.wait_for(run(), timeout=2.0))
    assert data == b""


def test_stream_mp3_happy_path_encodes_and_flushes(monkeypatch):
    from app.core.config import Mp3Config
    import app.services.mp3 as mp3

    dummy_hist = _DummyHistogram()
    dummy_ctr = _DummyCounter()
    monkeypatch.setattr(mp3, "AUDIO_ENCODE_SECONDS", dummy_hist)
    monkeypatch.setattr(mp3, "AUDIO_ENCODE_FAILURES_TOTAL", dummy_ctr)

    # Ensure enough PCM data for a real MP3 encoder to output frames.
    chunks = [
        np.zeros((16000,), dtype=np.float32),
        np.ones((16000,), dtype=np.float32) * 0.25,
    ]

    async def run() -> bytes:
        out = []
        async for b in mp3.stream_mp3(
            request=_FakeRequest(),
            wav_chunks=_agen(chunks),
            sample_rate=16000,
            mp3=Mp3Config(bitrate_kbps=192, quality=2),
        ):
            out.append(b)
        return b"".join(out)

    data = asyncio.run(run())
    assert data
    assert _looks_like_mp3(data)
    assert dummy_ctr.count == 0
    assert dummy_hist.observations  # observed per chunk


def test_stream_mp3_stops_on_disconnect(monkeypatch):
    from app.core.config import Mp3Config
    import app.services.mp3 as mp3

    monkeypatch.setattr(mp3, "AUDIO_ENCODE_SECONDS", _DummyHistogram())
    monkeypatch.setattr(mp3, "AUDIO_ENCODE_FAILURES_TOTAL", _DummyCounter())

    chunks = [np.zeros((8000,), dtype=np.float32) for _ in range(4)]

    async def run() -> bytes:
        out: list[bytes] = []
        async for b in mp3.stream_mp3(
            request=_FakeRequest(disconnect_after_checks=0),
            wav_chunks=_agen(chunks),
            sample_rate=16000,
            mp3=Mp3Config(bitrate_kbps=128, quality=2),
        ):
            out.append(b)
        return b"".join(out)

    data = asyncio.run(asyncio.wait_for(run(), timeout=2.0))
    # Disconnect can happen before any PCM is enqueued; we just require termination.
    assert data == b"" or _looks_like_mp3(data)


def test_stream_mp3_raises_if_pcm_conversion_fails(monkeypatch):
    from app.core.config import Mp3Config
    import app.services.mp3 as mp3

    dummy_ctr = _DummyCounter()
    monkeypatch.setattr(mp3, "AUDIO_ENCODE_SECONDS", _DummyHistogram())
    monkeypatch.setattr(mp3, "AUDIO_ENCODE_FAILURES_TOTAL", dummy_ctr)

    def boom(_: np.ndarray) -> bytes:
        raise ValueError("boom")

    monkeypatch.setattr(mp3, "float32_to_s16le_bytes", boom)

    chunks = [np.zeros((10,), dtype=np.float32)]

    async def run() -> None:
        async for _ in mp3.stream_mp3(
            request=_FakeRequest(),
            wav_chunks=_agen(chunks),
            sample_rate=16000,
            mp3=Mp3Config(bitrate_kbps=128, quality=2),
        ):
            pass

    with pytest.raises(RuntimeError, match="MP3 encoder failed"):
        asyncio.run(run())
    assert dummy_ctr.count == 1


def test_stream_mp3_survives_default_executor_starvation(monkeypatch):
    from app.core.config import Mp3Config
    import app.services.mp3 as mp3

    class FakeEncoder:
        def set_bit_rate(self, value):
            pass

        def set_in_sample_rate(self, value):
            pass

        def set_channels(self, value):
            pass

        def set_quality(self, value):
            pass

        def encode(self, pcm_bytes):
            return b"frame" if pcm_bytes else b""

        def flush(self):
            return b"flush"

    monkeypatch.setitem(sys.modules, "lameenc", types.SimpleNamespace(Encoder=FakeEncoder))
    monkeypatch.setattr(mp3, "AUDIO_ENCODE_SECONDS", _DummyHistogram())
    monkeypatch.setattr(mp3, "AUDIO_ENCODE_FAILURES_TOTAL", _DummyCounter())

    chunks = [np.zeros((128,), dtype=np.float32) for _ in range(2)]

    async def run() -> bytes:
        loop = asyncio.get_running_loop()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(executor)
        unblock_executor = threading.Event()
        executor_blocker = loop.run_in_executor(executor, unblock_executor.wait)
        await asyncio.sleep(0)

        try:
            out: list[bytes] = []
            async for b in mp3.stream_mp3(
                request=_FakeRequest(),
                wav_chunks=_agen(chunks),
                sample_rate=16000,
                mp3=Mp3Config(bitrate_kbps=128, quality=2),
            ):
                out.append(b)
            return b"".join(out)
        finally:
            unblock_executor.set()
            await executor_blocker
            executor.shutdown(wait=True)

    data = asyncio.run(asyncio.wait_for(run(), timeout=1.0))
    assert data == b"frameframeflush"


def test_stream_mp3_close_cleans_up_threads(monkeypatch):
    from app.core.config import Mp3Config
    import app.services.mp3 as mp3

    class FakeEncoder:
        def set_bit_rate(self, value):
            pass

        def set_in_sample_rate(self, value):
            pass

        def set_channels(self, value):
            pass

        def set_quality(self, value):
            pass

        def encode(self, pcm_bytes):
            return b"frame" if pcm_bytes else b""

        def flush(self):
            return b"flush"

    def live_mp3_threads() -> set[int | None]:
        return {
            thread.ident
            for thread in threading.enumerate()
            if thread.is_alive() and (thread.name == "mp3-encoder" or thread.name.startswith("mp3-queue"))
        }

    monkeypatch.setitem(sys.modules, "lameenc", types.SimpleNamespace(Encoder=FakeEncoder))
    monkeypatch.setattr(mp3, "AUDIO_ENCODE_SECONDS", _DummyHistogram())
    monkeypatch.setattr(mp3, "AUDIO_ENCODE_FAILURES_TOTAL", _DummyCounter())

    chunks = [np.zeros((128,), dtype=np.float32) for _ in range(32)]
    before = live_mp3_threads()

    async def run() -> bytes:
        stream = mp3.stream_mp3(
            request=_FakeRequest(),
            wav_chunks=_agen(chunks),
            sample_rate=16000,
            mp3=Mp3Config(bitrate_kbps=128, quality=2),
        )
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    assert asyncio.run(asyncio.wait_for(run(), timeout=1.0)) == b"frame"
    assert live_mp3_threads() == before
