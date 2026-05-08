import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")

import numpy as np

DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))


class _DummyHistogram:
    def __init__(self):
        self.observations: list[float] = []

    def observe(self, v: float) -> None:
        self.observations.append(float(v))

    def labels(self, **kwargs):
        return self


class _DummyCounter:
    def __init__(self):
        self.count = 0

    def inc(self) -> None:
        self.count += 1


class FakeASRClient:
    """Fake ASR client for testing."""

    def __init__(self, transcription: str = "Hello, how are you?"):
        self.transcription = transcription
        self.transcribe_called = False

    async def transcribe(self, audio_bytes: bytes, audio_format: str = "wav") -> str:
        self.transcribe_called = True
        return self.transcription


class FakeLLMClient:
    """Fake LLM client for testing."""

    def __init__(self, response_chunks: list[str] | None = None):
        self.response_chunks = response_chunks or ["Hello! ", "How can I help you today?"]
        self.stream_called = False

    async def stream(self, user_input: str, prompt_template_override: str | None = None) -> AsyncIterator[str]:
        self.stream_called = True
        for chunk in self.response_chunks:
            yield chunk


class FakeVoxCPMServer:
    """Fake VoxCPM server for testing."""

    def __init__(self):
        self.generate_called = False
        self.generated_texts: list[str] = []

    async def generate(self, target_text: str, **kwargs):
        self.generate_called = True
        self.generated_texts.append(target_text)
        # Yield fake audio chunks
        yield np.zeros((160,), dtype=np.float32)
        yield np.ones((160,), dtype=np.float32) * 0.5


class FakeWebSocket:
    """Fake WebSocket for testing."""

    def __init__(self):
        self.sent_json: list[dict] = []
        self.sent_bytes: list[bytes] = []

    async def send_json(self, data: dict) -> None:
        self.sent_json.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)


@pytest.fixture
def mock_metrics(monkeypatch):
    """Mock metrics to avoid Prometheus registration issues."""
    import app.services.pipeline as pipeline
    import app.services.asr as asr
    import app.services.llm as llm

    dummy_hist = _DummyHistogram()
    monkeypatch.setattr(pipeline, "PIPELINE_DURATION_SECONDS", dummy_hist)
    monkeypatch.setattr(pipeline, "PIPELINE_STAGE_DURATION_SECONDS", dummy_hist)
    monkeypatch.setattr(asr, "PIPELINE_STAGE_DURATION_SECONDS", dummy_hist)
    monkeypatch.setattr(llm, "PIPELINE_STAGE_DURATION_SECONDS", dummy_hist)

    return dummy_hist


@pytest.fixture
def mock_opus(monkeypatch):
    """Mock Opus encoder to avoid requiring opuslib."""
    import app.services.pipeline as pipeline

    async def fake_stream_opus(*, wav_chunks, sample_rate, opus_config):
        async for chunk in wav_chunks:
            # Just yield some fake bytes
            yield b"\x00\x10" + b"\x00" * 16  # Length header + data

    monkeypatch.setattr(pipeline, "stream_opus", fake_stream_opus)


def test_pipeline_orchestrator_runs_full_pipeline(mock_metrics, mock_opus):
    from app.core.config import OpusConfig
    from app.services.pipeline import PipelineOrchestrator

    websocket = FakeWebSocket()
    asr_client = FakeASRClient("Test transcription")
    llm_client = FakeLLMClient(["Hello.", " How are you?"])
    voxcpm_server = FakeVoxCPMServer()

    pipeline = PipelineOrchestrator(
        websocket=websocket,
        voxcpm_server=voxcpm_server,
        asr_client=asr_client,
        llm_client=llm_client,
        opus_config=OpusConfig(bitrate=64000, frame_ms=20),
        sample_rate=16000,
    )

    async def run():
        await pipeline.run(b"fake audio bytes")

    asyncio.run(run())

    # Verify ASR was called
    assert asr_client.transcribe_called

    # Verify LLM was called
    assert llm_client.stream_called

    # Verify TTS was called
    assert voxcpm_server.generate_called

    # Verify we sent audio bytes
    assert len(websocket.sent_bytes) > 0

    # Verify completion message was sent
    assert {"type": "complete"} in websocket.sent_json


def test_pipeline_orchestrator_sentence_boundary_detection(mock_metrics, mock_opus):
    from app.core.config import OpusConfig
    from app.services.pipeline import PipelineOrchestrator

    websocket = FakeWebSocket()
    asr_client = FakeASRClient("Test")

    # Response with clear sentence boundaries
    llm_client = FakeLLMClient([
        "First sentence.",
        " Second sentence!",
        " Third sentence?",
        " Final text"
    ])
    voxcpm_server = FakeVoxCPMServer()

    pipeline = PipelineOrchestrator(
        websocket=websocket,
        voxcpm_server=voxcpm_server,
        asr_client=asr_client,
        llm_client=llm_client,
        opus_config=OpusConfig(bitrate=64000, frame_ms=20),
        sample_rate=16000,
    )

    async def run():
        await pipeline.run(b"fake audio bytes")

    asyncio.run(run())

    # Should have generated TTS for each sentence/segment
    # The exact number depends on how sentence boundaries are detected
    assert voxcpm_server.generate_called
    assert len(voxcpm_server.generated_texts) >= 3  # At least 3 sentences


def test_pipeline_orchestrator_handles_empty_transcription(mock_metrics, mock_opus):
    from app.core.config import OpusConfig
    from app.services.pipeline import PipelineOrchestrator

    websocket = FakeWebSocket()
    asr_client = FakeASRClient("")  # Empty transcription
    llm_client = FakeLLMClient()
    voxcpm_server = FakeVoxCPMServer()

    pipeline = PipelineOrchestrator(
        websocket=websocket,
        voxcpm_server=voxcpm_server,
        asr_client=asr_client,
        llm_client=llm_client,
        opus_config=OpusConfig(bitrate=64000, frame_ms=20),
        sample_rate=16000,
    )

    async def run():
        await pipeline.run(b"fake audio bytes")

    asyncio.run(run())

    # Should send error message for empty transcription
    error_msgs = [m for m in websocket.sent_json if m.get("type") == "error"]
    assert len(error_msgs) == 1
    assert "No speech detected" in error_msgs[0]["error"]

    # LLM should not be called
    assert not llm_client.stream_called


def test_sentence_endings_constant():
    from app.services.pipeline import SENTENCE_ENDINGS

    # Verify expected sentence endings are present
    assert "." in SENTENCE_ENDINGS
    assert "!" in SENTENCE_ENDINGS
    assert "?" in SENTENCE_ENDINGS
    assert "\n" in SENTENCE_ENDINGS
