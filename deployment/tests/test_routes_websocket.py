import asyncio
import base64
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")

import numpy as np

DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))


class _DummyGauge:
    def __init__(self):
        self.value = 0

    def inc(self) -> None:
        self.value += 1

    def dec(self) -> None:
        self.value -= 1


class _DummyCounter:
    def __init__(self):
        self.counts: dict[tuple, int] = {}

    def labels(self, **kwargs) -> "_DummyCounter":
        self._current_labels = tuple(sorted(kwargs.items()))
        return self

    def inc(self) -> None:
        key = getattr(self, "_current_labels", ())
        self.counts[key] = self.counts.get(key, 0) + 1


def test_websocket_schemas_parse_correctly():
    from app.schemas.websocket import (
        AudioChunkMessage,
        AudioEndMessage,
        TerminateMessage,
        parse_client_message,
    )

    # Test audio_chunk parsing
    msg = parse_client_message({
        "type": "audio_chunk",
        "data": base64.b64encode(b"test audio").decode(),
        "format": "wav"
    })
    assert isinstance(msg, AudioChunkMessage)
    assert msg.type == "audio_chunk"
    assert msg.format == "wav"

    # Test audio_end parsing
    msg = parse_client_message({"type": "audio_end"})
    assert isinstance(msg, AudioEndMessage)

    # Test terminate parsing
    msg = parse_client_message({"type": "terminate"})
    assert isinstance(msg, TerminateMessage)


def test_websocket_schemas_invalid_type_raises():
    from app.schemas.websocket import parse_client_message

    with pytest.raises(ValueError, match="Unknown message type"):
        parse_client_message({"type": "invalid"})


def test_websocket_schemas_missing_fields_raises():
    from app.schemas.websocket import parse_client_message
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        parse_client_message({"type": "audio_chunk"})  # Missing 'data'


@pytest.fixture
def mock_websocket_metrics(monkeypatch):
    """Mock WebSocket metrics."""
    import app.api.routes.websocket as ws_route

    gauge = _DummyGauge()
    counter = _DummyCounter()

    monkeypatch.setattr(ws_route, "WEBSOCKET_CONNECTIONS_ACTIVE", gauge)
    monkeypatch.setattr(ws_route, "WEBSOCKET_CONNECTIONS_TOTAL", counter)
    monkeypatch.setattr(ws_route, "WEBSOCKET_MESSAGES_TOTAL", counter)

    return {"gauge": gauge, "counter": counter}


@pytest.fixture
def mock_pipeline_metrics(monkeypatch):
    """Mock pipeline metrics."""
    import app.services.pipeline as pipeline
    import app.services.asr as asr
    import app.services.llm as llm

    class DummyHist:
        def observe(self, v): pass
        def labels(self, **kwargs): return self

    hist = DummyHist()
    monkeypatch.setattr(pipeline, "PIPELINE_DURATION_SECONDS", hist)
    monkeypatch.setattr(pipeline, "PIPELINE_STAGE_DURATION_SECONDS", hist)
    monkeypatch.setattr(asr, "PIPELINE_STAGE_DURATION_SECONDS", hist)
    monkeypatch.setattr(llm, "PIPELINE_STAGE_DURATION_SECONDS", hist)


def test_websocket_endpoint_rejects_missing_config(mock_websocket_metrics):
    """Test that WebSocket rejects connection when config is missing."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.routes.websocket import router

    app = FastAPI()
    app.include_router(router)
    # Don't set app.state.cfg

    client = TestClient(app)

    # WebSocket connection should be closed with error
    with pytest.raises(Exception):
        with client.websocket_connect("/ws"):
            pass


def test_websocket_endpoint_rejects_missing_api_key(mock_websocket_metrics):
    """Test that WebSocket rejects connection when GEMINI_API_KEY is not set."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.routes.websocket import router
    from app.core.config import OpusConfig, ASRConfig, LLMConfig

    app = FastAPI()
    app.include_router(router)

    # Set config with empty API key
    class FakeConfig:
        opus = OpusConfig(bitrate=64000, frame_ms=20)
        asr = ASRConfig(api_url="http://localhost:8001", model=None, timeout=30.0, api_key="EMPTY")
        llm = LLMConfig(api_key="", model="gemini-2.5-flash", timeout=60.0, prompt_template="test")  # Empty!

    app.state.cfg = FakeConfig()

    # Mock server dependency
    async def fake_get_server():
        return MagicMock()

    from app.api import deps
    app.dependency_overrides[deps.get_server_ws] = fake_get_server

    client = TestClient(app)

    # WebSocket connection should be closed with error
    with pytest.raises(Exception):
        with client.websocket_connect("/ws"):
            pass


def test_audio_chunk_message_default_format():
    from app.schemas.websocket import AudioChunkMessage

    msg = AudioChunkMessage(data="dGVzdA==")  # base64 for "test"
    assert msg.format == "wav"  # Default format


def test_complete_message_structure():
    from app.schemas.websocket import CompleteMessage

    msg = CompleteMessage()
    assert msg.type == "complete"
    assert msg.model_dump() == {"type": "complete"}


def test_error_message_structure():
    from app.schemas.websocket import ErrorMessage

    msg = ErrorMessage(error="Something went wrong")
    assert msg.type == "error"
    assert msg.error == "Something went wrong"
