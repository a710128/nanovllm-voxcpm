from __future__ import annotations

import asyncio
import base64
import logging
import time

from openai import AsyncOpenAI

from app.core.config import ASRConfig
from app.core.metrics import PIPELINE_STAGE_DURATION_SECONDS

logger = logging.getLogger(__name__)


class ASRClient:
    """Async client for OpenAI-compatible multimodal ASR API (e.g., vLLM with Qwen3-ASR)."""

    def __init__(self, config: ASRConfig) -> None:
        self.model = config.model or "Qwen/Qwen3-ASR-1.7B"
        self.timeout = config.timeout
        self.client = AsyncOpenAI(
            base_url=f"{config.api_url.rstrip('/')}/v1",
            api_key=config.api_key or "EMPTY",
        )

    async def transcribe(self, audio_bytes: bytes, audio_format: str = "wav") -> str:
        """Transcribe a single audio chunk to text.

        Each chunk must be a self-contained audio file (e.g. a complete WAV
        with RIFF header).

        Args:
            audio_bytes: Audio data (WAV, MP3, etc.) — one complete file.
            audio_format: Audio format hint (default "wav").

        Returns:
            Transcribed text from the audio.
        """
        t0 = time.perf_counter()

        try:
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            audio_data_url = f"data:audio/{audio_format};base64,{audio_b64}"

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                            {"type": "text", "text": "Transcribe arabic audio only:"},
                        ],
                    }
                ],
                stream=True,
                timeout=self.timeout
              
            )

            transcription_parts: list[str] = []
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    transcription_parts.append(chunk.choices[0].delta.content)

            full_transcription = "".join(transcription_parts).strip()
            if "<asr_text>" in full_transcription:
                full_transcription = full_transcription.split("<asr_text>")[-1]
                print(full_transcription.strip())
            return full_transcription.strip()

        except Exception as e:
            raise RuntimeError(f"ASR transcription failed: {e}") from e
        finally:
            PIPELINE_STAGE_DURATION_SECONDS.labels(stage="asr").observe(time.perf_counter() - t0)

    def create_session(self) -> StreamingASRSession:
        """Create a new streaming ASR session.

        Returns a session that accepts audio chunks one at a time, transcribes
        each concurrently, and concatenates the results in order when finished.
        """
        return StreamingASRSession(self)


class StreamingASRSession:
    """Manages streaming transcription of audio chunks arriving over time.

    Each chunk is dispatched to the ASR backend immediately as an independent
    transcription task.  ``finish()`` waits for all pending tasks and returns
    the concatenated transcription in chunk order.
    """

    def __init__(self, client: ASRClient) -> None:
        self._client = client
        self._tasks: list[asyncio.Task[str]] = []
        self._chunk_count = 0

    def push_chunk(self, audio_bytes: bytes, audio_format: str = "wav") -> None:
        """Submit an audio chunk for transcription.

        The chunk is transcribed in the background immediately.  Each chunk
        must be a self-contained audio file (e.g. a complete WAV).

        Args:
            audio_bytes: Audio data for this chunk.
            audio_format: Audio format hint (default "wav").
        """
        idx = self._chunk_count
        self._chunk_count += 1

        async def _transcribe_chunk() -> str:
            logger.debug("ASR streaming: transcribing chunk %d (%d bytes)", idx, len(audio_bytes))
            return await self._client.transcribe(audio_bytes, audio_format)

        self._tasks.append(asyncio.create_task(_transcribe_chunk()))

    @property
    def chunk_count(self) -> int:
        """Number of chunks submitted so far."""
        return self._chunk_count

    async def finish(self) -> str:
        """Wait for all pending transcription tasks and return the full text.

        Partial transcriptions are concatenated in the order the chunks were
        submitted (preserving temporal order of the audio).

        Returns:
            The complete transcription, or empty string if no chunks were submitted.

        Raises:
            RuntimeError: If any chunk transcription failed.
        """
        if not self._tasks:
            return ""

        t0 = time.perf_counter()
        try:
            results = await asyncio.gather(*self._tasks, return_exceptions=True)

            parts: list[str] = []
            errors: list[str] = []
            for i, result in enumerate(results):
                if isinstance(result, BaseException):
                    errors.append(f"chunk {i}: {result}")
                elif result:
                    parts.append(result)

            if errors:
                logger.warning("ASR streaming: %d/%d chunks failed: %s", len(errors), len(results), "; ".join(errors))
                # If ALL chunks failed, raise
                if not parts:
                    raise RuntimeError(f"All ASR chunks failed: {'; '.join(errors)}")

            transcription = " ".join(parts)
            logger.info(
                "ASR streaming: %d chunks -> %d chars in %.2fs",
                len(results), len(transcription), time.perf_counter() - t0,
            )
            return transcription.strip()
        finally:
            self._tasks.clear()
