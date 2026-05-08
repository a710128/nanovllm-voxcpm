from __future__ import annotations

import asyncio
import inspect
import time
from typing import TYPE_CHECKING, Any

from app.core.metrics import PIPELINE_DURATION_SECONDS, PIPELINE_STAGE_DURATION_SECONDS, WS_PIPELINE_LATENCY_SECONDS
from app.services.asr import ASRClient
from app.services.llm import LLMClient
from app.services.opus import stream_opus

if TYPE_CHECKING:
    from fastapi import WebSocket

    from app.core.config import OpusConfig

# Sentence boundary characters that trigger immediate TTS
SENTENCE_ENDINGS = frozenset({".", "!", "?", "\n"})


class PipelineOrchestrator:
    """Orchestrates the ASR -> LLM -> TTS pipeline with sentence-boundary streaming.

    This class coordinates the full voice-to-voice flow:
    1. ASR: Transcribe audio to text
    2. LLM: Generate response (streaming)
    3. TTS: Convert response to audio (streaming, triggered on sentence boundaries)

    The key optimization is that TTS starts generating audio for completed sentences
    while the LLM is still producing subsequent sentences, reducing end-to-end latency.
    """

    def __init__(
        self,
        websocket: WebSocket,
        voxcpm_server: Any,
        asr_client: ASRClient,
        llm_client: LLMClient,
        opus_config: OpusConfig,
        sample_rate: int,
        tts_prompt_latents: bytes | None = None,
        tts_prompt_text: str = "",
        tts_ref_audio_latents: bytes | None = None,
        llm_prompt_template: str | None = None,
    ) -> None:
        self.websocket = websocket
        self.voxcpm_server = voxcpm_server
        self.asr_client = asr_client
        self.llm_client = llm_client
        self.opus_config = opus_config
        self.sample_rate = sample_rate
        self.tts_prompt_latents = tts_prompt_latents
        self.tts_prompt_text = tts_prompt_text
        self.tts_ref_audio_latents = tts_ref_audio_latents
        self.llm_prompt_template = llm_prompt_template
        self._interrupt_event = asyncio.Event()

    def interrupt(self) -> None:
        """Interrupt the ongoing pipeline generation."""
        self._interrupt_event.set()

    def reset_interrupt(self) -> None:
        """Reset the interrupt event for a new pipeline generation."""
        self._interrupt_event.clear()

    async def run_with_transcription(self, transcription: str) -> None:
        """Run the backend portion of the voice pipeline (LLM -> TTS).

        The ASR stage is expected to be handled upstream so chunks can be
        streamed to ASR as they arrive.

        Args:
            transcription: The completed text from the ASR step.
        """
        pipeline_start = time.perf_counter()

        try:
            if not transcription.strip():
                try:
                    await self.websocket.send_json({"type": "error", "error": "No speech detected"})
                except Exception:
                    pass
                return

            # Stage 2 & 3: LLM + TTS with sentence-boundary streaming
            await self._stream_llm_to_tts(transcription)

            # Send completion message
            try:
                await self.websocket.send_json({"type": "complete"})
            except Exception:
                pass

        finally:
            PIPELINE_DURATION_SECONDS.observe(time.perf_counter() - pipeline_start)

    async def _stream_llm_to_tts(self, transcription: str) -> None:
        """Stream LLM output to TTS with sentence-boundary detection.

        Sentences are sent to TTS as soon as they are complete, allowing
        audio generation to overlap with LLM generation.

        Args:
            transcription: The transcribed user speech.
        """
        # Queue for sentences ready to be spoken
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
        tts_error: list[Exception] = []
        first_audio_sent = False
        latency_start = time.perf_counter()

        async def tts_worker() -> None:
            """Consume sentences from queue and stream audio to client."""
            nonlocal first_audio_sent
            try:
                while True:
                    sentence = await tts_queue.get()
                    if sentence is None:  # Poison pill signals end
                        break

                    t0 = time.perf_counter()
                    try:
                        await self._generate_and_stream_audio(sentence, latency_start if not first_audio_sent else None)
                        first_audio_sent = True
                    finally:
                        PIPELINE_STAGE_DURATION_SECONDS.labels(stage="tts").observe(time.perf_counter() - t0)

            except Exception as e:
                tts_error.append(e)

        # Start TTS worker task
        tts_task = asyncio.create_task(tts_worker())

        try:
            text_buffer = ""

            async for chunk in self.llm_client.stream(transcription, self.llm_prompt_template, interrupt_event=self._interrupt_event):
                if self._interrupt_event.is_set():
                    break
                
                text_buffer += chunk

                # Check for sentence boundary
                if text_buffer and text_buffer.rstrip()[-1:] in SENTENCE_ENDINGS:
                    sentence = text_buffer.strip()
                    if sentence:
                        await tts_queue.put(sentence)
                    text_buffer = ""

            # Flush any remaining text
            remaining = text_buffer.strip()
            if remaining and not self._interrupt_event.is_set():
                await tts_queue.put(remaining)

            # Signal TTS worker to finish
            await tts_queue.put(None)
            await tts_task

            # Re-raise any TTS errors
            if tts_error:
                raise tts_error[0]

        except Exception:
            # Cancel TTS task on error
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
            raise

    async def _generate_and_stream_audio(self, text: str, latency_start: float | None = None) -> None:
        """Generate TTS audio for text and stream Opus frames to client.

        Args:
            text: Text to convert to speech.
            latency_start: If set, record latency from this timestamp on first frame.
        """
        # Generate audio using VoxCPM
        generate_kwargs: dict[str, Any] = {"target_text": text}

        if self.tts_prompt_latents is not None:
            if not self.tts_prompt_text:
                try:
                    await self.websocket.send_json({
                        "type": "error",
                        "error": "Server misconfigured: WS prompt latents were provided but prompt text is empty",
                    })
                except Exception:
                    pass
                return
            generate_kwargs["prompt_latents"] = self.tts_prompt_latents
            generate_kwargs["prompt_text"] = self.tts_prompt_text

        if self.tts_ref_audio_latents is not None:
            generate_params = inspect.signature(self.voxcpm_server.generate).parameters
            if "ref_audio_latents" not in generate_params:
                try:
                    await self.websocket.send_json({
                        "type": "error",
                        "error": "Reference audio is not supported by the loaded model",
                    })
                except Exception:
                    pass
                return
            generate_kwargs["ref_audio_latents"] = self.tts_ref_audio_latents

        audio_stream = self.voxcpm_server.generate(**generate_kwargs)

        first_frame = True
        # Encode to Opus and stream to client
        async for opus_frame in stream_opus(
            wav_chunks=audio_stream,
            sample_rate=self.sample_rate,
            opus_config=self.opus_config,
        ):
            if self._interrupt_event.is_set():
                break

            if first_frame and latency_start is not None:
                WS_PIPELINE_LATENCY_SECONDS.observe(time.perf_counter() - latency_start)
                first_frame = False
            
            try:
                await self.websocket.send_bytes(opus_frame)
            except Exception:
                # Connection likely closed
                break
