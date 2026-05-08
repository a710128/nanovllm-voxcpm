from __future__ import annotations

import base64
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
import traceback
from fastapi.responses import PlainTextResponse
from app.api.deps import get_server_ws
from app.core.metrics import (
    WEBSOCKET_CONNECTIONS_ACTIVE,
    WEBSOCKET_CONNECTIONS_TOTAL,
    WEBSOCKET_MESSAGES_TOTAL,
    WS_BYTES_IN_TOTAL,
    WS_BYTES_OUT_TOTAL,
)
from app.schemas.websocket import (
    AudioChunkMessage,
    AudioEndMessage,
    InitMessage,
    TerminateMessage,
    InterruptMessage,
    parse_client_message,
)
from app.services.asr import ASRClient
import asyncio
from app.services.pipeline import PipelineOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    server: Any = Depends(get_server_ws),
) -> None:
    """Real-time voice cloning WebSocket endpoint with speaker selection.

    This endpoint implements a streaming voice-to-voice pipeline:
    1. Client connects and sends an init message to select a speaker
    2. Client sends audio chunks
    3. Client sends audio_end to signal completion
    4. Server processes: ASR -> LLM -> TTS
    5. Server streams Opus audio back to client
    6. Server sends "complete" message
    7. Connection stays open for more interactions until "terminate" from client

    Message Protocol:
        Client -> Server:
            - {"type": "init", "speaker": "alaa"}           (REQUIRED first message)
            - {"type": "audio_chunk", "data": "<base64>", "format": "wav"}
            - {"type": "audio_end"}
            - {"type": "terminate"}

        Server -> Client:
            - <binary Opus frames> (raw bytes)
            - {"type": "complete"}
            - {"type": "error", "error": "..."}
    """
    # Get config from app state
    cfg = getattr(websocket.app.state, "cfg", None)
    if cfg is None:
        await websocket.close(code=1011, reason="Server misconfigured: missing app.state.cfg")
        return

    # Check model server is ready
    if server is None:
        await websocket.close(code=1011, reason="Model server not ready")
        return

    await websocket.accept()
    WEBSOCKET_CONNECTIONS_ACTIVE.inc()
    WEBSOCKET_CONNECTIONS_TOTAL.labels(status="connected").inc()

    # Initialize clients
    asr_client = ASRClient(cfg.asr)
    llm_client = getattr(websocket.app.state, "llm_client", None)
    if llm_client is None:
        await websocket.close(code=1011, reason="Server misconfigured: missing app.state.llm_client")
        WEBSOCKET_CONNECTIONS_ACTIVE.dec()
        WEBSOCKET_CONNECTIONS_TOTAL.labels(status="error").inc()
        return

    # Get model info for sample rate
    try:
        model_info = await server.get_model_info()
        sample_rate = int(model_info["sample_rate"])
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        await websocket.send_json({"type": "error", "error": f"Failed to get model info: {e}"})
        await websocket.close(code=1011, reason="Failed to get model info")
        WEBSOCKET_CONNECTIONS_ACTIVE.dec()
        WEBSOCKET_CONNECTIONS_TOTAL.labels(status="error").inc()
        return

    # ── Wait for init message (speaker selection) ──────────────────────
    speakers = getattr(websocket.app.state, "speakers", {})

    try:
        init_data = await websocket.receive_json()
        WEBSOCKET_MESSAGES_TOTAL.labels(direction="in", type="init").inc()
        init_msg = parse_client_message(init_data)
        if not isinstance(init_msg, InitMessage):
            await websocket.send_json({"type": "error", "error": "First message must be {\"type\": \"init\", \"speaker\": \"<name>\"}"})
            WEBSOCKET_MESSAGES_TOTAL.labels(direction="out", type="error").inc()
            await websocket.close(code=1008, reason="Expected init message")
            WEBSOCKET_CONNECTIONS_ACTIVE.dec()
            WEBSOCKET_CONNECTIONS_TOTAL.labels(status="error").inc()
            return
    except (ValueError, Exception) as e:
        logger.warning(f"Invalid init message: {e}")
        try:
            await websocket.send_json({"type": "error", "error": f"Invalid init message: {e}"})
        except Exception:
            pass
        await websocket.close(code=1008, reason="Invalid init message")
        WEBSOCKET_CONNECTIONS_ACTIVE.dec()
        WEBSOCKET_CONNECTIONS_TOTAL.labels(status="error").inc()
        return

    speaker_name = init_msg.speaker
    speaker_cfg = speakers.get(speaker_name)

    # Resolve TTS conditioning for this speaker
    if speaker_cfg is not None:
        tts_prompt_latents = speaker_cfg.prompt_latents
        tts_prompt_text = speaker_cfg.prompt_text
        tts_ref_audio_latents = speaker_cfg.ref_audio_latents
        llm_prompt_template = speaker_cfg.gemini_prompt_template or None
    else:
        # Fallback to global defaults
        tts_prompt_latents = getattr(websocket.app.state, "ws_prompt_latents", None)
        tts_prompt_text = getattr(websocket.app.state, "ws_prompt_text", "")
        tts_ref_audio_latents = getattr(websocket.app.state, "ws_ref_audio_latents", None)
        llm_prompt_template = None

    logger.info("WebSocket connection initialized with speaker '%s'", speaker_name)

    # Create pipeline orchestrator with speaker-specific config
    pipeline = PipelineOrchestrator(
        websocket=websocket,
        voxcpm_server=server,
        asr_client=asr_client,
        llm_client=llm_client,
        opus_config=cfg.opus,
        sample_rate=sample_rate,
        tts_prompt_latents=tts_prompt_latents,
        tts_prompt_text=tts_prompt_text,
        tts_ref_audio_latents=tts_ref_audio_latents,
        llm_prompt_template=llm_prompt_template,
    )

    pipeline_task = None
    ws_lock = asyncio.Lock()

    async def safe_send_json(data: dict) -> None:
        async with ws_lock:
            await websocket.send_json(data)

    async def run_pipeline_task(text: str) -> None:
        try:
            await pipeline.run_with_transcription(text)
            if not pipeline._interrupt_event.is_set():
                WEBSOCKET_MESSAGES_TOTAL.labels(direction="out", type="complete").inc()
        except Exception as e:
            logger.error(f"Pipeline error in task: {e}")
            try:
                await safe_send_json({"type": "error", "error": str(e)})
            except Exception:
                pass

    try:
        asr_session = asr_client.create_session()

        while True:
            try:
                raw_data = await websocket.receive_json()
                raw_str_len = len(str(raw_data))
                WS_BYTES_IN_TOTAL.inc(raw_str_len)

                msg = parse_client_message(raw_data)
                WEBSOCKET_MESSAGES_TOTAL.labels(direction="in", type=msg.type).inc()

                if isinstance(msg, AudioChunkMessage):
                    # Dispatch audio chunk to ASR immediately
                    print(f"Received audio chunk: {len(msg.data)} bytes (format={msg.format})")
                    try:
                        chunk_bytes = base64.b64decode(msg.data)
                        asr_session.push_chunk(chunk_bytes, msg.format)
                    except Exception as e:
                        logger.warning(f"Failed to decode audio chunk: {e}")
                        await safe_send_json({"type": "error", "error": f"Invalid audio chunk: {e}"})
                        WEBSOCKET_MESSAGES_TOTAL.labels(direction="out", type="error").inc()

                elif isinstance(msg, AudioEndMessage):
                    # Collect all ASR transcriptions
                    print("Received audio end message, finalizing ASR transcription")
                    if asr_session.chunk_count == 0:
                        await safe_send_json({"type": "error", "error": "No audio chunks received"})
                        WEBSOCKET_MESSAGES_TOTAL.labels(direction="out", type="error").inc()
                        continue

                    try:
                        transcription = await asr_session.finish()
                        
                        if pipeline_task and not pipeline_task.done():
                            pipeline.interrupt()
                            pipeline_task.cancel()
                        pipeline.reset_interrupt()
                        pipeline_task = asyncio.create_task(run_pipeline_task(transcription))
                        
                    except Exception as e:
                        logger.exception(f"Pipeline error: {e}")
                        try:
                            await safe_send_json({"type": "error", "error": str(e)})
                            WEBSOCKET_MESSAGES_TOTAL.labels(direction="out", type="error").inc()
                        except Exception:
                            pass
                    finally:
                        # Reset for next interaction
                        asr_session = asr_client.create_session()

                elif isinstance(msg, InterruptMessage):
                    print("Received interrupt message from client")
                    if pipeline_task and not pipeline_task.done():
                        pipeline.interrupt()
                
                elif isinstance(msg, TerminateMessage):
                    # Client requested connection close
                    print("Client requested connection termination")
                    break

                elif isinstance(msg, InitMessage):
                    # Init already handled; ignore duplicates
                    logger.warning("Received duplicate init message, ignoring")

            except ValueError as e:
                # Invalid message format
                logger.warning(f"Invalid message: {e}")
                try:
                    await safe_send_json({"type": "error", "error": f"Invalid message: {e}"})
                    WEBSOCKET_MESSAGES_TOTAL.labels(direction="out", type="error").inc()
                except Exception:
                    pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        WEBSOCKET_CONNECTIONS_TOTAL.labels(status="closed").inc()
    except Exception as e:
        if isinstance(e, RuntimeError) and "close message has been sent" in str(e):
            logger.info("WebSocket client disconnected (RuntimeError)")
            WEBSOCKET_CONNECTIONS_TOTAL.labels(status="closed").inc()
        else:
            logger.exception(f"WebSocket error: {e}")
            WEBSOCKET_CONNECTIONS_TOTAL.labels(status="error").inc()
            try:
                await websocket.send_json({"type": "error", "error": str(e)})
            except Exception:
                pass
    finally:
        WEBSOCKET_CONNECTIONS_ACTIVE.dec()
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/debug/tasks", response_class=PlainTextResponse)
async def list_running_tasks():
    """Diagnostic route to view all active asyncio tasks and where they are blocked."""
    tasks = asyncio.all_tasks()
    output = [f"Total Active Tasks: {len(tasks)}\n"]
    output.append("-" * 50)
    
    for task in tasks:
        task_name = task.get_name()
        try:
            coro_name = task.get_coro().__name__
        except AttributeError:
            coro_name = "Unknown Coro"
            
        output.append(f"Task Name: {task_name} | Coroutine: {coro_name}")
        
        stack = task.get_stack()
        if stack:
            formatted_stack = ""
            for frame in stack:
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                func_name = frame.f_code.co_name
                formatted_stack += f"  File \"{filename}\", line {lineno}, in {func_name}\n"
            output.append("Currently Blocked At:\n" + formatted_stack.strip())
        else:
            output.append("State: Pending/Not started or recently completed")
            
        output.append("-" * 50)
        
    return "\n".join(output)