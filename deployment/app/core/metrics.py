from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

HTTP_REQUESTS_TOTAL = Counter(
    "nanovllm_http_requests_total",
    "Total HTTP requests",
    labelnames=["route", "method", "status"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "nanovllm_http_request_duration_seconds",
    "HTTP request duration in seconds (includes streaming)",
    labelnames=["route", "method"],
)
INFLIGHT_REQUESTS = Gauge(
    "nanovllm_inflight_requests",
    "Number of in-flight HTTP requests",
    labelnames=["route"],
)
EXCEPTIONS_TOTAL = Counter(
    "nanovllm_exceptions_total",
    "Unhandled exceptions",
    labelnames=["route", "exception_type"],
)

GENERATE_TTFB_SECONDS = Histogram(
    "nanovllm_generate_ttfb_seconds",
    "Time-to-first-byte for /generate streaming responses",
)
GENERATE_AUDIO_SECONDS_TOTAL = Counter(
    "nanovllm_generate_audio_seconds_total",
    "Total generated audio duration in seconds",
)
GENERATE_STREAM_BYTES_TOTAL = Counter(
    "nanovllm_generate_stream_bytes_total",
    "Total bytes streamed by /generate",
)

AUDIO_ENCODE_FAILURES_TOTAL = Counter(
    "nanovllm_audio_encode_failures_total",
    "MP3 encoding failures",
)
AUDIO_ENCODE_SECONDS = Histogram(
    "nanovllm_audio_encode_seconds",
    "Time spent in MP3 encoder.encode() calls",
)

ENCODE_LATENTS_REQUESTS_TOTAL = Counter(
    "nanovllm_encode_latents_requests_total",
    "Total /encode_latents requests",
    labelnames=["status"],
)
ENCODE_LATENTS_DURATION_SECONDS = Histogram(
    "nanovllm_encode_latents_duration_seconds",
    "Latency of /encode_latents in seconds",
)

# WebSocket metrics
WEBSOCKET_CONNECTIONS_ACTIVE = Gauge(
    "nanovllm_websocket_connections_active",
    "Number of active WebSocket connections",
)
WEBSOCKET_CONNECTIONS_TOTAL = Counter(
    "nanovllm_websocket_connections_total",
    "Total WebSocket connections",
    labelnames=["status"],  # "connected", "error", "closed"
)
WEBSOCKET_MESSAGES_TOTAL = Counter(
    "nanovllm_websocket_messages_total",
    "Total WebSocket messages",
    labelnames=["direction", "type"],  # direction: "in"/"out", type: message type
)
PIPELINE_DURATION_SECONDS = Histogram(
    "nanovllm_pipeline_duration_seconds",
    "End-to-end pipeline duration (ASR -> LLM -> TTS)",
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)
PIPELINE_STAGE_DURATION_SECONDS = Histogram(
    "nanovllm_pipeline_stage_duration_seconds",
    "Duration of individual pipeline stages",
    labelnames=["stage"],  # "asr", "llm", "llm_ttft", "tts"
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)

# Latency: last audio chunk received -> first audio chunk sent back
WS_PIPELINE_LATENCY_SECONDS = Histogram(
    "nanovllm_ws_pipeline_latency_seconds",
    "Latency from audio_end to first response chunk sent",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)

# WebSocket network bytes
WS_BYTES_IN_TOTAL = Counter(
    "nanovllm_ws_bytes_in_total",
    "Total bytes received via WebSocket",
)
WS_BYTES_OUT_TOTAL = Counter(
    "nanovllm_ws_bytes_out_total",
    "Total bytes sent via WebSocket",
)

# GPU metrics (best-effort, updated periodically)
GPU_UTILIZATION_PERCENT = Gauge(
    "nanovllm_gpu_utilization_percent",
    "GPU utilization percentage",
    labelnames=["gpu"],
)
GPU_MEMORY_USED_BYTES = Gauge(
    "nanovllm_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    labelnames=["gpu"],
)
GPU_MEMORY_TOTAL_BYTES = Gauge(
    "nanovllm_gpu_memory_total_bytes",
    "GPU memory total in bytes",
    labelnames=["gpu"],
)


def update_gpu_metrics() -> None:
    """Update GPU metrics from pynvml. No-op if pynvml is unavailable."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_id = str(i)
            GPU_UTILIZATION_PERCENT.labels(gpu=gpu_id).set(util.gpu)
            GPU_MEMORY_USED_BYTES.labels(gpu=gpu_id).set(mem_info.used)
            GPU_MEMORY_TOTAL_BYTES.labels(gpu=gpu_id).set(mem_info.total)
    except Exception:
        pass  # pynvml not available or failed — silently skip


def install_metrics(app: FastAPI) -> None:
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        route = request.url.path
        method = request.method
        start = time.perf_counter()

        INFLIGHT_REQUESTS.labels(route=route).inc()
        try:
            response = await call_next(request)
        except Exception as e:
            EXCEPTIONS_TOTAL.labels(route=route, exception_type=type(e).__name__).inc()
            dur = time.perf_counter() - start
            HTTP_REQUEST_DURATION_SECONDS.labels(route=route, method=method).observe(dur)
            HTTP_REQUESTS_TOTAL.labels(route=route, method=method, status="500").inc()
            INFLIGHT_REQUESTS.labels(route=route).dec()
            raise

        status = str(response.status_code)

        if isinstance(response, StreamingResponse):
            original_iter = response.body_iterator

            async def wrapped_iter():
                try:
                    async for chunk in original_iter:
                        yield chunk
                except Exception as e:
                    EXCEPTIONS_TOTAL.labels(route=route, exception_type=type(e).__name__).inc()
                    raise
                finally:
                    dur = time.perf_counter() - start
                    HTTP_REQUEST_DURATION_SECONDS.labels(route=route, method=method).observe(dur)
                    HTTP_REQUESTS_TOTAL.labels(route=route, method=method, status=status).inc()
                    INFLIGHT_REQUESTS.labels(route=route).dec()

            response.body_iterator = wrapped_iter()
            return response

        dur = time.perf_counter() - start
        HTTP_REQUEST_DURATION_SECONDS.labels(route=route, method=method).observe(dur)
        HTTP_REQUESTS_TOTAL.labels(route=route, method=method, status=status).inc()
        INFLIGHT_REQUESTS.labels(route=route).dec()
        return response


def metrics_response() -> Response:
    update_gpu_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
