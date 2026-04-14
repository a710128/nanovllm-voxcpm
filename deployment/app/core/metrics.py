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
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
