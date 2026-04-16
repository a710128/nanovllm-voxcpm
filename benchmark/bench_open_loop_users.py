"""Benchmark VoxCPM TTFB under fixed request rate (open-loop).

This script is meant for the "long audio synthesis" load pattern: many
in-flight streaming generations with a fixed arrival rate (e.g. 30 RPS), and
we measure client-side TTFB (time to first generated chunk) from the model
server generator.

In-process (direct engine) run (recommended via uv):
  uv run python benchmark/bench_open_loop_users.py \
    --model ~/VoxCPM1.5 \
    --rps 30 --duration-s 60 \
    --target-text-file benchmark/target_text_100w_en.txt \
    --max-generate-length 8000

HTTP run (against deployment service /generate):
  uv run python benchmark/bench_open_loop_users.py \
    --url http://127.0.0.1:8000/generate \
    --rps 30 --duration-s 60 \
    --target-text-file benchmark/target_text_100w_en.txt \
    --max-generate-length 8000

Notes:
- This repo is GPU-centric; CPU-only execution is not supported.
- Metrics are end-to-end (parent process wall time) and include IPC overhead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterable, cast
from urllib.parse import urlparse

import torch

from lora_bench_utils import (
    add_lora_args,
    add_lora_to_payload,
    build_lora_config,
    choose_lora_name,
    make_lora_names,
    make_lora_rng,
    register_loras_http,
    register_loras_in_process,
    unregister_loras_http,
    unregister_loras_in_process,
    validate_lora_args,
)

if TYPE_CHECKING:
    from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool


DEFAULT_TEXT = "Hello world."


def _parse_devices(devices: str) -> list[int]:
    items = [x.strip() for x in devices.split(",") if x.strip()]
    if not items:
        return [0]
    return [int(x) for x in items]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if c == f:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def _stdev(xs: Iterable[float]) -> float:
    xs = list(xs)
    if len(xs) <= 1:
        return 0.0
    return statistics.stdev(xs)


def _fmt_float(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.4f}"


def _rtf_excluding_ttfb(wall_s: float, ttfb_s: float, audio_s: float) -> float:
    if audio_s <= 0:
        return float("inf")
    return max(wall_s - ttfb_s, 0.0) / audio_s


@dataclass(frozen=True)
class OneRequestResult:
    ok: bool
    dropped: bool
    error: str | None
    scheduled_t: float
    started_t: float | None
    ttfb_s: float | None
    wall_s: float | None
    total_samples: int
    total_bytes: int
    num_chunks: int


async def _consume_one(
    server: Any,
    *,
    target_text: str,
    max_generate_length: int,
    temperature: float,
    cfg_value: float,
    scheduled_t: float,
    lora_name: str | None,
) -> OneRequestResult:
    started_t = time.perf_counter()
    first_chunk_t: float | None = None
    total_samples = 0
    num_chunks = 0
    try:
        async for chunk in server.generate(
            target_text=target_text,
            max_generate_length=max_generate_length,
            temperature=temperature,
            cfg_value=cfg_value,
            lora_name=lora_name,
        ):
            if first_chunk_t is None:
                first_chunk_t = time.perf_counter()
            total_samples += int(chunk.shape[0])
            num_chunks += 1
        end_t = time.perf_counter()
        if first_chunk_t is None:
            first_chunk_t = end_t
        return OneRequestResult(
            ok=True,
            dropped=False,
            error=None,
            scheduled_t=scheduled_t,
            started_t=started_t,
            ttfb_s=first_chunk_t - started_t,
            wall_s=end_t - started_t,
            total_samples=total_samples,
            total_bytes=0,
            num_chunks=num_chunks,
        )
    except Exception as e:
        end_t = time.perf_counter()
        return OneRequestResult(
            ok=False,
            dropped=False,
            error=f"{type(e).__name__}: {e}",
            scheduled_t=scheduled_t,
            started_t=started_t,
            ttfb_s=(first_chunk_t - started_t) if first_chunk_t is not None else None,
            wall_s=end_t - started_t,
            total_samples=total_samples,
            total_bytes=0,
            num_chunks=num_chunks,
        )


def _http_post_stream_ttfb(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    consume_full: bool,
) -> tuple[bool, str | None, float | None, float | None, int]:
    """Blocking helper: POST JSON and measure time to first response body byte."""

    import http.client
    import ssl

    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        return False, f"unsupported URL scheme: {u.scheme}", None, None, 0

    host = u.hostname
    if not host:
        return False, "invalid URL host", None, None, 0

    port = u.port
    if port is None:
        port = 443 if u.scheme == "https" else 80

    path = u.path or "/"
    if u.query:
        path = f"{path}?{u.query}"

    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
        "Connection": "close",
    }

    ctx = ssl.create_default_context() if u.scheme == "https" else None
    if u.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(host, port=port, timeout=timeout_s, context=ctx)
    else:
        conn = http.client.HTTPConnection(host, port=port, timeout=timeout_s)

    start_t = time.perf_counter()
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()

        if resp.status != 200:
            try:
                err_b = resp.read(4096)
            except Exception:
                err_b = b""
            msg = err_b.decode("utf-8", errors="replace")
            return False, f"HTTP {resp.status}: {msg}".strip(), None, None, 0

        # Trigger TTFB by reading at least 1 byte.
        total_bytes = 0
        b = resp.read(1)
        first_byte_t = time.perf_counter()
        if b:
            total_bytes += len(b)

        if consume_full:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                total_bytes += len(chunk)

        end_t = time.perf_counter()
        return True, None, first_byte_t - start_t, end_t - start_t, total_bytes
    except Exception as e:
        end_t = time.perf_counter()
        return False, f"{type(e).__name__}: {e}", None, end_t - start_t, 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


@dataclass(frozen=True)
class BenchSummary:
    rps_target: float
    duration_s: float
    warmup_s: float
    max_inflight: int
    queue_on_overload: bool
    total_scheduled: int
    total_started: int
    total_dropped: int
    total_completed: int
    total_ok: int
    total_err: int
    achieved_rps_started: float
    ttfb_p50_s: float
    ttfb_p90_s: float
    ttfb_p95_s: float
    ttfb_p99_s: float
    ttfb_mean_s: float
    ttfb_stdev_s: float
    sample_rate: int | None
    audio_s_per_req_mean: float | None
    rtf_p50: float | None
    rtf_p90: float | None
    rtf_p95: float | None
    rtf_p99: float | None
    rtf_mean: float | None
    rtf_stdev: float | None


async def async_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark VoxCPM TTFB under fixed RPS (open-loop)")
    p.add_argument(
        "--url",
        default=None,
        help=(
            "If set, benchmark HTTP POST to this /generate endpoint instead of the in-process engine "
            "(e.g. http://127.0.0.1:8000/generate)"
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help="Local model directory (or HF repo id); required for in-process mode",
    )
    p.add_argument(
        "--devices",
        default="0",
        help="Comma-separated CUDA device indices, e.g. '0' or '0,1'",
    )
    p.add_argument("--inference-timesteps", type=int, default=10)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--enforce-eager", action="store_true")

    p.add_argument("--target-text", default=DEFAULT_TEXT)
    p.add_argument("--target-text-file", default=None, help="Read target text from file (UTF-8)")
    p.add_argument("--max-generate-length", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cfg-value", type=float, default=2.0)
    add_lora_args(p)

    p.add_argument(
        "--http-timeout-s",
        type=float,
        default=600.0,
        help="(HTTP mode) socket connect/read timeout per request",
    )
    p.add_argument(
        "--http-consume-full",
        action="store_true",
        help="(HTTP mode) consume full streamed response body (recommended)",
    )

    p.add_argument("--rps", type=float, default=30.0, help="Target requests/sec arrival rate")
    p.add_argument("--duration-s", type=float, default=60.0, help="Measurement window length")
    p.add_argument("--warmup-s", type=float, default=5.0, help="Warmup time before measurement")
    p.add_argument(
        "--max-inflight",
        type=int,
        default=512,
        help="Max in-flight requests allowed by the benchmark driver",
    )
    p.add_argument(
        "--queue-on-overload",
        action="store_true",
        help="If set, wait for an in-flight slot instead of dropping scheduled requests",
    )
    p.add_argument(
        "--drain-timeout-s",
        type=float,
        default=600.0,
        help="After scheduling ends, wait up to this long for in-flight requests to finish",
    )
    p.add_argument("--json-out", default=None, help="Write results JSON to this path")
    p.add_argument(
        "--print-top-errors",
        type=int,
        default=5,
        help="Print up to N most common error messages (0 to disable)",
    )
    args = p.parse_args(argv)

    if args.url is None and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this project does not support CPU-only in-process benchmarking")

    if args.target_text_file is not None:
        args.target_text = open(args.target_text_file, "r", encoding="utf-8").read().strip()
        if not args.target_text:
            raise ValueError("target text is empty")

    if args.rps <= 0:
        raise ValueError("--rps must be > 0")
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be > 0")
    if args.warmup_s < 0:
        raise ValueError("--warmup-s must be >= 0")
    if args.max_inflight <= 0:
        raise ValueError("--max-inflight must be >= 1")
    validate_lora_args(args)

    if args.url is None and not args.model:
        raise ValueError("in-process mode requires --model (or use --url for HTTP mode)")

    devices = _parse_devices(args.devices)
    lora_names = make_lora_names(args)
    lora_rng = make_lora_rng(args)

    server_pool: AsyncVoxCPMServerPool | None = None
    if args.url is None:
        # Import after arg parsing so `--help` works even if optional runtime deps are missing.
        from nanovllm_voxcpm import VoxCPM

        assert args.model is not None
        server_pool = cast(
            "AsyncVoxCPMServerPool",
            VoxCPM.from_pretrained(
                model=args.model,
                inference_timesteps=args.inference_timesteps,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_num_seqs=args.max_num_seqs,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                devices=devices,
                lora_config=build_lora_config(
                    args.model,
                    max_loras=int(args.max_loras),
                    max_lora_rank=int(args.max_lora_rank),
                ),
            ),
        )

    results: list[OneRequestResult] = []
    inflight = asyncio.Semaphore(args.max_inflight)
    tasks: set[asyncio.Task[OneRequestResult]] = set()
    unregister_errors: list[str] = []

    async def _start_one(scheduled_t: float) -> OneRequestResult:
        started_t = time.perf_counter()
        try:
            if args.url is not None:
                lora_name = choose_lora_name(lora_names, lora_rng)
                ok, err, ttfb_s, wall_s, total_bytes = await asyncio.to_thread(
                    _http_post_stream_ttfb,
                    args.url,
                    add_lora_to_payload(
                        {
                            "target_text": args.target_text,
                            "max_generate_length": int(args.max_generate_length),
                            "temperature": float(args.temperature),
                            "cfg_value": float(args.cfg_value),
                        },
                        lora_name,
                    ),
                    timeout_s=float(args.http_timeout_s),
                    consume_full=bool(args.http_consume_full),
                )
                return OneRequestResult(
                    ok=ok,
                    dropped=False,
                    error=err,
                    scheduled_t=scheduled_t,
                    started_t=started_t,
                    ttfb_s=ttfb_s,
                    wall_s=wall_s,
                    total_samples=0,
                    total_bytes=total_bytes,
                    num_chunks=0,
                )

            assert server_pool is not None
            return await _consume_one(
                server_pool,
                target_text=args.target_text,
                max_generate_length=args.max_generate_length,
                temperature=args.temperature,
                cfg_value=args.cfg_value,
                scheduled_t=scheduled_t,
                lora_name=choose_lora_name(lora_names, lora_rng),
            )
        finally:
            inflight.release()

    def _on_done(t: asyncio.Task[OneRequestResult]) -> None:
        tasks.discard(t)
        try:
            results.append(t.result())
        except Exception as e:
            # Should be rare because _consume_one catches, but keep this safe.
            results.append(
                OneRequestResult(
                    ok=False,
                    dropped=False,
                    error=f"{type(e).__name__}: {e}",
                    scheduled_t=float("nan"),
                    started_t=None,
                    ttfb_s=None,
                    wall_s=None,
                    total_samples=0,
                    total_bytes=0,
                    num_chunks=0,
                )
            )

    try:
        if server_pool is not None:
            await server_pool.wait_for_ready()
            await register_loras_in_process(server_pool, lora_names, args.lora_path)
        elif args.url is not None:
            await register_loras_http(args.url, lora_names, args.lora_path, timeout_s=float(args.http_timeout_s))

        sample_rate: int | None = None
        if server_pool is not None:
            try:
                model_info = await server_pool.get_model_info()
                sample_rate = int(model_info["sample_rate"])
            except Exception:
                sample_rate = None

        # Warmup (time-based, no measurements).
        if args.warmup_s > 0:
            warmup_end = time.perf_counter() + float(args.warmup_s)
            while time.perf_counter() < warmup_end:
                # Keep warmup lightweight: one request at a time.
                if args.url is not None:
                    lora_name = choose_lora_name(lora_names, lora_rng)
                    await asyncio.to_thread(
                        _http_post_stream_ttfb,
                        args.url,
                        add_lora_to_payload(
                            {
                                "target_text": args.target_text,
                                "max_generate_length": int(max(1, min(args.max_generate_length, 500))),
                                "temperature": float(args.temperature),
                                "cfg_value": float(args.cfg_value),
                            },
                            lora_name,
                        ),
                        timeout_s=float(args.http_timeout_s),
                        consume_full=False,
                    )
                else:
                    assert server_pool is not None
                    await _consume_one(
                        server_pool,
                        target_text=args.target_text,
                        max_generate_length=max(1, min(args.max_generate_length, 500)),
                        temperature=args.temperature,
                        cfg_value=args.cfg_value,
                        scheduled_t=time.perf_counter(),
                        lora_name=choose_lora_name(lora_names, lora_rng),
                    )

        # Measurement scheduling.
        start_t = time.perf_counter()
        end_t = start_t + float(args.duration_s)
        period_s = 1.0 / float(args.rps)
        k = 0
        while True:
            scheduled_t = start_t + (k * period_s)
            if scheduled_t >= end_t:
                break
            now = time.perf_counter()
            if scheduled_t > now:
                await asyncio.sleep(scheduled_t - now)

            # Enforce max in-flight. Default is drop-on-overload to keep open-loop arrivals.
            acquired = False
            if args.queue_on_overload:
                await inflight.acquire()
                acquired = True
            else:
                if inflight.locked():
                    acquired = False
                else:
                    await inflight.acquire()
                    acquired = True

            if not acquired:
                results.append(
                    OneRequestResult(
                        ok=False,
                        dropped=True,
                        error="dropped: max_inflight reached",
                        scheduled_t=scheduled_t,
                        started_t=None,
                        ttfb_s=None,
                        wall_s=None,
                        total_samples=0,
                        total_bytes=0,
                        num_chunks=0,
                    )
                )
                k += 1
                continue

            t = asyncio.create_task(_start_one(scheduled_t))
            tasks.add(t)
            t.add_done_callback(_on_done)
            k += 1

        # Drain in-flight.
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=float(args.drain_timeout_s))
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    finally:
        if server_pool is not None:
            unregister_errors = await unregister_loras_in_process(server_pool, lora_names)
            await server_pool.stop()
        elif args.url is not None:
            unregister_errors = await unregister_loras_http(args.url, lora_names, timeout_s=float(args.http_timeout_s))

    # Summarize.
    total_scheduled = len(results)
    total_dropped = sum(1 for r in results if r.dropped)
    total_started = sum(1 for r in results if r.started_t is not None)
    total_completed = sum(1 for r in results if r.wall_s is not None)
    total_ok = sum(1 for r in results if r.ok)
    total_err = sum(1 for r in results if (not r.ok) and (not r.dropped))

    ttfbs = [r.ttfb_s for r in results if r.ttfb_s is not None and r.ok]
    achieved_rps_started = total_started / float(args.duration_s)

    audio_s_per_req_mean: float | None = None
    rtfs: list[float] = []
    if sample_rate is not None and sample_rate > 0:
        audio_s = [
            (r.total_samples / float(sample_rate))
            for r in results
            if r.ok and r.wall_s is not None and r.total_samples > 0
        ]
        if audio_s:
            audio_s_per_req_mean = _mean(audio_s)
            rtfs = [
                _rtf_excluding_ttfb(r.wall_s, r.ttfb_s, r.total_samples / float(sample_rate))
                for r in results
                if r.ok and r.wall_s is not None and r.ttfb_s is not None and r.total_samples > 0
            ]

    summary = BenchSummary(
        rps_target=float(args.rps),
        duration_s=float(args.duration_s),
        warmup_s=float(args.warmup_s),
        max_inflight=int(args.max_inflight),
        queue_on_overload=bool(args.queue_on_overload),
        total_scheduled=total_scheduled,
        total_started=total_started,
        total_dropped=total_dropped,
        total_completed=total_completed,
        total_ok=total_ok,
        total_err=total_err,
        achieved_rps_started=achieved_rps_started,
        ttfb_p50_s=_percentile(ttfbs, 50),
        ttfb_p90_s=_percentile(ttfbs, 90),
        ttfb_p95_s=_percentile(ttfbs, 95),
        ttfb_p99_s=_percentile(ttfbs, 99),
        ttfb_mean_s=_mean(ttfbs),
        ttfb_stdev_s=_stdev(ttfbs),
        sample_rate=sample_rate,
        audio_s_per_req_mean=audio_s_per_req_mean,
        rtf_p50=_percentile(rtfs, 50) if rtfs else None,
        rtf_p90=_percentile(rtfs, 90) if rtfs else None,
        rtf_p95=_percentile(rtfs, 95) if rtfs else None,
        rtf_p99=_percentile(rtfs, 99) if rtfs else None,
        rtf_mean=_mean(rtfs) if rtfs else None,
        rtf_stdev=_stdev(rtfs) if rtfs else None,
    )

    print("Benchmark finished")
    if args.url is not None:
        print(f"  url: {args.url}")
    else:
        print(f"  model: {args.model}")
    print(f"  devices: {devices}")
    print(f"  target RPS: {args.rps}")
    if lora_names:
        print(f"  loras: {len(lora_names)} aliases from {args.lora_path}")
    if unregister_errors:
        print("  lora cleanup warnings:")
        for msg in unregister_errors:
            print(f"    {msg}")
    print(f"  duration_s: {args.duration_s} (warmup {args.warmup_s})")
    print(f"  max_inflight: {args.max_inflight} (queue={args.queue_on_overload})")
    print("Results")
    print(f"  scheduled: {summary.total_scheduled}")
    print(f"  started: {summary.total_started} (achieved {summary.achieved_rps_started:.2f} rps)")
    print(f"  dropped: {summary.total_dropped}")
    print(f"  ok: {summary.total_ok}")
    print(f"  err: {summary.total_err}")

    if args.print_top_errors and summary.total_err > 0:
        err_counter = Counter(r.error for r in results if (r.error is not None and (not r.ok) and (not r.dropped)))
        print("Top errors")
        for msg, cnt in err_counter.most_common(int(args.print_top_errors)):
            print(f"  {cnt}: {msg}")

    print("TTFB (seconds) over ok requests")
    print(f"  p50: {_fmt_float(summary.ttfb_p50_s)}")
    print(f"  p90: {_fmt_float(summary.ttfb_p90_s)}")
    print(f"  p95: {_fmt_float(summary.ttfb_p95_s)}")
    print(f"  p99: {_fmt_float(summary.ttfb_p99_s)}")
    print(f"  mean +/- stdev: {_fmt_float(summary.ttfb_mean_s)} +/- {_fmt_float(summary.ttfb_stdev_s)}")

    if summary.rtf_mean is not None:
        assert summary.sample_rate is not None
        print(f"Audio/RTF over ok requests (sample_rate={summary.sample_rate})")
        if summary.audio_s_per_req_mean is not None:
            print(f"  audio_s_per_req_mean: {_fmt_float(summary.audio_s_per_req_mean)}")
        print(f"  RTF p50: {_fmt_float(summary.rtf_p50 or float('nan'))}")
        print(f"  RTF p90: {_fmt_float(summary.rtf_p90 or float('nan'))}")
        print(f"  RTF p95: {_fmt_float(summary.rtf_p95 or float('nan'))}")
        print(f"  RTF p99: {_fmt_float(summary.rtf_p99 or float('nan'))}")
        print(f"  RTF mean +/- stdev: {_fmt_float(summary.rtf_mean)} +/- {_fmt_float(summary.rtf_stdev or 0.0)}")
    elif args.url is not None:
        print("Audio/RTF: unavailable in HTTP mode (MP3 duration not decoded)")

    payload: dict[str, Any] = {
        "args": vars(args),
        "devices": devices,
        "results": [asdict(r) for r in results],
        "summary": asdict(summary),
    }
    if args.json_out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
