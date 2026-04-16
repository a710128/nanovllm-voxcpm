"""Benchmark VoxCPM under closed-loop "N users" load.

This script simulates N independent users. Each user sends a new request
immediately after the previous one completes (closed-loop). We measure:

- TTFB: time-to-first generated chunk (in-process) OR time-to-first response byte (HTTP)
- RTF: (request wall_time - TTFB) / generated_audio_seconds

In-process (direct engine) run (recommended via uv):
  uv run python benchmark/bench_closed_loop_users.py \
    --model ~/VoxCPM1.5 \
    --num-users 100 --duration-s 60 --warmup-s 5 \
    --target-text-file benchmark/target_text_100w_en.txt \
    --max-generate-length 8000

HTTP run (against deployment service /generate):
  uv run python benchmark/bench_closed_loop_users.py \
    --url http://127.0.0.1:8000/generate \
    --num-users 100 --duration-s 60 --warmup-s 5 \
    --target-text-file benchmark/target_text_100w_en.txt \
    --max-generate-length 8000

Notes:
- This repo is GPU-centric; CPU-only execution is not supported for in-process mode.
- HTTP-mode RTF is estimated by parsing MP3 frame headers (no audio decode).
- Metrics are end-to-end (parent process wall time) and include IPC / networking overhead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
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
    user_id: int
    user_seq: int
    ok: bool
    error: str | None
    started_t: float
    ttfb_s: float | None
    wall_s: float | None
    total_samples: int
    total_bytes: int
    audio_s: float | None
    rtf: float | None


class _Mp3FrameCounter:
    """Estimate MP3 duration by parsing frame headers.

    This does not decode audio. It parses MPEG audio frame headers and sums
    samples-per-frame across frames.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._skip_bytes = 0
        self.sample_rate: int | None = None
        self.total_samples: int = 0
        self.total_frames: int = 0

    def feed(self, data: bytes) -> None:
        if not data:
            return
        self._buf.extend(data)
        self._parse()

    def finish(self) -> float | None:
        if self.sample_rate is None or self.sample_rate <= 0:
            return None
        if self.total_samples <= 0:
            return 0.0
        return self.total_samples / float(self.sample_rate)

    def _parse(self) -> None:
        # Handle a leading ID3v2 tag if present.
        while True:
            if self._skip_bytes > 0:
                if len(self._buf) <= self._skip_bytes:
                    self._skip_bytes -= len(self._buf)
                    self._buf.clear()
                    return
                del self._buf[: self._skip_bytes]
                self._skip_bytes = 0

            if len(self._buf) < 10:
                return
            if self._buf[0:3] == b"ID3":
                # ID3v2 header: 10 bytes; size is synchsafe int in bytes 6..9.
                tag_size = (
                    (self._buf[6] & 0x7F) << 21
                    | (self._buf[7] & 0x7F) << 14
                    | (self._buf[8] & 0x7F) << 7
                    | (self._buf[9] & 0x7F)
                )
                self._skip_bytes = 10 + int(tag_size)
                continue
            break

        # Parse MPEG frames from buffer.
        while True:
            if len(self._buf) < 4:
                return

            # Sync scan.
            if not (self._buf[0] == 0xFF and (self._buf[1] & 0xE0) == 0xE0):
                del self._buf[0]
                continue

            b1 = self._buf[1]
            b2 = self._buf[2]

            version_id = (b1 >> 3) & 0x03
            layer_id = (b1 >> 1) & 0x03
            if version_id == 0x01 or layer_id == 0x00:
                del self._buf[0]
                continue

            bitrate_idx = (b2 >> 4) & 0x0F
            sr_idx = (b2 >> 2) & 0x03
            padding = (b2 >> 1) & 0x01
            if bitrate_idx in (0, 0x0F) or sr_idx == 0x03:
                del self._buf[0]
                continue

            # Version mapping.
            if version_id == 0x03:
                mpeg_version = 1
                sr_table = (44100, 48000, 32000)
            elif version_id == 0x02:
                mpeg_version = 2
                sr_table = (22050, 24000, 16000)
            else:
                mpeg_version = 25  # 2.5
                sr_table = (11025, 12000, 8000)
            sample_rate = int(sr_table[sr_idx])

            # Layer mapping.
            if layer_id == 0x03:
                layer = 1
            elif layer_id == 0x02:
                layer = 2
            else:
                layer = 3

            bitrate_kbps = _mp3_bitrate_kbps(mpeg_version=mpeg_version, layer=layer, bitrate_idx=bitrate_idx)
            if bitrate_kbps is None:
                del self._buf[0]
                continue

            frame_len = _mp3_frame_length_bytes(
                mpeg_version=mpeg_version,
                layer=layer,
                bitrate_kbps=bitrate_kbps,
                sample_rate=sample_rate,
                padding=padding,
            )
            if frame_len is None or frame_len < 4 or frame_len > 64 * 1024:
                del self._buf[0]
                continue

            if len(self._buf) < frame_len:
                return

            samples_per_frame = _mp3_samples_per_frame(mpeg_version=mpeg_version, layer=layer)
            if samples_per_frame is None:
                del self._buf[0]
                continue

            if self.sample_rate is None:
                self.sample_rate = sample_rate
            self.total_samples += samples_per_frame
            self.total_frames += 1
            del self._buf[:frame_len]


def _mp3_bitrate_kbps(*, mpeg_version: int, layer: int, bitrate_idx: int) -> int | None:
    # Tables indexed by bitrate_idx (0..15). 0 and 15 are invalid/reserved.
    if layer == 1:
        if mpeg_version == 1:
            table = (
                0,
                32,
                64,
                96,
                128,
                160,
                192,
                224,
                256,
                288,
                320,
                352,
                384,
                416,
                448,
                0,
            )
        else:
            table = (
                0,
                32,
                48,
                56,
                64,
                80,
                96,
                112,
                128,
                144,
                160,
                176,
                192,
                224,
                256,
                0,
            )
    elif layer == 2:
        if mpeg_version == 1:
            table = (
                0,
                32,
                48,
                56,
                64,
                80,
                96,
                112,
                128,
                160,
                192,
                224,
                256,
                320,
                384,
                0,
            )
        else:
            table = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)
    else:
        # Layer III
        if mpeg_version == 1:
            table = (
                0,
                32,
                40,
                48,
                56,
                64,
                80,
                96,
                112,
                128,
                160,
                192,
                224,
                256,
                320,
                0,
            )
        else:
            table = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0)

    kbps = int(table[bitrate_idx])
    return kbps if kbps > 0 else None


def _mp3_samples_per_frame(*, mpeg_version: int, layer: int) -> int | None:
    if layer == 1:
        return 384
    if layer == 2:
        return 1152
    if layer == 3:
        # MPEG-1 Layer III: 1152 samples/frame; MPEG-2/2.5: 576 samples/frame
        return 1152 if mpeg_version == 1 else 576
    return None


def _mp3_frame_length_bytes(
    *,
    mpeg_version: int,
    layer: int,
    bitrate_kbps: int,
    sample_rate: int,
    padding: int,
) -> int | None:
    if sample_rate <= 0 or bitrate_kbps <= 0:
        return None
    bitrate = bitrate_kbps * 1000
    if layer == 1:
        return int(((12 * bitrate) // sample_rate + padding) * 4)
    if layer == 2:
        return int((144 * bitrate) // sample_rate + padding)
    if layer == 3:
        # For MPEG-2/2.5 Layer III, frame uses 576 samples => 72 multiplier.
        coef = 144 if mpeg_version == 1 else 72
        return int((coef * bitrate) // sample_rate + padding)
    return None


def _http_post_stream_metrics(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
) -> tuple[bool, str | None, float | None, float | None, int, float | None]:
    """Blocking helper: POST JSON and measure TTFB, wall time, and audio duration.

    Returns: ok, error, ttfb_s, wall_s, total_bytes, audio_s
    """

    import http.client
    import ssl

    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        return False, f"unsupported URL scheme: {u.scheme}", None, None, 0, None

    host = u.hostname
    if not host:
        return False, "invalid URL host", None, None, 0, None

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
    total_bytes = 0
    counter = _Mp3FrameCounter()
    first_byte_t: float | None = None

    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()

        if resp.status != 200:
            try:
                err_b = resp.read(4096)
            except Exception:
                err_b = b""
            msg = err_b.decode("utf-8", errors="replace")
            end_t = time.perf_counter()
            return (
                False,
                f"HTTP {resp.status}: {msg}".strip(),
                None,
                end_t - start_t,
                0,
                None,
            )

        b = resp.read(1)
        first_byte_t = time.perf_counter()
        if b:
            total_bytes += len(b)
            counter.feed(b)

        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            total_bytes += len(chunk)
            counter.feed(chunk)

        end_t = time.perf_counter()
        if first_byte_t is None:
            first_byte_t = end_t
        audio_s = counter.finish()
        return True, None, first_byte_t - start_t, end_t - start_t, total_bytes, audio_s
    except Exception as e:
        end_t = time.perf_counter()
        ttfb_s = (first_byte_t - start_t) if first_byte_t is not None else None
        return (
            False,
            f"{type(e).__name__}: {e}",
            ttfb_s,
            end_t - start_t,
            total_bytes,
            None,
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


async def _consume_one_in_process(
    server: Any,
    *,
    target_text: str,
    max_generate_length: int,
    temperature: float,
    cfg_value: float,
    lora_name: str | None,
) -> tuple[float, float, int]:
    """Return: ttfb_s, wall_s, total_samples."""

    start = time.perf_counter()
    first_chunk_t: float | None = None
    total_samples = 0

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

    end = time.perf_counter()
    if first_chunk_t is None:
        first_chunk_t = end
    return first_chunk_t - start, end - start, total_samples


@dataclass(frozen=True)
class BenchSummary:
    num_users: int
    duration_s: float
    warmup_s: float
    total_started_measured: int
    achieved_rps_started: float
    total_ok_measured: int
    total_err_measured: int
    ttfb_p50_s: float
    ttfb_p90_s: float
    ttfb_p95_s: float
    ttfb_p99_s: float
    ttfb_mean_s: float
    ttfb_stdev_s: float
    rtf_p50: float | None
    rtf_p90: float | None
    rtf_p95: float | None
    rtf_p99: float | None
    rtf_mean: float | None
    rtf_stdev: float | None


async def async_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark VoxCPM with closed-loop N-user load")
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
        "--num-users",
        type=int,
        default=100,
        help="Number of concurrent closed-loop users",
    )
    p.add_argument("--warmup-s", type=float, default=5.0, help="Warmup time before measurement")
    p.add_argument("--duration-s", type=float, default=60.0, help="Measurement window length")
    p.add_argument(
        "--drain-timeout-s",
        type=float,
        default=600.0,
        help="After measurement ends, wait up to this long for in-flight requests to finish",
    )

    p.add_argument(
        "--http-timeout-s",
        type=float,
        default=600.0,
        help="(HTTP mode) socket connect/read timeout per request",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Write results JSON to this path (includes per-request results)",
    )
    args = p.parse_args(argv)

    if args.url is None and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this project does not support CPU-only in-process benchmarking")

    if args.target_text_file is not None:
        args.target_text = open(args.target_text_file, "r", encoding="utf-8").read().strip()
        if not args.target_text:
            raise ValueError("target text is empty")

    if args.num_users <= 0:
        raise ValueError("--num-users must be >= 1")
    if args.warmup_s < 0:
        raise ValueError("--warmup-s must be >= 0")
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be > 0")
    if args.url is None and not args.model:
        raise ValueError("in-process mode requires --model (or use --url for HTTP mode)")
    validate_lora_args(args)

    devices = _parse_devices(args.devices)
    lora_names = make_lora_names(args)
    lora_rng = make_lora_rng(args)

    server_pool: AsyncVoxCPMServerPool | None = None
    sample_rate: int | None = None
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

    # Initialized after the engine is ready; defined here for closure capture.
    measure_start_t = 0.0
    measure_end_t = 0.0

    results: list[OneRequestResult] = []
    results_lock = asyncio.Lock()
    unregister_errors: list[str] = []

    async def _record(r: OneRequestResult) -> None:
        async with results_lock:
            results.append(r)

    async def _user_loop(user_id: int) -> None:
        user_seq = 0
        while True:
            started_t = time.perf_counter()
            if started_t >= measure_end_t:
                return

            if args.url is not None:
                lora_name = choose_lora_name(lora_names, lora_rng)
                ok, err, ttfb_s, wall_s, total_bytes, audio_s = await asyncio.to_thread(
                    _http_post_stream_metrics,
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
                )
                rtf = (
                    _rtf_excluding_ttfb(wall_s, ttfb_s, audio_s)
                    if (ok and wall_s is not None and ttfb_s is not None and audio_s is not None and audio_s > 0)
                    else None
                )
                await _record(
                    OneRequestResult(
                        user_id=user_id,
                        user_seq=user_seq,
                        ok=bool(ok),
                        error=err,
                        started_t=started_t,
                        ttfb_s=ttfb_s,
                        wall_s=wall_s,
                        total_samples=0,
                        total_bytes=int(total_bytes),
                        audio_s=audio_s,
                        rtf=rtf,
                    )
                )
                user_seq += 1
                continue

            assert server_pool is not None
            try:
                ttfb_s, wall_s, total_samples = await _consume_one_in_process(
                    server_pool,
                    target_text=args.target_text,
                    max_generate_length=int(args.max_generate_length),
                    temperature=float(args.temperature),
                    cfg_value=float(args.cfg_value),
                    lora_name=choose_lora_name(lora_names, lora_rng),
                )
                audio_s = (total_samples / float(sample_rate)) if (sample_rate and sample_rate > 0) else None
                rtf = _rtf_excluding_ttfb(wall_s, ttfb_s, audio_s) if (audio_s is not None and audio_s > 0) else None
                await _record(
                    OneRequestResult(
                        user_id=user_id,
                        user_seq=user_seq,
                        ok=True,
                        error=None,
                        started_t=started_t,
                        ttfb_s=ttfb_s,
                        wall_s=wall_s,
                        total_samples=int(total_samples),
                        total_bytes=0,
                        audio_s=audio_s,
                        rtf=rtf,
                    )
                )
            except Exception as e:
                end_t = time.perf_counter()
                await _record(
                    OneRequestResult(
                        user_id=user_id,
                        user_seq=user_seq,
                        ok=False,
                        error=f"{type(e).__name__}: {e}",
                        started_t=started_t,
                        ttfb_s=None,
                        wall_s=end_t - started_t,
                        total_samples=0,
                        total_bytes=0,
                        audio_s=None,
                        rtf=None,
                    )
                )
            user_seq += 1

    try:
        if server_pool is not None:
            await server_pool.wait_for_ready()
            await register_loras_in_process(server_pool, lora_names, args.lora_path)
            try:
                model_info = await server_pool.get_model_info()
                sample_rate = int(model_info["sample_rate"])
            except Exception:
                sample_rate = None
        elif args.url is not None:
            await register_loras_http(args.url, lora_names, args.lora_path, timeout_s=float(args.http_timeout_s))

        t0 = time.perf_counter()
        measure_start_t = t0 + float(args.warmup_s)
        measure_end_t = measure_start_t + float(args.duration_s)

        # Start all user loops immediately; warmup is handled by filtering on started_t.
        user_tasks = [asyncio.create_task(_user_loop(i)) for i in range(int(args.num_users))]

        # Wait for measurement window to end, then wait for loops to finish their last request.
        now = time.perf_counter()
        if measure_end_t > now:
            await asyncio.sleep(measure_end_t - now)

        try:
            await asyncio.wait_for(asyncio.gather(*user_tasks), timeout=float(args.drain_timeout_s))
        except asyncio.TimeoutError:
            for t in user_tasks:
                t.cancel()
            await asyncio.gather(*user_tasks, return_exceptions=True)
    finally:
        if server_pool is not None:
            unregister_errors = await unregister_loras_in_process(server_pool, lora_names)
            await server_pool.stop()
        elif args.url is not None:
            unregister_errors = await unregister_loras_http(args.url, lora_names, timeout_s=float(args.http_timeout_s))

    # Summarize (measured window only).
    measured = [r for r in results if r.started_t >= measure_start_t and r.started_t < measure_end_t]
    measured_ok = [r for r in measured if r.ok]
    measured_err = [r for r in measured if not r.ok]

    ttfbs = [r.ttfb_s for r in measured_ok if r.ttfb_s is not None]
    rtfs = [r.rtf for r in measured_ok if r.rtf is not None]

    total_started_measured = len(measured)
    achieved_rps_started = total_started_measured / float(args.duration_s)

    summary = BenchSummary(
        num_users=int(args.num_users),
        duration_s=float(args.duration_s),
        warmup_s=float(args.warmup_s),
        total_started_measured=total_started_measured,
        achieved_rps_started=achieved_rps_started,
        total_ok_measured=len(measured_ok),
        total_err_measured=len(measured_err),
        ttfb_p50_s=_percentile(ttfbs, 50),
        ttfb_p90_s=_percentile(ttfbs, 90),
        ttfb_p95_s=_percentile(ttfbs, 95),
        ttfb_p99_s=_percentile(ttfbs, 99),
        ttfb_mean_s=_mean(ttfbs),
        ttfb_stdev_s=_stdev(ttfbs),
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
    if sample_rate is not None:
        print(f"  sample_rate: {sample_rate}")
    print(f"  users: {summary.num_users}")
    if lora_names:
        print(f"  loras: {len(lora_names)} aliases from {args.lora_path}")
    if unregister_errors:
        print("  lora cleanup warnings:")
        for msg in unregister_errors:
            print(f"    {msg}")
    print(f"  duration_s: {summary.duration_s} (warmup {summary.warmup_s})")
    print("Results (measured window)")
    print(f"  started: {summary.total_started_measured} (achieved {summary.achieved_rps_started:.2f} rps)")
    print(f"  ok: {summary.total_ok_measured}")
    print(f"  err: {summary.total_err_measured}")
    print("TTFB (seconds) over ok requests")
    print(f"  p50: {_fmt_float(summary.ttfb_p50_s)}")
    print(f"  p90: {_fmt_float(summary.ttfb_p90_s)}")
    print(f"  p95: {_fmt_float(summary.ttfb_p95_s)}")
    print(f"  p99: {_fmt_float(summary.ttfb_p99_s)}")
    print(f"  mean +/- stdev: {_fmt_float(summary.ttfb_mean_s)} +/- {_fmt_float(summary.ttfb_stdev_s)}")
    if summary.rtf_mean is not None:
        print("RTF (wall/audio) over ok requests")
        print(f"  p50: {_fmt_float(summary.rtf_p50 or float('nan'))}")
        print(f"  p90: {_fmt_float(summary.rtf_p90 or float('nan'))}")
        print(f"  p95: {_fmt_float(summary.rtf_p95 or float('nan'))}")
        print(f"  p99: {_fmt_float(summary.rtf_p99 or float('nan'))}")
        print(f"  mean +/- stdev: {_fmt_float(summary.rtf_mean)} +/- {_fmt_float(summary.rtf_stdev or 0.0)}")
    else:
        print("RTF: unavailable (missing/failed audio duration estimate)")

    payload: dict[str, Any] = {
        "args": vars(args),
        "devices": devices,
        "sample_rate": sample_rate,
        "measure_start_t": measure_start_t,
        "measure_end_t": measure_end_t,
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
