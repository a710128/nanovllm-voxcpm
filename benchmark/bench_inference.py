"""Benchmark VoxCPM inference throughput/latency.

Run (recommended via uv):
  uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --concurrency 4 --iters 5 --warmup 1

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
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterable, cast

import torch

from lora_bench_utils import (
    add_lora_args,
    build_lora_config,
    choose_lora_name,
    make_lora_names,
    make_lora_rng,
    register_loras_in_process,
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


def _maybe_read_sample_rate(model: str) -> int | None:
    # Best-effort: read from local config.json if model resolves to a directory.
    model_path = os.path.expanduser(model)
    if not os.path.isdir(model_path):
        return None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        cfg = json.loads(open(config_path, "r", encoding="utf-8").read())
    except Exception:
        return None

    # VoxCPMConfig.audio_vae_config.sample_rate default is 16000.
    audio_cfg = cfg.get("audio_vae_config")
    if isinstance(audio_cfg, dict):
        sr = audio_cfg.get("sample_rate")
        if isinstance(sr, int):
            return sr
    return None


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


@dataclass(frozen=True)
class OneRequestResult:
    total_samples: int
    num_chunks: int
    ttfb_s: float
    wall_s: float


async def _consume_one(
    server: Any,
    *,
    target_text: str,
    max_generate_length: int,
    temperature: float,
    cfg_value: float,
    lora_name: str | None,
) -> OneRequestResult:
    start = time.perf_counter()
    first_chunk_t = None
    total_samples = 0
    num_chunks = 0

    async for chunk in server.generate(
        target_text=target_text,
        max_generate_length=max_generate_length,
        temperature=temperature,
        cfg_value=cfg_value,
        lora_name=lora_name,
    ):
        if first_chunk_t is None:
            first_chunk_t = time.perf_counter()

        # chunk is a float32 numpy array (waveform).
        total_samples += int(chunk.shape[0])
        num_chunks += 1

    end = time.perf_counter()
    if first_chunk_t is None:
        first_chunk_t = end
    return OneRequestResult(
        total_samples=total_samples,
        num_chunks=num_chunks,
        ttfb_s=first_chunk_t - start,
        wall_s=end - start,
    )


@dataclass(frozen=True)
class IterationResult:
    concurrency: int
    wall_s: float
    total_samples: int
    total_chunks: int
    ttfb_p50_s: float
    ttfb_p90_s: float
    # Per-request generated audio duration distribution (seconds) for this iteration.
    audio_s_p50: float | None
    audio_s_p90: float | None
    audio_s_p95: float | None
    audio_s_p99: float | None
    audio_s_mean: float | None
    audio_s_stdev: float | None
    # Average per-request generated audio duration (seconds) for this iteration.
    audio_s_per_req_mean: float | None
    # Average per-request RTF for this iteration.
    rtf_per_req_mean: float | None


async def _run_iteration(
    server_pool: Any,
    *,
    concurrency: int,
    target_text: str,
    max_generate_length: int,
    temperature: float,
    cfg_value: float,
    sample_rate: int | None,
    lora_names: list[str],
    lora_rng: Any,
) -> IterationResult:
    start = time.perf_counter()
    tasks = [
        asyncio.create_task(
            _consume_one(
                server_pool,
                target_text=target_text,
                max_generate_length=max_generate_length,
                temperature=temperature,
                cfg_value=cfg_value,
                lora_name=choose_lora_name(lora_names, lora_rng),
            )
        )
        for _ in range(concurrency)
    ]
    results = await asyncio.gather(*tasks)
    end = time.perf_counter()

    ttfbs = [r.ttfb_s for r in results]

    audio_s_per_req_mean: float | None
    rtf_per_req_mean: float | None
    audio_s_p50: float | None
    audio_s_p90: float | None
    audio_s_p95: float | None
    audio_s_p99: float | None
    audio_s_mean: float | None
    audio_s_stdev: float | None
    if sample_rate is not None and sample_rate > 0:
        audio_s_per_req = [r.total_samples / float(sample_rate) for r in results]
        rtfs = [_rtf_excluding_ttfb(r.wall_s, r.ttfb_s, a) for r, a in zip(results, audio_s_per_req)]
        audio_s_per_req_mean = _mean(audio_s_per_req)
        rtf_per_req_mean = _mean(rtfs)

        audio_s_p50 = _percentile(audio_s_per_req, 50)
        audio_s_p90 = _percentile(audio_s_per_req, 90)
        audio_s_p95 = _percentile(audio_s_per_req, 95)
        audio_s_p99 = _percentile(audio_s_per_req, 99)
        audio_s_mean = _mean(audio_s_per_req)
        audio_s_stdev = _stdev(audio_s_per_req)
    else:
        audio_s_per_req_mean = None
        rtf_per_req_mean = None
        audio_s_p50 = None
        audio_s_p90 = None
        audio_s_p95 = None
        audio_s_p99 = None
        audio_s_mean = None
        audio_s_stdev = None

    return IterationResult(
        concurrency=concurrency,
        wall_s=end - start,
        total_samples=sum(r.total_samples for r in results),
        total_chunks=sum(r.num_chunks for r in results),
        ttfb_p50_s=_percentile(ttfbs, 50),
        ttfb_p90_s=_percentile(ttfbs, 90),
        audio_s_p50=audio_s_p50,
        audio_s_p90=audio_s_p90,
        audio_s_p95=audio_s_p95,
        audio_s_p99=audio_s_p99,
        audio_s_mean=audio_s_mean,
        audio_s_stdev=audio_s_stdev,
        audio_s_per_req_mean=audio_s_per_req_mean,
        rtf_per_req_mean=rtf_per_req_mean,
    )


def _fmt_float(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.4f}"


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return statistics.mean(xs) if xs else float("nan")


def _stdev(xs: Iterable[float]) -> float:
    xs = list(xs)
    if len(xs) <= 1:
        return 0.0
    return statistics.stdev(xs)


def _rtf_excluding_ttfb(wall_s: float, ttfb_s: float, audio_s: float) -> float:
    if audio_s <= 0:
        return float("inf")
    return max(wall_s - ttfb_s, 0.0) / audio_s


async def async_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark VoxCPM inference speed")
    p.add_argument("--model", required=True, help="Local model directory (or HF repo id)")
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
    p.add_argument("--max-generate-length", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cfg-value", type=float, default=2.0)
    add_lora_args(p)

    p.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations (not included in stats)",
    )
    p.add_argument("--iters", type=int, default=5, help="Measured iterations")

    p.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Override sample rate for RTF calc; otherwise best-effort from config.json (fallback: omit RTF)",
    )
    p.add_argument("--json-out", default=None, help="Write results JSON to this path")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this project does not support CPU-only benchmarking")

    if args.target_text_file is not None:
        args.target_text = open(args.target_text_file, "r", encoding="utf-8").read().strip()
        if not args.target_text:
            raise ValueError("target text is empty")

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.iters <= 0:
        raise ValueError("--iters must be >= 1")
    validate_lora_args(args)

    sample_rate = args.sample_rate
    if sample_rate is None:
        # Best-effort: read from local config.json if model resolves to a directory.
        # We'll prefer the runtime value from model_info once the server is ready.
        sample_rate = _maybe_read_sample_rate(args.model)

    devices = _parse_devices(args.devices)
    lora_names = make_lora_names(args)
    lora_rng = make_lora_rng(args)

    # Import after arg parsing so `--help` works even if optional runtime deps are missing.
    from nanovllm_voxcpm import VoxCPM

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

    iters: list[IterationResult] = []
    try:
        # Async mode: from_pretrained returns AsyncVoxCPMServerPool.
        await server_pool.wait_for_ready()
        await register_loras_in_process(server_pool, lora_names, args.lora_path)

        # Prefer the runtime-reported sample rate from the model server.
        if args.sample_rate is None:
            try:
                model_info = await server_pool.get_model_info()
                sample_rate = int(model_info["sample_rate"])
            except Exception:
                # Keep best-effort config.json inference (or None).
                pass

        # Warmup
        for _ in range(args.warmup):
            await _run_iteration(
                server_pool,
                concurrency=args.concurrency,
                target_text=args.target_text,
                max_generate_length=args.max_generate_length,
                temperature=args.temperature,
                cfg_value=args.cfg_value,
                sample_rate=sample_rate,
                lora_names=lora_names,
                lora_rng=lora_rng,
            )

        # Measure
        for _ in range(args.iters):
            iters.append(
                await _run_iteration(
                    server_pool,
                    concurrency=args.concurrency,
                    target_text=args.target_text,
                    max_generate_length=args.max_generate_length,
                    temperature=args.temperature,
                    cfg_value=args.cfg_value,
                    sample_rate=sample_rate,
                    lora_names=lora_names,
                    lora_rng=lora_rng,
                )
            )
    finally:
        unregister_errors = await unregister_loras_in_process(server_pool, lora_names)
        await server_pool.stop()

    wall_s = [it.wall_s for it in iters]
    total_samples = [it.total_samples for it in iters]
    total_chunks = [it.total_chunks for it in iters]
    ttfb_p50_s = [it.ttfb_p50_s for it in iters]
    ttfb_p90_s = [it.ttfb_p90_s for it in iters]

    audio_s_p50 = [it.audio_s_p50 for it in iters if it.audio_s_p50 is not None]
    audio_s_p90 = [it.audio_s_p90 for it in iters if it.audio_s_p90 is not None]
    audio_s_p95 = [it.audio_s_p95 for it in iters if it.audio_s_p95 is not None]
    audio_s_p99 = [it.audio_s_p99 for it in iters if it.audio_s_p99 is not None]
    audio_s_mean = [it.audio_s_mean for it in iters if it.audio_s_mean is not None]
    audio_s_stdev = [it.audio_s_stdev for it in iters if it.audio_s_stdev is not None]

    audio_s_per_req_mean = [it.audio_s_per_req_mean for it in iters if it.audio_s_per_req_mean is not None]
    rtf_per_req_mean = [it.rtf_per_req_mean for it in iters if it.rtf_per_req_mean is not None]

    samples_per_s = [s / t for s, t in zip(total_samples, wall_s)]
    chunks_per_s = [c / t for c, t in zip(total_chunks, wall_s)]

    audio_s_total: list[float] | None
    if sample_rate is not None and sample_rate > 0:
        audio_s_total = [s / float(sample_rate) for s in total_samples]
    else:
        audio_s_total = None

    print("Benchmark finished")
    print(f"  model: {args.model}")
    print(f"  devices: {devices}")
    print(f"  concurrency: {args.concurrency}")
    if lora_names:
        print(f"  loras: {len(lora_names)} aliases from {args.lora_path}")
    if unregister_errors:
        print("  lora cleanup warnings:")
        for msg in unregister_errors:
            print(f"    {msg}")
    print(f"  iters: {args.iters} (warmup {args.warmup})")
    if sample_rate is not None:
        print(f"  sample_rate: {sample_rate}")
    else:
        print("  sample_rate: unknown (RTF omitted)")
    print("Metrics (mean +/- stdev over measured iterations)")
    print(f"  wall_s: {_fmt_float(_mean(wall_s))} +/- {_fmt_float(_stdev(wall_s))}")
    if audio_s_total is not None:
        print(f"  audio_s_total: {_fmt_float(_mean(audio_s_total))} +/- {_fmt_float(_stdev(audio_s_total))}")
    if audio_s_per_req_mean:
        print(
            f"  audio_s_per_req_mean: {_fmt_float(_mean(audio_s_per_req_mean))} +/- {_fmt_float(_stdev(audio_s_per_req_mean))}"
        )
    if audio_s_p50:
        print("  audio_s_per_req_dist (seconds):")
        print(f"    p50: {_fmt_float(_mean(audio_s_p50))} +/- {_fmt_float(_stdev(audio_s_p50))}")
        print(f"    p90: {_fmt_float(_mean(audio_s_p90))} +/- {_fmt_float(_stdev(audio_s_p90))}")
        print(f"    p95: {_fmt_float(_mean(audio_s_p95))} +/- {_fmt_float(_stdev(audio_s_p95))}")
        print(f"    p99: {_fmt_float(_mean(audio_s_p99))} +/- {_fmt_float(_stdev(audio_s_p99))}")
        print(f"    mean +/- stdev: {_fmt_float(_mean(audio_s_mean))} +/- {_fmt_float(_mean(audio_s_stdev))}")
    if rtf_per_req_mean:
        print(f"  RTF_per_req_mean: {_fmt_float(_mean(rtf_per_req_mean))} +/- {_fmt_float(_stdev(rtf_per_req_mean))}")
    print(f"  samples/s: {_fmt_float(_mean(samples_per_s))} +/- {_fmt_float(_stdev(samples_per_s))}")
    print(f"  chunks/s: {_fmt_float(_mean(chunks_per_s))} +/- {_fmt_float(_stdev(chunks_per_s))}")
    print(f"  TTFB p50 (s): {_fmt_float(_mean(ttfb_p50_s))} +/- {_fmt_float(_stdev(ttfb_p50_s))}")
    print(f"  TTFB p90 (s): {_fmt_float(_mean(ttfb_p90_s))} +/- {_fmt_float(_stdev(ttfb_p90_s))}")

    payload: dict[str, Any] = {
        "args": vars(args),
        "devices": devices,
        "sample_rate": sample_rate,
        "iterations": [asdict(it) for it in iters],
        "summary": {
            "wall_s_mean": _mean(wall_s),
            "wall_s_stdev": _stdev(wall_s),
            "samples_per_s_mean": _mean(samples_per_s),
            "samples_per_s_stdev": _stdev(samples_per_s),
            "chunks_per_s_mean": _mean(chunks_per_s),
            "chunks_per_s_stdev": _stdev(chunks_per_s),
            "ttfb_p50_s_mean": _mean(ttfb_p50_s),
            "ttfb_p50_s_stdev": _stdev(ttfb_p50_s),
            "ttfb_p90_s_mean": _mean(ttfb_p90_s),
            "ttfb_p90_s_stdev": _stdev(ttfb_p90_s),
        },
    }
    if audio_s_total is not None:
        payload["summary"].update(
            {
                "audio_s_total_mean": _mean(audio_s_total),
                "audio_s_total_stdev": _stdev(audio_s_total),
            }
        )
    if audio_s_p50:
        payload["summary"].update(
            {
                "audio_s_p50_mean": _mean(audio_s_p50),
                "audio_s_p50_stdev": _stdev(audio_s_p50),
                "audio_s_p90_mean": _mean(audio_s_p90),
                "audio_s_p90_stdev": _stdev(audio_s_p90),
                "audio_s_p95_mean": _mean(audio_s_p95),
                "audio_s_p95_stdev": _stdev(audio_s_p95),
                "audio_s_p99_mean": _mean(audio_s_p99),
                "audio_s_p99_stdev": _stdev(audio_s_p99),
                "audio_s_mean_mean": _mean(audio_s_mean),
                "audio_s_mean_stdev": _stdev(audio_s_mean),
                "audio_s_stdev_mean": _mean(audio_s_stdev),
                "audio_s_stdev_stdev": _stdev(audio_s_stdev),
            }
        )
    if audio_s_per_req_mean and rtf_per_req_mean:
        payload["summary"].update(
            {
                "audio_s_per_req_mean_mean": _mean(audio_s_per_req_mean),
                "audio_s_per_req_mean_stdev": _stdev(audio_s_per_req_mean),
                "rtf_per_req_mean_mean": _mean(rtf_per_req_mean),
                "rtf_per_req_mean_stdev": _stdev(rtf_per_req_mean),
            }
        )

    if args.json_out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
