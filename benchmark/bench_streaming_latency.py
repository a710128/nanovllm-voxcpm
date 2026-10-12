"""Compare API-level first-chunk and inter-chunk latency at concurrency one.

Run each backend in a separate process against the same local model and text.
Both APIs yield CPU waveforms; the nano measurement also includes its worker IPC.
Initialization is measured separately. Warmup records remain in the JSON so a
first-request cost cannot accidentally be presented as steady-state latency.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time


def summarize_request(start: float, arrivals: list[float], samples: list[int], sample_rate: int) -> dict:
    if not arrivals:
        raise RuntimeError("Generation produced no nonempty audio chunks")
    gaps_ms = [(current - previous) * 1000 for previous, current in zip(arrivals, arrivals[1:])]
    return {
        "ttfb_ms": (arrivals[0] - start) * 1000,
        "wall_s": time.perf_counter() - start,
        "chunk_arrival_ms": [(arrival - start) * 1000 for arrival in arrivals],
        "chunk_samples": samples,
        "inter_chunk_ms": gaps_ms,
        "first_chunk_audio_ms": samples[0] / sample_rate * 1000,
        "audio_s": sum(samples) / sample_rate,
        "num_chunks": len(samples),
    }


async def run_nano(args: argparse.Namespace, text: str) -> tuple[float, int, str, list[dict]]:
    import nanovllm_voxcpm
    from nanovllm_voxcpm import VoxCPM

    start = time.perf_counter()
    pool = VoxCPM.from_pretrained(
        model=args.model,
        devices=[0],
        inference_timesteps=args.inference_timesteps,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    records = []
    try:
        await pool.wait_for_ready()
        info = await pool.get_model_info()
        sample_rate = int(info["sample_rate"])
        initialization_s = time.perf_counter() - start
        for index in range(args.warmup + args.iters):
            if args.request_gap_ms:
                await asyncio.sleep(args.request_gap_ms / 1000)
            arrivals, samples = [], []
            start = time.perf_counter()
            async for chunk in pool.generate(
                target_text=text,
                max_generate_length=args.max_generate_length,
                temperature=1.0,
                cfg_value=args.cfg_value,
                seed=args.seed,
            ):
                if chunk.size:
                    arrivals.append(time.perf_counter())
                    samples.append(int(chunk.size))
            record = summarize_request(start, arrivals, samples, sample_rate)
            record.update(index=index, warmup=index < args.warmup)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        await pool.stop()
    return initialization_s, sample_rate, nanovllm_voxcpm.__file__, records


def run_native(args: argparse.Namespace, text: str) -> tuple[float, int, str, list[dict]]:
    if args.native_source:
        source = Path(args.native_source).resolve()
        if not (source / "voxcpm" / "__init__.py").is_file():
            raise FileNotFoundError(f"Expected a voxcpm package under {source}")
        sys.path.insert(0, str(source))

    import numpy as np
    import torch
    import voxcpm
    from voxcpm import VoxCPM

    start = time.perf_counter()
    model = VoxCPM.from_pretrained(
        args.model,
        load_denoiser=False,
        optimize=not args.enforce_eager,
    )
    vae = model.tts_model.audio_vae
    sample_rate = int(getattr(vae, "out_sample_rate", vae.sample_rate))
    torch.cuda.synchronize()
    initialization_s = time.perf_counter() - start
    records = []
    accepts_seed = "seed" in inspect.signature(model._generate).parameters
    for index in range(args.warmup + args.iters):
        if args.request_gap_ms:
            time.sleep(args.request_gap_ms / 1000)
        arrivals, samples = [], []
        # VoxCPM 2.0.3 predates the per-request seed argument.
        if not accepts_seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        stream = model.generate_streaming(
            text=text,
            max_len=args.max_generate_length,
            inference_timesteps=args.inference_timesteps,
            cfg_value=args.cfg_value,
            normalize=False,
            denoise=False,
            retry_badcase=False,
            **({"seed": args.seed} if accepts_seed else {}),
        )
        try:
            for chunk in stream:
                if chunk.size:
                    arrivals.append(time.perf_counter())
                    samples.append(int(chunk.size))
        finally:
            stream.close()
        record = summarize_request(start, arrivals, samples, sample_rate)
        record.update(index=index, warmup=index < args.warmup)
        record["seed_mode"] = "request_argument" if accepts_seed else "global_rng"
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return initialization_s, sample_rate, voxcpm.__file__, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nano", "native"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-source", help="Directory containing the official voxcpm package")
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument("--target-text", default="Hello world.")
    text_group.add_argument("--target-text-file")
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-generate-length", type=int, default=2000)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--request-gap-ms", type=float, default=0, help="Idle time before each request, outside timing")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be nonnegative and --iters must be positive")
    if args.request_gap_ms < 0:
        parser.error("--request-gap-ms must be nonnegative")
    text = (
        Path(args.target_text_file).read_text(encoding="utf-8").strip() if args.target_text_file else args.target_text
    )
    if not text.strip():
        parser.error("Target text must not be empty")
    args.model = str(Path(args.model).expanduser().resolve())
    config_path = Path(args.model) / "config.json"
    config_bytes = config_path.read_bytes()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    initialization_s, sample_rate, package_file, records = (
        asyncio.run(run_nano(args, text)) if args.backend == "nano" else run_native(args, text)
    )
    measured = [record for record in records if not record["warmup"]]
    gaps = [gap for record in measured for gap in record["inter_chunk_ms"]]
    summary = {
        "ttfb_ms_median": statistics.median(record["ttfb_ms"] for record in measured),
        "ttfb_ms_mean": statistics.mean(record["ttfb_ms"] for record in measured),
        "inter_chunk_ms_median": statistics.median(gaps) if gaps else None,
        "inter_chunk_ms_mean": statistics.mean(gaps) if gaps else None,
        "first_chunk_audio_ms_median": statistics.median(record["first_chunk_audio_ms"] for record in measured),
        "wall_s_mean": statistics.mean(record["wall_s"] for record in measured),
        "audio_s_mean": statistics.mean(record["audio_s"] for record in measured),
    }
    output = {
        "args": vars(args),
        "text": text,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "package_file": package_file,
            "model_config": json.loads(config_bytes),
            "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "nano_environment": {key: value for key, value in os.environ.items() if key.startswith("NANOVLLM_")},
        },
        "initialization_s": initialization_s,
        "sample_rate": sample_rate,
        "summary": summary,
        "requests": records,
    }
    output_path = Path(args.json_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
