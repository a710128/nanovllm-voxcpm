"""Generate a deterministic waveform and optionally compare it with a reference."""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np


def compare_waveforms(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int | bool]:
    if actual.size == 0 or reference.size == 0:
        raise ValueError("waveform comparison requires non-empty actual and reference arrays")
    common_samples = min(actual.size, reference.size)
    actual_common = actual[:common_samples].astype(np.float64)
    reference_common = reference[:common_samples].astype(np.float64)
    error = actual_common - reference_common
    centered_actual = actual_common - float(actual_common.mean())
    centered_reference = reference_common - float(reference_common.mean())
    denominator = np.linalg.norm(centered_actual) * np.linalg.norm(centered_reference)
    if denominator:
        correlation = float(np.dot(centered_actual, centered_reference) / denominator)
    else:
        correlation = 1.0 if np.array_equal(actual_common, reference_common) else 0.0
    return {
        "actual_samples": int(actual.size),
        "reference_samples": int(reference.size),
        "common_samples": common_samples,
        "length_match": actual.size == reference.size,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "abs_error_p99_9": float(np.percentile(np.abs(error), 99.9)),
        "max_abs_error": float(np.max(np.abs(error))),
        "correlation": correlation,
    }


def comparison_failures(
    comparison: dict[str, float | int | bool],
    *,
    require_same_length: bool,
    max_rmse: float | None,
    max_abs_error: float | None,
    min_correlation: float | None,
) -> list[str]:
    failures = []
    if require_same_length and not comparison["length_match"]:
        failures.append(
            f"sample count mismatch: actual={comparison['actual_samples']} reference={comparison['reference_samples']}"
        )
    if max_rmse is not None and comparison["rmse"] > max_rmse:
        failures.append(f"rmse {comparison['rmse']:.6g} exceeds {max_rmse:.6g}")
    if max_abs_error is not None and comparison["max_abs_error"] > max_abs_error:
        failures.append(f"max_abs_error {comparison['max_abs_error']:.6g} exceeds {max_abs_error:.6g}")
    if min_correlation is not None and comparison["correlation"] < min_correlation:
        failures.append(f"correlation {comparison['correlation']:.9g} is below {min_correlation:.9g}")
    return failures


async def async_main(args: argparse.Namespace) -> None:
    from nanovllm_voxcpm import VoxCPM

    if args.concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    target_text = (
        Path(args.target_text_file).read_text(encoding="utf-8").strip() if args.target_text_file else args.target_text
    )
    server_pool = VoxCPM.from_pretrained(
        model=args.model,
        inference_timesteps=args.inference_timesteps,
        max_num_batched_tokens=16384,
        max_num_seqs=512,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        devices=[args.device],
        enforce_eager=args.enforce_eager,
        enable_prefill_diffusion_cudagraph=not args.disable_prefill_diffusion_cudagraph,
    )

    async def generate_one(seed: int) -> np.ndarray:
        chunks = []
        async for chunk in server_pool.generate(
            target_text=target_text,
            max_generate_length=args.max_generate_length,
            temperature=args.temperature,
            cfg_value=args.cfg_value,
            seed=seed,
        ):
            chunks.append(np.asarray(chunk, dtype=np.float32))
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)

    try:
        await server_pool.wait_for_ready()
        waveforms = await asyncio.gather(*(generate_one(args.seed + index) for index in range(args.concurrency)))
    finally:
        await server_pool.stop()

    if args.concurrency == 1:
        np.save(args.output, waveforms[0])
    else:
        np.savez(args.output, **{f"request_{index}": waveform for index, waveform in enumerate(waveforms)})
    result: dict[str, object] = {
        "output": str(Path(args.output).resolve()),
        "num_samples": [int(waveform.size) for waveform in waveforms],
    }
    failures = []
    if args.reference:
        reference = np.load(args.reference)
        if args.concurrency == 1:
            comparisons = [compare_waveforms(waveforms[0], reference)]
            result["comparison"] = comparisons[0]
        else:
            comparisons = [
                compare_waveforms(waveform, reference[f"request_{index}"]) for index, waveform in enumerate(waveforms)
            ]
            result["comparison"] = comparisons
        for index, comparison in enumerate(comparisons):
            request_failures = comparison_failures(
                comparison,
                require_same_length=not args.allow_length_mismatch,
                max_rmse=args.max_rmse,
                max_abs_error=args.max_abs_error,
                min_correlation=args.min_correlation,
            )
            failures.extend(f"request_{index}: {failure}" for failure in request_failures)
        result["comparison_passed"] = not failures
        if failures:
            result["failures"] = failures
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--target-text", default="Hello world.")
    parser.add_argument("--target-text-file")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--max-generate-length", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefill-diffusion-cudagraph", action="store_true")
    parser.add_argument("--allow-length-mismatch", action="store_true")
    parser.add_argument("--max-rmse", type=float)
    parser.add_argument("--max-abs-error", type=float)
    parser.add_argument("--min-correlation", type=float)
    asyncio.run(async_main(parser.parse_args()))


if __name__ == "__main__":
    main()
