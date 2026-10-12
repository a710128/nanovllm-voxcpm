# Benchmark

End-to-end inference benchmarking for VoxCPM.

## First Audio Versus the Official Implementation

Run each backend in a separate process with the same local model weights:

```bash
uv run python benchmark/bench_streaming_latency.py --backend nano --model ~/VoxCPM2 \
  --target-text-file benchmark/target_text_100w_en.txt --warmup 5 --iters 20 \
  --max-generate-length 32 --request-gap-ms 50 --json-out nano-latency.json

# Use a Python environment containing the official voxcpm package, or pass
# --native-source /path/to/directory/containing/voxcpm to select its source.
uv run python benchmark/bench_streaming_latency.py --backend native --model ~/VoxCPM2 \
  --target-text-file benchmark/target_text_100w_en.txt --warmup 5 --iters 20 \
  --max-generate-length 32 --request-gap-ms 50 --json-out native-latency.json
```

TTFB ends when the first nonempty CPU waveform arrives. Nano includes worker IPC;
the official implementation is called in-process. Neither measurement includes HTTP.
The request gap is outside the timer. Keep the default queue-coalescing policy
unless comparing that policy explicitly. Each JSON contains warmup records,
per-chunk arrivals and sample counts, initialization time, and environment details.

The official backend disables normalization, denoising, and retry. Identical seeds
do not imply identical samples across implementations. Compare the first chunk's
audio duration as well as latency, and do not mix cold-request times with warmed
medians. The 32-chunk cap is useful for latency experiments; use
`--max-generate-length 2000` to compare complete generation.

## Run

```bash
uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --concurrency 4 --iters 5 --warmup 1 --seed 42
```

Fixed-RPS TTFB (open-loop) for long-audio load:

```bash
uv run python benchmark/bench_open_loop_users.py --model ~/VoxCPM1.5 --rps 30 --duration-s 60 \
  --target-text-file benchmark/target_text_100w_en.txt --max-generate-length 2000
```

In in-process mode, the script also reports RTF, computed as `(request_wall_time - TTFB) / generated_audio_seconds`.

You can also benchmark the deployment service endpoint:

```bash
uv run python benchmark/bench_open_loop_users.py --url http://127.0.0.1:8000/generate --rps 30 --duration-s 60 \
  --target-text-file benchmark/target_text_100w_en.txt --max-generate-length 2000 --http-consume-full
```

Key flags:

- `--concurrency`: number of concurrent `generate()` requests
- `--max-generate-length`: maximum number of generation steps per request
- `--devices`: CUDA devices that exist locally, e.g. `0` or `0,1`
- `--json-out`: write machine-readable results
- `--max-loras`: enable LoRA when greater than `0` and register that many aliases from `--lora-path`
- `--seed`: optional fixed random seed for reproducible generation

Advanced KV-cache tuning:

- `NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS` manually overrides automatic KV-cache sizing for in-process
  benchmark runs. Leave it unset for the normal safe memory calculation.
- On low-VRAM Windows development machines, this can be useful when you have validated a smaller block
  count locally. For example, in PowerShell: `$env:NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS = "64"`.
- Tensor parallelism is not supported on Windows; `--devices 0,1` requires both CUDA device ordinals to
  exist and does not make a single-GPU machine emulate tensor parallelism.

Closed-loop "N users" benchmark (each user sends the next request immediately after the previous finishes):

```bash
uv run python benchmark/bench_closed_loop_users.py --model ~/VoxCPM1.5 --num-users 60 --duration-s 60 --warmup-s 5 \
  --target-text-file benchmark/target_text_100w_en.txt --max-generate-length 2000
```

LoRA benchmark mode registers `--max-loras` names for one checkpoint path, then randomly assigns one registered
LoRA to every warmup and measured request:

```bash
uv run python benchmark/bench_open_loop_users.py --model ~/VoxCPM1.5 --rps 30 --duration-s 60 \
  --target-text-file benchmark/target_text_100w_en.txt --max-generate-length 2000 \
  --max-loras 4 --lora-path /path/to/lora/checkpoint
```

For HTTP mode, the deployment service must already be started with runtime LoRA enabled and enough capacity/rank to
accept those registrations.

## Notes

- Metrics are measured from the parent process wall time and include IPC overhead.
- If the model directory is local, the script reads `config.json` to infer `sample_rate` for RTF; otherwise provide `--sample-rate`.
- `RTF_per_req_mean` is computed as the average over requests of `((request_wall_time - TTFB) / request_audio_duration)`.
