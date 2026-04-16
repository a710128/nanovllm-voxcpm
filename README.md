# Nano-vLLM-VoxCPM

An inference engine for VoxCPM based on Nano-vLLM.

Features:
- Faster than the pytorch implementation
- Support concurrent requests
- Friendly async API (can be wrapped by an HTTP server; see `deployment/README.md`)

This repository contains a Python package (`nanovllm_voxcpm/`) plus an optional FastAPI demo.

## Installation

### Install from PyPI

Core package:

```bash
pip install nano-vllm-voxcpm
```

Or with `uv`:

```bash
uv pip install nano-vllm-voxcpm
```

Note: the optional FastAPI demo service (`deployment/`) is not published on PyPI.

### Prerequisites

- Linux + NVIDIA GPU (CUDA)
- Python >= 3.10
- `flash-attn` is required (the package imports it at runtime)

The runtime is GPU-centric (Triton + FlashAttention). CPU-only execution is not supported.

### Install from source (dev)

This repo uses `uv` and includes a lockfile (`uv.lock`).

```bash
uv sync --frozen
```

Dev deps (tests):

```bash
uv sync --frozen --dev
```

Note: `flash-attn` may require additional system CUDA tooling depending on your environment.

## Basic Usage

See `example.py` for an end-to-end async example.

Quickstart:

```bash
uv run python example.py
```

### Load a model

`VoxCPM.from_pretrained(...)` accepts either:

- a local model directory path, or
- a HuggingFace repo id (it will download via `huggingface_hub.snapshot_download`).

The model directory is expected to contain:

- `config.json`
- one or more `*.safetensors` weight files
- `audiovae.pth` (VAE weights)

### Generate (async)

If you call `from_pretrained()` inside an async event loop, it returns an `AsyncVoxCPMServerPool`.

```python
import asyncio
import numpy as np

from nanovllm_voxcpm import VoxCPM


async def main() -> None:
    server = VoxCPM.from_pretrained(
        model="/path/to/VoxCPM",
        devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.95,
    )
    await server.wait_for_ready()

    chunks = []
    async for chunk in server.generate(target_text="Hello world"):
        chunks.append(chunk)  # each chunk is a float32 numpy array

    wav = np.concatenate(chunks, axis=0)
    # Write with the model's sample rate (see your model's AudioVAE config; often 16000)
    # import soundfile as sf; sf.write("out.wav", wav, sample_rate)

    await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

### Generate (sync)

If you call `from_pretrained()` outside an event loop, it returns a `SyncVoxCPMServerPool`.

```python
import numpy as np

from nanovllm_voxcpm import VoxCPM


server = VoxCPM.from_pretrained(model="/path/to/VoxCPM", devices=[0])
chunks = []
for chunk in server.generate(target_text="Hello world"):
    chunks.append(chunk)
wav = np.concatenate(chunks, axis=0)
server.stop()
```

### Prompting and reference audio (optional)

The VoxCPM2 server supports these conditioning inputs:

- zero-shot: no prompt or reference audio
- prompt continuation: provide `prompt_latents` + `prompt_text`
- stored prompt: provide a `prompt_id` (via `add_prompt`) and then generate with that id
- reference audio: provide `ref_audio_latents` to add a separate reference-audio condition

`ref_audio_latents` is independent from `prompt_latents`:

- use `prompt_latents` when you want to continue from an existing audio prefix
- use `ref_audio_latents` when you want to provide extra reference audio without treating it as the decode prefix

See the public API in `nanovllm_voxcpm/models/voxcpm2/server.py` for details.

## FastAPI demo

The HTTP server demo is documented separately to keep this README focused:

- `deployment/README.md`

If you want the deployment server dependencies too, use:

```bash
uv sync --all-packages --frozen
```

## Benchmark

The `benchmark/` directory contains an end-to-end inference benchmark that drives
the public server API and reports throughput/latency metrics.

Quick run:

```bash
uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --devices 0 --concurrency 1 --warmup 1 --iters 5
```

Use a longer English prompt (~100 words) for more stable results:

```bash
uv run python benchmark/bench_inference.py --model ~/VoxCPM1.5 --devices 0 --concurrency 1 --warmup 1 --iters 5 \
  --target-text-file benchmark/target_text_100w_en.txt
```

See `benchmark/README.md` for more flags.

### Reference Results (RTX 4090)

All reference numbers in this section are measured on NVIDIA GeForce RTX 4090 with `openbmb/VoxCPM2`.
The benchmark now defines `RTF_per_req_mean` as the mean over requests of `((request_wall_time - TTFB) / request_audio_duration)` under the given concurrency.

Short prompt, no LoRA:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.1948 ± 0.0008 | 0.1948 ± 0.0008 | 0.0983 ± 0.0027 |
| 8 | 0.2062 ± 0.0053 | 0.2065 ± 0.0053 | 0.1429 ± 0.0046 |
| 16 | 0.1959 ± 0.0022 | 0.1963 ± 0.0022 | 0.2221 ± 0.0069 |
| 32 | 0.2133 ± 0.0011 | 0.2151 ± 0.0010 | 0.3927 ± 0.0108 |
| 64 | 0.2733 ± 0.0847 | 0.2767 ± 0.0849 | 0.6958 ± 0.0347 |

Long prompt, no LoRA:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.2067 ± 0.0036 | 0.2067 ± 0.0036 | 0.1252 ± 0.0005 |
| 8 | 0.3316 ± 0.0546 | 0.3322 ± 0.0548 | 0.2076 ± 0.0086 |
| 16 | 0.2449 ± 0.0236 | 0.2456 ± 0.0235 | 0.3223 ± 0.0054 |
| 32 | 0.3365 ± 0.0116 | 0.3393 ± 0.0118 | 0.5517 ± 0.0075 |
| 64 | 0.5795 ± 0.0546 | 0.5834 ± 0.0544 | 1.0146 ± 0.0077 |

Short prompt, LoRA enabled with 32 runtime slots:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.4568 ± 0.0048 | 0.4568 ± 0.0048 | 0.1495 ± 0.0028 |
| 8 | 0.6041 ± 0.1172 | 0.6045 ± 0.1172 | 0.2048 ± 0.0039 |
| 16 | 0.5892 ± 0.1392 | 0.5899 ± 0.1393 | 0.3025 ± 0.0040 |
| 32 | 0.6446 ± 0.2677 | 0.6460 ± 0.2679 | 0.5300 ± 0.0554 |
| 64 | 0.4904 ± 0.0579 | 0.4931 ± 0.0575 | 0.8623 ± 0.0131 |
| 128 | 0.7240 ± 0.2278 | 0.7805 ± 0.1791 | 1.7254 ± 0.0873 |

Long prompt, LoRA enabled with 32 runtime slots:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.4653 ± 0.0052 | 0.4653 ± 0.0052 | 0.1815 ± 0.0008 |
| 8 | 0.6196 ± 0.1261 | 0.6200 ± 0.1260 | 0.2550 ± 0.0028 |
| 16 | 0.6078 ± 0.1007 | 0.6082 ± 0.1007 | 0.3943 ± 0.0014 |
| 32 | 0.5718 ± 0.1215 | 0.5727 ± 0.1215 | 0.6410 ± 0.0028 |
| 64 | 0.7754 ± 0.0811 | 0.7785 ± 0.0814 | 1.1209 ± 0.0024 |

Closed-loop results:

| mode | users | registered LoRAs | started | achieved rps | ok | err |
|---|---:|---:|---:|---:|---:|---:|
| no LoRA | 60 | 0 | 65 | 1.08 | 65 | 0 |
| LoRA | 30 | 32 | 52 | 0.87 | 52 | 0 |
| LoRA | 30 | 128 | 48 | 0.80 | 48 | 0 |
| LoRA | 30 | 256 | 51 | 0.85 | 51 | 0 |

Closed-loop TTFB (seconds, ok requests):

| mode | users | registered LoRAs | p50 | p90 | p95 | p99 | mean | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no LoRA | 60 | 0 | 0.3867 | 0.4602 | 0.4649 | 0.4933 | 0.3819 | 0.0690 |
| LoRA | 30 | 32 | 0.5137 | 0.8142 | 0.8539 | 0.8970 | 0.5692 | 0.1243 |
| LoRA | 30 | 128 | 0.5595 | 0.9116 | 0.9310 | 0.9671 | 0.6267 | 0.1426 |
| LoRA | 30 | 256 | 0.5907 | 0.9143 | 0.9165 | 0.9737 | 0.6459 | 0.1487 |

Closed-loop RTF ((wall - TTFB)/audio, ok requests):

| mode | users | registered LoRAs | p50 | p90 | p95 | p99 | mean | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no LoRA | 60 | 0 | 1.0802 | 1.1958 | 1.2033 | 1.2090 | 1.0288 | 0.2077 |
| LoRA | 30 | 32 | 0.8037 | 0.9202 | 0.9375 | 0.9615 | 0.7665 | 0.1395 |
| LoRA | 30 | 128 | 0.8263 | 0.9532 | 0.9844 | 1.0188 | 0.7752 | 0.1598 |
| LoRA | 30 | 256 | 0.8300 | 0.9217 | 0.9632 | 0.9938 | 0.7741 | 0.1487 |

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

## License

MIT License

## Known Issue

If you see the errors below:
```
ValueError: Missing parameters: ['base_lm.embed_tokens.weight', 'base_lm.layers.0.self_attn.qkv_proj.weight', ... , 'stop_proj.weight', 'stop_proj.bias', 'stop_head.weight']
[rank0]:[W1106 07:26:04.469150505 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```

It's because nanovllm loads model parameters from `*.safetensors`, but some VoxCPM releases ship weights as `.pt`.

Fix:

- use a safetensors-converted checkpoint (or convert the checkpoint yourself)
- ensure the `*.safetensors` files live next to `config.json` in the model directory
