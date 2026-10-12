# Nano-vLLM-VoxCPM

An inference engine for VoxCPM based on Nano-vLLM.

Features:
- Faster than the pytorch implementation
- Support concurrent requests
- Friendly async API (can be wrapped by an HTTP server; see `deployment/README.md`)

This repository contains a Python package (`nanovllm_voxcpm/`) plus an optional FastAPI demo.

**Coverage**: ~71% combined (core + deployment) - see [CI coverage job](.github/workflows/ci.yml) for the latest.

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

- Linux / Windows + NVIDIA GPU (CUDA)
- Python >= 3.10
- `flash-attn` is required (the package imports it at runtime)

> ⚠️ **Important Note for Windows Users:** 
> Automated installation and compilation of `flash-attn` is bypassed on Windows during the package setup phase to prevent build isolation and compiler errors. 
> 
> **Please note that a standard `pip install nano-vllm-voxcpm` on Windows is NOT enough by itself to run the engine.** The package will fail immediately at runtime with a `ModuleNotFoundError` unless you install `flash-attn` separately in your active Python environment.
> 
> To resolve this, you must manually install a precompiled community wheel (highly recommended to avoid local MSVC/NVCC compilation headaches) matching your exact Python/PyTorch/CUDA version, or compile it locally from source.

The runtime is GPU-centric (Triton + FlashAttention). CPU-only execution is not supported.

Windows support notes:

- Tensor parallelism (`tensor_parallel_size > 1`) is not supported on Windows. This path requires CUDA
  tensor collectives through NCCL, which is not available on Windows; use single-GPU workers on Windows
  or a Linux environment for tensor parallelism.
- Advanced users can manually override automatic KV-cache sizing with `NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS`.
  Leave it unset for the normal safe memory calculation. Setting it bypasses that calculation and may cause
  CUDA OOM if the value is too high for the GPU.

### Install from source (dev)

This repo uses `uv` and includes a lockfile (`uv.lock`).

```bash
uv sync --frozen
```

Dev deps (tests):

```bash
uv sync --frozen --dev
```

Note: compiling `flash-attn` from source on Linux may require the native NVIDIA CUDA Toolkit (with `nvcc` and CUDA headers) to be present in your system PATH.

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

## Manual GPU Smoke Suite

Use `scripts/gpu_smoke.sh` for manual CUDA validation on an idle Linux GPU host. It checks CUDA,
FlashAttention, Triton, device visibility, and then runs the curated single-GPU or two-rank TP tests.
This suite requires real CUDA hardware and does not run in CI.

```bash
# Single-device smoke
CUDA_VISIBLE_DEVICES=0 bash scripts/gpu_smoke.sh --single

# Two-device tensor-parallel smoke
CUDA_VISIBLE_DEVICES=0,1 bash scripts/gpu_smoke.sh --tp

# Intentional hidden-device diagnostic
CUDA_VISIBLE_DEVICES="" bash scripts/gpu_smoke.sh --single

# Intentional insufficient-GPU failure for TP
CUDA_VISIBLE_DEVICES=0 bash scripts/gpu_smoke.sh --tp
```

### Reference Results (RTX 4090)

All reference numbers in this section are measured on NVIDIA GeForce RTX 4090 with `openbmb/VoxCPM2`.
The benchmark defines `RTF_per_req_mean` as the mean over requests of `((request_wall_time - TTFB) / request_audio_duration)` under the given concurrency.

Unless noted, runs use the default `gpu_memory_utilization=0.8`. Two high-concurrency LoRA
points (short prompt @ 128, long prompt @ 64) are measured at `gpu_memory_utilization=0.7`
(marked with `†`); at the default `0.8` they can OOM on a 24 GB card. See "Memory note" below.
LoRA tables are measured with 5 warmup iterations (instead of 1) to absorb first-touch
LoRA slot population; with a single warmup iteration the first measured iteration is biased high.

Short prompt, no LoRA:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.0362 ± 0.0012 | 0.0362 ± 0.0012 | 0.0859 ± 0.0003 |
| 8 | 0.0438 ± 0.0025 | 0.0440 ± 0.0026 | 0.1145 ± 0.0007 |
| 16 | 0.0567 ± 0.0049 | 0.0573 ± 0.0049 | 0.1583 ± 0.0012 |
| 32 | 0.0765 ± 0.0033 | 0.0771 ± 0.0032 | 0.2516 ± 0.0021 |
| 64 | 0.1576 ± 0.0189 | 0.1618 ± 0.0214 | 0.5382 ± 0.1000 |

Long prompt, no LoRA:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.0433 ± 0.0041 | 0.0433 ± 0.0041 | 0.1016 ± 0.0026 |
| 8 | 0.0603 ± 0.0099 | 0.0610 ± 0.0098 | 0.1373 ± 0.0067 |
| 16 | 0.0883 ± 0.0038 | 0.0888 ± 0.0039 | 0.1797 ± 0.0013 |
| 32 | 0.1898 ± 0.0119 | 0.1909 ± 0.0115 | 0.3061 ± 0.0146 |
| 64 | 0.3185 ± 0.0028 | 0.3206 ± 0.0027 | 0.5515 ± 0.0040 |

Short prompt, LoRA enabled with 32 runtime slots:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.0876 ± 0.0006 | 0.0876 ± 0.0006 | 0.1133 ± 0.0012 |
| 8 | 0.0986 ± 0.0201 | 0.0988 ± 0.0202 | 0.1498 ± 0.0019 |
| 16 | 0.1006 ± 0.0175 | 0.1010 ± 0.0176 | 0.1966 ± 0.0007 |
| 32 | 0.1125 ± 0.0048 | 0.1133 ± 0.0049 | 0.3065 ± 0.0014 |
| 64 | 0.1801 ± 0.0032 | 0.1819 ± 0.0029 | 0.5704 ± 0.0118 |
| 128 † | 0.3691 ± 0.0578 | 0.3735 ± 0.0591 | 1.2174 ± 0.1505 |

Long prompt, LoRA enabled with 32 runtime slots:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.0930 ± 0.0047 | 0.0930 ± 0.0047 | 0.1307 ± 0.0005 |
| 8 | 0.1171 ± 0.0352 | 0.1177 ± 0.0348 | 0.1763 ± 0.0039 |
| 16 | 0.1177 ± 0.0124 | 0.1192 ± 0.0134 | 0.2455 ± 0.0253 |
| 32 | 0.1869 ± 0.0140 | 0.1876 ± 0.0139 | 0.3625 ± 0.0049 |
| 64 † | 0.3913 ± 0.0334 | 0.3922 ± 0.0332 | 0.7433 ± 0.0192 |

`†` measured at `gpu_memory_utilization=0.7`.

Closed-loop results:

| mode | users | registered LoRAs | started | achieved rps | ok | err |
|---|---:|---:|---:|---:|---:|---:|
| no LoRA | 60 | 0 | 180 | 3.00 | 180 | 0 |
| LoRA | 30 | 32 | 120 | 2.00 | 120 | 0 |
| LoRA | 30 | 128 | 120 | 2.00 | 120 | 0 |
| LoRA | 30 | 256 | 120 | 2.00 | 120 | 0 |

Closed-loop TTFB (seconds, ok requests):

| mode | users | registered LoRAs | p50 | p90 | p95 | p99 | mean | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no LoRA | 60 | 0 | 0.3379 | 0.3686 | 0.3739 | 0.3745 | 0.3341 | 0.0333 |
| LoRA | 30 | 32 | 0.1359 | 0.1764 | 0.2912 | 0.6074 | 0.1557 | 0.0833 |
| LoRA | 30 | 128 | 0.1836 | 0.2793 | 0.3044 | 0.3528 | 0.1969 | 0.0551 |
| LoRA | 30 | 256 | 0.1944 | 0.2603 | 0.2872 | 0.3198 | 0.2072 | 0.0464 |

Closed-loop RTF ((wall - TTFB)/audio, ok requests):

| mode | users | registered LoRAs | p50 | p90 | p95 | p99 | mean | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no LoRA | 60 | 0 | 0.5853 | 0.5887 | 0.5887 | 0.5897 | 0.5851 | 0.0028 |
| LoRA | 30 | 32 | 0.4131 | 0.4269 | 0.4323 | 0.4363 | 0.4119 | 0.0144 |
| LoRA | 30 | 128 | 0.4279 | 0.4416 | 0.4431 | 0.4466 | 0.4247 | 0.0169 |
| LoRA | 30 | 256 | 0.4250 | 0.4377 | 0.4416 | 0.4462 | 0.4215 | 0.0166 |

Memory note: this release adds a prefill diffusion CUDA graph that improves latency/throughput
but increases steady-state VRAM by roughly 2.5 GB (the extra graph pool is not yet accounted for
in the automatic KV-cache budget). On a 24 GB card at high concurrency with LoRA (e.g. short
prompt @ 128, long prompt @ 64), the default `gpu_memory_utilization=0.9` can OOM; lower it
(e.g. `0.7`) or reduce `max_num_seqs` to run those configurations.

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
