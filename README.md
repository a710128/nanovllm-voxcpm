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

Short prompt, no LoRA:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.0672 ± 0.0018 | 0.0672 ± 0.0018 | 0.1027 ± 0.0012 |
| 8 | 0.0789 ± 0.0033 | 0.0790 ± 0.0033 | 0.1307 ± 0.0006 |
| 16 | 0.0860 ± 0.0008 | 0.0864 ± 0.0009 | 0.1764 ± 0.0005 |
| 32 | 0.1142 ± 0.0023 | 0.1148 ± 0.0024 | 0.2842 ± 0.0026 |
| 64 | 0.1885 ± 0.0024 | 0.1907 ± 0.0025 | 0.6054 ± 0.0989 |

Long prompt, no LoRA:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.0768 ± 0.0022 | 0.0768 ± 0.0022 | 0.1163 ± 0.0006 |
| 8 | 0.0865 ± 0.0030 | 0.0867 ± 0.0031 | 0.1492 ± 0.0007 |
| 16 | 0.1346 ± 0.0017 | 0.1349 ± 0.0017 | 0.2017 ± 0.0011 |
| 32 | 0.2677 ± 0.0010 | 0.2684 ± 0.0009 | 0.3334 ± 0.0071 |
| 64 | 0.5510 ± 0.0182 | 0.5544 ± 0.0211 | 0.6724 ± 0.0134 |

Short prompt, LoRA enabled with 32 runtime slots:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.1375 ± 0.0038 | 0.1375 ± 0.0038 | 0.1284 ± 0.0003 |
| 8 | 0.2442 ± 0.0675 | 0.2444 ± 0.0675 | 0.1639 ± 0.0024 |
| 16 | 0.3771 ± 0.3279 | 0.3774 ± 0.3278 | 0.2168 ± 0.0021 |
| 32 | 0.2358 ± 0.0560 | 0.2366 ± 0.0560 | 0.3419 ± 0.0040 |
| 64 | 0.3287 ± 0.0825 | 0.3312 ± 0.0822 | 0.6400 ± 0.0192 |
| 128 † | 0.4712 ± 0.0513 | 0.4749 ± 0.0533 | 1.3215 ± 0.0421 |

Long prompt, LoRA enabled with 32 runtime slots:

| concurrency | TTFB p50 (s) | TTFB p90 (s) | RTF_per_req_mean |
|---:|---:|---:|---:|
| 1 | 0.1444 ± 0.0013 | 0.1444 ± 0.0013 | 0.1495 ± 0.0004 |
| 8 | 0.2559 ± 0.0817 | 0.2561 ± 0.0817 | 0.1894 ± 0.0004 |
| 16 | 0.3636 ± 0.3142 | 0.3653 ± 0.3137 | 0.2541 ± 0.0028 |
| 32 | 0.4441 ± 0.1444 | 0.4451 ± 0.1442 | 0.4028 ± 0.0025 |
| 64 † | 0.5850 ± 0.0438 | 0.5865 ± 0.0436 | 0.7403 ± 0.0045 |

`†` measured at `gpu_memory_utilization=0.7`.

Closed-loop results:

| mode | users | registered LoRAs | started | achieved rps | ok | err |
|---|---:|---:|---:|---:|---:|---:|
| no LoRA | 60 | 0 | 180 | 3.00 | 180 | 0 |
| LoRA | 30 | 32 | 103 | 1.72 | 103 | 0 |
| LoRA | 30 | 128 | 90 | 1.50 | 90 | 0 |
| LoRA | 30 | 256 | 60 | 1.00 | 60 | 0 |

Closed-loop TTFB (seconds, ok requests):

| mode | users | registered LoRAs | p50 | p90 | p95 | p99 | mean | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no LoRA | 60 | 0 | 0.5135 | 0.5572 | 0.5581 | 0.5584 | 0.5263 | 0.0213 |
| LoRA | 30 | 32 | 0.1788 | 0.3038 | 0.6535 | 0.6544 | 0.2208 | 0.1448 |
| LoRA | 30 | 128 | 0.3960 | 1.0322 | 1.9344 | 2.0003 | 0.5718 | 0.5049 |
| LoRA | 30 | 256 | 0.4576 | 1.3177 | 1.3184 | 1.3192 | 0.5969 | 0.3841 |

Closed-loop RTF ((wall - TTFB)/audio, ok requests):

| mode | users | registered LoRAs | p50 | p90 | p95 | p99 | mean | stdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no LoRA | 60 | 0 | 0.6737 | 0.6942 | 0.6943 | 0.6943 | 0.6785 | 0.0114 |
| LoRA | 30 | 32 | 0.4440 | 0.4589 | 0.4626 | 0.4684 | 0.4237 | 0.0570 |
| LoRA | 30 | 128 | 0.5067 | 0.5372 | 0.5479 | 0.5726 | 0.5005 | 0.0350 |
| LoRA | 30 | 256 | 0.6370 | 0.7082 | 0.7123 | 0.7235 | 0.6331 | 0.0621 |

Memory note: this release adds a prefill diffusion CUDA graph that improves latency/throughput
but increases steady-state VRAM by roughly 2.5 GB (the extra graph pool is not yet accounted for
in the automatic KV-cache budget). On a 24 GB card at high concurrency with LoRA (e.g. short
prompt @ 128, long prompt @ 64), the default `gpu_memory_utilization=0.9` can OOM; lower it
(e.g. `0.7`), reduce `max_num_seqs`, or pass `enforce_dit_prefill_eager=True` to retain
decode CUDA Graphs without allocating the prefill diffusion graph.

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
