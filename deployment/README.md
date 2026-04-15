# VoxCPM FastAPI Service

This folder contains a production-oriented FastAPI wrapper around
`nanovllm_voxcpm.models.voxcpm.server.AsyncVoxCPMServerPool`.

Key properties:

- Stateless API (no `prompt_id`, no prompt pool endpoints)
- Runtime LoRA management via `/loras`
- `/generate` streams MP3 (`audio/mpeg`) encoded server-side via `lameenc`

## Install (uv)

This repo uses `uv` and `deployment/` is a uv workspace member.

Install workspace dependencies at the repo root:

```bash
uv sync --all-packages --frozen
```

Alternatively, to sync only the deployment service dependencies:

```bash
uv sync --package nano-vllm-voxcpm-deployment --frozen
```

Note: `uv sync --frozen` (without `--all-packages/--package`) only syncs the root package by default.

## Configure

Environment variables:

- `NANOVLLM_MODEL_PATH` (default `~/VoxCPM1.5`)
- MP3 encoding (read at startup):
  - `NANOVLLM_MP3_BITRATE_KBPS` (int, default `192`)
  - `NANOVLLM_MP3_QUALITY` (int, default `2`, allowed `0..2`)
- LoRA startup preload env vars are removed. Register adapters at runtime via `POST /loras`.
- Runtime LoRA capacity (read at startup):
  - `NANOVLLM_LORA_ENABLED` (bool, default `false`; must be `true` to register adapters)
  - `NANOVLLM_LORA_MAX_LORAS` (int, default `1`)
  - `NANOVLLM_LORA_MAX_LORA_RANK` (int, default `32`)
  - `NANOVLLM_LORA_ENABLE_LM` (bool override; default enables LM LoRA)
  - `NANOVLLM_LORA_ENABLE_DIT` (bool override; default enables DiT LoRA)
  - `NANOVLLM_LORA_ENABLE_PROJ` (bool override; default enables projection LoRA)
  - `NANOVLLM_LORA_TARGET_MODULES_LM` (comma-separated override; default enables all supported LM targets)
  - `NANOVLLM_LORA_TARGET_MODULES_DIT` (comma-separated override; default enables all supported DiT targets)
  - `NANOVLLM_LORA_TARGET_PROJ_MODULES` (comma-separated override; default is architecture-specific)

- Server pool startup (read at startup):
  - `NANOVLLM_SERVERPOOL_MAX_NUM_BATCHED_TOKENS` (int, default `8192`)
  - `NANOVLLM_SERVERPOOL_MAX_NUM_SEQS` (int, default `16`)
  - `NANOVLLM_SERVERPOOL_MAX_MODEL_LEN` (int, default `4096`)
  - `NANOVLLM_SERVERPOOL_GPU_MEMORY_UTILIZATION` (float, default `0.95`, allowed `(0, 1]`)
  - `NANOVLLM_SERVERPOOL_ENFORCE_EAGER` (bool, default `false`; accepts `1/0,true/false,yes/no,on/off`)
  - `NANOVLLM_SERVERPOOL_DEVICES` (comma-separated ints, default `0`; e.g. `0,1`)

LoRA checkpoint layout (recommended):

```
step_0002000/
  lora_weights.safetensors
  lora_config.json
```

If `lora_config.json` exists, the core loader reads adapter rank/alpha from it during `POST /loras` registration.

## Run

From the repo root:

```bash
uv run fastapi run deployment/app/main.py --host 0.0.0.0 --port 8000
```

Alternatively (matches the container entrypoint):

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

OpenAPI:

- http://localhost:8000/docs

## Tests

```bash
uv run pytest deployment/tests -q
```

## Docker (k8s-ready)

This repo ships a multi-stage CUDA image at `deployment/Dockerfile`.

Build from the repo root (important: build context is `.`):

```bash
docker build -f deployment/Dockerfile -t nano-vllm-voxcpm-deployment:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -e NANOVLLM_MODEL_PATH=/models/VoxCPM1.5 \
  -e NANOVLLM_CACHE_DIR=/var/cache/nanovllm \
  -v /path/to/models:/models \
  nano-vllm-voxcpm-deployment:latest
```

Notes:

- GPU: on a GPU node you typically need `--gpus all` (Docker) or the NVIDIA device plugin (k8s).
- The container runs as a non-root user (uid `10001`) and uses `NANOVLLM_CACHE_DIR` for writable cache.
- Probes: use `GET /health` (liveness) and `GET /ready` (readiness).

## Client example

`deployment/client.py` demonstrates calling `/encode_latents` and `/generate` and writes MP3 files:

It expects a prompt audio file at `deployment/prompt_audio.wav`.

```bash
uv run python deployment/client.py
```

Outputs:

- `out_zero_shot.mp3`
- `out_prompted.mp3`

## API

### Health

- `GET /health` (liveness): returns `{"status":"ok"}`
- `GET /ready` (readiness): returns 200 only after the model is loaded

### Info

`GET /info`

Returns model metadata from core (`sample_rate/channels/feat_dim/...`) plus MP3 encoder config.

### Metrics

`GET /metrics`

Prometheus metrics.

### Encode prompt wav to latents

`POST /encode_latents`

Request body (JSON):

- `wav_base64`: base64-encoded bytes of the *entire audio file* (not a data URI)
- `wav_format`: container format for decoding (e.g. `wav`, `flac`, `mp3`; passed to torchaudio)

Response body (JSON):

- `prompt_latents_base64`: base64-encoded float32 bytes
- `feat_dim`: reshape with `np.frombuffer(bytes, np.float32).reshape(-1, feat_dim)`
- `latents_dtype`: `"float32"`
- `sample_rate`: output sample rate (from the model)
- `channels`: `1`

### Generate (streaming MP3)

`POST /generate`

Request body (JSON):

- `target_text`: required
- Prompt (optional, mutually exclusive):
  - wav prompt: `prompt_wav_base64` + `prompt_wav_format` + `prompt_text`
  - latents prompt: `prompt_latents_base64` + `prompt_text`
  - zero-shot: omit all prompt fields
- Reference audio (optional, mutually exclusive):
  - wav reference: `ref_audio_wav_base64` + `ref_audio_wav_format`
  - latents reference: `ref_audio_latents_base64`

`ref_audio_*` is independent from the prompt fields, so you can combine reference audio with either zero-shot or prompted generation.

Response:

- `Content-Type: audio/mpeg`
- body is a streamed MP3 byte stream
- headers:
  - `X-Audio-Sample-Rate`
  - `X-Audio-Channels`
