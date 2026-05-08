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

### LoRA Management

Runtime LoRA adapters can be registered/unregistered dynamically without restarting the server.

**Enable LoRA at startup:**
```bash
export NANOVLLM_LORA_ENABLED=true
```

#### Register LoRA Adapter

`POST /loras`

```bash
curl -X POST http://localhost:8000/loras \
  -H "Content-Type: application/json" \
  -d '{"name": "my_lora", "path": "/path/to/lora_weights.safetensors"}'
```

Request body:
- `name`: Unique identifier for this adapter
- `path`: Absolute path to LoRA checkpoint directory or `.safetensors` file

Response: `{"name": "my_lora"}`

#### List LoRA Adapters

`GET /loras`

```bash
curl http://localhost:8000/loras
```

Response: `[{"name": "my_lora"}, {"name": "another_lora"}]`

#### Use LoRA in Generation

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"target_text": "Hello world", "lora_name": "my_lora"}'
```

#### Unregister LoRA Adapter

`DELETE /loras/{name}`

```bash
curl -X DELETE http://localhost:8000/loras/my_lora
```

---

## WebSocket Voice Pipeline

Real-time voice-to-voice pipeline: **Audio → ASR → LLM → TTS → Audio**

### Quick Start

1. **Start ASR server** (vLLM with Qwen3-ASR):
```bash
vllm serve Qwen/Qwen3-ASR-1.7B --port 8001
```

2. **Set environment variables**:
```bash
export GEMINI_API_KEY=<your-key>
export NANOVLLM_ASR_API_URL=http://localhost:8001
```

3. **Run the server**:
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### WebSocket Endpoint

**URL:** `ws://localhost:8000/ws`

**Protocol:**

```
Client → Server:
  {"type": "audio_chunk", "data": "<base64>", "format": "wav"}
  {"type": "audio_end"}
  {"type": "terminate"}

Server → Client:
  <binary Opus frames>
  {"type": "complete"}
  {"type": "error", "error": "..."}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NANOVLLM_ASR_API_URL` | `http://localhost:8001` | ASR vLLM server URL |
| `NANOVLLM_ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | ASR model name |
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model |
| `NANOVLLM_OPUS_BITRATE` | `64000` | Opus bitrate (bps) |
| `NANOVLLM_OPUS_FRAME_MS` | `20` | Opus frame duration |

#### Persistent TTS voice conditioning (WebSocket)

These env vars are **optional** and are loaded **once at server startup**. If set, they apply to **every** WebSocket interaction and condition the TTS voice.

- `NANOVLLM_WS_PROMPT_AUDIO_PATH`: server-local path to prompt audio (e.g. `/data/prompt.mp3`)
- `NANOVLLM_WS_PROMPT_TEXT_PATH`: server-local path to a text file containing the transcript of the prompt audio (required when prompt audio is set)
- `NANOVLLM_WS_PROMPT_AUDIO_FORMAT`: optional override (otherwise inferred from file suffix)

Optional reference audio (VoxCPM2 only):

- `NANOVLLM_WS_REF_AUDIO_PATH`: server-local path to reference audio
- `NANOVLLM_WS_REF_AUDIO_FORMAT`: optional override

### Test with Python

```python
import asyncio
import base64
import websockets
import json

async def test():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Send audio
        with open("test.wav", "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        
        await ws.send(json.dumps({"type": "audio_chunk", "data": audio_b64, "format": "wav"}))
        await ws.send(json.dumps({"type": "audio_end"}))
        
        # Receive response
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                print(f"Received {len(msg)} bytes of audio")
            else:
                data = json.loads(msg)
                print(data)
                if data.get("type") == "complete":
                    break

asyncio.run(test())
```

### Metrics

Available at `GET /metrics`:

- `nanovllm_websocket_connections_active` - Active connections
- `nanovllm_websocket_connections_total` - Total connections by status
- `nanovllm_websocket_messages_total` - Messages by direction/type
- `nanovllm_pipeline_duration_seconds` - End-to-end latency
- `nanovllm_pipeline_stage_duration_seconds` - Per-stage latency (asr/llm/tts)
