# WebSocket Client API Guide

## Connection

Connect via WebSocket to:

```
ws://<host>:<port>/ws
```

Default: `ws://localhost:8001/ws`

## Message Flow

```
Client                          Server
  │                               │
  │──── 1. init ─────────────────►│  (required first message)
  │                               │
  │──── 2. audio_chunk ──────────►│  (one or more)
  │──── 3. audio_chunk ──────────►│
  │──── 4. audio_end ────────────►│  (triggers processing)
  │                               │
  │◄─── 5. <binary opus frames>──│  (streamed audio response)
  │◄─── 6. <binary opus frames>──│
  │◄─── 7. {"type":"complete"} ──│  (processing done)
  │                               │
  │  (repeat steps 2-7 for more)  │
  │                               │
  │──── 8. terminate ────────────►│  (close connection)
  │                               │
```

## Client → Server Messages

### 1. `init` (required first message)

Select a speaker voice. Must be sent **before** any audio.

```json
{"type": "init", "speaker": "alaa"}
```

| Field     | Type   | Required | Values                              |
|-----------|--------|----------|-------------------------------------|
| `type`    | string | yes      | `"init"`                            |
| `speaker` | string | yes      | `"alaa"`, `"hammad"`, `"hanan"`, `"khalil"` |

### 2. `audio_chunk`

Send audio data as base64-encoded chunks.

```json
{
  "type": "audio_chunk",
  "data": "<base64-encoded audio bytes>",
  "format": "wav"
}
```

| Field    | Type   | Required | Default | Description                 |
|----------|--------|----------|---------|-----------------------------|
| `type`   | string | yes      |         | `"audio_chunk"`             |
| `data`   | string | yes      |         | Base64-encoded audio bytes  |
| `format` | string | no       | `"wav"` | Audio format hint           |

You can send multiple chunks — they are concatenated server-side.

### 3. `audio_end`

Signals that all audio has been sent. Triggers the pipeline.

```json
{"type": "audio_end"}
```

### 4. `terminate`

Close the connection gracefully.

```json
{"type": "terminate"}
```

## Server → Client Messages

### Binary Frames (Opus Audio)

Audio response is streamed as **raw binary WebSocket frames** containing Opus-encoded audio. Decode with Opus at 48kHz mono.

### `complete`

Sent after all audio frames for the current interaction.

```json
{"type": "complete"}
```

### `error`

Sent if any stage of the pipeline fails.

```json
{"type": "error", "error": "description of what went wrong"}
```

## Opus Audio Decoding

Response audio is Opus-encoded with these defaults:

| Parameter   | Value   |
|-------------|---------|
| Sample rate | 48000 Hz |
| Channels    | 1 (mono) |
| Frame size  | 20 ms    |
| Bitrate     | 64 kbps  |

## Quick Example (Python)

```python
import asyncio, json, base64
import websockets

async def main():
    async with websockets.connect("ws://localhost:8001/ws") as ws:
        # 1. Select speaker
        await ws.send(json.dumps({"type": "init", "speaker": "alaa"}))

        # 2. Send audio
        audio = open("question.wav", "rb").read()
        await ws.send(json.dumps({
            "type": "audio_chunk",
            "data": base64.b64encode(audio).decode(),
            "format": "wav"
        }))
        await ws.send(json.dumps({"type": "audio_end"}))

        # 3. Receive response
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                # Opus audio frame — decode with opuslib
                pass
            else:
                data = json.loads(msg)
                if data["type"] == "complete":
                    break
                elif data["type"] == "error":
                    print(f"Error: {data['error']}")
                    break

        # 4. Close
        await ws.send(json.dumps({"type": "terminate"}))

asyncio.run(main())
```

## Error Handling

- If the first message is not `init`, the server closes with code `1008`.
- If `speaker` is invalid, validation error is returned and connection closes.
- Pipeline errors (ASR/LLM/TTS failures) are sent as `{"type": "error"}` messages — the connection stays open for retry.
- Server misconfiguration (missing API keys, model not ready) closes immediately with code `1011`.
