"""VoxCPM TTS API Server - OpenAI Compatible"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool
import base64
from pydantic import BaseModel
from typing import Optional
import torch
import os
import asyncio
import numpy as np
import io
import soundfile

MODEL_PATH = os.environ.get("VOXCPM_MODEL_PATH", "/app/VoxCPM-0.5B")
FRONTEND_PATH = "/app/frontend/index.html"
SAMPLE_RATE = 16000

global_instances = {}
voice_cache = {}  # {voice_name: {"prompt_id": str, "prompt_text": str}}

def get_gpu_devices():
    env_devices = os.environ.get("VOXCPM_DEVICES")
    if env_devices:
        return [int(d) for d in env_devices.split(",")]
    gpu_count = torch.cuda.device_count()
    return list(range(gpu_count)) if gpu_count > 0 else [0]

async def warmup_all_gpus(gpu_count: int):
    """并发预热所有 GPU"""
    async def single_warmup(i):
        async for _ in global_instances["server"].generate(
            target_text=f"预热{i}",
            prompt_latents=None,
            prompt_text="",
            prompt_id=None,
            max_generate_length=50,
            temperature=1.0,
            cfg_value=1.5,
        ):
            pass
    await asyncio.gather(*[single_warmup(i) for i in range(gpu_count)])
    print(f"[VoxCPM] 所有 {gpu_count} 张 GPU 预热完成")

@asynccontextmanager
async def lifespan(app: FastAPI):
    devices = get_gpu_devices()
    print(f"[VoxCPM] 使用 GPU 设备: {devices}")
    
    server = AsyncVoxCPMServerPool(
        model_path=MODEL_PATH,
        devices=devices,
    )
    global_instances["server"] = server
    
    # 智能预热
    await warmup_all_gpus(len(devices))
    
    yield
    global_instances.clear()
    voice_cache.clear()

app = FastAPI(title="VoxCPM TTS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Web UI ====================

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    if os.path.exists(FRONTEND_PATH):
        with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>VoxCPM TTS API</h1><p>Frontend not found</p>")

# ==================== Health & Info ====================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "voxcpm-0.5b",
        "voices_count": len(voice_cache)
    }

@app.get("/voices")
async def list_voices():
    return {"voices": list(voice_cache.keys()), "count": len(voice_cache)}

# ==================== Voice Management ====================

class CreateVoiceRequest(BaseModel):
    voice_name: str
    prompt_wav_base64: Optional[str] = None
    prompt_wav_path: Optional[str] = None
    prompt_wav_format: str = "wav"
    prompt_text: str
    replace: bool = False

@app.post("/v1/voices")
async def create_voice(request: CreateVoiceRequest):
    if request.voice_name in voice_cache and not request.replace:
        raise HTTPException(status_code=409, detail=f"Voice '{request.voice_name}' already exists")
    
    wav_data = None
    wav_format = request.prompt_wav_format
    
    if request.prompt_wav_base64:
        try:
            wav_data = base64.b64decode(request.prompt_wav_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {e}")
    elif request.prompt_wav_path:
        if not os.path.exists(request.prompt_wav_path):
            raise HTTPException(status_code=404, detail=f"Audio file not found: {request.prompt_wav_path}")
        try:
            with open(request.prompt_wav_path, "rb") as f:
                wav_data = f.read()
            ext = os.path.splitext(request.prompt_wav_path)[1].lower().lstrip(".")
            if ext in ["wav", "mp3", "flac", "ogg"]:
                wav_format = ext
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read audio file: {e}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either prompt_wav_base64 or prompt_wav_path")
    
    prompt_text = request.prompt_text
    if os.path.exists(prompt_text):
        try:
            with open(prompt_text, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        except:
            pass
    
    try:
        server = global_instances["server"]
        prompt_id = await server.add_prompt(wav_data, wav_format, prompt_text)
        
        voice_cache[request.voice_name] = {
            "prompt_id": prompt_id,
            "prompt_text": prompt_text
        }
        
        return {"status": "success", "voice_name": request.voice_name, "prompt_id": prompt_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/v1/voices/{voice_name}")
async def delete_voice(voice_name: str):
    if voice_name not in voice_cache:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")
    
    try:
        prompt_id = voice_cache[voice_name]["prompt_id"]
        server = global_instances["server"]
        await server.remove_prompt(prompt_id)
        del voice_cache[voice_name]
        return {"status": "success", "message": f"Voice '{voice_name}' deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== OpenAI Compatible TTS API ====================

class SpeechRequest(BaseModel):
    model: str = "voxcpm-0.5b"
    input: str
    voice: Optional[str] = None
    response_format: str = "wav"
    max_length: int = 2000
    temperature: float = 1.0
    cfg_value: float = 1.5

@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    server = global_instances["server"]
    
    prompt_id = None
    prompt_text = ""
    if request.voice and request.voice in voice_cache:
        prompt_id = voice_cache[request.voice]["prompt_id"]
    
    chunks = []
    async for chunk in server.generate(
        target_text=request.input,
        prompt_latents=None,
        prompt_text=prompt_text,
        prompt_id=prompt_id,
        max_generate_length=request.max_length,
        temperature=request.temperature,
        cfg_value=request.cfg_value,
    ):
        chunks.append(chunk)
    
    audio = np.concatenate(chunks)
    
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, SAMPLE_RATE, format=request.response_format.upper())
    buffer.seek(0)
    
    media_types = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}
    return Response(
        content=buffer.read(),
        media_type=media_types.get(request.response_format, "audio/wav"),
        headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"}
    )

@app.post("/v1/audio/speech/stream")
async def create_speech_stream(request: SpeechRequest):
    server = global_instances["server"]
    
    prompt_id = None
    prompt_text = ""
    if request.voice and request.voice in voice_cache:
        prompt_id = voice_cache[request.voice]["prompt_id"]
    
    async def audio_generator():
        chunk_index = 0
        async for chunk in server.generate(
            target_text=request.input,
            prompt_latents=None,
            prompt_text=prompt_text,
            prompt_id=prompt_id,
            max_generate_length=request.max_length,
            temperature=request.temperature,
            cfg_value=request.cfg_value,
        ):
            # 确保是 float32 类型
            chunk = chunk.astype(np.float32)
            
            # 安全转换：clipping 防止溢出
            pcm_16bit = np.clip(chunk * 32767, -32768, 32767).astype(np.int16)
            yield pcm_16bit.tobytes()
            chunk_index += 1
    
    return StreamingResponse(
        audio_generator(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(SAMPLE_RATE), "X-Channels": "1", "X-Bit-Depth": "16"}
    )

# ==================== Legacy API (Backward Compatibility) ====================

class GenerateRequest(BaseModel):
    target_text: str
    prompt_id: Optional[str] = None
    prompt_text: str = ""
    max_generate_length: int = 2000
    temperature: float = 1.0
    cfg_value: float = 1.5

@app.post("/generate")
async def generate_legacy(request: GenerateRequest):
    server = global_instances["server"]
    chunks = []
    async for chunk in server.generate(
        target_text=request.target_text,
        prompt_latents=None,
        prompt_text=request.prompt_text,
        prompt_id=request.prompt_id,
        max_generate_length=request.max_generate_length,
        temperature=request.temperature,
        cfg_value=request.cfg_value,
    ):
        chunks.append(chunk)
    
    audio = np.concatenate(chunks)
    buffer = io.BytesIO()
    soundfile.write(buffer, audio, SAMPLE_RATE, format="WAV")
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="audio/wav")
