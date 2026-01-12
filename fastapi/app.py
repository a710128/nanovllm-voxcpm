from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool
from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig
import base64
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI()

# ==================== Configuration ====================
MODEL_PATH = "~/VoxCPM1.5"
LORA_PATH = None

# LoRA configuration (set to None to disable LoRA structure)
# LORA_CONFIG = LoRAConfig(
#     enable_lm=True,
#     enable_dit=True,
#     enable_proj=False,
#     r=32,
#     alpha=16.0,
#     dropout=0.0,
#     target_modules_lm=["q_proj", "k_proj", "v_proj", "o_proj"],
#     target_modules_dit=["q_proj", "k_proj", "v_proj", "o_proj"],
# )
# If LoRA is not needed, set to None:
LORA_CONFIG = None
# ================================================

global_instances = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global_instances["server"] = AsyncVoxCPMServerPool(
        model_path=MODEL_PATH,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        devices=[0],
        lora_config=LORA_CONFIG,  # Add LoRA config
    )
    await global_instances["server"].wait_for_ready()  # Wait for model to load first
    
    # Then load LoRA weights (optional)
    if LORA_PATH:
        await global_instances["server"].load_lora(LORA_PATH)
        await global_instances["server"].set_lora_enabled(True)
    yield
    await global_instances["server"].stop()
    del global_instances["server"]

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}


# ==================== LoRA Management API ====================

class LoadLoRARequest(BaseModel):
    lora_path: str

@app.post("/lora/load")
async def load_lora(request: LoadLoRARequest):
    """Load LoRA weights"""
    server: AsyncVoxCPMServerPool = global_instances["server"]
    result = await server.load_lora(request.lora_path)
    return result

class SetLoRAEnabledRequest(BaseModel):
    enabled: bool

@app.post("/lora/set_enabled")
async def set_lora_enabled(request: SetLoRAEnabledRequest):
    """Enable/disable LoRA"""
    server: AsyncVoxCPMServerPool = global_instances["server"]
    result = await server.set_lora_enabled(request.enabled)
    return result

@app.post("/lora/reset")
async def reset_lora():
    """Reset LoRA weights (equivalent to unloading)"""
    server: AsyncVoxCPMServerPool = global_instances["server"]
    result = await server.reset_lora()
    return result


# ==================== Original API ====================

class AddPromptRequest(BaseModel):
    wav_base64: str
    wav_format: str
    prompt_text: str

@app.post("/add_prompt")
async def add_prompt(request: AddPromptRequest):
    wav = base64.b64decode(request.wav_base64)
    server: AsyncVoxCPMServerPool = global_instances["server"]

    prompt_id = await server.add_prompt(wav, request.wav_format, request.prompt_text)
    return {"prompt_id": prompt_id}

class RemovePromptRequest(BaseModel):
    prompt_id: str

@app.post("/remove_prompt")
async def remove_prompt(request: RemovePromptRequest):
    server: AsyncVoxCPMServerPool = global_instances["server"]
    await server.remove_prompt(request.prompt_id)
    return {"status": "ok"}


class GenerateRequest(BaseModel):
    target_text: str
    prompt_id: str | None = None
    max_generate_length: int = 2000
    temperature: float = 1.0
    cfg_value: float = 1.5


async def numpy_to_bytes(gen):
    async for data in gen:
        yield data.tobytes()

@app.post("/generate")
async def generate(request: GenerateRequest):
    server: AsyncVoxCPMServerPool = global_instances["server"]
    return StreamingResponse(
        numpy_to_bytes(
            server.generate(
                target_text=request.target_text,
                prompt_latents=None,
                prompt_text="",
                prompt_id=request.prompt_id,
                max_generate_length=request.max_generate_length,
                temperature=request.temperature,
                cfg_value=request.cfg_value,
            )
        ),
        media_type="audio/raw",
    )
