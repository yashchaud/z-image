"""
OpenAI-compatible image generation server using FastFusion pattern.
Supports Z-Image Turbo and other HuggingFace Diffusers models.
"""

import os
import json
import asyncio
import base64
import time
import logging
from io import BytesIO
from typing import Optional, List, Literal
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Global model holder
class ModelManager:
    def __init__(self):
        self.pipeline = None
        self.model_id = None
        self.device = None
        self.config = None
        self._lock = asyncio.Lock()

    async def load_model(self, model_id: str, config: dict):
        async with self._lock:
            if self.pipeline is not None and self.model_id == model_id:
                logger.info("model_already_loaded", model_id=model_id)
                return

            logger.info("loading_model", model_id=model_id)
            start = time.time()

            try:
                # Import appropriate pipeline
                from diffusers import AutoPipelineForText2Image

                # Determine device and dtype
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype_str = config.get("torch_dtype", "float16")
                dtype = getattr(torch, dtype_str, torch.float16)

                # Load pipeline
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    cache_dir=os.environ.get("HF_HOME", "/app/models"),
                    use_safetensors=True
                )

                # Move to device
                self.pipeline = self.pipeline.to(self.device)

                # Enable memory optimizations
                if config.get("enable_vae_slicing", True):
                    self.pipeline.enable_vae_slicing()
                if config.get("enable_vae_tiling", True):
                    self.pipeline.enable_vae_tiling()
                if config.get("enable_cpu_offload", False):
                    self.pipeline.enable_model_cpu_offload()

                self.model_id = model_id
                self.config = config

                elapsed = time.time() - start
                logger.info("model_loaded", model_id=model_id, device=self.device, elapsed_seconds=elapsed)

            except Exception as e:
                logger.error("model_load_failed", model_id=model_id, error=str(e))
                raise

model_manager = ModelManager()

# Load config
def load_config():
    config_path = os.environ.get("CONFIG_PATH", "/app/config/model_config.json")
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("config_not_found", path=config_path)
        return {"default_model": "z-image-turbo", "models": {}}

config = load_config()

# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load default model
    default_model = config.get("default_model", "z-image-turbo")
    model_config = config.get("models", {}).get(default_model, {})
    model_id = model_config.get("model_id", os.environ.get("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo"))

    try:
        await model_manager.load_model(model_id, model_config)
    except Exception as e:
        logger.error("startup_model_load_failed", error=str(e))

    yield

    # Shutdown: cleanup
    logger.info("shutting_down")

app = FastAPI(
    title="Z-Image Server",
    description="OpenAI-compatible image generation API",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response Models (OpenAI-compatible)
class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation")
    model: Optional[str] = Field(None, description="Model to use")
    n: int = Field(1, ge=1, le=4, description="Number of images to generate")
    size: str = Field("1024x1024", description="Image size (WxH)")
    response_format: Literal["url", "b64_json"] = Field("b64_json", description="Response format")
    quality: Optional[str] = Field("standard", description="Image quality")
    style: Optional[str] = Field(None, description="Image style hint")

class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageData]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_id: Optional[str]
    device: Optional[str]
    gpu_memory_used: Optional[str]

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_memory = None
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_memory = f"{allocated:.1f}GB / {total:.1f}GB"

    return HealthResponse(
        status="healthy" if model_manager.pipeline else "loading",
        model_loaded=model_manager.pipeline is not None,
        model_id=model_manager.model_id,
        device=model_manager.device,
        gpu_memory_used=gpu_memory
    )

@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_images(request: ImageGenerationRequest):
    if model_manager.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info("generation_request", prompt=request.prompt[:100], n=request.n, size=request.size)
    start = time.time()

    try:
        # Parse size
        width, height = map(int, request.size.split("x"))

        # Get default params from config
        default_params = model_manager.config.get("default_params", {})
        guidance_scale = default_params.get("guidance_scale", 0.0)
        num_steps = default_params.get("num_inference_steps", 9)

        # Generate images
        images = []
        for i in range(request.n):
            with torch.inference_mode():
                result = model_manager.pipeline(
                    prompt=request.prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                )
                images.append(result.images[0])

        # Encode response
        data = []
        for img in images:
            if request.response_format == "b64_json":
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                b64 = base64.b64encode(buffer.getvalue()).decode()
                data.append(ImageData(b64_json=b64, revised_prompt=request.prompt))
            else:
                # URL format would require storage backend
                raise HTTPException(status_code=400, detail="URL format not supported")

        elapsed = time.time() - start
        logger.info("generation_complete", elapsed_seconds=elapsed, n=len(images))

        return ImageGenerationResponse(
            created=int(time.time()),
            data=data
        )

    except Exception as e:
        logger.error("generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_manager.model_id or "z-image-turbo",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "z-image"
            }
        ]
    }
