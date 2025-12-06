"""
OpenAI-compatible multimodal image generation server.
Supports text-to-image, image-to-image, and image editing with comprehensive logging.
"""

import os
import json
import asyncio
import base64
import time
import traceback
import uuid
from io import BytesIO
from typing import Optional, List, Literal, Union
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import structlog

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# ============================================================================
# UTILITIES
# ============================================================================

def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image with validation."""
    try:
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data))
        image.verify()  # Verify it's a valid image
        image = Image.open(BytesIO(image_data))  # Reopen after verify
        return image.convert("RGB")
    except Exception as e:
        logger.error("image_decode_failed", error=str(e))
        raise ValueError(f"Invalid image data: {str(e)}")

def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()

async def process_upload_file(file: UploadFile) -> Image.Image:
    """Process uploaded image file with validation."""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # Validate image
        if image.format not in ["PNG", "JPEG", "JPG", "WEBP"]:
            raise ValueError(f"Unsupported image format: {image.format}")

        # Validate size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise ValueError("Image size exceeds 10MB limit")

        return image.convert("RGB")
    except Exception as e:
        logger.error("file_upload_failed", filename=file.filename, error=str(e))
        raise ValueError(f"Invalid image file: {str(e)}")

# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """Manages model loading and pipeline operations with comprehensive logging."""

    def __init__(self):
        self.text2img_pipeline = None
        self.img2img_pipeline = None
        self.inpaint_pipeline = None
        self.model_id = None
        self.device = None
        self.config = None
        self._lock = asyncio.Lock()
        self._stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "total_inference_time": 0.0
        }

    async def load_model(self, model_id: str, config: dict):
        """Load all required pipelines with error handling."""
        async with self._lock:
            if self.text2img_pipeline is not None and self.model_id == model_id:
                logger.info("model_already_loaded", model_id=model_id)
                return

            logger.info("loading_model", model_id=model_id, config=config)
            start = time.time()

            try:
                from diffusers import (
                    AutoPipelineForText2Image,
                    AutoPipelineForImage2Image,
                    AutoPipelineForInpainting
                )

                # Determine device and dtype
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype_str = config.get("torch_dtype", "float16")
                dtype = getattr(torch, dtype_str, torch.float16)

                logger.info("device_config", device=self.device, dtype=dtype_str)

                cache_dir = os.environ.get("HF_HOME", "./models")

                # Load text-to-image pipeline
                logger.info("loading_text2img_pipeline")
                self.text2img_pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    cache_dir=cache_dir,
                    use_safetensors=True
                )
                self.text2img_pipeline = self.text2img_pipeline.to(self.device)

                # Load image-to-image pipeline
                logger.info("loading_img2img_pipeline")
                try:
                    self.img2img_pipeline = AutoPipelineForImage2Image.from_pipe(
                        self.text2img_pipeline
                    )
                except Exception as e:
                    logger.warning("img2img_pipeline_unavailable", error=str(e))
                    self.img2img_pipeline = None

                # Load inpainting pipeline
                logger.info("loading_inpaint_pipeline")
                try:
                    self.inpaint_pipeline = AutoPipelineForInpainting.from_pipe(
                        self.text2img_pipeline
                    )
                except Exception as e:
                    logger.warning("inpaint_pipeline_unavailable", error=str(e))
                    self.inpaint_pipeline = None

                # Enable memory optimizations
                for pipeline in [self.text2img_pipeline, self.img2img_pipeline, self.inpaint_pipeline]:
                    if pipeline is not None:
                        try:
                            if config.get("enable_vae_slicing", True):
                                pipeline.enable_vae_slicing()
                            if config.get("enable_vae_tiling", True):
                                pipeline.enable_vae_tiling()
                            if config.get("enable_cpu_offload", False):
                                pipeline.enable_model_cpu_offload()
                        except Exception as e:
                            logger.warning("optimization_failed", error=str(e))

                self.model_id = model_id
                self.config = config

                elapsed = time.time() - start
                logger.info("model_loaded_successfully",
                           model_id=model_id,
                           device=self.device,
                           text2img=self.text2img_pipeline is not None,
                           img2img=self.img2img_pipeline is not None,
                           inpaint=self.inpaint_pipeline is not None,
                           elapsed_seconds=round(elapsed, 2))

            except Exception as e:
                logger.error("model_load_failed",
                            model_id=model_id,
                            error=str(e),
                            traceback=traceback.format_exc())
                raise

    def get_stats(self):
        """Return current statistics."""
        return {
            **self._stats,
            "avg_inference_time": (
                self._stats["total_inference_time"] / self._stats["requests_total"]
                if self._stats["requests_total"] > 0 else 0
            )
        }

model_manager = ModelManager()

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config():
    """Load configuration with fallback."""
    config_path = os.environ.get("CONFIG_PATH", "./config/model_config.json")
    try:
        with open(config_path) as f:
            cfg = json.load(f)
            logger.info("config_loaded", path=config_path)
            return cfg
    except FileNotFoundError:
        logger.warning("config_not_found", path=config_path, using_defaults=True)
        return {
            "default_model": "z-image-turbo",
            "models": {},
            "server": {"host": "0.0.0.0", "port": 8000},
            "generation": {
                "max_concurrent_requests": 4,
                "enable_vae_slicing": True,
                "enable_vae_tiling": True
            }
        }

config = load_config()

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ImageGenerationRequest(BaseModel):
    """Text-to-image generation request."""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Text prompt")
    model: Optional[str] = Field(None, description="Model to use")
    n: int = Field(1, ge=1, le=4, description="Number of images")
    size: str = Field("1024x1024", description="Image size (WxH)")
    response_format: Literal["b64_json"] = Field("b64_json", description="Response format")
    quality: Optional[str] = Field("standard", description="Image quality")
    style: Optional[str] = Field(None, description="Image style")

    # Additional parameters
    guidance_scale: Optional[float] = Field(None, ge=0.0, le=20.0, description="Guidance scale")
    num_inference_steps: Optional[int] = Field(None, ge=1, le=100, description="Inference steps")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    @validator('size')
    def validate_size(cls, v):
        try:
            w, h = map(int, v.split('x'))
            if w < 64 or h < 64 or w > 2048 or h > 2048:
                raise ValueError("Size must be between 64x64 and 2048x2048")
            return v
        except:
            raise ValueError("Size must be in format WIDTHxHEIGHT (e.g., 1024x1024)")

class ImageEditRequest(BaseModel):
    """Image editing/inpainting request."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    image: str = Field(..., description="Base64 encoded image")
    mask: Optional[str] = Field(None, description="Base64 encoded mask")
    n: int = Field(1, ge=1, le=4)
    size: Optional[str] = Field(None, description="Output size (WxH)")
    response_format: Literal["b64_json"] = Field("b64_json")

    guidance_scale: Optional[float] = Field(None, ge=0.0, le=20.0)
    num_inference_steps: Optional[int] = Field(None, ge=1, le=100)
    strength: Optional[float] = Field(0.8, ge=0.0, le=1.0, description="Transformation strength")
    seed: Optional[int] = Field(None)

class ImageVariationRequest(BaseModel):
    """Image variation request."""
    image: str = Field(..., description="Base64 encoded image")
    n: int = Field(1, ge=1, le=4)
    size: Optional[str] = Field(None)
    response_format: Literal["b64_json"] = Field("b64_json")

    strength: Optional[float] = Field(0.75, ge=0.0, le=1.0)
    seed: Optional[int] = Field(None)

class ImageData(BaseModel):
    """Image response data."""
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    """Standard image generation response."""
    created: int
    data: List[ImageData]

class ErrorResponse(BaseModel):
    """Error response."""
    error: dict

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_id: Optional[str]
    device: Optional[str]
    gpu_memory_used: Optional[str]
    pipelines_available: dict
    stats: dict

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with model loading."""
    logger.info("server_starting")

    # Startup: load model
    default_model = config.get("default_model", "z-image-turbo")
    model_config = config.get("models", {}).get(default_model, {})
    model_id = model_config.get("model_id", os.environ.get("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo"))

    try:
        await model_manager.load_model(model_id, model_config)
        logger.info("server_ready", model_id=model_id)
    except Exception as e:
        logger.error("startup_failed", error=str(e), traceback=traceback.format_exc())

    yield

    # Shutdown
    logger.info("server_shutting_down", stats=model_manager.get_stats())

app = FastAPI(
    title="Multimodal Image Generation Server",
    description="OpenAI-compatible API for text-to-image, image-to-image, and image editing",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    request_id = str(uuid.uuid4())
    start = time.time()

    logger.info("request_started",
               request_id=request_id,
               method=request.method,
               path=request.url.path,
               client=request.client.host if request.client else None)

    try:
        response = await call_next(request)
        elapsed = time.time() - start

        logger.info("request_completed",
                   request_id=request_id,
                   status_code=response.status_code,
                   elapsed_seconds=round(elapsed, 3))

        return response
    except Exception as e:
        elapsed = time.time() - start
        logger.error("request_failed",
                    request_id=request_id,
                    error=str(e),
                    elapsed_seconds=round(elapsed, 3),
                    traceback=traceback.format_exc())
        raise

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "name": "Multimodal Image Generation Server",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            "text_to_image": "/v1/images/generations",
            "image_editing": "/v1/images/edits",
            "image_variations": "/v1/images/variations",
            "health": "/health",
            "models": "/v1/models"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check."""
    gpu_memory = None
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_memory = f"{allocated:.2f}GB / {total:.2f}GB"
        except Exception as e:
            logger.warning("gpu_memory_check_failed", error=str(e))
            gpu_memory = "unavailable"

    return HealthResponse(
        status="healthy" if model_manager.text2img_pipeline else "loading",
        model_loaded=model_manager.text2img_pipeline is not None,
        model_id=model_manager.model_id,
        device=model_manager.device,
        gpu_memory_used=gpu_memory,
        pipelines_available={
            "text2img": model_manager.text2img_pipeline is not None,
            "img2img": model_manager.img2img_pipeline is not None,
            "inpaint": model_manager.inpaint_pipeline is not None
        },
        stats=model_manager.get_stats()
    )

@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_images(request: ImageGenerationRequest):
    """
    Generate images from text prompts (text-to-image).

    OpenAI-compatible endpoint for creating images from text descriptions.
    """
    request_id = str(uuid.uuid4())
    logger.info("text2img_request",
               request_id=request_id,
               prompt=request.prompt[:100],
               n=request.n,
               size=request.size)

    if model_manager.text2img_pipeline is None:
        logger.error("pipeline_not_loaded", request_id=request_id, pipeline="text2img")
        raise HTTPException(status_code=503, detail="Model not loaded. Please wait.")

    start = time.time()
    model_manager._stats["requests_total"] += 1

    try:
        # Parse size
        width, height = map(int, request.size.split("x"))

        # Get generation params
        default_params = model_manager.config.get("default_params", {})
        guidance_scale = request.guidance_scale or default_params.get("guidance_scale", 7.5)
        num_steps = request.num_inference_steps or default_params.get("num_inference_steps", 30)

        # Set seed if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=model_manager.device).manual_seed(request.seed)
            logger.info("using_seed", request_id=request_id, seed=request.seed)

        # Generate images
        images = []
        for i in range(request.n):
            logger.info("generating_image", request_id=request_id, index=i+1, total=request.n)

            with torch.inference_mode():
                result = model_manager.text2img_pipeline(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    generator=generator
                )
                images.append(result.images[0])

        # Encode response
        data = []
        for i, img in enumerate(images):
            b64 = encode_image_to_base64(img)
            data.append(ImageData(b64_json=b64, revised_prompt=request.prompt))
            logger.info("image_encoded", request_id=request_id, index=i+1, size_kb=len(b64)//1024)

        elapsed = time.time() - start
        model_manager._stats["requests_success"] += 1
        model_manager._stats["total_inference_time"] += elapsed

        logger.info("text2img_complete",
                   request_id=request_id,
                   elapsed_seconds=round(elapsed, 2),
                   n=len(images),
                   avg_time_per_image=round(elapsed/len(images), 2))

        return ImageGenerationResponse(
            created=int(time.time()),
            data=data
        )

    except Exception as e:
        model_manager._stats["requests_failed"] += 1
        logger.error("text2img_failed",
                    request_id=request_id,
                    error=str(e),
                    traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.post("/v1/images/edits", response_model=ImageGenerationResponse)
async def edit_images(request: ImageEditRequest):
    """
    Edit images using prompts (image-to-image or inpainting).

    Supports both image-to-image transformation and inpainting with masks.
    """
    request_id = str(uuid.uuid4())
    logger.info("image_edit_request",
               request_id=request_id,
               prompt=request.prompt[:100],
               has_mask=request.mask is not None,
               n=request.n)

    # Choose appropriate pipeline
    if request.mask:
        if model_manager.inpaint_pipeline is None:
            raise HTTPException(status_code=501, detail="Inpainting not supported by this model")
        pipeline = model_manager.inpaint_pipeline
        mode = "inpaint"
    else:
        if model_manager.img2img_pipeline is None:
            raise HTTPException(status_code=501, detail="Image-to-image not supported by this model")
        pipeline = model_manager.img2img_pipeline
        mode = "img2img"

    start = time.time()
    model_manager._stats["requests_total"] += 1

    try:
        # Decode input image
        logger.info("decoding_input_image", request_id=request_id)
        input_image = decode_base64_image(request.image)

        # Decode mask if provided
        mask_image = None
        if request.mask:
            logger.info("decoding_mask", request_id=request_id)
            mask_image = decode_base64_image(request.mask).convert("L")

        # Get params
        default_params = model_manager.config.get("default_params", {})
        guidance_scale = request.guidance_scale or default_params.get("guidance_scale", 7.5)
        num_steps = request.num_inference_steps or default_params.get("num_inference_steps", 30)
        strength = request.strength

        # Set seed
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=model_manager.device).manual_seed(request.seed)

        # Generate images
        images = []
        for i in range(request.n):
            logger.info("editing_image", request_id=request_id, mode=mode, index=i+1, total=request.n)

            with torch.inference_mode():
                if mode == "inpaint":
                    result = pipeline(
                        prompt=request.prompt,
                        image=input_image,
                        mask_image=mask_image,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                        strength=strength,
                        generator=generator
                    )
                else:
                    result = pipeline(
                        prompt=request.prompt,
                        image=input_image,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                        strength=strength,
                        generator=generator
                    )
                images.append(result.images[0])

        # Encode response
        data = [ImageData(b64_json=encode_image_to_base64(img)) for img in images]

        elapsed = time.time() - start
        model_manager._stats["requests_success"] += 1
        model_manager._stats["total_inference_time"] += elapsed

        logger.info("image_edit_complete",
                   request_id=request_id,
                   mode=mode,
                   elapsed_seconds=round(elapsed, 2),
                   n=len(images))

        return ImageGenerationResponse(created=int(time.time()), data=data)

    except ValueError as e:
        model_manager._stats["requests_failed"] += 1
        logger.error("validation_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        model_manager._stats["requests_failed"] += 1
        logger.error("image_edit_failed",
                    request_id=request_id,
                    error=str(e),
                    traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Image editing failed: {str(e)}")

@app.post("/v1/images/variations", response_model=ImageGenerationResponse)
async def create_variations(request: ImageVariationRequest):
    """
    Create variations of an image.

    Generates similar images based on an input image.
    """
    request_id = str(uuid.uuid4())
    logger.info("variation_request", request_id=request_id, n=request.n)

    if model_manager.img2img_pipeline is None:
        raise HTTPException(status_code=501, detail="Image variations not supported by this model")

    start = time.time()
    model_manager._stats["requests_total"] += 1

    try:
        # Decode input
        logger.info("decoding_input_image", request_id=request_id)
        input_image = decode_base64_image(request.image)

        # Set seed
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=model_manager.device).manual_seed(request.seed)

        # Generate variations
        images = []
        for i in range(request.n):
            logger.info("creating_variation", request_id=request_id, index=i+1, total=request.n)

            with torch.inference_mode():
                result = model_manager.img2img_pipeline(
                    prompt="",  # No text prompt for variations
                    image=input_image,
                    strength=request.strength,
                    generator=generator
                )
                images.append(result.images[0])

        # Encode response
        data = [ImageData(b64_json=encode_image_to_base64(img)) for img in images]

        elapsed = time.time() - start
        model_manager._stats["requests_success"] += 1
        model_manager._stats["total_inference_time"] += elapsed

        logger.info("variation_complete",
                   request_id=request_id,
                   elapsed_seconds=round(elapsed, 2),
                   n=len(images))

        return ImageGenerationResponse(created=int(time.time()), data=data)

    except ValueError as e:
        model_manager._stats["requests_failed"] += 1
        logger.error("validation_error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        model_manager._stats["requests_failed"] += 1
        logger.error("variation_failed",
                    request_id=request_id,
                    error=str(e),
                    traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Variation generation failed: {str(e)}")

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_manager.model_id or "unknown",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "z-image",
                "capabilities": {
                    "text_to_image": model_manager.text2img_pipeline is not None,
                    "image_to_image": model_manager.img2img_pipeline is not None,
                    "inpainting": model_manager.inpaint_pipeline is not None
                }
            }
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get server statistics."""
    return {
        "stats": model_manager.get_stats(),
        "device": model_manager.device,
        "model": model_manager.model_id
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with logging."""
    logger.warning("http_exception",
                  status_code=exc.status_code,
                  detail=exc.detail,
                  path=request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "http_error", "code": exc.status_code}}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    logger.error("unhandled_exception",
                error=str(exc),
                path=request.url.path,
                traceback=traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "server_error", "code": 500}}
    )

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))

    logger.info("starting_server", host=host, port=port)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
