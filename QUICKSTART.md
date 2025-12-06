# Quick Start Guide - Local Development

This guide will help you run the multimodal image generation server locally (without Docker).

## Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support (optional, will fall back to CPU)
- 16GB+ RAM (32GB recommended)
- ~10-20GB disk space for models

## Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to set your model
# Optional: Add HuggingFace token if using gated models
```

Example `.env`:
```bash
MODEL_ID=Tongyi-MAI/Z-Image-Turbo
HF_TOKEN=your_hf_token_here  # Optional
PORT=8000
LOG_LEVEL=INFO
```

## Step 3: Start the Server

```bash
# Run the server
python src/server.py
```

The server will:
1. Load configuration from `config/model_config.json`
2. Download the model (first run only, takes 5-15 minutes)
3. Load model into memory
4. Start API server on http://localhost:8000

### Expected Output

```json
{"event": "server_starting", "timestamp": "2025-12-06T..."}
{"event": "loading_model", "model_id": "Tongyi-MAI/Z-Image-Turbo", ...}
{"event": "device_config", "device": "cuda", "dtype": "float16"}
{"event": "loading_text2img_pipeline"}
{"event": "loading_img2img_pipeline"}
{"event": "model_loaded_successfully", "elapsed_seconds": 45.2}
{"event": "server_ready", "model_id": "Tongyi-MAI/Z-Image-Turbo"}
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 4: Test the Server

### Option A: Use the Test Script

```bash
# In a new terminal
python test_server.py
```

This will test all endpoints and save output images.

### Option B: Manual Testing

**Check health:**
```bash
curl http://localhost:8000/health
```

**Generate an image:**
```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "n": 1,
    "size": "1024x1024"
  }' > response.json
```

**Using Python (OpenAI SDK compatible):**
```python
import base64
import requests

# Generate image
response = requests.post("http://localhost:8000/v1/images/generations", json={
    "prompt": "A serene lake with mountains in the background",
    "n": 1,
    "size": "512x512",
    "num_inference_steps": 20
})

# Save image
data = response.json()
img_data = base64.b64decode(data['data'][0]['b64_json'])
with open("output.png", "wb") as f:
    f.write(img_data)

print("Image saved to output.png")
```

## API Endpoints

### 1. Text-to-Image Generation

**POST** `/v1/images/generations`

```json
{
  "prompt": "A serene mountain landscape",
  "n": 1,
  "size": "1024x1024",
  "guidance_scale": 7.5,
  "num_inference_steps": 30,
  "negative_prompt": "blurry, low quality",
  "seed": 42
}
```

### 2. Image Editing (Image-to-Image)

**POST** `/v1/images/edits`

```json
{
  "prompt": "Transform into a sunset scene",
  "image": "base64_encoded_image_here",
  "n": 1,
  "strength": 0.7,
  "guidance_scale": 7.5,
  "num_inference_steps": 30,
  "seed": 123
}
```

### 3. Image Variations

**POST** `/v1/images/variations`

```json
{
  "image": "base64_encoded_image_here",
  "n": 2,
  "strength": 0.75,
  "seed": 456
}
```

### 4. Health Check

**GET** `/health`

Returns server status, GPU memory usage, available pipelines, and statistics.

### 5. List Models

**GET** `/v1/models`

Returns information about loaded models and their capabilities.

### 6. Server Statistics

**GET** `/stats`

Returns request statistics and performance metrics.

## Logging

The server uses structured JSON logging for easy parsing:

```json
{
  "event": "text2img_request",
  "request_id": "uuid-here",
  "prompt": "A beautiful...",
  "n": 1,
  "size": "1024x1024",
  "timestamp": "2025-12-06T14:30:00Z"
}
```

### Log Levels

Set via environment variable:
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Key Log Events

- `server_starting` - Server initialization
- `model_loaded_successfully` - Model ready
- `request_started` - New request received
- `text2img_request` - Text-to-image generation started
- `generating_image` - Image generation in progress
- `text2img_complete` - Generation finished with timing
- `request_completed` - Request finished
- `text2img_failed` - Error during generation (includes traceback)

## Troubleshooting

### Model Not Loading

**Check logs for errors:**
```json
{"event": "model_load_failed", "error": "...", "traceback": "..."}
```

**Common issues:**
- Not enough GPU memory → Use CPU or smaller model
- No internet → Model needs to download first
- Invalid model ID → Check MODEL_ID in .env

### Out of Memory

**Solution 1: Enable CPU offload**

Edit `config/model_config.json`:
```json
{
  "generation": {
    "enable_cpu_offload": true,
    "enable_vae_slicing": true,
    "enable_vae_tiling": true
  }
}
```

**Solution 2: Use smaller image sizes**
```json
{
  "size": "512x512"  // Instead of 1024x1024
}
```

**Solution 3: Reduce inference steps**
```json
{
  "num_inference_steps": 20  // Instead of 50
}
```

### Slow Generation (CPU Mode)

If no GPU is detected, generation will be slow (5-10 minutes per image).

**Check device:**
```bash
curl http://localhost:8000/health | jq '.device'
```

Expected: `"cuda"` (GPU) or `"cpu"` (CPU fallback)

### Connection Refused

Make sure the server is running:
```bash
# Check if server is listening
netstat -an | findstr "8000"  # Windows
netstat -an | grep "8000"     # Linux/Mac
```

## Advanced Usage

### Change Model

Edit `.env`:
```bash
MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
```

Or update `config/model_config.json`:
```json
{
  "default_model": "sdxl",
  "models": {
    "sdxl": {
      "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
      "torch_dtype": "float16",
      "default_params": {
        "guidance_scale": 7.5,
        "num_inference_steps": 30
      }
    }
  }
}
```

Restart the server for changes to take effect.

### Multiple Workers

For production with multiple GPUs:
```bash
# NOT recommended for single GPU
WORKERS=2 python src/server.py
```

**Note:** Keep WORKERS=1 for single GPU to avoid memory issues.

### Custom Port

```bash
PORT=9000 python src/server.py
```

## Production Deployment

For production deployment:

1. Use a process manager (systemd, supervisor)
2. Enable HTTPS with reverse proxy (nginx, caddy)
3. Set up proper firewall rules
4. Monitor logs and metrics
5. Implement rate limiting
6. Add authentication if needed

Example systemd service:
```ini
[Unit]
Description=Z-Image Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/z-image
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python src/server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Next Steps

- Read the full API documentation in [README.md](README.md)
- Explore example requests in [test_server.py](test_server.py)
- Check model configuration in [config/model_config.json](config/model_config.json)
- Review logs for optimization opportunities

## Support

If you encounter issues:

1. Check the logs for error messages
2. Verify GPU/CPU availability
3. Ensure sufficient memory
4. Check model download completed
5. Review configuration files

For detailed logging, set `LOG_LEVEL=DEBUG` in `.env`.
