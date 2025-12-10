# Docker Deployment Guide

Complete guide for deploying z-image server with Docker (RunPod & local).

## Quick Start

### Option 1: Pre-built Image (GitHub Container Registry)

```bash
# Pull the latest image
docker pull ghcr.io/yourusername/z-image:latest

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=your_huggingface_token \
  -e OPENROUTER_API_KEY=your_openrouter_key \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/assets:/app/assets \
  ghcr.io/yourusername/z-image:latest
```

### Option 2: Build Locally

```bash
# Clone repo
git clone https://github.com/yourusername/z-image.git
cd z-image

# Build image
docker build -t z-image-server:latest .

# Run
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=your_token \
  -e OPENROUTER_API_KEY=your_key \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/assets:/app/assets \
  z-image-server:latest
```

### Option 3: Docker Compose (Recommended)

```bash
# Create .env file
cat > .env << EOF
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_key
MODEL_ID=Qwen/Qwen-Image-Edit-2509
PORT=8000
LOG_LEVEL=INFO
EOF

# Start with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## RunPod Deployment

### Method 1: Using Docker Image (Fastest)

**1. Create RunPod Template:**
- Container Image: `ghcr.io/yourusername/z-image:latest`
- Docker Command: Leave empty (uses default CMD)
- Container Disk: 20GB minimum
- Volume Disk: 50GB+ (for models)

**2. Environment Variables:**
```
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_key
MODEL_ID=Qwen/Qwen-Image-Edit-2509
```

**3. Exposed Ports:**
- HTTP Service: Port 8000

**4. Volume Mounts:**
```
Container Path: /app/models
Mount Path: /workspace/models

Container Path: /app/assets
Mount Path: /workspace/assets
```

**5. Start Pod:**
- Select GPU (recommend: A100 40GB or RTX A6000)
- Deploy!
- Server auto-starts on pod start

### Method 2: Build on RunPod

```bash
# SSH into RunPod pod
cd /workspace

# Clone repo
git clone https://github.com/yourusername/z-image.git
cd z-image

# Build Docker image
docker build -t z-image-server:latest .

# Run container
docker run -d --name z-image \
  --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  -v /workspace/models:/app/models \
  -v /workspace/assets:/app/assets \
  --restart unless-stopped \
  z-image-server:latest

# Check logs
docker logs -f z-image
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ Yes | - | HuggingFace API token |
| `OPENROUTER_API_KEY` | ✅ Yes | - | OpenRouter API key (for LinkedIn pipeline) |
| `MODEL_ID` | No | Qwen/Qwen-Image-Edit-2509 | HuggingFace model to use |
| `PORT` | No | 8000 | Server port |
| `HOST` | No | 0.0.0.0 | Server host |
| `LOG_LEVEL` | No | INFO | Logging level |
| `CUDA_VISIBLE_DEVICES` | No | 0 | GPU device selection |

### Volumes

| Container Path | Purpose | Size Needed |
|----------------|---------|-------------|
| `/app/models` | Model cache | 20-50GB |
| `/app/assets` | Generated images | 10GB+ |

### Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 8000 | HTTP | API server |

---

## GitHub Actions Auto-Build

Every push to `main` branch automatically builds and publishes Docker image.

### Setup

**1. Enable GitHub Packages:**
- Go to repo Settings → Actions → General
- Enable "Read and write permissions"

**2. Push to trigger build:**
```bash
git add .
git commit -m "Update z-image"
git push origin main
```

**3. Image published to:**
```
ghcr.io/yourusername/z-image:latest
ghcr.io/yourusername/z-image:main
ghcr.io/yourusername/z-image:sha-abc1234
```

### Manual Build Trigger

Go to: Actions → Build and Push Docker Image → Run workflow

---

## Usage

### Check Health

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_id": "Qwen/Qwen-Image-Edit-2509",
  "device": "cuda",
  "gpu_memory_used": "12.3GB / 40.0GB"
}
```

### Generate Image

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset",
    "size": "1024x1024"
  }'
```

### LinkedIn Pipeline

```bash
# Encode reference image
REF_IMAGE=$(base64 -w 0 reference.png)

# Generate LinkedIn variants
curl -X POST http://localhost:8000/v1/linkedin/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"Announcing our new product launch\",
    \"reference_image\": \"$REF_IMAGE\",
    \"size\": \"1024x1024\"
  }"
```

---

## Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker logs z-image-server
# or
docker-compose logs
```

**Common issues:**

1. **GPU not available**
```bash
# Test GPU in container
docker run --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

2. **Port already in use**
```bash
# Find process using port 8000
lsof -i :8000
# Kill it
kill -9 <PID>
```

3. **Out of disk space**
```bash
# Check disk usage
df -h
# Clean up Docker
docker system prune -a
```

### Model Download Fails

**Check HuggingFace token:**
```bash
docker exec z-image-server bash -c 'echo $HF_TOKEN'
```

**Manually download model:**
```bash
docker exec z-image-server bash -c '
  python -c "from huggingface_hub import snapshot_download; \
  snapshot_download(\"Qwen/Qwen-Image-Edit-2509\", cache_dir=\"/app/models\")"
'
```

### Server Not Responding

**Check if running:**
```bash
docker ps | grep z-image
```

**Restart container:**
```bash
docker restart z-image-server
# or
docker-compose restart
```

**Check resource usage:**
```bash
docker stats z-image-server
```

---

## Performance Optimization

### For 40GB+ GPUs (A100, A6000)

```yaml
# docker-compose.yml
shm_size: '16gb'
environment:
  - CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all GPUs
```

### For 24GB GPUs (RTX 4090, A5000)

```yaml
shm_size: '8gb'
environment:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### For Multiple GPUs

```bash
# Use specific GPUs
docker run --gpus '"device=0,1"' ...

# Use all GPUs
docker run --gpus all ...
```

---

## Updating

### Pull Latest Image

```bash
docker pull ghcr.io/yourusername/z-image:latest
docker-compose down
docker-compose up -d
```

### Rebuild from Source

```bash
git pull origin main
docker-compose build --no-cache
docker-compose up -d
```

---

## Monitoring

### Real-time Logs

```bash
# Docker
docker logs -f z-image-server

# Docker Compose
docker-compose logs -f
```

### Container Stats

```bash
docker stats z-image-server
```

### GPU Monitoring

```bash
# From host
nvidia-smi -l 1

# Inside container
docker exec z-image-server nvidia-smi
```

---

## Cleanup

### Remove Container

```bash
docker-compose down
# or
docker stop z-image-server
docker rm z-image-server
```

### Remove Images

```bash
# Remove specific image
docker rmi z-image-server:latest

# Remove all unused images
docker image prune -a
```

### Clean Volumes (⚠️ Deletes models)

```bash
docker volume rm z-image_model-cache
docker volume rm z-image_assets
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start | `docker-compose up -d` |
| Stop | `docker-compose down` |
| Logs | `docker-compose logs -f` |
| Restart | `docker-compose restart` |
| Rebuild | `docker-compose build --no-cache` |
| Shell | `docker exec -it z-image-server bash` |
| Health | `curl http://localhost:8000/health` |
| Stats | `docker stats z-image-server` |

---

## Production Checklist

- [ ] Set strong HuggingFace token
- [ ] Set OpenRouter API key
- [ ] Configure reverse proxy (nginx/Caddy)
- [ ] Enable HTTPS
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log rotation
- [ ] Set resource limits
- [ ] Enable automatic restarts
- [ ] Back up model cache
- [ ] Test failover scenarios

---

Your z-image server is now Dockerized and ready for deployment! 🚀
