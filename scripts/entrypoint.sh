#!/bin/bash
set -euo pipefail

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting Z-Image Server..."

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    log "WARNING: No GPU detected. Running on CPU (will be slow)."
    export CUDA_VISIBLE_DEVICES=""
else
    log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    nvidia-smi --query-gpu=memory.total,memory.free --format=csv
fi

# Set HuggingFace token if provided
if [ -n "${HF_TOKEN:-}" ]; then
    log "HuggingFace token configured"
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Download model if not cached
MODEL_ID="${MODEL_ID:-Tongyi-MAI/Z-Image-Turbo}"
log "Checking model: ${MODEL_ID}"

python -c "
from huggingface_hub import snapshot_download
import os

model_id = os.environ.get('MODEL_ID', 'Tongyi-MAI/Z-Image-Turbo')
cache_dir = os.environ.get('HF_HOME', '/app/models')

print(f'Downloading {model_id} to {cache_dir}...')
try:
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,
        local_files_only=False
    )
    print('Model download complete.')
except Exception as e:
    print(f'Download error: {e}')
    # Continue anyway - model might already be cached
"

log "Starting API server on port ${PORT:-8000}..."

# Start the server
exec python -m uvicorn src.server:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --workers "${WORKERS:-1}" \
    --timeout-keep-alive "${TIMEOUT:-300}"
