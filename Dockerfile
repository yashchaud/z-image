# Lightweight base with CUDA support
# Using NVIDIA's public registry (no authentication required)
FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Build args for flexibility
ARG PYTHON_VERSION=3.11
ARG MODEL_ID=Tongyi-MAI/Z-Image-Turbo
ARG HF_HOME=/app/models

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=${HF_HOME} \
    MODEL_ID=${MODEL_ID}

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install FastFusion and dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY src/ /app/src/

# Copy configuration and scripts
COPY config/ /app/config/
COPY scripts/ /app/scripts/
RUN chmod +x /app/scripts/*.sh

# Create assets directory for serving images
RUN mkdir -p /app/assets && chmod 777 /app/assets

# Create volume for persistent assets storage
VOLUME ["/app/assets"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python /app/scripts/healthcheck.py || exit 1

# Expose API port
EXPOSE 8000

# Entrypoint handles model download and server start
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
