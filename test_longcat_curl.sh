#!/bin/bash

# Sample API calls for testing LongCat Image Edit model
# Make sure the server is running at http://localhost:8000

BASE_URL="http://localhost:8000"

echo "=============================================="
echo "  LongCat Image Edit Model - Sample API Calls"
echo "=============================================="

# 1. Check server health
echo -e "\n[1] Checking Server Health..."
echo "----------------------------------------"
curl -s "${BASE_URL}/health" | jq .

# 2. Text-to-Image Generation
echo -e "\n[2] Text-to-Image Generation..."
echo "----------------------------------------"
curl -s -X POST "${BASE_URL}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset with a clear lake reflection",
    "n": 1,
    "size": "1024x1024",
    "response_format": "url",
    "guidance_scale": 4.5,
    "num_inference_steps": 50,
    "seed": 42
  }' | jq .

# 3. Image Editing with URL
echo -e "\n[3] Image Editing with URL..."
echo "----------------------------------------"
curl -s -X POST "${BASE_URL}/v1/images/edits" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Make this image look like a vibrant watercolor painting",
    "image_url": "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg",
    "n": 1,
    "response_format": "url",
    "guidance_scale": 4.5,
    "num_inference_steps": 50,
    "strength": 0.8,
    "seed": 123
  }' | jq .

# 4. List all available models
echo -e "\n[4] Available Models..."
echo "----------------------------------------"
curl -s "${BASE_URL}/v1/models" | jq .

# 5. Server statistics
echo -e "\n[5] Server Statistics..."
echo "----------------------------------------"
curl -s "${BASE_URL}/stats" | jq .

echo -e "\n=============================================="
echo "  Tips:"
echo "  - View generated images: ${BASE_URL}/assets/"
echo "  - List all assets: ${BASE_URL}/debug/assets"
echo "  - API documentation: ${BASE_URL}/docs"
echo "=============================================="
