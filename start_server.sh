#!/bin/bash
set -e

echo "========================================="
echo "Z-Image Server Startup"
echo "========================================="

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt --no-cache-dir

echo ""
echo "Starting server..."
python src/server.py
