#!/bin/bash

# Setup Systemd Service for Z-Image Server
# This creates a proper system service that auto-starts

echo "Setting up Z-Image as systemd service..."

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH=$(which python3 || which python)
USER=$(whoami)

echo "Project directory: $PROJECT_DIR"
echo "Python path: $PYTHON_PATH"
echo "User: $USER"

# Create systemd service file
sudo tee /etc/systemd/system/z-image.service > /dev/null << EOF
[Unit]
Description=Z-Image Server - AI Image Generation API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="HF_HOME=/workspace/models"
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PYTHON_PATH $PROJECT_DIR/src/server.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/z-image-server.log
StandardError=append:/var/log/z-image-error.log

# Resource limits (adjust based on your GPU)
MemoryLimit=40G
CPUQuota=400%

[Install]
WantedBy=multi-user.target
EOF

# Create log files with proper permissions
sudo touch /var/log/z-image-server.log
sudo touch /var/log/z-image-error.log
sudo chown $USER:$USER /var/log/z-image-server.log
sudo chown $USER:$USER /var/log/z-image-error.log

# Reload systemd
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable z-image.service

# Start the service
sudo systemctl start z-image.service

# Check status
sleep 3
sudo systemctl status z-image.service

echo ""
echo "================================================"
echo "✅ Systemd service configured!"
echo "================================================"
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start z-image"
echo "  Stop:    sudo systemctl stop z-image"
echo "  Restart: sudo systemctl restart z-image"
echo "  Status:  sudo systemctl status z-image"
echo "  Logs:    sudo journalctl -u z-image -f"
echo "  or:      tail -f /var/log/z-image-server.log"
echo ""
echo "The service will auto-start on pod restart!"
echo "================================================"
