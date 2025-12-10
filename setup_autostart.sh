#!/bin/bash

# Setup Auto-Start for Z-Image Server on RunPod
# Run this script once to configure auto-start

echo "Setting up auto-start for Z-Image server..."

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# Create the auto-start script in bashrc
BASHRC_FILE="$HOME/.bashrc"

# Remove old auto-start entries if they exist
sed -i '/# Z-IMAGE AUTO-START/,/# END Z-IMAGE AUTO-START/d' "$BASHRC_FILE"

# Add new auto-start entry
cat >> "$BASHRC_FILE" << 'EOF'

# Z-IMAGE AUTO-START
# Automatically start z-image server on pod startup
if [ ! -f /tmp/z-image-server.pid ] || ! ps -p $(cat /tmp/z-image-server.pid 2>/dev/null) > /dev/null 2>&1; then
    if [ -f "/workspace/z-image/start_server.sh" ]; then
        echo "Auto-starting Z-Image server..."
        bash /workspace/z-image/start_server.sh
    elif [ -f "$HOME/z-image/start_server.sh" ]; then
        echo "Auto-starting Z-Image server..."
        bash $HOME/z-image/start_server.sh
    fi
fi
# END Z-IMAGE AUTO-START

EOF

echo "✅ Auto-start configured!"
echo ""
echo "The server will automatically start when you:"
echo "  1. SSH into the pod"
echo "  2. Open a new terminal"
echo "  3. Restart the pod"
echo ""
echo "To test, run: source ~/.bashrc"
