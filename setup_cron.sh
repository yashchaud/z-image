#!/bin/bash

# Setup Cron Auto-Start for Z-Image Server
# This uses cron @reboot to start server on pod restart

echo "Setting up cron auto-start..."

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create cron job
CRON_CMD="@reboot sleep 30 && cd $PROJECT_DIR && bash $PROJECT_DIR/start_server.sh >> /tmp/z-image-cron.log 2>&1"

# Check if cron job already exists
(crontab -l 2>/dev/null | grep -v "z-image"; echo "$CRON_CMD") | crontab -

echo "✅ Cron job added!"
echo ""
echo "Current crontab:"
crontab -l
echo ""
echo "The server will start 30 seconds after pod reboot."
echo "To remove: crontab -e (and delete the z-image line)"
