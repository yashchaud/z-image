# RunPod Deployment Guide for Z-Image Server

Complete guide to deploy and auto-start z-image server on RunPod.

## Quick Start

### 1. Upload Code to RunPod

**Option A: Using Git (Recommended)**
```bash
# SSH into your RunPod instance
cd /workspace  # or ~/
git clone https://github.com/yourusername/z-image.git
cd z-image
```

**Option B: Using RunPod File Upload**
- Upload the entire `z-image` folder via RunPod's web interface
- Place it in `/workspace/z-image`

### 2. Install Dependencies

```bash
cd /workspace/z-image

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy and edit .env file
cp .env.example .env
nano .env  # or vim .env

# Set your API keys:
# - OPENROUTER_API_KEY=your_key_here
# - HF_TOKEN=your_huggingface_token
```

### 4. Choose Auto-Start Method

Pick ONE of these methods:

#### Method 1: Systemd Service (Most Reliable) ⭐ Recommended

```bash
chmod +x setup_systemd.sh
sudo ./setup_systemd.sh

# Server will auto-start on every pod restart
# Control with:
sudo systemctl start z-image
sudo systemctl stop z-image
sudo systemctl status z-image
```

#### Method 2: RunPod Start Command

```bash
chmod +x start_server.sh

# In RunPod web console:
# Template Settings → Docker Command
# Set to: /workspace/z-image/start_server.sh
```

#### Method 3: Cron @reboot

```bash
chmod +x setup_cron.sh
./setup_cron.sh

# Server starts 30 seconds after pod boot
```

#### Method 4: .bashrc Auto-Start

```bash
chmod +x setup_autostart.sh
./setup_autostart.sh

# Server starts when you SSH into pod
```

### 5. Start Server Now

```bash
chmod +x manage_server.sh
./manage_server.sh start

# Check status
./manage_server.sh status

# View logs
./manage_server.sh logs -f
```

---

## Verification

### Test the Server

```bash
# Check health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","model_loaded":true,...}
```

### Access from Outside RunPod

RunPod provides public URLs for your pod:

1. Go to RunPod web console
2. Find your pod
3. Look for "HTTP Service [Port 8000]"
4. Click to get public URL: `https://your-pod-id.proxy.runpod.net`

Test it:
```bash
curl https://your-pod-id.proxy.runpod.net/health
```

---

## Server Management

### Using manage_server.sh

```bash
# Start server
./manage_server.sh start

# Stop server
./manage_server.sh stop

# Restart server
./manage_server.sh restart

# Check status
./manage_server.sh status

# View logs
./manage_server.sh logs

# Follow logs in real-time
./manage_server.sh logs -f
```

### Using systemd (if installed)

```bash
# Start
sudo systemctl start z-image

# Stop
sudo systemctl stop z-image

# Restart
sudo systemctl restart z-image

# Status
sudo systemctl status z-image

# Logs
sudo journalctl -u z-image -f
```

---

## Troubleshooting

### Server Won't Start

**Check logs:**
```bash
tail -f /tmp/z-image-server.log
# or
sudo journalctl -u z-image -n 100
```

**Common issues:**

1. **GPU not available**
   ```bash
   nvidia-smi  # Check if GPU is detected
   ```

2. **Missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Port already in use**
   ```bash
   # Kill existing process
   lsof -ti:8000 | xargs kill -9
   ```

4. **Missing .env file**
   ```bash
   cp .env.example .env
   nano .env  # Edit with your keys
   ```

### Server Stops After Pod Restart

**Check auto-start setup:**

```bash
# For systemd:
sudo systemctl is-enabled z-image
# Should output: enabled

# For cron:
crontab -l
# Should show @reboot line

# For .bashrc:
grep "Z-IMAGE AUTO-START" ~/.bashrc
# Should show auto-start block
```

### Model Not Loading

**Check disk space:**
```bash
df -h /workspace
```

**Check HuggingFace token:**
```bash
grep HF_TOKEN .env
# Should have valid token
```

**Manually download model:**
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen-Image-Edit-2509', cache_dir='/workspace/models')"
```

---

## RunPod-Specific Tips

### 1. Use /workspace for Persistence

RunPod's `/workspace` directory persists across pod stops/starts:

```bash
# Good: Model cache in /workspace
export HF_HOME=/workspace/models

# Bad: Temporary location
export HF_HOME=/tmp/models
```

### 2. Expose Port in Template

In your RunPod template:
- Add port mapping: `8000:8000`
- Enable "HTTP Service" for port 8000

### 3. Save as Template

After setup:
1. Stop your pod
2. Go to "My Pods" → Your Pod → "Save as Template"
3. Next time: Deploy from template (pre-configured!)

### 4. Network Storage

For models shared across pods:
```bash
# In RunPod template, add network storage
# Mount at: /workspace/models
# All pods share same model cache
```

### 5. Multiple GPU Setup

If using multiple GPUs:

```bash
# Edit .env
CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 GPUs

# Or specific GPUs
CUDA_VISIBLE_DEVICES=0,1  # Use first 2 GPUs
```

---

## Performance Optimization

### For 40GB+ GPUs (A100, A6000)

```bash
# Edit config/model_config.json
{
  "generation": {
    "max_concurrent_requests": 4,  # Handle 4 concurrent requests
    "enable_cpu_offload": false    # Keep everything on GPU
  }
}
```

### For 24GB GPUs (RTX 4090, A5000)

```bash
# Edit config/model_config.json
{
  "generation": {
    "max_concurrent_requests": 2,  # Safer for 24GB
    "enable_vae_slicing": true,
    "enable_vae_tiling": true
  }
}
```

### For 16GB GPUs (RTX 4080, 3090)

```bash
# Use smaller model or enable CPU offload
{
  "generation": {
    "max_concurrent_requests": 1,
    "enable_cpu_offload": true,
    "enable_vae_slicing": true,
    "enable_vae_tiling": true
  }
}
```

---

## Monitoring

### Real-time GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Server Metrics

```bash
# Get server stats
curl http://localhost:8000/stats

# Expected response:
# {
#   "requests_total": 42,
#   "requests_success": 40,
#   "requests_failed": 2,
#   "total_inference_time": 123.45,
#   "avg_inference_time": 3.09
# }
```

### Auto-Monitoring Script

```bash
# Add to cron for health checks
*/5 * * * * curl -s http://localhost:8000/health || /workspace/z-image/manage_server.sh restart
```

---

## Security

### API Key Protection

```bash
# Never commit .env to git
echo ".env" >> .gitignore

# Restrict .env permissions
chmod 600 .env
```

### Firewall (Optional)

```bash
# Only allow local access
# Edit server.py:
# host="127.0.0.1" instead of "0.0.0.0"

# Use RunPod proxy for external access
```

---

## Cost Optimization

### Auto-Stop When Idle

```bash
# Create idle checker script
cat > /workspace/check_idle.sh << 'EOF'
#!/bin/bash
# Stop pod if no requests in last 30 minutes

STATS=$(curl -s http://localhost:8000/stats)
REQUESTS=$(echo $STATS | jq -r '.requests_total')

# Save current count
echo $REQUESTS > /tmp/request_count

# Check if count changed in 30 min
sleep 1800
NEW_REQUESTS=$(curl -s http://localhost:8000/stats | jq -r '.requests_total')

if [ "$REQUESTS" == "$NEW_REQUESTS" ]; then
    echo "No activity, stopping pod..."
    # Use RunPod API to stop pod
fi
EOF

chmod +x /workspace/check_idle.sh
```

---

## Backup & Recovery

### Backup Configuration

```bash
# Backup important files
tar -czf z-image-backup.tar.gz \
    .env \
    config/model_config.json \
    src/pipelines/

# Download from RunPod or upload to cloud storage
```

### Quick Recovery

```bash
# After pod restart:
cd /workspace/z-image
./manage_server.sh start

# Should auto-start if systemd/cron configured
```

---

## Support

### Check Server Status

```bash
./manage_server.sh status
```

### View Recent Errors

```bash
./manage_server.sh logs | grep -i error
```

### Full Health Check

```bash
curl http://localhost:8000/health
nvidia-smi
df -h /workspace
./manage_server.sh status
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start server | `./manage_server.sh start` |
| Stop server | `./manage_server.sh stop` |
| View logs | `./manage_server.sh logs -f` |
| Check status | `./manage_server.sh status` |
| Test API | `curl http://localhost:8000/health` |
| Check GPU | `nvidia-smi` |
| Check disk | `df -h /workspace` |

---

## Next Steps

1. ✅ Deploy code to RunPod
2. ✅ Install dependencies
3. ✅ Configure .env
4. ✅ Choose auto-start method
5. ✅ Test server
6. ✅ Save as RunPod template
7. ✅ Monitor and optimize

Your z-image server is now ready for production on RunPod! 🚀
