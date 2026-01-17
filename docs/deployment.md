# Deployment Guide

Complete guide for deploying AI Forge to production.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Configuration](#production-configuration)
5. [Backup & Recovery](#backup--recovery)

---

## Local Deployment

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| macOS | 13.0 (Ventura) | 14.0+ (Sonoma) |
| Chip | M1 | M3 Pro/Max |
| RAM | 16GB | 32GB+ |
| Storage | 50GB free | 100GB+ SSD |
| Python | 3.11 | 3.12 |

### Step 1: System Preparation

```bash
# Update Homebrew
brew update && brew upgrade

# Install system dependencies
brew install git python@3.11 ollama

# Verify installations
python3.11 --version
ollama --version
```

### Step 2: Clone and Install

```bash
# Clone repository
git clone https://github.com/ai-forge/ai-forge.git
cd ai-forge

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -e ".[all]"
```

### Step 3: Configuration

Create `.env` file:

```bash
# .env
AI_FORGE_HOST=0.0.0.0
AI_FORGE_PORT=8000
AI_FORGE_DEBUG=false
AI_FORGE_OLLAMA_HOST=http://localhost:11434
AI_FORGE_LOG_LEVEL=INFO
```

### Step 4: Start Services

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start AI Forge
source .venv/bin/activate
uvicorn conductor.service:app --host 0.0.0.0 --port 8000
```

### Step 5: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test Ollama connection
curl http://localhost:11434

# Test API endpoint
curl -X POST http://localhost:8000/v1/retrain \
  -H "Content-Type: application/json" \
  -d '{"project_path": ".", "force": false}'
```

### Step 6: Run as Service (launchd)

Create `/Library/LaunchDaemons/com.aiforge.service.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aiforge.service</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/ai-forge/.venv/bin/uvicorn</string>
        <string>conductor.service:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/ai-forge</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/path/to/ai-forge/.venv/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/aiforge/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/aiforge/stderr.log</string>
</dict>
</plist>
```

Enable:
```bash
sudo mkdir -p /var/log/aiforge
sudo launchctl load /Library/LaunchDaemons/com.aiforge.service.plist
```

---

## Docker Deployment

### Build Image

```bash
docker build -t ai-forge:latest .
```

### Run Container

```bash
# Run with GPU (requires NVIDIA GPU)
docker run -d \
  --name ai-forge \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  ai-forge:latest

# Run CPU-only (Mac compatible)
docker run -d \
  --name ai-forge \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  ai-forge:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Cloud Deployment

### AWS EC2

**Recommended Instance:**
- Type: `g5.xlarge` (GPU) or `m6i.2xlarge` (CPU)
- AMI: Ubuntu 22.04 LTS
- Storage: 100GB gp3

**Setup:**
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Install NVIDIA drivers (for GPU instances)
sudo apt install -y nvidia-driver-535

# Deploy
docker-compose up -d
```

### Google Cloud

**Recommended Instance:**
- Type: `n1-standard-8` with NVIDIA T4
- OS: Ubuntu 22.04

**Setup:**
```bash
# Install NVIDIA drivers
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py | sudo python3

# Deploy
docker-compose up -d
```

### Azure

**Recommended Instance:**
- Type: `Standard_NC6s_v3` (GPU) or `Standard_D8s_v3` (CPU)
- OS: Ubuntu 22.04

### Cost Optimization

| Provider | Instance | Cost/month | Best For |
|----------|----------|------------|----------|
| AWS | g5.xlarge | ~$500 | Production |
| AWS | m6i.2xlarge | ~$200 | Dev/Test |
| GCP | n1-standard-8 + T4 | ~$450 | Production |
| Azure | NC6s_v3 | ~$600 | Enterprise |

**Tips:**
1. Use spot/preemptible instances for training
2. Scale down during off-hours
3. Use reserved instances for 40% savings
4. Consider Mac Mini cluster for cost efficiency

---

## Production Configuration

### Environment Variables

```bash
# Required
AI_FORGE_HOST=0.0.0.0
AI_FORGE_PORT=8000
AI_FORGE_OLLAMA_HOST=http://localhost:11434

# Recommended
AI_FORGE_DEBUG=false
AI_FORGE_LOG_LEVEL=WARNING
AI_FORGE_MAX_WORKERS=4
AI_FORGE_TIMEOUT=300

# Security
AI_FORGE_API_KEY=your-secret-key
AI_FORGE_ALLOWED_ORIGINS=https://yourdomain.com
```

### Reverse Proxy (Nginx)

```nginx
upstream aiforge {
    server 127.0.0.1:8000;
    keepalive 64;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://aiforge;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}
```

### SSL/TLS Setup

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d api.yourdomain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

---

## Backup & Recovery

### What to Backup

| Item | Location | Frequency |
|------|----------|-----------|
| Trained models | `./output/` | After each training |
| Configuration | `.env`, `config/` | On change |
| Training data | `./data/` | Daily |
| Ollama models | `~/.ollama/` | Weekly |

### Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/ai-forge/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup models
cp -r ./output "$BACKUP_DIR/"

# Backup config
cp -r ./config "$BACKUP_DIR/"
cp .env "$BACKUP_DIR/"

# Backup Ollama models
cp -r ~/.ollama/models "$BACKUP_DIR/ollama-models"

# Compress
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup complete: $BACKUP_DIR.tar.gz"
```

### Restore Script

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: ./restore.sh /path/to/backup.tar.gz"
    exit 1
fi

# Extract
tar -xzf "$BACKUP_FILE" -C /tmp/

# Restore
BACKUP_DIR="/tmp/$(basename "$BACKUP_FILE" .tar.gz)"
cp -r "$BACKUP_DIR/output"/* ./output/
cp -r "$BACKUP_DIR/config"/* ./config/
cp "$BACKUP_DIR/.env" ./
cp -r "$BACKUP_DIR/ollama-models"/* ~/.ollama/models/

echo "Restore complete"
```

### Disaster Recovery

1. **Model Corruption:**
   ```bash
   # Remove corrupted model
   ollama rm corrupted-model
   
   # Restore from backup
   cp backup/model.gguf ~/.ollama/models/
   ollama create model-name -f Modelfile
   ```

2. **Service Failure:**
   ```bash
   # Restart services
   docker-compose down
   docker-compose up -d
   ```

3. **Data Loss:**
   ```bash
   # Restore from latest backup
   ./restore.sh /backups/ai-forge/latest.tar.gz
   ```

---

## Health Checks

### Automated Health Check

```bash
#!/bin/bash
# healthcheck.sh

API_URL="http://localhost:8000"
OLLAMA_URL="http://localhost:11434"

# Check API
if curl -sf "$API_URL/health" > /dev/null; then
    echo "✅ API: healthy"
else
    echo "❌ API: unhealthy"
    exit 1
fi

# Check Ollama
if curl -sf "$OLLAMA_URL" > /dev/null; then
    echo "✅ Ollama: healthy"
else
    echo "❌ Ollama: unhealthy"
    exit 1
fi

echo "All systems operational"
```

---

## Next Steps

- [Production Checklist](production_checklist.md) - Pre-deployment checklist
- [Monitoring](monitoring.md) - Observability setup
- [Troubleshooting](troubleshooting.md) - Common issues
