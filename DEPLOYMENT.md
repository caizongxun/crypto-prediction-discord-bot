# 🚀 Deployment Guide

## Local Development

### Requirements
- Python 3.8+
- 2GB RAM (minimum)
- Stable internet connection

### Setup

```bash
# Clone
git clone https://github.com/caizongxun/crypto-prediction-discord-bot.git
cd crypto-prediction-discord-bot

# Virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your tokens
```

### Run

```bash
# Terminal 1: Discord Bot
python discord_bot.py

# Terminal 2: Web Dashboard
python web_dashboard.py
```

---

## Linux Server Deployment

### 1. Install Dependencies

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

### 2. Setup Project

```bash
cd /opt
sudo git clone https://github.com/caizongxun/crypto-prediction-discord-bot.git
cd crypto-prediction-discord-bot

sudo python3 -m venv venv
sudo source venv/bin/activate
sudo pip install -r requirements.txt

# Configure
sudo cp .env.example .env
sudo nano .env  # Edit with your tokens
```

### 3. Run with Screen (Simple)

```bash
# Terminal 1: Discord Bot
screen -S crypto-bot
source venv/bin/activate
python discord_bot.py
# Press Ctrl+A+D to detach

# Terminal 2: Dashboard
screen -S crypto-dash
source venv/bin/activate
python web_dashboard.py
# Press Ctrl+A+D to detach

# Reattach
screen -r crypto-bot
screen -r crypto-dash
```

### 4. Run with Systemd (Recommended)

#### Create Bot Service

```bash
sudo tee /etc/systemd/system/crypto-bot.service > /dev/null <<EOF
[Unit]
Description=Crypto Prediction Discord Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/crypto-prediction-discord-bot
Environment="PATH=/opt/crypto-prediction-discord-bot/venv/bin"
ExecStart=/opt/crypto-prediction-discord-bot/venv/bin/python discord_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

#### Create Dashboard Service

```bash
sudo tee /etc/systemd/system/crypto-dashboard.service > /dev/null <<EOF
[Unit]
Description=Crypto Prediction Web Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/crypto-prediction-discord-bot
Environment="PATH=/opt/crypto-prediction-discord-bot/venv/bin"
ExecStart=/opt/crypto-prediction-discord-bot/venv/bin/python web_dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

#### Enable and Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable crypto-bot crypto-dashboard
sudo systemctl start crypto-bot crypto-dashboard

# Check status
sudo systemctl status crypto-bot crypto-dashboard

# View logs
sudo journalctl -u crypto-bot -f
sudo journalctl -u crypto-dashboard -f
```

### 5. Setup Nginx Reverse Proxy

```bash
sudo apt install -y nginx
sudo tee /etc/nginx/sites-available/crypto-dashboard > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/crypto-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. SSL Certificate (Let's Encrypt)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## Docker Deployment

### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for web dashboard
EXPOSE 5000

# Run both bot and dashboard
CMD python discord_bot.py & python web_dashboard.py
```

### 2. Create .dockerignore

```
venv
.env
.git
__pycache__
*.pyc
.DS_Store
models/
```

### 3. Build and Run

```bash
# Build image
docker build -t crypto-bot .

# Run container
docker run -d \\
  --name crypto-bot \\
  --env-file .env \\
  -p 5000:5000 \\
  -v $(pwd)/models:/app/models \\
  crypto-bot

# Check logs
docker logs -f crypto-bot
```

### 4. Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  crypto-bot:
    build: .
    container_name: crypto-bot
    env_file: .env
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
    restart: always
    networks:
      - crypto-network

  nginx:
    image: nginx:latest
    container_name: crypto-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - crypto-bot
    networks:
      - crypto-network

networks:
  crypto-network:
    driver: bridge
```

Run:
```bash
docker-compose up -d
```

---

## Cloud Deployment

### AWS EC2

1. Launch Ubuntu 20.04 instance (t2.medium or larger)
2. Security group: Allow ports 22, 80, 443, 5000
3. SSH and follow Linux deployment above

### Google Cloud

```bash
# Create VM
gcloud compute instances create crypto-bot \\
  --image-family=ubuntu-2004-lts \\
  --image-project=ubuntu-os-cloud \\
  --machine-type=e2-medium

# SSH
gcloud compute ssh crypto-bot

# Follow Linux deployment
```

### DigitalOcean

1. Create Droplet (Ubuntu 20.04, 2GB RAM)
2. SSH and follow Linux deployment
3. Use ufw for firewall:

```bash
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 5000
sudo ufw enable
```

### Heroku

Create `Procfile`:

```
worker: python discord_bot.py
web: python web_dashboard.py
```

Deploy:

```bash
heroku login
heroku create crypto-bot
git push heroku main
heroku config:set DISCORD_BOT_TOKEN=your_token
```

---

## Monitoring

### Check Bot Status

```bash
# Systemd
sudo systemctl status crypto-bot

# Docker
docker ps -a
docker logs crypto-bot

# Process
ps aux | grep python
```

### View Logs

```bash
# Systemd
sudo journalctl -u crypto-bot -n 100 -f

# Docker
docker logs -f crypto-bot

# File
tail -f /var/log/crypto-bot.log
```

### Health Check

```bash
# Test Discord bot
curl http://localhost:5000/api/status

# Test web dashboard
curl http://localhost:5000/
```

---

## Performance Tuning

### Increase Model Loading Timeout

```python
# In model_manager.py
huggingface_hub.hf_hub_download(..., timeout=60)
```

### Cache Optimization

```python
# In data_fetcher.py
self.cache_duration = 120  # Increase to 2 minutes
```

### Auto-Restart on Crash

**Systemd** (already configured)

**Screen**:
```bash
while true; do
  python discord_bot.py
  sleep 10
done
```

---

## Troubleshooting

### Bot crashes on startup

```bash
# Check logs
sudo journalctl -u crypto-bot -n 50

# Verify .env
cat .env

# Test imports
python -c "import torch; import discord; print('OK')"
```

### High memory usage

```bash
# Monitor
watch -n 1 'free -h && ps aux | grep python'

# Reduce cache duration
# Disable model caching (re-download each time)
```

### Slow predictions

```bash
# Check system resources
htop

# Optimize features
# Reduce from 100 to 50 candles in data_fetcher.py
```

---

## Backup & Recovery

### Backup Configuration

```bash
cp .env .env.backup
cp -r models models.backup
```

### Restore

```bash
cp .env.backup .env
cp -r models.backup models
sudo systemctl restart crypto-bot
```

---

## Security Best Practices

1. ✅ Use environment variables for secrets
2. ✅ Rotate Discord token regularly
3. ✅ Use HTTPS with SSL certificates
4. ✅ Restrict firewall to necessary ports
5. ✅ Keep dependencies updated:
   ```bash
   pip list --outdated
   pip install --upgrade -r requirements.txt
   ```
6. ✅ Monitor error logs regularly
7. ✅ Use strong server passwords

---

**Questions?** Open an issue on GitHub!
