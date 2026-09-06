#!/bin/bash
# ============================================================
# Trading Bot VPS Setup Script
# Run this ONCE on your cloud VPS to deploy the bot
# ============================================================
set -e

echo "=========================================="
echo "  Trading Bot Deployment"
echo "=========================================="

# 1. Install Docker
echo "[1/5] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "Docker installed. You may need to log out and back in."
else
    echo "Docker already installed."
fi

# 2. Install Docker Compose
echo "[2/5] Installing Docker Compose..."
if ! command -v docker compose &> /dev/null; then
    sudo apt-get install -y docker-compose-plugin
else
    echo "Docker Compose already installed."
fi

# 3. Create project directory
echo "[3/5] Setting up project directory..."
DEPLOY_DIR="$HOME/trading-bot"
mkdir -p "$DEPLOY_DIR"
cd "$DEPLOY_DIR"

# 4. Copy files (assuming you've uploaded the project)
echo "[4/5] Preparing files..."
if [ ! -f "deploy/Dockerfile" ]; then
    echo "ERROR: Please upload the trading bot project to $DEPLOY_DIR first."
    echo "  You can use: scp -r /path/to/project/* user@vps:$DEPLOY_DIR/"
    exit 1
fi

# Set bot mode to paper (safety first!)
sed -i 's/mode: paper/mode: paper/' configs/bot_live.yaml 2>/dev/null || true

# 5. Build and start
echo "[5/5] Building and starting bot..."
cd "$DEPLOY_DIR"
docker compose -f deploy/docker-compose.yml up -d --build

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "  Bot is running in PAPER mode."
echo ""
echo "  Commands:"
echo "    View logs:    docker compose -f deploy/docker-compose.yml logs -f"
echo "    Stop bot:     docker compose -f deploy/docker-compose.yml down"
echo "    Restart bot:  docker compose -f deploy/docker-compose.yml restart"
echo "    Check status: docker compose -f deploy/docker-compose.yml ps"
echo ""
echo "  To go live:"
echo "    1. Edit configs/bot_live.yaml on the VPS"
echo "    2. Set bot.mode: live"
echo "    3. Set exchange.api_key and exchange.api_secret"
echo "    4. Set exchange.sandbox: false"
echo "    5. Restart: docker compose -f deploy/docker-compose.yml restart"
echo ""
