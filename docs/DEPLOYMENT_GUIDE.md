# Deployment Guide: Running the Bot 24/7 Without touching Your Computer

> **⚠️ ARCHIVED (2026-09):** The bot now runs on **GitHub Actions only** (`.github/workflows/bot.yml`, every 15 min) — see README "Deployment". This VPS guide and `deploy/` were moved to `deploy/disabled/`; the paths below work only if you restore the original layout.


## Overview

The bot needs to run on a **cloud server (VPS)** that stays online 24/7.
Your personal computer is not suitable because it sleeps, restarts, and loses internet.

**Recommended setup:** A $5-10/month cloud VPS running Docker.

---

## Option 1: Cloud VPS + Docker (Recommended)

### Step 1: Get a Cloud VPS

| Provider | Cheapest Plan | Monthly Cost | Best For |
|----------|--------------|-------------|----------|
| **DigitalOcean** | Basic $6/mo | $6 | Simple, reliable |
| **Hetzner** | CX22 | $5 | Best value (EU) |
| **Vultr** | $6/mo | $6 | Good global coverage |
| **AWS Lightsail** | $5/mo | $5 | AWS ecosystem |
| **Linode** | Nanode $5/mo | $5 | Good support |

**Recommended:** DigitalOcean or Hetzner, 2 vCPU, 2GB RAM, Ubuntu 22.04.

### Step 2: Upload the Project

From your computer (Git Bash / terminal):

```bash
# Replace with your VPS IP and username
VPS_IP="your.vps.ip.address"
VPS_USER="root"

# Upload the project
scp -r ./* ${VPS_USER}@${VPS_IP}:~/trading-bot/
```

Or use git if your project is in a repository:

```bash
ssh ${VPS_USER}@${VPS_IP}
cd ~/trading-bot
git clone https://your-repo.git .
```

### Step 3: Deploy

```bash
ssh ${VPS_USER}@${VPS_IP}
cd ~/trading-bot
chmod +x deploy/disabled/vps/setup_vps.sh
bash deploy/disabled/vps/setup_vps.sh
```

That's it. The bot starts automatically in **paper mode**.

### Step 4: Monitor

```bash
# Watch live logs
docker compose -f deploy/disabled/vps/docker-compose.yml logs -f

# Check if bot is running
docker compose -f deploy/disabled/vps/docker-compose.yml ps

# View last 50 log lines
docker compose -f deploy/disabled/vps/docker-compose.yml logs --tail 50
```

### Step 5: Go Live (When Ready)

```bash
# 1. Edit the config
nano ~/trading-bot/configs/bot_live.yaml

# Change these lines:
#   bot.mode: live
#   exchange.sandbox: false
#   exchange.api_key: "YOUR_BINANCE_API_KEY"
#   exchange.api_secret: "YOUR_BINANCE_API_SECRET"

# 2. Restart the bot
docker compose -f deploy/disabled/vps/docker-compose.yml restart

# 3. Verify it restarted correctly
docker compose -f deploy/disabled/vps/docker-compose.yml logs --tail 20
```

---

## Option 2: Windows Task Scheduler (If You Must Use Your PC)

If you don't want a VPS, you can run the bot on your Windows PC using Task Scheduler.
**Warning:** The bot stops when your PC sleeps, restarts, or loses internet.

### Setup

1. Create a batch file `C:\TradingBot\start_bot.bat`:
```batch
@echo off
cd /d C:\TradingBot
"D:\Program Files\Python\python.exe" -u scripts/run_paper_trading.py --continuous >> data\bot.log 2>&1
```

2. Open Task Scheduler:
   - Create Basic Task
   - Name: "Trading Bot"
   - Trigger: "When the computer starts"
   - Action: "Start a program" → `C:\TradingBot\start_bot.bat`
   - Check: "Run whether user is logged on or not"
   - Check: "Run with highest privileges"

3. Configure reliability:
   - Right-click task → Properties
   - Settings: Uncheck "Stop task if it runs longer than"
   - Settings: Check "If the running task ends, restart every: 1 minute"
   - Settings: Check "Stop the task if it runs longer than: 72 hours" → change to "Do not stop"

**Limitations:**
- Bot stops during Windows updates / restarts
- Bot stops during sleep mode
- No automatic recovery from crashes
- Internet disconnection = missed signals

---

## Option 3: Cloud Functions (Advanced)

For maximum reliability, you could use AWS Lambda or Google Cloud Functions to check signals every 4 hours. However, this requires significant refactoring of the bot code and is more complex to maintain.

**Not recommended** for the current codebase.

---

## Monitoring Setup

### Telegram Notifications (Recommended)

Add to `configs/bot_live.yaml`:
```yaml
bot:
  notification_enabled: true
  notification_webhook: "https://api.telegram.org/botYOUR_TOKEN/sendMessage?chat_id=YOUR_CHAT_ID"
```

The bot sends alerts for:
- Trade executions
- Daily P&L summary
- Emergency stops
- Errors

### Simple Health Check Script

Create `deploy/disabled/vps/health_check.sh` on the VPS:
```bash
#!/bin/bash
# Run this via cron every hour: 0 * * * * /root/trading-bot/deploy/disabled/vps/health_check.sh

CONTAINER="trading-bot"
if ! docker ps | grep -q $CONTAINER; then
    echo "$(date): Bot is DOWN. Restarting..." >> /root/trading-bot/data/health.log
    cd /root/trading-bot
    docker compose -f deploy/disabled/vps/docker-compose.yml up -d
fi
```

Add to crontab:
```bash
crontab -e
# Add this line:
0 * * * * /root/trading-bot/deploy/disabled/vps/health_check.sh
```

---

## Security Checklist

Before going live with real money:

- [ ] **API Key permissions:** Create a dedicated API key with ONLY futures trading permission (no withdrawal)
- [ ] **IP whitelist:** Restrict API key to your VPS IP only
- [ ] **No withdrawal permission:** Never grant withdrawal access to the bot
- [ ] **Firewall:** Only open SSH (port 22) and nothing else
- [ ] **SSH keys:** Disable password authentication, use SSH keys only
- [ ] **Fail2ban:** Install to block brute-force SSH attempts
- [ ] **HTTPS:** If adding a web dashboard, use Let's Encrypt
- [ ] **Backup:** Regularly backup `data/bot_state.json` and `configs/`

```bash
# Quick security setup on VPS:
sudo apt install -y fail2ban
sudo systemctl enable fail2ban

# Firewall (UFW)
sudo ufw allow ssh
sudo ufw enable
```

---

## Cost Summary

| Item | Monthly Cost | Notes |
|------|-------------|-------|
| Cloud VPS | $5-6 | DigitalOcean / Hetzner |
| Domain (optional) | $1 | For monitoring dashboard |
| Telegram bot | Free | For notifications |
| **Total** | **$5-7/month** | |

---

## Troubleshooting

### Bot crashes on start
```bash
# Check logs
docker compose -f deploy/disabled/vps/docker-compose.yml logs

# Common fix: data directory permissions
sudo chown -R 1000:1000 ~/trading-bot/data/
```

### Bot stops trading
```bash
# Check if it's actually running
docker compose -f deploy/disabled/vps/docker-compose.yml ps

# Check for errors
docker compose -f deploy/disabled/vps/docker-compose.yml logs --tail 100

# Restart
docker compose -f deploy/disabled/vps/docker-compose.yml restart
```

### Exchange connection issues
```bash
# Test API connectivity from VPS
docker compose -f deploy/disabled/vps/docker-compose.yml exec trading-bot python -c "
import ccxt
exchange = ccxt.binance({'options': {'defaultType': 'future'}})
print(exchange.fetch_ticker('ETH/USDT:USDT'))
"
```

### Bot uses too much memory
```bash
# Check memory usage
docker stats trading-bot

# If > 500MB, restart periodically
# Add to crontab: 0 */6 * * * docker compose -f /root/trading-bot/deploy/disabled/vps/docker-compose.yml restart
```
