# GitHub Actions Deployment Guide

## Why GitHub Actions?

- **100% free** — no credit card, no hidden fees
- **Runs every 4 hours** via cron (matches your 4h candle strategy)
- **No server to maintain** — GitHub handles everything
- **Secure** — API keys stored as encrypted secrets
- **Logs everything** — trade history saved as artifacts

## Setup Steps (10 minutes)

### Step 1: Create a GitHub Repository

1. Go to https://github.com/new
2. Name it something like `trading-bot` (can be private)
3. Upload your project:

```bash
# From your project directory
git init
git add .
git commit -m "Initial trading bot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/trading-bot.git
git push -u origin main
```

### Step 2: Add the GitHub Actions Workflow

1. In your repo, create folder `.github/workflows/`
2. Copy `deploy/github_actions.yml` to `.github/workflows/bot.yml`

```bash
mkdir -p .github/workflows
cp deploy/github_actions.yml .github/workflows/bot.yml
git add .github/workflows/bot.yml
git commit -m "Add GitHub Actions workflow"
git push
```

### Step 3: Add API Keys as Secrets

1. Go to your repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add:
   - Name: `EXCHANGE_API_KEY`, Value: your Binance API key
   - Name: `EXCHANGE_API_SECRET`, Value: your Binance API secret

**Important:** Never commit API keys to the repo!

### Step 4: Enable the Workflow

1. Go to Actions tab in your repo
2. Click "I understand my workflows, go ahead and enable them"
3. The bot will now run automatically every 4 hours

### Step 5: Monitor

**Check if it's running:**
- Go to Actions tab → click on any workflow run
- See the output in real-time

**Check trade logs:**
- Go to Actions → click a run → scroll to "Artifacts"
- Download `trade-log-*` to see all trades

## Manual Trigger

You can also run the bot manually anytime:
1. Go to Actions → "Trading Bot Runner"
2. Click "Run workflow"
3. Click the green "Run workflow" button

## Going Live

When ready to trade with real money:

1. Edit `.github/workflows/bot.yml`:
   ```yaml
   env:
     BOT_MODE: live  # Change from 'paper' to 'live'
   ```

2. Commit and push:
   ```bash
   git add .github/workflows/bot.yml
   git commit -m "Switch to live trading"
   git push
   ```

## Cost

**$0 forever.** GitHub Actions gives:
- **2,000 minutes/month** on private repos (enough for 4h runs)
- **Unlimited minutes** on public repos

Each run takes ~30 seconds, so 6 runs/day × 30 days = 180 minutes/month. You have plenty of headroom.

## Troubleshooting

**Bot not running?**
- Check Actions tab for failed runs
- Ensure workflow file is at `.github/workflows/bot.yml`

**API errors?**
- Verify secrets are set correctly (Settings → Secrets)
- Check API key has futures trading permission

**Need more frequent runs?**
- Edit the cron in `.github/workflows/bot.yml`:
  ```yaml
  - cron: '0 */2 * * *'  # Every 2 hours
  ```
