# Disabled bot runners (archived)

**GitHub Actions (`.github/workflows/bot.yml`) is the single, canonical runner
for the trading bot.** Everything in this directory is archived because it
either ran the bot from a second filesystem (causing duplicate-position
races) or was a duplicate of an existing workflow.

The observed bug this prevents: `paper_trader.py` was executed from GitHub
Actions (`bot.yml`), GitHub Actions again (`position_tracker.yml`),
Railway, Fly.io, a VPS docker-compose, and Telegram `/restart` — up to six
filesystems mutating `data/results/paper_state.json` with no coordination.
The run lock (`scripts/paper_trader.py` → `acquire_lock`) is a safety net,
but a single runner is the real fix.

## What was archived

| Archived here | Was | What it did |
|---|---|---|
| `workflows/position_tracker.yml` | `.github/workflows/position_tracker.yml` | Re-ran `paper_trader.py` every 15 min → duplicate trader |
| `github_actions/` | `deploy/github_actions.yml`, `deploy/health_check.yml` | Duplicate GH Actions runner + health check from the `deploy/` dir |
| `railway/` | `railway.toml`, `railway.json`, root `Dockerfile` | Long-running `bot_scheduler.py` on Railway |
| `fly/` | `fly.toml` | Long-running `bot_scheduler.py` on Fly.io (also used the root `Dockerfile`) |
| `vps/` | `deploy/{docker-compose.yml,Dockerfile,setup_vps.sh,auto_start.bat}` | Self-hosted VPS deployment |

`scripts/bot_scheduler.py` and the Dockerfile(s) still exist in the repo but
are dormant — nothing references them anymore.

## Platform-side teardown (still running apps)

Moving these files prevents *re*-deploys; it does **not** stop apps that are
already running on the platforms. To fully stop the other runners:

- **Railway** — delete the project (or `railway down`/pause the service).
- **Fly.io** — `fly apps destroy trading-bot-sonsquared` (after confirming
  you no longer want it) or `fly scale count 0`.
- **VPS** — `docker compose -f deploy/disabled/vps/docker-compose.yml down`
  and remove the cron/systemd entry that starts `bot_scheduler.py`.

## Re-enabling a platform

Move its files back to the original locations (the file contents reference
the original layout):

- Railway: `railway.toml`, `railway.json`, root `Dockerfile`
- Fly: `fly.toml`, root `Dockerfile`
- VPS: `deploy/docker-compose.yml`, `deploy/Dockerfile`, `deploy/setup_vps.sh`
- GH Actions duplicate: `.github/workflows/position_tracker.yml`

Then remove the `run-bot` job from `.github/workflows/bot.yml` so the two
runners can't overlap. If you switch platforms, update the docs in
`README.md` (see "Deployment" section).