# Rotating the leaked Telegram bot token

**Why this is urgent:** the token `8783971913:AAH1ZdvtKvHjgVuC2c9LLebYnM-o8gBMQaY`
was committed to this repository and is present in git *history*. Even though
it has now been removed from every source file, anyone with repo access (or a
leaked clone) still has a working token until it is revoked on Telegram's
side. Revoking makes the old token permanently dead regardless of history.

Do these steps in order. Steps 1–2 take ~2 minutes; the rest is copy-paste.

## 1. Revoke the old token (do this FIRST)

1. Open Telegram and message **@BotFather**.
2. Send `/mybots` and pick your bot (e.g. "trading-bot-sonsquared").
3. Tap **API Token**.
4. Tap **Revoke current token** and confirm.
   - This instantly kills `8783971913:AAH1…` — the leaked token is now
     worthless. Any bot that still uses it will get `401 Unauthorized`.
5. Tap **Generate new token** (or `/token`) and copy the new token.
   - It looks like `123456789:AAH...` — treat it like a password from here on.

## 2. Confirm your chat ID

1. Message your bot (the one you just made) and send `/start`.
2. Get your chat id via `https://api.telegram.org/bot<NEW_TOKEN>/getUpdates`
   (open in a browser) → the `"chat":{"id":...}` field. The current config
   uses `5421461006`; verify it still matches, or update it.

## 3. Update GitHub Actions secrets

1. Open the repo on GitHub → **Settings → Secrets and variables → Actions**.
2. Edit **`TELEGRAM_BOT_TOKEN`** → paste the NEW token.
3. Edit **`TELEGRAM_CHAT_ID`** → your chat id.
4. (If they don't exist, create them.)

These secrets feed `bot.yml`, `health_check.yml`, `monthly_optimize.yml`,
`reoptimize.yml`, and `weekly_chart.yml`. Nothing else needs changing in the
workflows — they already read the token from secrets only.

## 4. Update Railway (if the app still runs)

1. Open the Railway project → **Variables**.
2. Update `TELEGRAM_BOT_TOKEN` to the new value, then redeploy.

> Note: Railway is archived in `deploy/disabled/railway/` — the repo no
> longer deploys there. Update the variable only if the old app is still
> running, and consider deleting the project (see
> `deploy/disabled/README.md`).

## 5. Update Fly.io (if the app still runs)

```bash
fly secrets set TELEGRAM_BOT_TOKEN="<NEW_TOKEN>" TELEGRAM_CHAT_ID="5421461006" -a trading-bot-sonsquared
fly deploy -a trading-bot-sonsquared
```

> Note: Fly is archived in `deploy/disabled/fly/` — update the secret only
> if the old app is still running, then destroy it
> (`fly apps destroy trading-bot-sonsquared`) per the consolidation docs.

## 6. No hardcoded token remains

- `configs/bot_live.yaml` now has `bot_token: ""` — the bot reads
  `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` from the environment only
  (`scripts/paper_trader.py::load_telegram_config`, `scripts/telegram_bot.py`,
  `scripts/position_tracker.py`).
- If a token is missing, the scripts print a loud warning and skip Telegram —
  they never fall back to a hardcoded default.

## 7. Verify

1. Trigger the bot manually: GitHub → **Actions → Trading Bot Runner →
   Run workflow**.
2. In Telegram, send `/status` to the bot. You should get a portfolio reply
   within a few minutes (the next scheduled run answers commands).
3. Confirm the OLD token is dead:
   ```
   curl https://api.telegram.org/bot8783971913:AAH1ZdvtKvHjgVuC2c9LLebYnM-o8gBMQaY/getMe
   ```
   Expect `{"ok":false,...}` (401). The NEW token should return
   `{"ok":true,...}`.

## Keeping it safe going forward

- Never put a token in a YAML/JSON/py file that gets committed. Env vars or
  platform secrets only.
- The bot's own config loader is env-first by design; keep it that way.
- If you ever see a token in a diff again, revoke it immediately — history
  never forgets.