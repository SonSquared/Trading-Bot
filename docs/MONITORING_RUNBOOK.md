# Monitoring Runbook

How to read every Telegram message the bot sends, what is normal, what to
ignore, and when to intervene. Written for day-to-day operation, not
development.

---

## 1. The deployment, in one paragraph

The bot runs **only on GitHub Actions** (`.github/workflows/bot.yml`).
Each run: restore state from the `bot-state` git branch → schedule gate →
fetch closed candles → run strategies → open/close paper positions → save
state atomically → send the cycle message to Telegram → run the watchdog →
**persist all history files back to the `bot-state` branch** (runs even if
the watchdog fails). All state lives in `data/results/` on that branch:
`paper_state.json`, `paper_trades.jsonl`, `run_history.jsonl`,
`position_tracker.json`, `watchdog_last_ok.json` — one commit per trading
run, full history preserved. Fly/Railway/Docker deployments are disabled
(`deploy/disabled/`) so only one filesystem ever trades.

### Schedule: two firings per even hour, gated

GitHub cron on the free tier **drops slots wholesale** — verified
2026-09-07/08: a single 2-hourly cron produced 5–7h gaps (the 02/04/08/10
UTC slots never fired). The workflow now fires twice per even hour (`:23`
and `:47`) and `scripts/schedule_gate.py` skips any firing within 100
minutes of the last successful run. A dropped slot now costs ≤24 minutes
of delay instead of half a trading day. Redundant firings cost ~15s each
(they exit before pip install). 4h candles close at 00/04/08/12/16/20 UTC,
so the even-hour schedule never misses a trading boundary. The accounting
fix (entry fees + funding booked into P&L the moment they hit cash) landed
2026-09-06 — everything before that date predates the honest cost model.

## 1b. Day zero: the 90-day forward proof (2026-09-08)

The paper record restarts from zero on this date — deliberately.

- **Start:** 2026-09-08, capital **$97.00**, no open positions, empty
  trade log. The first Actions run on/after this date is day zero.
- **Why reset:** the old ledger carried pre-fix accounting gaps, and the
  Actions state cache had a bug that silently discarded every run's
  trades (immutable cache entries + a stable key — see the comment in
  `bot.yml`). A clean start means every number in the 90-day window
  comes from the complete, repaired ledger, and
  `python scripts/audit_equity.py --strict` can verify the whole window
  exactly.
- **Archived records** (local, gitignored):
  `paper_trades_archived_2026-09-08.jsonl`,
  `run_history_archived_2026-09-08.jsonl`,
  `paper_state_archived_2026-09-08.json`,
  `paper_summary_archived_2026-09-08.json` — plus the earlier archives
  `paper_trades_archived_2026-09-04.jsonl` and the
  `paper_trades_test_pollution_2026-09-05.jsonl` quarantine.
- **The production ledger now lives on the `bot-state` git branch**
  (seeded 2026-09-08 from the last cloud artifact, $63.03 cash + ETH
  SHORT; one commit per trading run, plus 90-day artifacts). Local copies
  are archives only — do not run the trader from this machine (one
  filesystem rule). If a run ever reports `FRESH-START REFUSED`, the state
  files were lost — restore them from the latest `bot-state` commit; never
  let the bot silently re-fabricate a day zero.
- **Success criteria (window ends ~2026-12-07):** equity net of all
  costs (fees + slippage + funding) above $97, drawdown within the risk
  manager's limits, and consistency across both halves of the window.
  A negative or rule-breaking window means the portfolio does not
  graduate to real capital.

## 2. Message catalog

### Cycle messages (every ~2 hours)

| Message | Meaning | Normal? |
|---|---|---|
| `PORTFOLIO ... Equity / P&L / positions` | End-of-cycle report. **Equity − start = realized + unrealized exactly** (the additive identity; a `(Reconciles: +/- 0.00)` line appears when positions are open and it doesn't hold — that would itself be an anomaly). | Yes |
| `+ OPENED LONG [PAPER] <pair> @ $...` | A position opened at the close of the triggering candle. | Yes, at most a few per day |
| `- CLOSED [PAPER] <pair> $entry -> $exit / P&L / Reason` | Position closed. Reasons you'll see: `signal`, `stop_loss`, `trailing_stop`, `disaster_stop`, `drawdown_close_all`. | Yes |
| `🛑 STOP LOSS` / `🎯 TAKE PROFIT` | SL/TP manager triggered a protective close. | Yes |
| `EMERGENCY STOP: ...` | Disaster circuit breaker fired (e.g. cash/heat invariant violated). Bot refuses to trade until the cause is fixed. | **No — act** |

### Commands (reply within ~15s, only checked while a run is active)

`/status`, `/equity`, `/balance`, `/positions`, `/pnl`, `/trades`,
`/signals`, `/restart`, `/dashboard`, `/help`.

Key reading rules:

- **Equity vs P&L.** `Equity` is marked-to-market (cash + open-position marks).
  `P&L` in `/status` is realized (net of fees/funding) + gross unrealized —
  the two reconcile exactly. `/pnl` per-position lines are **net of funding
  charged for that position**.
- **`/trades`** shows the last 5 events from the trade log, which is the
  ground truth; it is reconciled against `paper_state.json` by the audit tool.
- **`/dashboard`** shows run health from `run_history.jsonl`: status of the
  last run, success rate, consecutive failures.

### Alert messages (should be rare)

| Message | Meaning | Action |
|---|---|---|
| `HEALTH ALERT: No bot runs for 8+ hours` (from the health-check workflow) | The scheduler hasn't produced a run. | See §4.1 |
| Watchdog anomaly report | One of: missed scheduled run, stale state file, ledger desync, desync storm, duplicate-open signature, equity/P&L inconsistency. | See §4.2 |

## 3. What is normal

- 0–3 cycles with no trades (strategies idle). Silence is fine.
- Equity wobble of ±1–2% intraday — 3× leveraged crypto with funding costs.
- Occasional `partial` run status (a fetch retry succeeded on the second
  attempt).
- Funding charges every 8h slightly reducing cash while positions are open
  (now visible in P&L, not hidden).
- A `(Reconciles: +0.00)` line in cycle messages with open positions.

## 4. When to intervene

### 4.1 No messages for >8 hours

The bot can't phone home if the scheduler is dead. Diagnosis order:

1. **GitHub Actions minutes exhausted** (private repo, 2,000 min/month, reset
   on the 1st). This killed the bot Sep 1–5, 2026. Check
   *Settings → Billing → Actions*. If exhausted, wait for reset or reduce
   cadence further. The 2-hour cadence should fit the budget with margin;
   verify nothing re-added high-frequency schedules.
2. **Workflow disabled**: *Actions tab → bot.yml → enable*. (Forked/inactive
   repos get disabled automatically after 60 days.)
3. **Local ahead of origin**: pushes may not have happened. `git status -sb`
   should show `...origin/main` in sync.

### 4.2 Watchdog anomaly in Telegram

| Finding | Cause | Action |
|---|---|---|
| `missed run` | Scheduler gap | §4.1 |
| `stale state` | No state write for 24h+ | §4.1, or the bot crashed mid-run — check the Actions log |
| `ledger desync` / `desync storm` | Trade log and state file disagree (crash between write and save, or two writers) | Do NOT trade live. Inspect `data/results/paper_trades.jsonl` vs `paper_state.json`; run `python scripts/audit_equity.py` |
| `duplicate opens` | Two instances opened the same pair | Check that only GitHub Actions is enabled (`deploy/disabled/` untouched, no local scheduler running) |
| `equity/P&L inconsistency` | Accounting drift between messages and ledger | Run `python scripts/audit_equity.py --strict`; if it fails, halt and reconcile the ledger before any live consideration |

### 4.3 Hard rules

- If the trade log shows opens you didn't expect (burst of same-second
  entries, unknown cash values), **quarantine first, investigate second**:
  archive the suspicious tail to
  `paper_trades_<reason>_<date>.jsonl` (atomic rename, keep the ledger
  chain intact up to the last trustworthy entry), then rerun
  `python scripts/audit_equity.py`.
- Never reconcile by hand-editing `paper_state.json`. Rebuild from the log,
  or reset the state file and let the bot start from capital again — and say
  so in the log archive name.
- The watchdog and health checks are your tripwires; if you silence them,
  you own the monitoring they did.

## 5. Tools

| Command | Purpose |
|---|---|
| `python scripts/audit_equity.py` | Full ledger reconciliation report (trade log vs state vs run history). `--strict` exits 1 on hard failures; `--log <archive>` audits a historical window. |
| `python scripts/watchdog.py` | Run the production watchdog checks locally. |
| `python -m pytest tests/ -q` | Full test suite. The conftest sandboxes all data paths, so running tests never touches production data (a session tripwire fails loudly if it ever does). |

## 6. Current status (2026-09-08)

- Live cloud state: cash **$63.03**, one ETH SHORT (Donchian_Breakout,
  $33.95 notional), equity ~$97.14. Ledger reconciles exactly
  (`python scripts/audit_equity.py --strict`).
- **State carrier migrated cache → `bot-state` git branch** after the
  Sep 8 incident chain: run #114's watchdog failure discarded its state
  save, and run #116's cache miss silently restarted the bot from $97
  while trading was under way. The branch is seeded from run #117's
  artifact; one commit per trading run.
- Scheduler: two gated firings per even hour (`:23`/`:47`) after cron was
  proven to drop whole slots.
- Pre-fix history archived: `paper_trades_archived_2026-09-04.jsonl` and
  `paper_trades_test_pollution_2026-09-05.jsonl` (test-suite pollution —
  since prevented by the conftest sandbox).
- NOTE: run #116's fresh-$97 restart means the cloud ledger (cash $63.03
  + a fresh ETH SHORT) carries one stranded position from the lost #114
  era (the original ETH LONG opened by the rejected Bollinger config).
  Its cost basis is preserved in the archived local files and the #112
  artifacts; the promoted portfolio now controls all trading.
