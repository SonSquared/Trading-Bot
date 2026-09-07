# Monitoring Runbook

How to read every Telegram message the bot sends, what is normal, what to
ignore, and when to intervene. Written for day-to-day operation, not
development.

---

## 1. The deployment, in one paragraph

The bot runs **only on GitHub Actions** (`.github/workflows/bot.yml`, every
2 hours). Each run: fetch closed candles → run strategies → open/close paper
positions → save state atomically → send the cycle message to Telegram → run
the watchdog. All state lives in `data/results/paper_state.json` (persisted
across runs via the workflow cache). Fly/Railway/Docker deployments are
disabled (`deploy/disabled/`) so only one filesystem ever trades. The
accounting fix (entry fees + funding booked into P&L the moment they hit
cash) landed 2026-09-06 — everything before that date predates the honest
cost model.

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
- **The production ledger now lives on GitHub Actions** (cache entry
  `paper-state-v4-<run_number>` + 90-day artifacts). Local copies are
  archives only — do not run the trader from this machine (one
  filesystem rule).
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

## 6. Current status (2026-09-07)

- Real trade log: 2 opens (Sep 3), both still open, cash $29.06, state
  reconciled to the cent.
- Pre-fix history archived: `paper_trades_archived_2026-09-04.jsonl` and
  `paper_trades_test_pollution_2026-09-05.jsonl` (test-suite pollution —
  since prevented by the conftest sandbox).
- Scheduler: repaired to 2-hour cadence after the September Actions-minutes
  exhaustion; needs one push to re-arm.
