# AI Trading Bot — User Guide

An LLM-driven crypto trading bot (the Nate Herk "6 daily wakeups" plan, adapted to run on **Google Gemini's free tier** — `gemini-2.5-flash` via its OpenAI-compatible endpoint; no GPT-6 Astra, no API bill) for Binance USDⓈ-M perpetual futures, with hard risk rules enforced in code that the AI cannot override.

---

## How it works

```
   every 4 hours (6x daily)          each wakeup is STATELESS
  ┌────────────────────────┐        ┌──────────────────────────────────┐
  │ Scheduler              │───────▶│ 1. Read strategy.json (rules)    │
  │ 00:00  Asia open       │        │ 2. Read progress.json (handoff)  │
  │ 06:00  Asia/London     │        │ 3. Check paper SL/TP triggers    │
  │ 08:00  London open     │        │ 4. Fetch live Binance data       │
  │ 14:00  US open         │        │ 5. Compute indicators            │
  │ 20:00  US midday       │        │ 6. Ask the LLM for a decision    │
  │ 23:00  Daily close     │        │ 7. Validate vs hard risk rules   │
  └────────────────────────┘        │ 8. Execute (closes, then opens)  │
                                    │ 9. Write handoff + journal       │
                                    └──────────────────────────────────┘
```

The AI never has memory between wakeups. Continuity lives entirely in shared files in `data/ai_bot/` — so a crash, restart, or new machine changes nothing.

### Hard risk rules (code-enforced, AI-proof)

| Rule | Default | Behavior |
|---|---|---|
| Tradable pairs | BTC, ETH perps | Anything outside the list is **rejected**, even if the AI asks |
| Max open positions | 3 | 4th entry rejected |
| Max position size | 30% of equity — **derived** | Oversized entry rejected. The risk rules alone would allow `max_risk_per_trade_pct / stop_loss_max_pct * 100 = 40`, but a single position cannot exceed the total-exposure cap, so the smaller number is what ships and what binds |
| Venue minimum order | Binance `MIN_NOTIONAL` after lot-step rounding | Orders the exchange would reject are **refused and journaled**, in paper as well as live (see *Account scale and venue minimums*) |
| Stop-loss | ≤ 5% away, always | Entries without valid SL rejected |
| Risk/reward | ≥ 1.5 | 3% SL needs ≥ 4.5% TP |
| Min confidence | 60 | Low-confidence trades dropped |
| Max drawdown | 10% from peak | **Trading halts** — closes only |
| Portfolio heat | ≤ 30% of equity | Total exposure (notional) cap, **enforced on every entry** — re-checked per entry including positions already open and anything approved earlier in the same wakeup |
| Risk per trade | ≤ 2% of equity | Enforced **directly** as `size% × stop%`, not left implied by the size and stop caps agreeing |

---

## Quick start (5 minutes)

```bash
# 1. Install (already done if you followed the build):
pip install -r requirements.txt

# 2. Add a FREE Gemini key — copy .env.example to .env and set:
#    GEMINI_API_KEY=...      (from https://aistudio.google.com/apikey — no credit card)
# Paper mode needs NO Binance keys (public market data only).

# 3. Run one wakeup right now:
python scripts/start_ai_bot.py once

# 4. Watch it think:
python scripts/start_ai_bot.py journal
```

The first wakeup creates `data/ai_bot/strategy.json` (the AI's rulebook), starts a paper account at `bot.paper_starting_equity` (**$97** — the account size comes from that one config key and nowhere else), fetches real BTC/ETH data from Binance, and asks the model for a decision. Every action and rejection is journaled.

---

## Commands

```
python scripts/start_ai_bot.py once [NAME]   Run one wakeup now (NAME = schedule entry)
python scripts/start_ai_bot.py run           Continuous schedule (Ctrl+C to stop)
python scripts/start_ai_bot.py next          Wait for & run the next scheduled wakeup
python scripts/start_ai_bot.py status        Equity, positions, win rate, last handoff
python scripts/start_ai_bot.py journal [N]   Show the last N wakeup decisions
python scripts/ai_venue_check.py             What this account size can legally trade
```

Named wakeups: `once us_open`, `once daily_close`, etc. (see `configs/ai_bot.yaml`).

---

## Configuration (`configs/ai_bot.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `bot.mode` | `paper` | `paper` (simulated) or `live` (real money) |
| `bot.pairs` | BTC, ETH | The ONLY pairs the AI may trade |
| `bot.timeframe` | `1h` | Candle timeframe for analysis |
| `bot.data_dir` | `data/ai_bot` | Continuity files directory |
| `bot.paper_starting_equity` | `97` | Paper account start — the **only** owner of the account's scale (see below). Required: a missing key is an error, not a default |
| `bot.sandbox` | `false` | `true` = Binance **testnet** rehearsal (keys: `BINANCE_TESTNET_*`) |
| `bot.tz_offset_hours` | `0` | Local offset from UTC for schedule times |
| `ai.model` | `gemini-flash-latest` | FREE tier. The alias tracks the newest Gemini flash, so pinning never goes stale. Any Gemini or OpenAI model works |
| `risk.*` | see table above | Hard limits — edit freely, they're enforced either way |
| `telegram.enabled` | `true` | Report-only notifications (env: `AI_TELEGRAM_BOT_TOKEN`, `AI_TELEGRAM_CHAT_ID`; the unprefixed `TELEGRAM_*` names work too) — fails safe, sends nothing if secrets are absent |

---

## The shared files (`data/ai_bot/`)

| File | What it is |
|---|---|
| `strategy.json` | The AI's rulebook — strategy-level fields (pairs, preferences). Rules in the config's `risk:` section override the copy here |
| `progress.json` | Handoff notes the next wakeup reads first |
| `journal.jsonl` | **The audit trail** — one entry per wakeup, success or failure |
| `decisions.json` | The latest raw AI decision (what it wanted to do) |
| `trades.jsonl` | Every executed trade |
| `paper_ledger.json` | Paper account: cash, positions, closed-trade history |

Reading a journal entry:

```json
{
  "status": "success",
  "market_outlook": "bullish",
  "actions_requested": 2,
  "actions_approved": 1,
  "rejections": ["SOL/USDT:USDT long: pair not in tradable list ..."],
  "ai_reasoning": "BTC reclaiming EMA21 with rising volume...",
  "equity": 10012.40
}
```

`status: "error"` entries always name the exact problem. A wakeup is never silent.

---

## Account scale and venue minimums

The account starts at **$97** — the user's real capital. `bot.paper_starting_equity`
in `configs/ai_bot.yaml` is the **only** place that number lives: the agent reads it
directly and raises if it is missing, with no fallback. (There used to be one: the
agent looked for a `paper.starting_equity` key no config ever had and silently
defaulted to `10000`, so a $10,000 book ran for a $97 balance and nothing ever
failed. That is the failure mode this section exists to prevent.)

An existing ledger keeps the origin it was created with — **history is never
rewritten**. When the ledger's origin and the configured account disagree, the
journal, the daily digest and the weekly report label the history `LEGACY SCALE`
so old and current figures are never quietly mixed into one number.

**The exchange's minimum order size, not the strategy, decides what is tradeable at
the bottom of the account.** `python scripts/ai_venue_check.py` prints this from the
venue's own filters (falling back to a dated builtin table when the venue is
unreachable, always saying which it used):

| Pair | MIN_NOTIONAL | qty step | Tradeable at $97? |
|---|---|---|---|
| `BTC/USDT:USDT` | 50 USDT | 0.001 | **No** — a position may reach at most $29.10, and BTC needs ~$166.67 of equity |
| `ETH/USDT:USDT` | 20 USDT | 0.001 | **Yes** — legal orders run $20.00–$29.10 |

The size cap is **derived**, not picked by hand. Two rules bound a single position:

| Rule | Allows |
|---|---|
| Risk per trade | `max_risk_per_trade_pct / stop_loss_max_pct * 100` = 2/5 × 100 = **40%** |
| Total exposure | `max_portfolio_heat_pct` = **30%** |

so the smaller, **30%**, is what ships — quoting 40 would name a size no position can
hold. The old flat 10% was a third, arbitrary cap unrelated to either rule: at $97 it
permitted $9.70, under ETH's $20 minimum, so *no order the bot could size was legal at
all*.

The 2% risk cap is the outer bound on loss and is enforced **directly**
(`size% × stop%`), so no future edit to either adjacent number can raise it by
default. Be precise about what binds at this size, though: at the 30% cap and a 5%
stop a trade risks **1.5%** of equity ($1.46), because the exposure cap stops the
position before the risk cap does. Both are enforced; the exposure cap is the one
that bites here, and the per-trade loss stays under 2% either way.

Orders below the venue floor are refused in **paper mode too**: a simulated fill of
an order the exchange would reject is a lie, and it is exactly how the $9.70 problem
stayed invisible for weeks. The refusal is journaled with the arithmetic, and the
floor is stated to the AI in the risk context so it can size correctly:

```
BTC/USDT:USDT long: venue minimum $50.00 for BTC/USDT:USDT exceeds the $29.10 a
  position may reach at $97.00 equity (risk 2% of equity per trade, exposure cap
  30% of equity; needs ~$166.67)
```

**What a $97 month actually looks like.** Sizing every trade to the $29.10 cap, at
~20 trades/month, a 50% win rate and R:R 1.5, a trade risks **$1.46**, a win nets
**$2.15** and a loss costs **$1.48** (fees included) — expectancy about **+$0.33 per
trade → roughly $6.70/month (+6.9% of equity)**: single-digit dollars. Fees are
$0.0291 per round trip on a $29.10 order, which is 2% of the amount risked. A smaller
position scales every figure down proportionally. That is arithmetic on the rules,
**not a forecast** — the realized sample is a handful of trades, far too small to
estimate a win rate, and the same rules can give it back. Run
`scripts/ai_venue_check.py` for the current numbers at the current price.

---

## Paper vs live

**Paper mode** (default): simulated fills with 0.05% taker fees, SL/TP triggers checked against real prices on every wakeup, real equity tracking — and the same venue-minimum refusal as live, so a paper fill always represents an order the exchange would have accepted. The AI sees honest P&L.

**Live mode** requires three things, deliberately:
1. `bot.mode: live` in the config
2. `--live` flag on the CLI command
3. `AI_BOT_LIVE_CONFIRMED=1` in `.env`

...and real Binance keys with **trading permission only — never withdrawal**. Every live entry automatically gets a `STOP_MARKET` stop-loss and `TAKE_PROFIT_MARKET` order attached (reduce-only, mark-price protected). An AI position never sits naked.

**Recommended progression**: paper for at least 2–4 weeks → Binance testnet → live with money you can afford to lose. Testnet rehearsal is built in: set `bot.sandbox: true` in the config and put free testnet keys (`BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET`, from [testnet.binancefuture.com](https://testnet.binancefuture.com)) in `.env` — same order flow as live, fake money.

---

## Running 24/7 in the cloud (GitHub Actions, free)

The repo ships `.github/workflows/ai_bot.yml`, following the same battle-tested pattern as the main bot:

- **6 wakeup slots/day** (`00/06/08/14/20/23 UTC`), **plus a self-healing chain**: GitHub's cron is unreliable on this repo (2026-09-13: four consecutive wakeups never fired; the 23:50 digest ran ~2h late, three days running; only ~5 of 12 scheduled firings were delivered on 09-20/21), so other AI-suite workflows' completions kick the next wakeup check via `workflow_run` — the Telegram poller's generation is the heartbeat, and since 2026-09-22 that heartbeat **no longer depends on cron at all**: each poller generation relaunches the next one with `workflow_dispatch` (the one self-trigger GitHub honours from `GITHUB_TOKEN`; a workflow listing *itself* under `workflow_run` is silently ignored)
- **Cron re-timed 2026-09-22 for punctuality**: the slot hours fire at `:02/:12/:22/:32/:42/:52` (first one serves the slot, the rest are free catch-ups) and the hour BEFORE each slot fires at `:32/:42/:52` — 28/18/8 minutes early, i.e. inside the gate's relay window, so a single delivered firing makes the wakeup land exactly on time. Before this, the 08:00 slot ran +108/+126/+160 min late (and was missed once) for four days straight
- **`scripts/ai_bot_gate.py` is strictly SLOT-BASED**: it runs a wakeup only when the most recent schedule slot has no journal entry (on time, or catch-up for a slot GitHub dropped), **relays** (sleeps) up to **30 min** toward a slot that is about to come due — deliberately wider than the ~5-20 min heartbeat, which is the punctuality mechanism — and otherwise exits immediately. That is what keeps the bot at exactly **6 decisions/day** — the mesh kicks it far more often than that, and a stale-journal trigger under those kicks produced **22-25 wakeups/day** until the 2026-09-16 audit fixed it
- **Continuity files persist to the `ai-bot-state` branch** (git, not the unreliable Actions cache), restored on every run — including failure journals
- `workflow_dispatch` lets you trigger a wakeup manually from the Actions tab
- A `quality` job runs the AI bot test suite on every push that touches bot code
- **Daily digest at 23:50 UTC** (`ai_daily_digest.yml`): one Telegram heartbeat per day — how many of the **6 scheduled slots ran**, failures, P&L, equity, open positions, the AI's last reasoning. The report is anchored to *the day whose last slot (23:00) has passed*, so the ~2h cron slip that used to make it describe the wrong day is harmless. A missed slot is named (`⚠️ ... 3 never fired: 06:00 UTC, ...`) instead of being hidden behind a green tick, a slot served by a late catch-up is reported as LATE (`✅ 6/6 slots ran — 1 ran late: 08:00 UTC` — GitHub delivered that cron 3h late on 2026-09-20 and the old digest wrongly said "never fired"), a day with zero wakeups shouts "NO WAKEUPS RAN", and `digest_sent.txt` on the state branch keeps it idempotent. It is also kicked by every wakeup completion, so the report lands minutes after the last slot instead of hours later
- **Weekly report Sundays 17:00 UTC** (`ai_weekly_report.yml`): week P&L, win rate, AI notes, plus a **rolling performance review** (`trading_system/bot/perf_stats.py`) — 7d/30d/all-time win rate, profit factor, expectancy and realized R:R, realized **max drawdown** (measured from the true pre-trade peak), per-slot attribution (which wakeup slot earns and which never trades), planned-vs-realized R:R, week-over-week trend, and a ledger check that reports any drift between quoted P&L and actual cash instead of quietly rounding it away
- **Hourly health check** (`ai_health_check.yml`): watchdog that alerts if successful wakeups silently stop — then waits ~6 minutes for the self-healing chain to recover and goes green if a fresh success lands, so a fixed incident never leaves a red patrol on the board

Setup:

1. Push this repo to GitHub
2. Add repo secret: `GEMINI_API_KEY` (Settings → Secrets and variables → Actions) — free from https://aistudio.google.com/apikey
3. Actions tab → **AI Trading Bot** → enable scheduled workflows
4. Optionally add `AI_TELEGRAM_BOT_TOKEN` / `AI_TELEGRAM_CHAT_ID` for phone alerts (the `AI_` prefix keeps them separate from the main bot's `TELEGRAM_*` secrets)

⚠️ GitHub disables scheduled workflows after 60 days of repo inactivity — an occasional commit keeps it alive. (For zero-maintenance 24/7, any $5 VPS running `python scripts/start_ai_bot.py run` under systemd works too.)

---

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `GEMINI_API_KEY is required for gemini models` | No key in `.env` — the bot refuses to guess. Free Gemini key: https://aistudio.google.com/apikey |
| `market data unavailable for all pairs` | Binance down or blocked in your region; check `status` |
| `AI engine failed: ...` in journal, status ERROR | All models/attempts exhausted (retired model → fallback chain → retry/backoff); read the error, it names the last cause. A single 404/503 no longer kills the wakeup |
| Gate prints "relaying: sleeping …" | Normal — a kick landed within 30 min of a slot; this run waits and fires the wakeup **at** slot time |
| Gate prints "already served ... SKIPPING" | Normal — a redundant trigger between slots. This is the guard that keeps it at 6 decisions/day |
| Digest says "⚠️ ... never fired: 06:00 UTC" | A slot's wakeup never ran (GitHub dropped the trigger and nothing kicked it in time). Not fatal — the next trigger catches up — but repeated misses are worth a look |
| Digest says "✅ ... N ran late: 08:00 UTC" | The slot's wakeup ran as a late catch-up (GitHub delivered the cron trigger hours late). Informational — the slot WAS served |
| Digest says "Not due: digest for … already delivered" | Normal — the digest is re-triggered after the report went out; it exits without sending |
| Digest workflow shows red at "Persist digest marker" | Must never happen — the job needs `contents: write` to push the marker. If it does, the anti-duplicate guard is inert until it's fixed |
| Equity looks reset to $10,000 | The `data/ai_bot/` dir was deleted; paper state lives there |
| Want to start over | `rm -rf data/ai_bot` and run again |

---

## FAQ

**Why not GPT-6 Astra?** You don't have it — and don't need it. The engine speaks the OpenAI chat API and defaults to `gemini-flash-latest` on Google's **free tier** (free AI Studio key, no credit card; 6 wakeups/day fit far inside the free limits). The alias always tracks the newest flash model, so Google's version retirements never break the bot. To use OpenAI instead: set `ai.model` in `configs/ai_bot.yaml` and `OPENAI_API_KEY` in `.env`.

**Can the AI "escape" the rules?** No. Validation happens in code after the model responds — bad pairs, oversized positions, missing stop-losses, and low confidence are rejected and journaled with reasons.

**Does it always trade?** No. Most wakeups correctly decide to do nothing — empty `actions` is a valid, journaled outcome.

**What does it cost?** The LLM: **$0** — `gemini-flash-latest` on Google's free tier comfortably covers 6 wakeups/day. Everything else (GitHub Actions, Binance public data, Telegram) is free too.
