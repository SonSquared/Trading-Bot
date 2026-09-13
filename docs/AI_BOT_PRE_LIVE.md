# AI Bot — Pre-Live Checklist

**Read this before `bot.mode: live`.** Paper P&L proves the *strategy
loop* works. It proves nothing about the live execution path, which has
run **zero times**. This document is an honest audit of that path as of
2026-09-13: what is verified in code (**GREEN**), what is workable but
risky (**YELLOW**), and what blocks real money (**RED**).

The three locks (config `mode: live`, CLI `--live`,
`AI_BOT_LIVE_CONFIRMED=1`) gate *entry* into live mode. They say nothing
about whether live mode is *safe* once entered. This page does.

---

## Where live can even run (structural fact)

**Live mode cannot run on GitHub Actions.** Binance futures geo-blocks
GitHub's US runners (HTTP 451 — proven by run #10). The Kraken fallback
covers *market data only*; orders and balances always need Binance
reachable. Live therefore runs on **your PC or a VPS in an allowed
region**, via `python scripts/start_ai_bot.py run` (or `once` under
cron/systemd). The cloud keeps doing paper; live is a separate machine
with a separate `data_dir` (see RED-4).

---

## GREEN — verified in code, keep as-is

| # | Protection | Evidence |
|---|---|---|
| G1 | Triple lock to enter live: config + `--live` + `AI_BOT_LIVE_CONFIRMED=1`; paper is the default | `scripts/start_ai_bot.py` (`_build_agent`) |
| G2 | Live refuses a blind start: unreachable exchange → hard error; `equity == 0` → loud `RuntimeError` | `start_ai_bot.py`; `run_wakeup` step 2 |
| G3 | Every live entry attaches reduce-only `STOP_MARKET` + `TAKE_PROFIT_MARKET` (mark-price, price-protected) — the position is exchange-protected even if the bot dies | `_execute_open`, `place_stop_market_order` |
| G4 | Hard validation before ANY order; the AI cannot override: pair whitelist, min confidence, SL bounds, risk/reward ≥ 1.5, size cap, max positions, no duplicate pair, drawdown halt | `_approve_actions` |
| G5 | Drawdown halt works in live too: peak equity persists in `progress.json` and is re-read each wakeup; ≥10% DD → entries rejected, closes only | `_risk_snapshot` + `next_progress` |
| G6 | Nothing fails silently: every error path journals `status: "error"` and fires the ⚠️ Telegram alert | `run_wakeup` except-block |
| G7 | AI decisions are bounded: 2% risk/trade, 10% max position, whitelist-only pairs | `configs/ai_bot.yaml` → `risk:` |
| G8 | Alerting layers independent of the trader: watchdog (hourly), daily digest (23:50 UTC), always-on poller | `ai_health_check.yml`, `ai_daily_digest.yml`, `ai_poller.yml` |

---

## YELLOW — works, but know the caveat before you trust it

| # | Finding | Caveat | Mitigation |
|---|---|---|---|
| Y1 | Live peak-equity lags one wakeup on new highs | A violent crash inside one wakeup gap (up to 4h) is measured from the *previous* wakeup's equity, not the true high — the halt trips one wakeup late. Paper mode has no lag (`update_peak` runs before the snapshot) | Acceptable with 6 wakeups/day + exchange-side SL (G3). Optionally mirror `update_peak` for live |
| Y2 | `consecutive_losses` is always 0 in live | The 5-loss streak breaker exists only in paper (it reads the ledger) | Add loss-streak tracking to `progress.json` before live |
| Y3 | AI-open price vs fill price | Live fills at market (slippage), paper fills at decision-time price. Paper results slightly overstate live fills | Judge paper results net of ~0.05–0.1% slippage per side |
| Y4 | Gemini free-tier outage during open positions | Exchange-side SL/TP still protect positions (G3); new decisions pause; watchdog + digest tell you | None needed — by design |
| Y5 | `min_confidence_to_trade: 60` and the AI's current discipline | The AI currently declines ~everything (quiet market). Live with this calibration may simply never trade | That is a feature. Loosen only with evidence |

---

## RED — must fix/do before real money

| # | Blocker | Why it bites | Fix |
|---|---|---|---|
| R1 | **Failed protective-order placement leaves a naked position** | If SL or TP placement fails after a live fill, the code only logs `protective_orders_missing` — no Telegram alert, no error in the journal, the position stays open with NO stop until a human notices | On any protective failure: append to `result["errors"]`, `notify_error(...)`, and refuse further entries for that wakeup; consider an emergency market close |
| R2 | **Live execution path has zero real runs** | `place_market_order` amounts are unrounded (`size_usd / price`), and the live close reads `pos.get("size")` from ccxt — the exact field name on real `binanceusdm` payloads is unverified. A wrong field = silent `amount <= 0` no-op close | **Testnet rehearsal** (below) must complete end-to-end: open → SL/TP visible on testnet → AI close → exchange position flat |
| R3 | **Leverage is never set by the bot** | ccxt does not set leverage; the account's *existing* exchange-side setting applies (whatever it was last set to manually). 20x from an old experiment would silently multiply every position | Set leverage explicitly at connect (`set_leverage(pair, 1)` — or 2 max) and verify on testnet |
| R4 | **Shared `data_dir` seeds live risk state from paper history** | `progress.json` (peak, day anchors) and the journal carry over. A live account smaller than the paper peak can trip an *instant permanent* trading halt; paper errors pollute the live audit trail | Live runs with a **fresh, separate** `data_dir` (e.g. `data/ai_bot_live`); never `rm` paper history |
| R5 | **Key hygiene** | Live keys with withdrawal permission or no IP whitelist are an account-drain waiting for a leak | Binance sub-account, **trading permission only**, IP-whitelisted to the live machine; keys never in code/logs/secrets-of-other-repos |
| R6 | **No paper evidence yet** | The 2–4 week paper bar hasn't even started: the AI's discipline means ~0 closed trades so far — there is no win rate to point at | Define the gate (below) and let paper accumulate real closed trades |
| R7 | **No documented kill switch** | When things go wrong at 3am you do not want to improvise | Document + rehearse: cancel all orders (`cancel_all_orders`) and flatten every position (reduce-only market close), plus disabling live (remove `AI_BOT_LIVE_CONFIRMED`) |

---

## Testnet rehearsal (do this before R2 is "done")

1. Free keys: https://testnet.binancefuture.com → `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET` in `.env`
2. `configs/ai_bot.yaml`: `bot.sandbox: true`, `bot.mode: paper` first — verify data + decisions flow against testnet endpoints
3. Then `bot.mode: live` + `--live` + `AI_BOT_LIVE_CONFIRMED=1` **with testnet keys**: real order flow, fake money
4. Run ≥ 2 weeks or ≥ 20 executed testnet trades, whichever is longer, including at least one SL or TP firing *on the exchange*
5. Success criteria: every open visible on the testnet UI with both protective orders; every AI close leaves the exchange position flat; zero naked positions; journal and Telegram alerts match the exchange UI exactly

---

## Go-live gate (all must be true)

- [ ] R1–R7 all closed
- [ ] Paper: ≥ 4 weeks elapsed AND ≥ 30 closed trades AND positive net P&L AND max drawdown < 10%
- [ ] Testnet: rehearsal complete per the criteria above, on the *same machine class* that will run live (a VPS, if live will be a VPS)
- [ ] Live `data_dir` created fresh; paper untouched; `git status` clean
- [ ] Kill switch rehearsed once (for real, on testnet)
- [ ] Keys: sub-account, trading-only, IP-whitelisted
- [ ] You can afford to lose the entire live balance — literally, not figuratively

---

## Bottom line

The *decision* layer is genuinely solid — validation, hard limits, the
drawdown halt, and no-silent-failure journaling are all real and tested
(G1–G8). The *execution* layer is unproven (R2), has one dangerous edge
(R1), and one operational footgun (R4). Close the REDs, rehearse on
testnet, let paper build evidence — then live with money you can afford
to lose.
