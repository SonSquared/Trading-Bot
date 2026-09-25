# AI Bot — Pre-Live Checklist

**Read this before `bot.mode: live`.** Paper P&L proves the *strategy
loop* works. It proves nothing about the live execution path, which has
run **zero times**. This document is an honest audit of that path as of
2026-09-13: what is verified in code (**GREEN**), what is workable but
risky (**YELLOW**), and what blocks real money (**RED**).

**Re-reviewed 2026-09-22** against a week of live paper data — see
[the dated section at the end](#re-review--2026-09-22-one-week-of-real-paper-evidence)
for the regraded RED/YELLOW/GREEN table and the updated go-live gate.

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
| G7 | AI decisions are bounded: 2% risk/trade (enforced directly), 30% max position, ≤30% total exposure, whitelist-only pairs | `configs/ai_bot.yaml` → `risk:`; enforced in `AIAgent._validate_decision` |
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

---

# Re-review — 2026-09-22 (one week of real paper evidence)

**Fresh eyes, real data.** Every line below was re-graded against the cloud
journal, the paper ledger, the entry records and the Actions run history for
**2026-09-14 → 09-22** — not against the code's intentions. Totals at review
time: **4 closed trades, 3 wins (75%), +$80.25 net (+0.80%), realized max
drawdown 0.18%, 0 failed wakeups, ~50 decisions, equity $10,079.75.**

## What the week promoted (new GREEN)

| # | Now GREEN | Evidence |
|---|---|---|
| G9 | Reports tell the truth about themselves | The digest distinguishes ran / **ran late** / never fired and was verified against the same journal twice; the weekly report now carries rolling stats **and** a ledger check that flags P&L that does not reconcile with cash |
| G10 | The watchdog actually catches silent death | It paged during the real 09-13 model outage, and the digest independently named the dropped 08:00 slot on 09-20 instead of showing a green tick |
| G11 | Model-retirement resilience | Google retired every pinned flash name (404) and the engine now discovers the live model from the API — **0 model failures in the last 6 days** |
| G12 | Money math is net of both fees | +6.27% / +3.84% / +4.47% wins and a −1.72% stop-out all reproduce from the ledger to the cent; a −1.72% alert and its $ figure describe the same thing |
| G13 | The AI does trade, and respects its box | 4 entries / ~50 decisions (8%), every one inside the 2% risk / R:R≥1.5 rules and the 10% size cap then in force (now 30% — see R8); 3 take-profits and 1 exchange-style stop trigger |

G1–G8 stand unchanged (the decision layer held: no rule was ever bypassed in
production).

## Regraded YELLOW

| # | Item | Was | Now | Evidence / why it is not GREEN yet |
|---|---|---|---|---|
| Y1 | Live peak-equity lags one wakeup | YELLOW | **YELLOW** | Still untested — no live run has ever happened |
| Y2 | `consecutive_losses` always 0 in live | YELLOW | **YELLOW** | Unchanged in code |
| Y3 | Paper fills overstate live fills | YELLOW | **YELLOW** | Reinforced this week: paper models the taker fee but zero slippage, and a legacy record was found quoting P&L $0.50 above actual cash — fee accounting had gaps |
| Y4 | Model outage mid-position | YELLOW | **YELLOW** | No outage in 6 days, but exchange-side SL/TP is still the only protection when the LLM is down |
| Y5 | AI discipline / trade frequency | YELLOW | **YELLOW (improving)** | No longer "never trades": 8% of decisions opened a position. 4 trades is still far too small a sample to call the calibration right |
| Y6 | **Scheduler punctuality** (new) | — | **YELLOW** | Real defect found: the 08:00 slot ran **+108 / +126 / +160 min late** (missed outright on 09-20) and 14:00 slipped up to +172 min, because the poller heartbeat died for 3-4h at a time (its continuity rested on cron). Fixed 09-22: 30-min relay window, pre-slot cron deliveries, poller self-relaunch via `workflow_dispatch`. **Needs ≥7 days of observed on-time slots** before live — a late wakeup means a position is managed late |
| Y7 | **P&L ↔ cash reconciliation** (new) | — | **YELLOW** | The weekly report now flags any drift (currently $0.50, one legacy 09-14 record). Live must start from a clean `data_dir` so this never carries over |

## Regraded RED (all still open — verified in code today)

| # | Blocker | Status 2026-09-22 |
|---|---|---|
| R1 | Failed protective-order placement leaves a naked position | **CLOSED 2026-09-25.** `protective_orders_missing` no longer exists: the stop is placed first and sized to the *filled* amount, and if it cannot be placed the position is flattened immediately (reduce-only market), recorded in `result["errors"]`, alerted on Telegram, and further entries are blocked for that wakeup. See [the 2026-09-25 (R1) re-review](#re-review--2026-09-25-r1-an-unprotected-live-position-is-now-impossible-by-construction) |
| R2 | Live execution path has zero real runs | **OPEN.** No testnet rehearsal has happened (`bot.sandbox: false`, no `BINANCE_TESTNET_*` keys). Amount rounding and the ccxt position-size field are still unverified against a real exchange |
| R3 | Leverage is never set by the bot | **OPEN.** `set_leverage()` exists in `exchange.py` but the AI bot never calls it (only the main bot does). Whatever leverage is set on the account applies to every AI position |
| R4 | Shared `data_dir` seeds live state from paper | **OPEN (mitigable).** `_build_agent(data_dir_override=...)` exists but is not exposed as a CLI flag; the clean path is a separate `configs/ai_bot_live.yaml` with its own `data_dir` |
| R5 | Key hygiene | **OPEN.** No evidence of a trading-only, IP-whitelisted sub-account |
| R6 | Paper evidence gate | **OPEN: 1 of 4 criteria met.** 1 week (needs ≥4), 4 closed trades (needs ≥30), positive net P&L ✅, realized max DD 0.18% ✅ |
| R7 | No documented kill switch | **OPEN.** Still no AI-bot runbook with a rehearsed cancel-all + flatten procedure |

## Go-live gate (updated)

- [ ] R1–R7 all closed
- [ ] Paper: **≥ 4 weeks** AND **≥ 30 closed trades** AND positive net P&L AND realized max drawdown < 10%  
      *(today: 1 week / 4 trades / +$80.25 / 0.18%)*
- [ ] **≥ 7 consecutive days with every slot on time** (no slot more than ~15 min late) — new,
      from the 08:00/14:00 lateness found on 09-22
- [ ] Testnet rehearsal complete, on the same machine class that will run live
- [ ] Live `data_dir` created fresh, paper history untouched, `git status` clean
- [ ] Kill switch rehearsed once, for real, on Testnet
- [ ] Binance sub-account: trading permission only, IP-whitelisted, keys never in code or logs
- [ ] You can afford to lose the entire live balance — literally, not figuratively

## Bottom line, 2026-09-22

The week moved this system from "a bot that has never traded" to "a bot with a
verifiable 4-trade record, honest reporting and a self-healing heartbeat" — the
decision layer is now backed by evidence, not just tests. What has **not** moved
is the execution layer: **R1, R2 and R3 are unchanged**, and those are exactly
the three that can lose money silently. The next real milestone is not more
paper weeks — it is the **Testnet rehearsal**, which is the only thing that can
close R2 and prove R1/R3 are dead. Paper continues meanwhile; it needs ~26 more
closed trades and ~3 more weeks to satisfy the gate above.

---

# Re-review — 2026-09-25 (account scale and venue legality)

**A different question, asked properly this time.** The 09-13 and 09-22 reviews
audited the *execution* path. Neither asked whether the account could place an
order **at all**, and that turned out to be the load-bearing question. Totals at
review time: **6 closed trades, 3 wins (50%), +$49.02 net (+0.49%), equity
$10,048.52 — on a $10,000 book.**

## What was wrong

| # | Blocker | Finding |
|---|---|---|
| R8 | **The paper account was 103x the user's real capital, and no order the bot could size was legal at the real size** | `configs/ai_bot.yaml` said `paper_starting_equity: 10000`, but `ai_agent.py` read `config["paper"]["starting_equity"]` — a key no config ever had — and silently defaulted to 10000. The configured value was dead code, so a $10,000 book ran for a $97 balance and *nothing failed*. Worse: at $97 the flat `max_position_size_pct: 10` allowed $9.70, under Binance's $20 (ETH) / $50 (BTC) `MIN_NOTIONAL`, so **every** order the bot could size would have been rejected by the venue. Paper fills have no venue, so the simulation reported all of this as a normal, profitable strategy. **Closed 2026-09-25** — and a hostile re-review of the fix found two further holes, both now shut: the size cap was still *unreachable* above 30% because total exposure caps a single position (so 40 was a number that could never be used, and heat was displayed but never enforced on entries), and the feasibility projection quietly computed its expectancy from the 2% risk cap on a position that can only risk 1.5% — overstating a typical month by a third. The caps now bind, the number shipped is the one that binds, and the projection uses the risk the described trade actually takes |

Both halves were invisible to the earlier reviews: this document contained
**zero** mentions of "notional", "minimum", "step size", "$97" or "10000".
G1–G13 verified the bot would behave correctly *if* it could trade; nothing
verified that it could.

## What changed (2026-09-25)

- **The config owns the account's scale.** `bot.paper_starting_equity: 97` is read
  directly; a missing key raises instead of inventing an account
  (`trading_system/bot/account.py`).
- **Venue limits are enforced in paper and live**
  (`trading_system/bot/venue_limits.py`, `ExchangeInterface.get_market_limits`).
  An order below `MIN_NOTIONAL` *after* lot-step rounding is refused and journaled
  with the arithmetic, instead of being reported as a fill.
- **`max_position_size_pct` is derived, not hand-picked**: the risk rules allow
  `2/5 × 100 = 40%`, the total-exposure cap allows 30%, and 30% is what ships —
  because a single position cannot exceed total exposure either. **Both caps are now
  enforced on every entry**, including positions already open and entries approved
  earlier in the same wakeup, and the 2% risk cap is checked directly as
  `size% × stop%` so it cannot drift when an adjacent number is edited. What binds
  at $97 is the exposure cap (a $29.10 position at a 5% stop risks 1.5%, inside the
  2% cap), and that is stated rather than implied.
- **The config's `risk:` rules now win over `strategy.json`**, which carried its own
  copy of every rule and silently shadowed the config.
- **Live order amounts are quantised** to the venue's lot step and the same quantity
  is used for the entry and both protective orders (this closes R2's "amounts are
  unrounded" evidence gap; R2's "zero real runs" half stands).
- **Legacy history is labelled, never rewritten**: a ledger seeded at $10,000
  reported against a $97 config prints `LEGACY SCALE ...` in the journal, the daily
  digest and the weekly report.
- `scripts/ai_venue_check.py` answers the question from live filters on demand.

## What this does NOT change

R2–R7 are **unchanged and still open** (R1 closed later the same day — see the next
section). At $97 only **ETH** clears the floor; BTC
needs ~$166.67 of equity, so BTC entries are refused with that reason. The achievable
scale is single-digit dollars per month, and the sample is still 6 trades — nothing
here is evidence of a profitable edge, only that the account can now place a legal
order and that its numbers are finally about the right account.

## Go-live gate (2026-09-25 additions)

- [ ] R2–**R8** all closed (**R1** closed 2026-09-25)
- [ ] `scripts/ai_venue_check.py` reports at least one tradable pair at the live equity
- [ ] Live equity is the real balance and `bot.paper_starting_equity` matches it
- [ ] Everything in the 2026-09-22 gate still applies

---

# Re-review — 2026-09-25 (R1: an unprotected live position is now impossible by construction)

**The one defect that could wipe the account, closed.** R1 said: if the
protective stop fails to place after a live fill, the code only logs
`protective_orders_missing` and builds the trade anyway. At $10,000 that was a
bug; at **$97** it is the entire loss budget standing open with no stop and no
one told. The old code path no longer exists.

## The invariant

> **A live position may not exist without a confirmed protective stop, and
every failure mode ends in ONE defined outcome — flatten, or refuse the entry.**

| Failure mode | Defined outcome |
|---|---|
| Entry order not accepted / connection dropped | **Refuse** — nothing was opened. A dropped call is not trusted either: the position is read back, and if a fill is hiding behind the error it is adopted and protected |
| Fill unconfirmed and no position visible | **Refuse** — nothing to protect |
| Stop rejected or timed out | **Flatten** the filled size (reduce-only market), record, alert, block further entries this wakeup |
| Stop AND flatten both rejected | **Emergency alert** — the only state we cannot repair is announced loudly, not swallowed |
| Partial fill | Protective orders cover the **filled** size, never the request |
| Partial fill below the venue minimum | The position cannot even carry a stop → **flatten and refuse** |
| Take-profit rejected | **Keep** the (stop-protected) position, record + alert; the AI closes it next wakeup |
| Close rejected / dropped | Bounded retry, then record + alert; **never** a journal entry for a close that did not happen |
| Close only partially fills | Re-close the remainder; if it will not close, **leave the protective orders in place** so the remainder is not naked |

Two supporting rules: the **stop is placed before the take-profit** (shortest
possible naked window), and a failed protection **blocks further entries for the
rest of that wakeup** — a venue that just refused a stop gets no more orders.
Guard state is reset at the start of every wakeup, so a fault never leaks into
the next one.

## Evidence

**21 unit tests** (`tests/test_ai_live_safety.py`) drive the real
`_execute_open` / `_execute_close` / `run_wakeup` against a fault-injecting fake
venue, one test per row of the table above. They run in CI on every AI-bot push.

**14/14 hostile checks against the REAL `ExchangeInterface`** (a temporary
driver, deleted after it ran; the permanent artifact is the test file). It
connected to the live exchange for real market data and read the real venue
filters, then injected a fault into every order call:

```
Live ETH price: $2,693.37
LIVE ETH limits : min_notional=20.0 step=0.001 taker=0.05% source=exchange
LIVE BTC limits : min_notional=50.0 step=0.001 taker=0.05% source=exchange
$97, 30% size, 5% stop -> 0.01 ETH = $26.93
  risk = $1.3465 (cap $1.94)  OK
  round-trip taker cost = $0.0269

[1]  real run_wakeup, healthy venue ..... entry executed, stop+tp attached;
                                          journal carries live venue_limits
                                          (source=exchange) and equity 97.0
[2]  stop REJECTED ...................... refused, position flat, flatten
                                          sized to the fill, alert w/ cost
[3]  stop TIMED OUT (raised) ............ same outcome; 2nd entry blocked
[4]  stop + flatten rejected ............ emergency alert fired
[5]  partial fill below venue min ....... flattened and refused
[6]  take-profit rejected ............... position kept WITH its stop
[7]  entry connection drop, fill hiding . read back and protected
[8]  close rejected repeatedly .......... retried, no phantom close,
                                          stop left in place
[9]  partial close ...................... remainder closed, orders retired
[10] close only ever half-fills ......... no phantom close; protective
                                          orders kept for the remainder
RESULT: 14/14 checks passed
```

## What this does NOT change

- **R2's "zero real runs" half stands.** This is fault injection against the
  real interface, not a testnet rehearsal; no credential exists on this machine
  and no order was placed on any exchange. The rehearsal is still what closes R2
  and what confirms R1 against a real venue's order lifecycle.
- **Nothing here is a profit claim.** The honesty numbers are unchanged: the
  cost of refusing an entry is ~$0.027 of fees at $97, and the risk a protected
  trade carries is $1.35 of a $1.94 cap.
- R3–R8 remain open as listed.
