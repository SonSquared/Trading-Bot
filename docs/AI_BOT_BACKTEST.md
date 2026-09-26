# AI Bot — Backtest of the Execution Mechanics

**Read this before you read a number in it.** This page is the evidence behind the
claim "the mechanics work", and it is deliberately narrow: it tests the *machinery*
around a decision — venue filters, fees, the risk and exposure caps, stop/target
geometry, the six daily slots, the drawdown halt — driven by the **production
`AIAgent` in paper mode** over 4.6 years of real BTC/ETH candles.

It does **not** test the model's judgement, and it cannot. An LLM's output depends on
everything it believes at the moment it is asked, so replaying it over past candles
means handing it the answer sheet. Any number here that is described as "the
strategy's result" is a *mechanical stand-in* for the strategy, declared and not
fitted, and it says nothing about what the model would have done.

Written 2026-09-25, by the pass that added `trading_system/bot/backtest/` on top of
commit `844f086`, and re-measured by the pass that made the drawdown halt re-arm. Every
number below came out of that harness on real BTC/ETH 1h candles; none of them was
typed in by hand, and the ones the report generates (including the $97 tradability
verdict) are computed from the config and the venue helpers at run time.

---

## The one-line version

> The bookkeeping is provably correct: at $97 every legal ETH order is allowed, every
> illegal one is refused with its arithmetic, fees are charged twice, stops and
> targets fire at the declared geometry, and the same command always produces the
> same bytes. **Nothing here shows the bot makes money** — and the only real forward
> evidence (6 round trips, +$49 on a legacy $10,000 paper account) was produced at a
> position size the $97 account can no longer place.
>
> The loudest thing in the results is not about a rule at all: the 10% drawdown halt
> was measured against an all-time peak that only rises, so on the pre-fix tree
> **every mechanical source stopped trading in 2022 and never resumed**. It now
> re-arms after a served cool-off, and the runs below show entries resuming — 153
> halts, 153 ended, none unresolved. What still ends an account at this size is not
> the halt but the **$20 venue floor**: under $66.67 of equity there is no legal order
> size on any pair the bot trades. See
> [the halt finding](#the-halt-finding-a-one-way-door-fixed-and-re-measured).

| Question | Answer |
|---|---|
| Can this account place legal orders at all? | **Yes on ETH, no on BTC** — measured, not asserted |
| Are the caps, fees, stops and halt enforced on every entry? | **Yes** — checked on each executed entry, not assumed |
| Is there look-ahead in the replay? | **No violations over the whole span** — counted, not promised |
| Does it reproduce? | **Byte-identical** — same command, same SHA-256 |
| Does the halt still end the experiment? | **No** — 153 halts, 153 ended, entries resume after each |
| What ends an account at $97, then? | **The venue floor** — below $66.67 no legal size exists |
| Were the caps weakened to allow re-entry? | **No** — risk, exposure, stop and venue all still bind |
| Does the bot make money? | **Not supported.** Not measured either |
| Does the model have an edge? | **Not measured. This cannot measure it** |

---

## How to rerun it

```bash
# Data first (once). The harness reads data/raw/<SYMBOL>/klines_1h.parquet.
python scripts/download_data.py --pairs "BTC/USDT:USDT,ETH/USDT:USDT" --timeframes 1h

# The canonical suite: every shipped decision source, real data, real config.
python scripts/ai_backtest.py

# The pieces, and the useful variants:
python scripts/ai_backtest.py --list-sources            # provenance of each source
python scripts/ai_backtest.py --quick                   # last ~365 days, 2 folds
python scripts/ai_backtest.py --synthetic               # no data files needed
python scripts/ai_backtest.py --source trend --trigger-model intrabar
python scripts/ai_backtest.py --source journal --journal data/ai_bot/journal.jsonl
python scripts/ai_backtest.py --json results.json --with-curve
```

Zero network, zero LLM calls, no orders. `--json` writes machine-readable results;
`--with-curve` adds the equity curve. Runtime is stated below — it is not fast, and
deliberately so: it runs the real wakeup, not a vectorised approximation of it.

---

## What is replayed, and what that means

The engine does **not** re-implement the strategy. It builds a production `AIAgent`
in paper mode with a simulated clock and a historical exchange, and calls the
production `run_wakeup()` once per schedule slot. So these are the *same code paths*
the bot runs:

| Mechanic | Where it comes from |
|---|---|
| The six daily slots and their times | `scripts/ai_bot_gate.py` (`_load_slots`) — the schedule the cloud uses |
| Venue `MIN_NOTIONAL` / lot step / taker fee | `venue_limits.resolve_limits`, reconciled against the exchange, falling back to the dated builtin table |
| The refusal of an illegal order (paper included) | `AIAgent._venue_rejection` — same function that journals refusals in production |
| Risk 2%/trade, 30% position, 30% heat, ≤5% stop, R:R ≥ 1.5, 3 positions, 10% DD halt (cool-off → release or re-arm, budgeted) | `configs/ai_bot.yaml`, enforced in `_validate_decision` / `_risk_snapshot` / `_drawdown_halt` |
| Fees on both sides, stops and targets, the ledger, the journal | `PaperLedger` and `run_wakeup` |
| Account scale ($97) | `bot.paper_starting_equity` via `account.resolve_starting_equity` |

What the *engine* adds, and labels: the timeline; window orchestration; and an
optional `--trigger-model intrabar` that fills a resting stop at its own level
(a labelled sensitivity — the bot's paper mode fills at the price the wakeup
observes, so a gap through the stop costs more than the stop distance).

**The order path is unreachable, not unused.** Every order-ish method on the
historical exchange raises `OrderPathTouched` and increments a counter, and the report
prints `order-path attempts: 0`. That is a measurement over every replayed slot.

---

## The decision sources, and their provenance

Every result is printed with the source that produced it and whether it was fitted to
the data. `python scripts/ai_backtest.py --list-sources` prints this table from the
code:

| Source | Label | What it is | Fitted to this data? |
|---|---|---|---|
| `null` | null | never trades — the control | no |
| `random` | null | seeded 50/50 long/short at the same stop/target — the cost baseline | no |
| `trend` | mechanical | EMA 9/21/50 stacked + ADX ≥ 20, exit on flip | no |
| `rsi-reversion` | mechanical | RSI ≤ 30 / ≥ 70, exit at RSI 50 | no |
| `journal` | recorded-LLM | replays decisions the model actually made, when a journal recorded them | no |

**Nothing is fitted.** No parameter search, no threshold tweak, no selection of the
best window. That is the point: with nothing fitted there is no train/test split to
get wrong, and the walk-forward windows below are out-of-sample by construction
rather than by design. A rule that loses money here is not evidence that a *better*
rule loses money — it is evidence that the skeleton carries that rule without
breaking its own constraints.

### The `journal` source, honestly

It replays `actions` / `actions_proposed` from a JSONL log. Check what yours can
actually replay before quoting a result: the cloud journal up to 2026-09-25 held
**118 wakeups and 0 replayable actions**, because the older format stored
`actions_requested: 1` — a *count*. The report says so in as many words
("`N` records carried NO readable action … NOT evidence that the model chose to sit
flat"). This pass added `actions_proposed` — the model's full proposed action list —
to the agent's journal, so wakeups recorded after that change ships are replayable
as-is, and a count-only row can never again be mistaken for a decision.

---

## The window design: one account, four folds, five regimes

```
  2022-01-08 ─────────────────────────────────────────────────────── 2026-08-22
  continuous: one $97 account for the whole span        (the headline + the action log)
  fold 1/4:   $97 fresh   fold 2/4: $97 fresh   fold 3/4   fold 4/4
  2022 bear | 2023 recovery | 2024 bull | 2025 mixed | 2026 YTD   (each $97 fresh)
```

* **Continuous** — one account carried through the whole span. The only window whose
  trade log is reported as a sequence; the folds and regimes overlap it, so their
  trades are not concatenated into it.
* **Folds** — four contiguous, non-overlapping, equal-duration spans. This is the
  walk-forward view, and each fold starts a **fresh $97** so a lucky first year cannot
  fund the rest.
* **Regimes** — calendar-year blocks, declared up front to cover *different market
  states* (bear, recovery, bull, chop, current), each with a fresh $97. Both are
  needed: folds ask "does it hold up sequentially", regimes ask "where does it work".

---

## Results

### The continuous window — one $97 account, 10,121 slots each

Span: 2022-01-08 → 2026-08-22. Return is on a $97 base, so +41.66% is +$39.03.
These are the numbers **after** the halt fix below — the run in which a halted account
resumes instead of going quiet forever.

| Source | Round trips | Win | Net $ | Return | Max DD | Fees | Expectancy/trade |
|---|---|---|---|---|---|---|---|
| `null` | 0 | — | +0.00 | +0.00% | 0.00% | 0.0000 | — |
| `trend` (slot) | 500 | 33% | −27.61 | **−4.10%** | 31.37% | 14.7814 | −0.0552 |
| `random` (slot, seed 7) | 217 | 39% | −31.01 | **−31.97%** | 43.39% | 5.8439 | −0.1429 |
| `random` (intrabar) | 67 | 25% | −30.35 | −31.29% | 34.53% | 1.6458 | −0.4530 |
| `rsi-reversion` (slot) | 320 | 64% | +39.03 | **+41.66%** | 13.38% | 11.0581 | +0.1220 |
| `rsi-reversion` (intrabar) | 112 | 38% | −31.08 | −32.04% | 34.13% | 2.6749 | −0.2775 |

`null` is the control and it behaved: 0 trades, 0 fees, $97.00 → $97.00 through
10,121 wakeups and 8,096,604 candles fetched. Nothing invented a trade on its own.

With the halt re-arming, the sources are no longer killed by the guard and start being
graded by the market instead: two of the three trade the whole span and lose money
doing it. That is a statement about declared mechanical rules, not about the model,
and it is exactly why nothing here is allowed to grade a strategy.

### The halt finding: a one-way door, fixed and re-measured

**Before the fix, every trading source made its last trade in 2022.** The `trend`
rule's last action was **2022-02-04**, `random`'s was 2022-04-27, `rsi-reversion`'s was
2022-07-18 — and then the account sat idle for the remaining four years.

The reason was the 10% drawdown halt, and it is worth stating precisely because it was
easy to misread: the halt is measured against the **all-time peak equity, and the peak
only ever rises**. Once the account is ≥10% below its peak it refuses new entries
until equity climbs back above 90% of that peak. The `trend` account ended at $90.57
against a peak of about $101, so it was **still halted four and a half years later**,
having refused **12,588 entries in that window alone** (30,542 across all ten of its
windows).

That is not a guard doing its job, it is a deadlock: a *flat* account cannot trade,
so its equity cannot climb back above the line, so it stays halted for good. At $97
the account was one bad month away from being permanently finished, and the only ways
out were editing `progress.json` by hand or starting again on a fresh `data_dir`.

#### Before and after, same source / data / schedule / caps (continuous-window refusals)

| Source | Pre-fix: last action | Pre-fix: end | Pre-fix: halt refusals | Post-fix: last action | Post-fix: end | Post-fix: halt refusals |
|---|---|---|---|---|---|---|
| `null` | — | $97.00 | 0 | — | $97.00 | 0 |
| `trend` | 2022-02-04 | $90.57 | 12,588 | **2026-08-22** | $93.02 | 96 |
| `random` | 2022-04-27 | $103.51 | 4,764 | 2025-04-01 | $65.99 | 30 |
| `rsi-reversion` | 2022-07-18 | $99.82 | 1,143 | **2026-08-21** | $137.41 | 1 |
| `rsi-reversion` (intrabar) | 2022-02-04 | $89.36 | 1,349 | 2023-04-13 | $65.92 | 3 |

The remaining halt refusals are the ones the guard is *supposed* to make: entries
refused while a 24-hour cool-off is being served, not entries refused forever. `trend`
went from 14 round trips in one month to **500 across the span**; `rsi-reversion` from
60 to **320**. (`trend` intrabar ran in the pre-fix suite but not the post-fix one, so
it has no after column.)

#### The lifecycle, measured

The fix makes the halt a served cool-off with two defined exits — release if the
account is back inside the limit, re-arm to the current equity if it is not, bounded
by a budget of two re-arms per rolling 30 days (full semantics in
[`AI_BOT.md`](AI_BOT.md#the-drawdown-halt-is-a-cooldown-not-a-shutdown)). Across all
six post-fix runs (ten windows each, each window a fresh $97 account):

| Source | Engages | Released | Re-armed | Held (budget spent) | Halt refusals | Cool-off wakeups | Unresolved |
|---|---|---|---|---|---|---|---|
| `null` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `trend` | 41 | 0 | 41 | 0 | 277 | 205 | 0 |
| `random` | 55 | 19 | 36 | 0 | 111 | 275 | 0 |
| `random` (intrabar) | 39 | 6 | 33 | 0 | 116 | 195 | 0 |
| `rsi-reversion` | 4 | 0 | 4 | 0 | 3 | 20 | 0 |
| `rsi-reversion` (intrabar) | 14 | 1 | 13 | 0 | 25 | 70 | 0 |

**153 halts, 153 ended, none unresolved, no account left in limbo.** Every halt that
started also finished: 26 of them released because the account came back inside the
limit by itself (no re-arm spent), 127 re-armed onto the equity they resumed at, and
every one of the 153 served a full cool-off first — 765 cool-off wakeups over 153
halts is exactly 5 wakeups each, which is 24 hours at six slots a day. Twelve of the
re-arms spent the second of the two re-arms inside a 30-day window (`rearms_used=2`),
so the budget was exercised without ever being exhausted in these runs; the exhausted
branch is pinned by tests, not by these runs, and says so.

The `trend` continuous window is the one the pre-fix run died in: its last **entry**
was 2022-02-03 08:00, the halt first engaged at 2022-02-04 06:00, and the close that
followed at 08:00 was allowed through (a halt blocks entries, not exits) — then the
account never traded again. Here is what it does now, episode by episode:

| # | Halt engaged | Drawdown | Ended by | Resumes at | First entry after |
|---|---|---|---|---|---|
| 1 | 2022-02-04 06:00 | 10.02% | re-armed | 2022-02-05 06:00 | 2022-02-11 06:00 |
| 2 | 2022-03-03 20:00 | 10.28% | re-armed | 2022-03-04 20:00 | 2022-03-05 06:00 |
| 3 | 2022-05-05 20:00 | 10.90% | re-armed | 2022-05-06 20:00 | 2022-05-09 08:00 |
| 4 | 2023-12-20 08:00 | 10.13% | re-armed | 2023-12-21 08:00 | 2023-12-22 06:00 |
| 5 | 2024-03-24 00:00 | 10.25% | re-armed | 2024-03-25 00:00 | 2024-03-25 20:00 |
| 6 | 2024-06-29 14:00 | 11.03% | re-armed | 2024-06-30 14:00 | 2024-07-03 00:00 |
| 7 | 2024-11-20 00:00 | 10.49% | re-armed | 2024-11-21 00:00 | 2024-11-21 08:00 |
| 8 | 2025-01-20 00:00 | 10.29% | re-armed | 2025-01-21 00:00 | 2025-01-21 06:00 |
| 9 | 2025-02-08 06:00 | 10.30% | re-armed | 2025-02-09 06:00 | 2025-02-09 08:00 |
| 10 | 2025-02-20 00:00 | 11.88% | re-armed (2nd in window) | 2025-02-21 00:00 | 2025-02-24 20:00 |
| 11 | 2025-04-24 14:00 | 10.69% | re-armed | 2025-04-25 14:00 | 2025-04-28 20:00 |
| 12 | 2025-09-04 06:00 | 10.18% | re-armed | 2025-09-05 06:00 | 2025-09-05 14:00 |
| 13 | 2026-02-22 14:00 | 10.05% | re-armed | 2026-02-23 14:00 | 2026-02-25 14:00 |
| 14 | 2026-04-23 06:00 | 10.13% | re-armed | 2026-04-24 06:00 | 2026-04-26 14:00 |

Every halt ends, and an entry follows each one — the gap between "resumes at" and
"first entry" is the rule's own patience, not the guard: a released halt only
*permits* trades, and this rule waits for its stack to line up again. `random`
supplies the other exit: 18 episodes in its continuous window, of which 7 **released**
by themselves because the account was back inside the limit when the cool-off ended.

#### Where entries did NOT resume — and it is not the halt

Honesty first: **three of the five traded runs do eventually stop entering**, and no
account trades to the end of the data except `trend` and `rsi-reversion`. The cause is
not the drawdown halt. Here is the continuous window counted **after its last halt
resolution** (i.e. with the halt already out of the way — it can only re-engage on a
fresh 10% breach):

| Source | Last resolution | Last entry | Entries after | Wakeups after | Halt refusals | Venue refusals | End |
|---|---|---|---|---|---|---|---|
| `trend` | 2026-04-24 | 2026-08-22 | 47 | 720 | 0 | 458 | $93.02 |
| `rsi-reversion` | 2022-07-19 | 2026-08-21 | 261 | 8,966 | 0 | 521 | $137.41 |
| `random` | 2025-04-02 | 2025-04-01 | **0** | 3,040 | **0** | 1,542 | $65.99 |
| `random` (intrabar) | 2022-03-22 | 2022-03-20 | **0** | 9,684 | **0** | 4,879 | $66.65 |
| `rsi-reversion` (intrabar) | 2022-06-26 | 2023-04-13 | 26 | 9,107 | **0** | 984 | $65.92 |

For the three that stop, **zero entries were blocked by the halt after it resolved**.
They stop because the account is no longer big enough for the venue: ETH's floor is
$20 and the exposure cap is 30% of equity, so **under $66.67 equity there is no legal
order size on any pair the bot will trade** (BTC's $50 floor needs $166.67). The
refusals say so in their own words, and they keep coming — 1,542 of them in `random`
over its last 17 months:

```
ETH/USDT:USDT: venue minimum $20.00 for ETH/USDT:USDT exceeds the $19.80
                a position may reach at $65.99 equity (risk 2% of equity per
                trade, exposure cap 30% of equity; needs ~$66.67)
```

That is a $97-account problem, not a halt problem: a strategy that draws down ~32%
lands under the smallest order the venue accepts, and re-arming the halt cannot fix
that. It is also the honest answer to "does the bot have a stalemate?": at this
account size the *venue floor* is a floor on the experiment, and the last 17 months of
the `random` run are that floor, refusing every one of 1,542 attempts with its
arithmetic. Deciding what to do about it (more equity, a different pair, or accepting
that a $97 account has a ~32%-loss limit) is a product decision, not a backtest one.

#### The trade-off, and what limits re-entering the regime that caused the halt

A halt that can release will, by construction, be able to re-enter a losing regime.
That cost is accepted, and it is bounded by four things, all of them measured above:

* **A served cool-off before any release** — 24 hours flat, and the period cannot be
  cancelled by a wiggle back inside the limit (that was the pre-fix behaviour the
  tests now refuse). 153 halts, 153 full cool-offs.
* **A budget** — at most 2 re-arms per rolling 30 days, so at most two fresh 10%
  drawdowns per window before the halt holds instead.
* **The caps underneath, untouched** — risk per trade, exposure, stop distance and the
  venue floor all still bind (see [the mechanics section](#the-mechanics-are-enforced-not-assumed)).
* **A floor under the experiment** — the venue floor stops trading before the account
  reaches it, so the worst case here is an account parked under $66.67, not a halted
  account draining to zero.

### Was it a simulation artefact? No — it is in the committed production code

The harness calls the production `run_wakeup()`, so it cannot loop through code the
bot does not run. But "the harness did it" is still worth refuting directly, because a
replay bug and a one-way door look identical from inside a result table.

In the pre-fix commit (`952f29a`) the agent's peak equity is written in exactly two
places: it is seeded, and it is **raised** on a new high
(`PaperLedger.update_peak`). Nothing on the AI bot's path lowers it. A repo-wide
grep finds a peak re-arm in one place only — `scripts/paper_trader.py`, the *sibling*
bot, which has always re-armed after its drawdown stop — so the AI path was the odd
one out, not the harness. Meanwhile `_validate_decision` refuses every entry while
`trading_halted` is true, and `next_progress` re-persists the same peak, so a flat
account below the line could not trade, could not recover, and had no operator route
out except editing `progress.json` by hand or starting again on a fresh `data_dir`.

`tests/test_ai_dd_rearm.py::test_a_halt_with_no_release_is_permanent` drives that on
the production path — a real `AIAgent.run_wakeup()`, 40 wakeups, **no harness at
all** — and pins it: every entry refused, cash frozen at $100.00, peak unmoved at
$125.00, and no trade file written. A halt with no release condition is exactly what
it looks like.

### Walk-forward folds — four fresh $97 accounts

Post-fix numbers. Where a fold never contained a halt, its number is **unchanged**
from the pre-fix table — `rsi-reversion`'s folds 2–4 are byte-identical, which is what
a change confined to the halt should look like. Where a halt was inside the span, the
fold changes, because the account no longer stops trading at it.

| Source | fold 1 (22-01→23-03) | fold 2 (23-03→24-04) | fold 3 (24-04→25-06) | fold 4 (25-06→26-08) | net | positive |
|---|---|---|---|---|---|---|
| `random` | −3.71% | −22.11% | +7.55% | −16.55% | −$34.06 | 1/4 |
| `trend` | +18.06% | −7.33% | −11.19% | −1.19% | −$1.00 | 1/4 |
| `rsi-reversion` | +6.90% | +8.79% | +21.24% | −0.20% | +$34.68 | 3/4 |

### Regimes — five fresh $97 accounts, no carry-over

| Source | 2022 bear | 2023 recovery | 2024 bull | 2025 mixed | 2026 YTD | net | positive | spread |
|---|---|---|---|---|---|---|---|---|
| `random` | −0.04% | +28.28% | −5.15% | −15.12% | −14.40% | −$7.05 | 1/5 | **43.40 pp** |
| `trend` | +14.26% | +2.43% | −6.27% | −9.14% | −6.79% | −$3.92 | 2/5 | 23.40 pp |
| `rsi-reversion` | +12.05% | −3.89% | +10.88% | +25.39% | −6.69% | +$35.73 | 3/5 | 32.08 pp |

### The fill model changes the sign

The bot's paper mode fills a stop at the price the *wakeup observes*; a real exchange
fills it the moment the level trades. Those two are not the same thing, so the
harness can run either, and the difference is not a rounding error:

| Source | Slot model (the bot's paper behaviour) | Intrabar model (a labelled sensitivity) |
|---|---|---|
| `random` | −31.97%, 217 trades, 39% win | −31.29%, 67 trades, 25% win |
| `rsi-reversion` | **+41.66%**, 320 trades, 64% win | **−32.04%**, 112 trades, 38% win |

The sign of the `rsi-reversion` result depends entirely on which assumption you pick —
and at this stop distance (5% of a ~$29 position) the intrabar model is not a rounding
of the slot model, it is a different answer.
That alone is enough to know that none of these numbers should be used to choose
anything — and it is the honest reason the verdict below refuses to grade a strategy.

### What the dispersion says (the statistical part)

* **The coin flip still had the widest spread of all** — 43.40 percentage points
  across five regimes, against 23.40 pp for `trend` and 32.08 pp for `rsi-reversion`. A
  source with no edge at all produced a range that contains every result the rules
  produced.
* Expectancy per trade was −$0.14 (`random`, 217 trades), −$0.06 (`trend`, 500) and
  +$0.12 (`rsi-reversion`, 320). Taking 5% stops at a ~$29 notional, one trade swings
  about ±$1.50, so the standard error on 320 trades is roughly ±$0.08 — which is the
  whole estimated edge of the best rule, i.e. about one standard error. Even before the
  fill assumption above gets a vote, that is not an edge.
* Fees are small in absolute terms at this size ($1.65–$14.78 over the whole span) but
  they are the whole story where a rule trades often: `trend` paid **$14.78 in fees
  against a gross of −$12.83**, and `rsi-reversion`'s +$39.03 net came from a +$50.09
  gross, so fees ate **22%** of everything it made. A rule with no real edge cannot pay
  for itself here, which is what the `random` baseline demonstrates cleanly.

### The mechanics are enforced, not assumed

Counters from every run (per source, then four sources):

| | |
|---|---|
| Slots replayed | 30,363 per source — 182,178 for the six-run suite |
| Distinct candle windows | 60,744 per source — 364,464 for six |
| Candles served (agent + engine) | 24,289,812 per source — 145,738,872 for six |
| **Look-ahead violations** | **0** — a candle after the cursor was served, counted at the exchange |
| **Order-path attempts** | **0** — every place/cancel call raises `OrderPathTouched` |
| Errors recorded | **0** — `errors: []` in every window of every run |
| Peak observed exposure | 30.33–31.59% of equity — the 30% cap is applied at entry and equity moves after it, so the *observed* peak drifts above the cap the position was sized under |
| Exposure-cap refusals | **41** — the cap refusing an entry rather than bounding one: *portfolio heat 60.0% (open 30.0% + this 30%) would exceed the 30% exposure cap* |
| Risk-cap / size-cap refusals | **0** — nothing was ever refused for breaking them; the risk cap is slack by arithmetic (30% exposure × 5% stop = 1.5% < 2%) and the size cap is what *sizes* the trade rather than something it violates |
| Fees | charged both sides at 0.05% taker on every closed trade |
| Refusals, bucketed (all six runs) | 42,630 venue floor, 532 drawdown halt (every one of them while a cool-off was being served), 169 "close requested with no position" (the sources' own sloppiness caught, not acted on), 41 exposure cap |
| BTC entries refused | 18,377 in `trend` alone, every one with the arithmetic: *venue minimum $50.00 … exceeds the $29.10 a position may reach at $97.00 equity … needs ~$166.67* |

And the caps as measured on **every executed entry** of the continuous window — not
sampled, not assumed:

| Source | Entries | Worst risk (size×stop) | Least risk-cap headroom | Worst size | Past any cap | Stops used | Smallest size | BTC entries |
|---|---|---|---|---|---|---|---|---|
| `trend` (slot) | 790 | 1.500% | $0.41 | 30.003% | **0** | 5.0% | $21.50 | 0 |
| `random` (slot) | 217 | 1.499% | $0.34 | 29.990% | **0** | 5.0% | $20.00 | 0 |
| `random` (intrabar) | 67 | 1.499% | $0.34 | 29.982% | **0** | 5.0% | $20.11 | 0 |
| `rsi-reversion` (slot) | 321 | 1.500% | $0.49 | 29.991% | **0** | 5.0% | $25.98 | 0 |
| `rsi-reversion` (intrabar) | 112 | 1.499% | $0.34 | 29.990% | **0** | 5.0% | $20.00 | 0 |

Risk is `size × stop` as a share of the equity at that slot (cap 2%); size is the
notional against 30% of that slot's equity; "past any cap" counts entries that breached
risk, size, stop or the venue floor. Three `trend` entries sit 1–3 **tenths of a cent**
past 30% of the recorded equity (`$34.90` against `$116.33` at one of them): the agent
sizes from its own equity read (`$34.90 / 0.30 = $116.3333`), the harness records 2-dp
equity, and the gap is that rounding — not a cap being ridden. `random`'s smallest
entry is exactly **$20.00**, the venue floor, which is the floor acting as the lower
bound while the exposure cap acts as the upper one.

And the $97 tradability claim, computed from the venue helpers rather than typed:

```
SUPPORTED:  at $97.00 the account can place legal orders on ETH/USDT:USDT (venue
            floor $20.00, ceiling $29.10). It cannot on BTC/USDT:USDT, whose $50.00
            venue floor is above the $29.10 a position may reach at this equity
            (that pair needs about $166.67). Every entry respects the risk and
            exposure caps, fees are charged on both sides, stops and targets fire
            at the declared geometry, the drawdown halt engages and then releases or
            re-arms on its two defined routes after a served cool-off, and the same
            code produces the same numbers on every rerun.
```

### How long a rerun takes

One `--source` process = three passes over the span (continuous + folds + regimes) =
30,363 slot replays. Measured on this machine, **six processes in parallel: 51–55
minutes per source** (~87 ms per slot — the six ran 17:36:04 → 18:27–18:30 on
2026-09-25), **~21 minutes** for a continuous-window-only run, and about **five hours**
for the canonical six-run suite in a single process (the sources share one indicator
memo there, so it is a little less than six solo runs). Splitting by source across
processes produces the same numbers about six times faster — the runs are independent
and share nothing but the read-only parquet files.

---

## The forward record — and why it is not evidence for $97

The only real trading evidence that existed before this pass is the cloud paper
account, and it was produced on a **legacy $10,000 book**, not a $97 one.

| | |
|---|---|
| Wakeups | 118 (2026-09-12T18:17Z → 2026-09-25T08:01Z, 12.6 days) |
| Status | 114 `success`, **4 `error`** (all model-provider failures: three 503 "high demand", one 404 for a retired model) |
| Actions | 6 proposed → 6 approved → 6 executed |
| Round trips | 6 — 3 take-profit, 3 stop-loss |
| Net | **+$49.02** (fees $3.64), $10,000.00 → $10,048.52 (+0.49%), peak $10,083.05 |

Those 6 trades were **$499–$802 of notional each** (5–8% of a $10,000 account). At
$97 the same percentages are $4.85–$7.76 — below ETH's $20 venue floor — and the
account's ceiling is $29.10 while BTC's floor alone is $50. So the forward sample:

1. describes an account scale the $97 configuration **cannot reproduce**, and
2. is six trades. Six round trips cannot estimate a win rate, and three of them won
   only because a 7.5% target was reached inside a 4-day hold.

**Whether the model still sizes correctly at $97 is NOT MEASURED.** It is told the
floor in its risk context, and the first wakeups on the $97 config are the only thing
that can answer it. That is a forward question, not a backtest question.

---

## How to reproduce the numbers exactly

The replay is deterministic: same command, same data, same seed → same bytes.

```
$ python scripts/ai_backtest.py --source random,trend --seed 7 \
    --start 2025-01-01 --end 2025-03-01 --no-folds --no-regimes \
    --json det1.json   # run twice
$ sha256sum det1.json det2.json
f77ec22b6737edf43365370b05c2f977e4eb8ac05cde75477f8ae8501bc8487d  det1.json
f77ec22b6737edf43365370b05c2f977e4eb8ac05cde75477f8ae8501bc8487d  det2.json
```

(The hash changes when behaviour changes — it moved in the pass that made the halt
re-arm, which is why this example is worth re-running rather than quoting.)

`random` is seeded (`--seed`); the other sources are functions of the candles alone.
Fresh state per window is enforced by construction: each window gets a new ledger,
a new strategy file and a new journal directory, so nothing carries a peak equity —
and therefore a drawdown halt — in from another window.

---

## What would change the answer

| Missing | Why it matters | Cost to close |
|---|---|---|
| Forward wakeups on the $97 config | The only way to observe the model's sizing and judgement at this scale | Free — it runs on schedule; 14:00Z is the first slot after the $97 commit |
| Slippage and spread | Paper fills at the decision-time price; live fills at market | Model it, or measure the first live fills |
| Funding | The ledger does not charge it; the report quantifies the omission from stored rates | Already quantified per run, not charged |
| Testnet rehearsal | The live order path has still run **zero** times | Needs testnet keys (see `AI_BOT_PRE_LIVE.md`, R-items) |
| A bigger sample | 6 round trips is not a sample | Time, or more pairs the account can afford |
| What to do about the $66.67 venue floor | Below that equity no legal size exists on any pair the bot trades, so a ~32% drawdown parks the account for good — the halt now re-arms, the venue does not | A product decision: more equity, a different pair, or accept that a $97 account has a hard ~32% floor |

---

## Tests that lock this down

`tests/test_ai_backtest.py` (44 tests) — the tests check the things that could make
the harness *lie*: the historical surface never serves a candle after its cursor and
refuses a timeframe it does not hold; the order path raises; the venue floor, risk
cap, exposure cap, lot step and two-sided fees are enforced on every executed entry
(measured); two full-size proposals cannot both open; the slot model fills at the
observed price including a gap through the stop; the intrabar model catches a stop the
slot model misses; the same seed replays identically and the indicator memo changes
nothing; the venue verdict is computed from the config rather than typed; the report
labels its sources, its synthetic runs and its own limits. Plus the agent-side tests
that the journal now records what the model proposed (`tests/test_ai_agent.py`).

`tests/test_ai_dd_rearm.py` (15 tests) pins the halt lifecycle itself, on the
production path and **without the harness** — a real `AIAgent.run_wakeup()`, a
simulated clock and a price series, nothing else. It pins that a cool-off must be
*served* (`test_a_recovery_above_the_line_does_not_shorten_the_cool_off`) and that a
price recovery alone is not a release (`test_the_release_is_not_a_price_recovery`);
that a calm wakeup re-arms nothing (`test_a_calm_wakeup_re_arms_nothing`) and a
restart cannot cancel a cool-off (`test_the_cooldown_survives_a_restart`); that a
spent budget bounds repeats **without ever deadlocking**
(`test_the_budget_bounds_repeats_without_ever_deadlocking`) and a dead account comes
back (`test_a_dead_account_comes_back`); that the live peak follows new highs and
still trips (`test_the_live_peak_tracks_new_highs_and_still_trips`); and — the point
of the whole pass — that the caps are **not** weakened to allow re-entry:
`test_the_size_cap_still_binds`, `test_the_stop_and_risk_caps_still_bind`,
`test_the_venue_floor_still_binds`, `test_the_heat_cap_still_binds_across_entries`.
The pre-fix permanence stays pinned as the regression it is:
`test_a_halt_with_no_release_is_permanent`.

`scripts/ai_backtest.py` and this whole module are linted in CI
(`.github/workflows/bot.yml`) and run in the AI bot workflow
(`.github/workflows/ai_bot.yml`).
