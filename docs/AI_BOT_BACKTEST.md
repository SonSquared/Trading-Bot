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
commit `844f086`. Every number below came out of that harness on real BTC/ETH 1h
candles; none of them was typed in by hand, and the ones the report generates
(including the $97 tradability verdict) are computed from the config and the venue
helpers at run time.

---

## The one-line version

> The bookkeeping is provably correct: at $97 every legal ETH order is allowed, every
> illegal one is refused with its arithmetic, fees are charged twice, stops and
> targets fire at the declared geometry, and the same command always produces the
> same bytes. **Nothing here shows the bot makes money** — and the only real forward
> evidence (6 round trips, +$49 on a legacy $10,000 paper account) was produced at a
> position size the $97 account can no longer place.
>
> The loudest thing in the results is not about a rule at all: **every mechanical
> source stopped trading in 2022 and never resumed**, because the 10% drawdown halt
> is measured against an all-time peak that only rises. See
> [the halt finding](#the-finding-that-matters-most-the-halt-ends-the-experiment).

| Question | Answer |
|---|---|
| Can this account place legal orders at all? | **Yes on ETH, no on BTC** — measured, not asserted |
| Are the caps, fees, stops and halt enforced on every entry? | **Yes** — checked on each executed entry, not assumed |
| Is there look-ahead in the replay? | **No violations over the whole span** — counted, not promised |
| Does it reproduce? | **Byte-identical** — same command, same SHA-256 |
| Does the halt matter more than the entries? | **Yes** — it ended all three trading runs in 2022 |
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
| Risk 2%/trade, 30% position, 30% heat, ≤5% stop, R:R ≥ 1.5, 3 positions, 10% DD halt | `configs/ai_bot.yaml`, enforced in `_validate_decision` / `_risk_snapshot` |
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

Span: 2022-01-08 → 2026-08-22. Return is on a $97 base, so +6.71% is +$6.51.

| Source | Round trips | Win | Net $ | Return | Max DD | Fees | Expectancy/trade |
|---|---|---|---|---|---|---|---|
| `null` | 0 | — | +0.00 | +0.00% | 0.00% | 0.0000 | — |
| `random` (seed 7) | 28 | 46% | +6.51 | **+6.71%** | 11.21% | 0.8632 | +0.2323 |
| `trend` | 14 | 21% | −6.43 | **−6.62%** | 10.37% | 0.3817 | −0.4590 |
| `rsi-reversion` | 60 | 60% | +2.82 | **+2.90%** | 11.24% | 1.7650 | +0.0469 |

`null` is the control and it behaved: 0 trades, 0 fees, $97.00 → $97.00 through
10,121 wakeups and 8,096,604 candles fetched. Nothing invented a trade on its own.

### The finding that matters most: the halt ends the experiment

**Every trading source made its last trade in 2022.** The `trend` rule's last action
was **2022-02-04**, `random`'s was 2022-04-27, `rsi-reversion`'s was 2022-07-18 —
and then the account sat idle for the remaining four years.

The reason is the 10% drawdown halt, and it is worth stating precisely because it is
easy to misread: the halt is measured against the **all-time peak equity, and the peak
only ever rises**. Once the account is ≥10% below its peak it refuses new entries
until equity climbs back above 90% of that peak. The `trend` account ended at $90.57
against a peak of about $101, so it was **still halted four and a half years later**,
having refused **12,588 entries in that window alone** (30,542 across all ten of its
windows).

So the continuous column is not "four and a half years of a rule trading". It is "a
few weeks in 2022, then the guard doing its job" — which is why all three rows above
are numerically **identical to their fold 1 and their 2022-bear regime**. The
fresh-$97 folds and regimes exist exactly so each period can be seen rather than
hidden behind one inherited peak.

This is a property of the shipped guard, not a bug in the replay, and it is not a
reason to loosen the guard. It does mean that at $97 a single bad month can end the
account's trading for a long time, and that fact deserves its own look before any
real money is involved.

### Walk-forward folds — four fresh $97 accounts

| Source | fold 1 (22-01→23-03) | fold 2 (23-03→24-04) | fold 3 (24-04→25-06) | fold 4 (25-06→26-08) | net | positive |
|---|---|---|---|---|---|---|
| `random` | +6.71% | −6.82% | −8.40% | −5.57% | −$13.66 | 1/4 |
| `trend` | −6.62% | +0.37% | −6.91% | +3.45% | −$9.42 | 2/4 |
| `rsi-reversion` | +2.90% | +8.79% | +21.24% | −0.20% | +$30.80 | 3/4 |

### Regimes — five fresh $97 accounts, no carry-over

| Source | 2022 bear | 2023 recovery | 2024 bull | 2025 mixed | 2026 YTD | net | positive | spread |
|---|---|---|---|---|---|---|---|---|
| `random` | +6.71% | +28.28% | −4.69% | −10.13% | −10.40% | +$9.08 | 2/5 | **38.68 pp** |
| `trend` | −6.62% | +3.60% | −3.94% | −6.22% | +1.91% | −$10.94 | 2/5 | 10.22 pp |
| `rsi-reversion` | +2.90% | −3.89% | +10.88% | +10.54% | −6.69% | +$12.45 | 3/5 | 17.57 pp |

### The fill model changes the sign

The bot's paper mode fills a stop at the price the *wakeup observes*; a real exchange
fills it the moment the level trades. Those two are not the same thing, so the
harness can run either, and the difference is not a rounding error:

| Source | Slot model (the bot's paper behaviour) | Intrabar model (a labelled sensitivity) |
|---|---|---|
| `trend` | −6.62%, 14 trades, 21% win | −10.32%, 10 trades, 10% win |
| `rsi-reversion` | **+2.90%**, 60 trades, 60% win | **−7.88%**, 20 trades, 30% win |

The sign of the `rsi-reversion` result depends entirely on which assumption you pick.
That alone is enough to know that none of these numbers should be used to choose
anything — and it is the honest reason the verdict below refuses to grade a strategy.

### What the dispersion says (the statistical part)

* **The coin flip had the widest spread of all** — 38.68 percentage points across five
  regimes, against 10.22 pp for `trend` and 17.57 pp for `rsi-reversion`. A source with
  no edge at all produced a range that contains every result the rules produced.
* Expectancy per trade was +$0.23 (`random`, 28 trades), −$0.46 (`trend`, 14) and
  +$0.05 (`rsi-reversion`, 60). Taking 5% stops at a ~$29 notional, one trade swings
  about ±$1.50, so the standard error on 60 trades is roughly ±$0.19 — several times
  the whole estimated edge of the best rule.
* Fees are small in absolute terms at this size ($0.38–$1.77 over the entire span)
  but they are a large share of gross where the rule actually trades: for
  `rsi-reversion`, $1.77 of $4.59 gross — **39%**. A rule with no real edge cannot pay
  for itself here, which is the one thing the `random` baseline demonstrates cleanly.

### The mechanics are enforced, not assumed

Counters from every run (per source, then four sources):

| | |
|---|---|
| Slots replayed | 30,363 per source — 121,452 in total |
| Distinct candle windows | 60,744 per source — 242,976 in total |
| Candles served (agent + engine) | 24,289,812 per source — 97,159,248 in total |
| **Look-ahead violations** | **0** — a candle after the cursor was served, counted at the exchange |
| **Order-path attempts** | **0** — every place/cancel call raises `OrderPathTouched` |
| Peak observed exposure | 30.4–30.6% of equity (the 30% cap binds at entry; equity moves after it) |
| Fees | charged both sides at 0.05% taker on every closed trade |
| Refusals, bucketed | drawdown halt, below the pair's `MIN_NOTIONAL`, "close requested with no position" (19 in `trend`, 32 in `rsi-reversion` — the sources' own sloppiness caught, not acted on) |
| BTC entries refused | 3,678 (`trend`) with the arithmetic: *venue minimum $50.00 … exceeds the $29.10 a position may reach at $97.00 equity … needs ~$166.67* |

And the $97 tradability claim, computed from the venue helpers rather than typed:

```
SUPPORTED:  at $97.00 the account can place legal orders on ETH/USDT:USDT (venue
            floor $20.00, ceiling $29.10). It cannot on BTC/USDT:USDT, whose $50.00
            venue floor is above the $29.10 a position may reach at this equity
            (that pair needs about $166.67). Every entry respects the risk and
            exposure caps, fees are charged on both sides, stops and targets fire
            at the declared geometry, the drawdown halt engages, and the same code
            produces the same numbers on every rerun.
```

### How long a rerun takes

One `--source` process = three passes over the span (continuous + folds + regimes) =
30,363 slot replays. Measured on this machine, four to six processes in parallel:
**~44 minutes per source** (~87 ms per slot), **~21 minutes** for a
continuous-window-only run, and nearly **three hours** for the canonical four-source
suite in a single process (the sources share one indicator memo there, so it is a
little less than four solo runs). Splitting by source across processes produces the
same numbers about four times faster — the runs are independent and share nothing but
the read-only parquet files.

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
1bfe7e493943b75687742835f6e34935244196260de9f22625239aaefe0ecebd  det1.json
1bfe7e493943b75687742835f6e34935244196260de9f22625239aaefe0ecebd  det2.json
```

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
| How far the halt should reach | As shipped, one 10% drawdown from an all-time peak can stop entries indefinitely — at $97 that is a $9.70 loss ending the experiment | A product decision, not a backtest: decide whether the halt should decay, reset on a schedule, or be measured from a rolling peak |

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

`scripts/ai_backtest.py` and this whole module are linted in CI
(`.github/workflows/bot.yml`) and run in the AI bot workflow
(`.github/workflows/ai_bot.yml`).
