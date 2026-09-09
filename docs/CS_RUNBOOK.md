# crypto_system Runbook

Operator documentation for the audited platform in `src/crypto_system`.

> **NO PROFIT GUARANTEE.** This system is built for bounded risk and
> reproducible evidence. It does not promise profit; it can lose money,
> including all capital deployed to it. Every automation target in this
> document is paper trading.

---

## 1. What this platform is

A second, strictly-audited trading platform alongside the legacy paper bot:

| Layer | Path | Role |
|---|---|---|
| Models/config | `src/crypto_system/models.py`, `config.py` | Immutable schemas; risk caps enforced at construction (leverage ≤ 50x, per-trade risk ≤ 0.5%) |
| Audit | `audit/` | Hash-chained append-only ledger; state is a *cache*, ledger is truth |
| Data | `data/` | Quality gates (reject gaps/dupes/skew/impossible prices), point-in-time catalogue |
| Execution | `execution/` | Deterministic seeded paper fills (spread, impact, latency, partials, funding, margin, stops) |
| Risk | `risk/` | Governor is the ONLY authorizer; regime scaling bounded [0,1]; latching kill switch |
| Research | `research/` | Purged walk-forward, bootstrap stats, league with hard gates, veto, quarantine |
| Live | `live/` | Human-approved, reconciled transport — **disabled by default** |
| Reporting | `reporting/` | Redacted, read-only; Telegram is report-only and dry-run by default |

**Paper and live state never share a write path.** The live service is
unconstructible unless `CS_MODE=live` **and** `CS_LIVE_ENABLED=true`, and
every live order additionally requires a human-approved, unexpired,
single-use approval bound to an immutable intent digest.

## 2. Daily operation (paper)

Nothing is required. The scheduled workflows do the work:

| Workflow | Cadence | What it does | What it must never do |
|---|---|---|---|
| `cs_ci.yml` | push/PR touching cs paths | full suite + ruff + mypy + quick run | — |
| `cs_weekly_report.yml` | Mondays 06:13 UTC | render + print report (**report-only**) | promote, send, touch live |
| `cs_monthly_optimize.yml` | 1st of month 05:37 UTC | ledger verify → league → promote-if-gates-pass (paper) | enable live, push legacy bot state |

These crons are deliberately unaligned with the legacy bot's crons
(`:23`/`:47` even hours) so they never compete for scheduler slots.

## 3. Running on your PC

```bash
# Deterministic, offline, zero side effects — safe to run anytime:
python scripts/improve.py --quick --report-only

# Full run (requires paper state on disk; verifies ledger first):
python scripts/improve.py
```

Quick mode replays fixtures only: no network (test-pinned with a socket
tripwire), no promotion, no state push, no Telegram send.

## 4. Configuration & secrets

- Copy `.env.example` to `.env` (git-ignored). Fill only what you use.
- Secrets are `SecretStr`-contained: they never appear in repr, dumps,
  reports, or ledger records (redaction is applied at append and render).
- Extra/typo'd `CS_*` env keys are **rejected** at load — a mistyped
  variable fails loudly instead of silently defaulting.
- If you ever (deliberately, knowingly) enable live mode: keys MUST be
  trading-only (no withdrawal/transfer permissions), testnet first.

## 5. The self-improvement loop

1. **Data** passes quality gates (corruption is rejected, staleness flagged).
2. **League** evaluates candidates with purged, embargoed walk-forward;
   parameter search is confined to training folds; OOS scoring happens once.
3. **Gates** require: enough independent OOS trades, positive mean AND
   median, fold consistency, bounded worst fold (tail control), bounded
   drawdown, sharpe floor.
4. **Veto**: if nothing passes, nothing is promoted — the least-bad
   candidate is never crowned.
5. **Scoreboard**: a live incumbent whose closed-trade record persistently
   underperforms is quarantined; its slot opens at the NEXT monthly league
   only.
6. **Ledger**: every decision and result is hash-chained; tampering,
   deletion, or reordering is detectable; recovery replays the ledger.

The cadence is monthly for promotion — weekly runs are report-only. This
asymmetry is deliberate: reaction speed is not an edge; evidence is.

## 6. Halts and recovery

- Halts latch: daily loss beyond 3% of equity, drawdown beyond 10%, data
  faults, ledger faults, reconciliation mismatches, order ambiguity.
- A latched halt blocks ALL new risk until a human acknowledges with an
  operator identity (`KillSwitch.acknowledge(code, operator=...)`).
- Post-submit timeout = order MAY be live: the `order_ambiguous` halt is
  tripped and the order is **never** retried automatically. Recovery:
  reconcile exchange positions, resolve by hand, acknowledge.
- Ledger verification failure blocks everything (fail closed). A crash-
  truncated tail can be repaired with `Ledger.recover_to_consistent_tail()`;
  anything else is investigated, never auto-"fixed".

## 7. Rollback

Promotion is a paper-state change recorded in the ledger. Rollback =
replay the ledger to the last pre-promotion sequence and restore the
previous paper config; the hash chain proves exactly which records
post-date the decision being rolled back.

## 8. Testnet smoke (hand-held only)

The signed Binance transport (`execution/binance_live.py`) is testnet-only
by construction; production endpoints require `allow_production=True`, and
no repository code path ever sets it. The one and only way the transport is
constructed is the smoke command:

```bash
CS_SMOKE_TESTNET_CONFIRM=yes CS_BINANCE_API_KEY=... CS_BINANCE_API_SECRET=... \
  python scripts/cs_smoke_testnet.py
```

It exercises signing, a client-ID'd market order, ack handling, and
reconciliation, ledgering every step. On a lost ack it returns exit 3 with
an `order_ambiguous`-style refusal to retry. Nothing schedules it; no
workflow invokes it; credentials never leave the environment. Graduate to
production keys only after repeated clean testnet smokes — and never give
those keys withdrawal/transfer permissions.

## 9. Lock conflicts

The improvement runner holds a TTL lease (`improve.lock`). A second runner
fails closed with `LockHeld`; an expired lease (dead runner) can be taken
over. Never run two improvement jobs against the same state directory.

## 10. Known limitations (honest list)

- Execution realism is modeled (spread, impact, latency, partials, funding,
  margin), but real venue micro-structure can still surprise; the live
  adapter exists and is tested against mocks only — treat testnet smoke as
  the first real gate before any production key is ever involved.
- Backtest edges decay; the league's veto and quarantine exist because
  most candidate strategies SHOULD fail. A veto month is normal, not a bug.
- The legacy Kraken paper bot (`scripts/paper_trader.py`, `bot.yml`) runs
  independently; this platform does not read or write its state.
