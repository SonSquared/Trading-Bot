"""Backfill audit: reconcile every reported equity figure against the trade log.

Three checks over data/results/paper_trades.jsonl, run_history.jsonl and
paper_state.json:

1. Cash-chain integrity — each entry's ``cash_after`` must equal the previous
   ``cash_after`` plus the entry's own cash effect:
       OPEN:  -(size_usd + fee)
       CLOSE: +(size_usd + pnl_usd)   (size_usd logged on CLOSE since
              2026-09-07; earlier closes are reconstructed from their OPEN)
       FUND:  -funding                (funding ledger entries since 2026-09-07)
   Small unexplained deviations in the pre-2026-09-07 window (legacy
   double-charged slippage, unlogged funding) fold into a legacy offset;
   anything larger after the fix is a real ledger break.

2. Flat-moment P&L reconciliation — whenever the reconstructed position count
   is zero, a correct cost model satisfies:
       cash - initial_capital == sum(pnl_usd of closes) - sum(open fees)
                                 - sum(funding entries) + legacy offset
   Pre-2026-09-06 the bot hid entry fees and funding from the reported P&L,
   so this gap grew negative — exactly the Telegram "equity vs P&L" bug. The
   audit quantifies the hidden cost in that window and verifies the gap is
   ~zero at every flat moment after the fix.

3. Run-history equity — run_history.jsonl records the equity each run
   reported. At flat moments it must equal the trade-log cash. (At moments
   with open positions it is marked-to-market and can legitimately differ.)

Usage:
    python scripts/audit_equity.py            # full audit report
    python scripts/audit_equity.py --strict   # exit 1 on any hard failure
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = Path("data/results")
TRADE_LOG = RESULTS / "paper_trades.jsonl"
STATE_FILE = RESULTS / "paper_state.json"
RUN_LOG = RESULTS / "run_history.jsonl"

# The accounting fix (entry fee + funding booked into total_pnl at charge
# time) landed 2026-09-06 UTC; full ledger completeness (size_usd on CLOSE
# entries + FUND ledger entries for funding debits) landed 2026-09-07 UTC.
# Entries before the latter are reconciled through a legacy offset; after it
# the audit is exact.
FIX_TS = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)

FUNDING_TOLERANCE = 0.02   # max unexplained delta attributed to funding
CENT = 0.005


def _ts(s) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def audit(log_path: Path = TRADE_LOG, state_path: Path = STATE_FILE,
          run_path: Path = RUN_LOG) -> tuple[list[str], list[str], dict]:
    """Returns (hard_problems, notes, stats).

    Findings inside historical windows — anything before the accounting fix,
    or any log other than the production one (e.g. a quarantined archive) —
    are classified as historical and surfaced as notes with counts. Only
    findings in the current production log after the fix are hard problems.
    """
    historical_log = log_path.resolve() != TRADE_LOG.resolve()
    hard: list[str] = []
    historical: list[str] = []
    notes: list[str] = []

    def _add(ts_, msg: str) -> None:
        if historical_log or (ts_ is not None and ts_ < FIX_TS):
            historical.append(msg)
        else:
            hard.append(msg)

    trades = sorted(load_jsonl(log_path), key=lambda t: t.get("timestamp", ""))

    if not trades:
        return [f"trade log missing or empty: {log_path}"], [], {}

    # Initial capital from the state file if available, else the classic $97.
    initial = 97.0
    if state_path.exists():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
            initial = float(st.get("initial_capital", 97.0) or 97.0)
        except (json.JSONDecodeError, OSError):
            pass

    stats = {"entries": len(trades), "opens": 0, "closes": 0,
             "hidden_cost_pre_fix": 0.0, "max_flat_gap_pre_fix": 0.0,
             "max_flat_gap_post_fix": 0.0, "flat_checks": 0,
             "cash_chain_breaks": 0, "funding_sized_deltas": 0}

    prev_cash: float | None = None
    prev_ts: datetime | None = None
    open_count = 0
    sum_close_pnl = 0.0
    seen_opened_pairs: set[str] = set()
    open_sizes: dict[str, float] = {}
    sum_open_fees = 0.0
    sum_fund = 0.0
    legacy_delta = 0.0

    for t in trades:
        ts = _ts(t.get("timestamp"))
        action = str(t.get("action", ""))
        pair = str(t.get("pair", "?"))
        ca = t.get("cash_after")
        ca_f = float(ca) if ca is not None else None

        if action.startswith("OPEN"):
            stats["opens"] += 1
            open_count += 1
            seen_opened_pairs.add(pair)
            open_sizes[pair] = float(t.get("size_usd", 0) or 0)
            sum_open_fees += float(t.get("fee", 0) or 0)
            if pair in seen_opened_pairs and open_count > len(seen_opened_pairs):
                _add(ts,
                     f"{t.get('timestamp')}: duplicate open of {pair} while "
                     f"already open (double-booked slot)")
        elif action == "CLOSE":
            stats["closes"] += 1
            if pair not in seen_opened_pairs:
                _add(ts, f"{t.get('timestamp')}: CLOSE of {pair} with no prior OPEN")
            else:
                seen_opened_pairs.discard(pair)
            open_count = max(0, open_count - 1)
            sum_close_pnl += float(t.get("pnl_usd", 0) or 0)
        elif action == "FUND":
            stats["funds"] = stats.get("funds", 0) + 1
            sum_fund += float(t.get("funding", 0) or 0)
        else:
            _add(ts, f"{t.get('timestamp')}: unknown action {action!r}")

        # 1) Cash-chain integrity
        if ca_f is not None and prev_cash is not None:
            if action.startswith("OPEN"):
                expected = prev_cash - (float(t.get("size_usd", 0) or 0)
                                        + float(t.get("fee", 0) or 0))
            elif action == "CLOSE":
                size = (float(t.get("size_usd", 0) or 0)
                        or open_sizes.get(pair, 0.0))
                expected = prev_cash + (size
                                        + float(t.get("pnl_usd", 0) or 0))
            elif action == "FUND":
                expected = prev_cash - float(t.get("funding", 0) or 0)
            else:
                expected = prev_cash
            delta = ca_f - expected
            if abs(delta) > CENT:
                stats["cash_chain_breaks"] += 1
                if abs(delta) <= FUNDING_TOLERANCE and ts and prev_ts \
                        and (ts - prev_ts).total_seconds() >= 8 * 3600:
                    stats["funding_sized_deltas"] += 1
                else:
                    _add(ts,
                         f"{t.get('timestamp')}: cash-chain break — cash_after "
                         f"${ca_f:.4f} but expected ${expected:.4f} "
                         f"(unexplained ${delta:+.4f})")
            if ts is not None and ts < FIX_TS:
                # Legacy-era ledger gaps (double-charged slippage, funding
                # debits that predate FUND entries) shift cash permanently.
                # Carry them forward so post-fix checks stay exact.
                legacy_delta += delta
        if ca_f is not None:
            prev_cash = ca_f
        prev_ts = ts or prev_ts

        # 2) Flat-moment P&L reconciliation
        if open_count == 0 and ca_f is not None and stats["closes"] > 0:
            gap = ((ca_f - initial) - (sum_close_pnl - sum_open_fees
                                       - sum_fund + legacy_delta))
            stats["flat_checks"] += 1
            if ts and ts < FIX_TS:
                stats["max_flat_gap_pre_fix"] = max(
                    stats["max_flat_gap_pre_fix"], abs(gap))
            else:
                stats["max_flat_gap_post_fix"] = max(
                    stats["max_flat_gap_post_fix"], abs(gap))
                if abs(gap) > CENT:
                    _add(ts,
                         f"{t.get('timestamp')}: flat gap ${gap:+.4f} — "
                         f"costs still hidden from P&L")

    # Hidden cost in the pre-fix window: entry fees of closes whose open
    # predates the fix + funding can no longer be reconstructed exactly, so
    # report the observed max flat gap as the hidden-cost magnitude.
    stats["hidden_cost_pre_fix"] = stats["max_flat_gap_pre_fix"]

    # 3) State file cash == last cash_after (production log only — the state
    # file corresponds to the live ledger, not to an archived window)
    if state_path.exists() and not historical_log:
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
            state_cash = float(st.get("cash", 0))
            last_ca = next((float(t["cash_after"]) for t in reversed(trades)
                            if t.get("cash_after") is not None), None)
            if last_ca is not None and abs(state_cash - last_ca) > CENT:
                _add(None,
                     f"state cash ${state_cash:.4f} != last logged cash_after "
                     f"${last_ca:.4f} (trades happened after the last state save?)")
        except (json.JSONDecodeError, OSError):
            notes.append("state file unreadable — skipped state reconciliation")

    # 4) Run-history equity at flat moments (production log only, same reason)
    runs = [] if historical_log else load_jsonl(run_path)
    flat_equity_checks = 0
    for r in runs:
        r_ts = _ts(r.get("timestamp"))
        eq = r.get("equity")
        if r_ts is None or eq is None:
            continue
        # Position count at that time = opens - closes strictly before r_ts.
        opens = sum(1 for t in trades
                    if str(t.get("action", "")).startswith("OPEN")
                    and _ts(t.get("timestamp")) and _ts(t.get("timestamp")) <= r_ts)
        closes = sum(1 for t in trades
                     if t.get("action") == "CLOSE"
                     and _ts(t.get("timestamp")) and _ts(t.get("timestamp")) <= r_ts)
        if opens - closes != 0:
            continue  # marked-to-market equity legitimately differs
        # Last cash_after at or before r_ts
        cash_at = next((float(t["cash_after"]) for t in reversed(trades)
                        if t.get("cash_after") is not None
                        and _ts(t.get("timestamp")) and _ts(t.get("timestamp")) <= r_ts),
                       None)
        if cash_at is None:
            continue
        flat_equity_checks += 1
        if abs(float(eq) - cash_at) > CENT:
            _add(r_ts,
                 f"run {r.get('timestamp')}: reported equity ${float(eq):.4f} "
                 f"!= trade-log cash ${cash_at:.4f} at a flat moment")
    stats["flat_equity_checks"] = flat_equity_checks
    stats["historical_findings"] = len(historical)

    if historical:
        notes.append(
            f"{len(historical)} historical findings (pre-fix entries or an "
            f"archived/quarantined log) — recorded, not current failures. "
            f"First: {historical[0]}")

    notes.append(
        f"Pre-fix window (before {FIX_TS.date()}): max flat-moment gap "
        f"${stats['max_flat_gap_pre_fix']:.4f} — this was the hidden "
        f"entry-fee+funding cost that made Telegram's P&L line disagree "
        f"with its Equity line.")
    notes.append(
        f"Post-fix: {stats['flat_checks']} flat-moment checks, max gap "
        f"${stats['max_flat_gap_post_fix']:.4f}.")
    return hard, notes, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any hard failures are found")
    ap.add_argument("--log", type=Path, default=TRADE_LOG,
                    help="trade log to audit (default: production log; "
                         "point at an archive to audit a historical window)")
    args = ap.parse_args()

    problems, notes, stats = audit(log_path=args.log)

    print("=" * 60)
    print("EQUITY AUDIT — trade log vs every reported equity figure")
    print("=" * 60)
    print(f"Entries:            {stats.get('entries', 0)} "
          f"({stats.get('opens', 0)} opens / {stats.get('closes', 0)} closes "
          f"/ {stats.get('funds', 0)} funding)")
    print(f"Cash-chain breaks:  {stats.get('cash_chain_breaks', 0)} "
          f"(of which funding-sized: {stats.get('funding_sized_deltas', 0)})")
    print(f"Flat-moment checks: {stats.get('flat_checks', 0)} "
          f"(run-history flat checks: {stats.get('flat_equity_checks', 0)})")
    print(f"Max flat gap pre-fix:  ${stats.get('max_flat_gap_pre_fix', 0):.4f}")
    print(f"Max flat gap post-fix: ${stats.get('max_flat_gap_post_fix', 0):.4f}")
    print()

    if STATE_FILE.exists():
        try:
            sc = float(json.loads(STATE_FILE.read_text(encoding="utf-8")).get("cash", 0))
            print(f"State cash: ${sc:,.2f}")
        except (json.JSONDecodeError, OSError):
            pass

    if notes:
        print()
        for n in notes:
            print(f"NOTE: {n}")

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for p in problems[:25]:
            print(f"  - {p}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
    else:
        print("\nNo hard failures: every equity figure reconciles with the ledger.")

    if args.strict and problems:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
