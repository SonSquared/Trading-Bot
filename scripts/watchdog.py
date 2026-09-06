"""
Production watchdog: surfaces anomalies before they become losses.

Checks (all pure functions over files/args, trivially testable):
  1. Missed runs          - run_history.jsonl heartbeat too old vs schedule
  2. Stale state          - paper_state.json missing/corrupt/not updated
  3. Ledger desync        - positions in state without a matching recent OPEN
  4. Desync cleanup storm - repeated "(ledger desync cleared)" events
  5. Duplicate opens      - same pair+side opened twice inside the guard window
  6. Marker age           - marked equity older than the last successful run

Exit codes: 0 = healthy (or warnings below threshold), 1 = anomalies found.
Sends one Telegram message when anything is found, plus a periodic
heartbeat OK so silence itself is diagnostic.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, ".")

RESULTS = Path("data/results")
RUN_LOG = RESULTS / "run_history.jsonl"
STATE_FILE = RESULTS / "paper_state.json"
TRADE_LOG = RESULTS / "paper_trades.jsonl"

# The scheduled cadence in bot.yml is every 15 minutes.
EXPECTED_RUN_INTERVAL_MINUTES = float(os.getenv("WATCHDOG_RUN_INTERVAL_MINUTES", "15"))
# How late a heartbeat may run before alerting (GH runners queue; be lenient)
MISSED_RUN_FACTOR = float(os.getenv("WATCHDOG_MISSED_RUN_FACTOR", "6"))
DUPLICATE_WINDOW_MINUTES = 60
DESYNC_STORM_THRESHOLD = 3          # cleanup events within the lookback window
DESYNC_STORM_LOOKBACK_HOURS = 48
HEARTBEAT_EVERY_HOURS = 12          # send an OK heartbeat at most this often

OK_STAMP = RESULTS / "watchdog_last_ok.json"


def _iso(s: str) -> datetime | None:
    """Parse an ISO timestamp; naive values are treated as UTC."""
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


def read_jsonl(path: Path, limit: int = 500) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_missed_runs(now: datetime, run_log: Path = RUN_LOG,
                      interval_min: float = EXPECTED_RUN_INTERVAL_MINUTES,
                      factor: float = MISSED_RUN_FACTOR) -> str | None:
    """Alert when the last bot run is older than interval * factor."""
    runs = read_jsonl(run_log, limit=5)
    if not runs:
        return "No run history at all — the bot has never completed a run"
    last = _iso(runs[-1].get("timestamp", ""))
    if last is None:
        return "Latest run-history entry has no parseable timestamp"
    age_min = (now - last).total_seconds() / 60.0
    allowed = interval_min * factor
    if age_min > allowed:
        hours = age_min / 60.0
        return (f"Missed runs: last bot run was {hours:.1f}h ago "
                f"(expected every ~{interval_min:.0f}m). Scheduler may be dead.")
    return None


def check_stale_state(now: datetime, state_file: Path = STATE_FILE) -> str | None:
    """Alert when state is missing, corrupt, or not written by the last run."""
    if not state_file.exists():
        return "paper_state.json MISSING — bot would reset to $97 and re-open positions"
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return f"paper_state.json unreadable ({e}) — corrupt state risks a reset"
    last_run = _iso(state.get("last_run_at", "") or "")
    if last_run is None:
        return "State file has no last_run_at — written by an old code version"
    age_h = (now - last_run).total_seconds() / 3600.0
    if age_h > 24:
        return f"State not updated for {age_h:.0f}h — runs are failing before save"
    return None


def check_ledger_desync(now: datetime, trade_log: Path = TRADE_LOG,
                        state_file: Path = STATE_FILE) -> str | None:
    """Alert when the state holds a position whose OPEN was never logged."""
    del now  # uniform signature; this check is order-free
    if not trade_log.exists() or not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    positions = state.get("positions", {})
    if not positions:
        return None
    logged: set[tuple[str, str]] = set()
    for t in read_jsonl(trade_log, limit=500):
        action = str(t.get("action", ""))
        if action.startswith("OPEN"):
            logged.add((str(t.get("pair", "")), action))
    missing = [p for p in positions
               if (p, "OPEN_LONG") not in logged and (p, "OPEN_SHORT") not in logged]
    if missing:
        return (f"Ledger desync: {len(missing)} position(s) in state with no "
                f"logged OPEN: {', '.join(sorted(missing))}")
    return None


def check_desync_storm(now: datetime, trade_log: Path = TRADE_LOG,
                       threshold: int = DESYNC_STORM_THRESHOLD,
                       lookback_h: float = DESYNC_STORM_LOOKBACK_HOURS) -> str | None:
    """Alert when the close path keeps hitting '(ledger desync cleared)'."""
    if not trade_log.exists():
        return None
    count = 0
    for t in read_jsonl(trade_log, limit=1000):
        if "ledger desync cleared" in str(t.get("reason", "")):
            ts = _iso(t.get("timestamp", ""))
            if ts and (now - ts) <= timedelta(hours=lookback_h):
                count += 1
    if count >= threshold:
        return (f"Desync cleanup storm: {count} ledger-desync clears in "
                f"{lookback_h:.0f}h — the close path is fighting the exchange")
    return None


def check_duplicate_opens(now: datetime, trade_log: Path = TRADE_LOG,
                          window_min: float = DUPLICATE_WINDOW_MINUTES) -> str | None:
    """Alert on the Sep-3 signature: same pair+side opened twice within minutes."""
    if not trade_log.exists():
        return None
    opens: list[tuple[str, str, datetime]] = []
    for t in read_jsonl(trade_log, limit=1000):
        action = str(t.get("action", ""))
        if (action.startswith("OPEN")
                and "paper-state-consistency" not in str(t.get("note", ""))):
            ts = _iso(t.get("timestamp", ""))
            if ts and (now - ts) <= timedelta(days=7):
                opens.append((str(t.get("pair", "")), action, ts))
    seen: dict[tuple[str, str], list[datetime]] = {}
    for pair, action, ts in opens:
        seen.setdefault((pair, action), []).append(ts)
    for (pair, action), times in sorted(seen.items()):
        times.sort()
        for a, b in zip(times, times[1:]):
            if (b - a) <= timedelta(minutes=window_min):
                mins = (b - a).total_seconds() / 60.0
                return (f"Duplicate-open signature: {pair} {action} opened twice "
                        f"within {mins:.0f}min — concurrency guard may be bypassed")
    return None


def check_marker_age(now: datetime, state_file: Path = STATE_FILE,
                     run_log: Path = RUN_LOG) -> str | None:
    """Alert when marked equity is far older than the last successful run."""
    if not state_file.exists() or not run_log.exists():
        return None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    marked = _iso(state.get("last_run_at", "") or "")
    if marked is None:
        return None
    runs = read_jsonl(run_log, limit=5)
    if not runs:
        return None
    last_ok = next((_iso(r.get("timestamp", "")) for r in reversed(runs)
                    if r.get("status") in ("success", "partial")), None)
    if last_ok and (last_ok - marked) > timedelta(hours=6):
        return "Marked equity is stale while runs succeeded — save path regressed?"
    return None


def check_equity_consistency(now: datetime, state_file: Path = STATE_FILE) -> str | None:
    """Alert when realized P&L is wildly inconsistent with cash-based equity.

    With all costs (entry/exit fees, funding) booked into total_pnl, the
    identity is: equity == cash (flat) and total_pnl == equity - start
    when flat. A large gap means some cost is being charged to cash but
    hidden from the reported P&L — the "equity vs P&L mismatch" bug class.
    Tolerances are loose to tolerate mark-to-market (positions valued at
    live prices while total_pnl is realized-only).
    """
    if not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    cash = float(state.get("cash", 0.0))
    equity = float(state.get("last_equity_marked", cash) or cash)
    pnl = float(state.get("total_pnl", 0.0) or 0.0)
    start = float(state.get("initial_capital", 0.0) or 0.0)
    if start <= 0:
        start = 97.0
    # When FLAT, the identity is exact: equity == cash and pnl == eq - start.
    if not state.get("positions"):
        if abs(equity - cash) > 0.01:
            return (f"Flat-account inconsistency: equity ${equity:,.2f} != cash "
                    f"${cash:,.2f} (state may be corrupt or partially written)")
        gap = equity - start - pnl
        if abs(gap) > max(0.05, abs(equity - start) * 0.05):
            return (f"Equity-vs-P&L mismatch while flat: equity-start "
                    f"${equity - start:+,.2f} but reported P&L ${pnl:+,.2f} "
                    f"(gap ${gap:+,.2f}). A cost is charged to cash but hidden "
                    f"from reported P&L.")
    else:
        # With open positions, only sanity-check that reported realized P&L
        # does not exceed the account's total gain by more than the total
        # position notional (would mean costs hidden from P&L or double-count).
        open_notional = sum(float(p.get("size_usd", 0) or 0)
                            for p in state["positions"].values())
        gap = (equity - start) - pnl
        if abs(gap) > open_notional + max(0.05, abs(equity - start) * 0.05):
            return (f"Equity-vs-P&L gap ${gap:+,.2f} exceeds total open notional "
                    f"${open_notional:,.2f} — accounting is inconsistent.")
    return None


ALL_CHECKS = [
    ("missed_runs", check_missed_runs),
    ("stale_state", check_stale_state),
    ("ledger_desync", check_ledger_desync),
    ("desync_storm", check_desync_storm),
    ("duplicate_opens", check_duplicate_opens),
    ("marker_age", check_marker_age),
    ("equity_consistency", check_equity_consistency),
]


def run_all_checks(now: datetime | None = None) -> list[str]:
    """Run every check; one broken check must never hide the others."""
    now = now or datetime.now(timezone.utc)
    findings: list[str] = []
    for _name, fn in ALL_CHECKS:
        try:
            msg = fn(now)
        except Exception as e:
            msg = f"watchdog check {fn.__name__} itself failed: {e}"
        if msg:
            findings.append(msg)
    return findings


# --------------------------------------------------------------------------
# Telegram + CLI
# --------------------------------------------------------------------------

def send_telegram(token: str, chat_id: str, text: str) -> bool:
    from scripts.paper_trader import tg_send_message
    return tg_send_message(token, chat_id, text)


def should_send_heartbeat(now: datetime,
                          stamp_file: Path = OK_STAMP) -> bool:
    """Send an OK heartbeat at most once per HEARTBEAT_EVERY_HOURS."""
    try:
        last = json.loads(stamp_file.read_text(encoding="utf-8"))
        ts = _iso(last.get("ok_at", ""))
        if ts and (now - ts) < timedelta(hours=HEARTBEAT_EVERY_HOURS):
            return False
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    return True


def record_heartbeat(now: datetime, stamp_file: Path = OK_STAMP) -> None:
    stamp_file.parent.mkdir(parents=True, exist_ok=True)
    stamp_file.write_text(json.dumps({"ok_at": now.isoformat()}), encoding="utf-8")


def _safe_print(text: str) -> None:
    """Console print that survives codepages without emoji (Windows cp1252)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def main() -> int:
    now = datetime.now(timezone.utc)
    findings = run_all_checks(now)

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"

    if findings:
        msg = ("🚨 <b>WATCHDOG ALERT</b>\n"
               f"{now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
               + "\n".join(f"• {f}" for f in findings))
        _safe_print(msg)
        if token and chat_id and enabled:
            try:
                send_telegram(token, chat_id, msg)
            except Exception as e:
                _safe_print(f"Telegram send failed: {e}")
        return 1

    _safe_print("Watchdog: all checks passed.")
    if token and chat_id and enabled and should_send_heartbeat(now):
        try:
            send_telegram(token, chat_id,
                          "🐕 Watchdog heartbeat: all checks passed.")
            record_heartbeat(now)
        except Exception as e:
            _safe_print(f"Telegram heartbeat failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
