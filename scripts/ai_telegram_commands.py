#!/usr/bin/env python3
"""Answer Telegram commands for the AI trading bot (report-only).

Two responders share this module's routing and credentials:
  - scripts/ai_poller.py (always-on, launched in 15-min generations by
    ai_poller.yml) long-polls getUpdates and answers within seconds;
  - the scheduled wakeup path (below) answers anything pending at each
    wakeup as a fallback (e.g. while the poller is disabled).
Both dedupe via the shared update offset in data/ai_bot.

Commands (accepted ONLY from AI_TELEGRAM_CHAT_ID):
    /status     equity, open positions, last wakeup outcome
    /positions  open-position detail (entry, SL/TP, unrealized P&L)
    /last       the AI's latest decision and reasoning
    /help       command list

Manual use (e.g. answer pending commands right now):
    python scripts/ai_telegram_commands.py

Reads paper_ledger.json / journal.jsonl from data/ai_bot; live prices
come from ExchangeInterface (Binance futures with the Kraken spot
fallback, so this works from GitHub's geo-blocked runners too).

Exit codes: 0 = fine (including "no pending commands" and "Telegram
not configured" — command answering must never fail a trading run);
1 = real infrastructure error, surfaced loudly in CI logs.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(os.environ.get("AI_BOT_DATA_DIR", "data/ai_bot"))
OFFSET_FILE = DATA_DIR / "telegram_offset.txt"
MAX_MESSAGE_CHARS = 4000  # Telegram hard limit is 4096

WEEK_DAYS = 7


def _creds() -> tuple[str, str]:
    """AI_-prefixed env wins so both bots keep separate credentials."""
    token = os.environ.get("AI_TELEGRAM_BOT_TOKEN") or os.environ.get(
        "TELEGRAM_BOT_TOKEN", ""
    )
    chat_id = os.environ.get("AI_TELEGRAM_CHAT_ID") or os.environ.get(
        "TELEGRAM_CHAT_ID", ""
    )
    return token, chat_id


def _tg(token: str, method: str, **params) -> dict | None:
    """Call the Bot API; None on any failure (never raises)."""
    import requests

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/{method}",
            json=params,
            timeout=25,
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            return data
        if resp.status_code == 409 and method == "getUpdates":
            # The always-on poller (scripts/ai_poller.py) holds the
            # long-poll — commands are answered within seconds by it.
            # This is the healthy steady state, not a warning.
            print("getUpdates: always-on poller active (HTTP 409) — "
                  "it answers commands; nothing to do here.")
            return None
        print(f"WARNING: {method} failed: HTTP {resp.status_code} {data}", file=sys.stderr)
        return None
    except Exception as e:  # noqa: BLE001 — report-only, must not crash the run
        print(f"WARNING: {method} error: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Shared files
# ---------------------------------------------------------------------------

def _load_ledger() -> dict:
    path = DATA_DIR / "paper_ledger.json"
    try:
        return json.loads(path.read_text()) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _last_journal() -> dict | None:
    path = DATA_DIR / "journal.jsonl"
    if not path.exists():
        return None
    try:
        lines = path.read_text().strip().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def _current_prices(pairs: list[str]) -> dict[str, float]:
    """Live prices via ExchangeInterface (Kraken fallback included)."""
    prices: dict[str, float] = {}
    if not pairs:
        return prices
    try:
        from trading_system.bot.exchange import ExchangeInterface
        from trading_system.config import ExchangeConfig

        iface = ExchangeInterface(ExchangeConfig())
        for pair in pairs:
            ticker = iface.get_ticker(pair)
            if ticker.get("last", 0) > 0:
                prices[pair] = float(ticker["last"])
    except Exception as e:  # noqa: BLE001 — fall back to entry prices
        print(f"WARNING: price fetch failed: {e}", file=sys.stderr)
    return prices


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _unrealized(pos: dict, price: float) -> float:
    direction = 1.0 if pos.get("side") in ("buy", "long") else -1.0
    return direction * (price - float(pos["entry_price"])) * float(pos["amount"])


def _positions_lines(ledger: dict, prices: dict[str, float]) -> list[str]:
    lines: list[str] = []
    for pair, pos in ledger.get("positions", {}).items():
        side = "LONG" if pos.get("side") in ("buy", "long") else "SHORT"
        entry = float(pos["entry_price"])
        price = prices.get(pair, entry)
        pnl = _unrealized(pos, price)
        pnl_pct = (pnl / (entry * float(pos["amount"])) * 100) if entry else 0.0
        lines.append(
            f"  {pair} {side}: entry ${entry:,.2f} now ${price:,.2f} "
            f"SL {pos.get('stop_loss_pct', '?')}% TP {pos.get('take_profit_pct', '?')}% "
            f"-> {pnl:+.2f}$ ({pnl_pct:+.2f}%)"
        )
    return lines


def _closed_stats(ledger: dict) -> tuple[int, float, float]:
    """(closed count, win rate %, trailing-week net P&L)."""
    from datetime import datetime, timedelta, timezone

    closed = ledger.get("closed_trades", [])
    wins = sum(1 for t in closed if float(t.get("net_pnl", 0)) > 0)
    wr = (wins / len(closed) * 100) if closed else 0.0

    cutoff = datetime.now(timezone.utc) - timedelta(days=WEEK_DAYS)
    week_pnl = 0.0
    for t in closed:
        ts = t.get("close_time", "")
        try:
            if datetime.fromisoformat(ts.replace("Z", "+00:00")) >= cutoff:
                week_pnl += float(t.get("net_pnl", 0))
        except (ValueError, TypeError):
            continue
    return len(closed), wr, week_pnl


def cmd_status(ledger: dict, journal: dict | None, prices: dict[str, float]) -> str:
    positions = ledger.get("positions", {})
    cash = float(ledger.get("cash", 0))
    unreal = sum(
        _unrealized(pos, prices.get(pair, float(pos["entry_price"])))
        for pair, pos in positions.items()
    )
    equity = cash + unreal
    start = float(ledger.get("start_equity", 0)) or equity
    ret = ((equity - start) / start * 100) if start else 0.0
    closed, wr, week_pnl = _closed_stats(ledger)

    lines = [
        "AI BOT STATUS (paper)",
        f"Equity: ${equity:,.2f}  ({ret:+.2f}% all-time, ${week_pnl:+,.2f} this week)",
        f"Open positions: {len(positions)}",
        *_positions_lines(ledger, prices),
        f"Closed trades: {closed} | Win rate {wr:.0f}%",
    ]
    if journal:
        lines.append(
            f"Last wakeup: {journal.get('wakeup_id', '?')} "
            f"[{journal.get('status', '?')}] outlook={journal.get('market_outlook', '?')} "
            # EXECUTED actions, not merely approved ones — "approved" read as
            # a trade count while an approved-then-rejected action is not one.
            f"trades={journal.get('actions_executed', 0)}"
        )
    else:
        lines.append("Last wakeup: none journaled yet")
    lines.append("/positions /last /help")
    return "\n".join(lines)


def cmd_positions(ledger: dict, prices: dict[str, float]) -> str:
    positions = ledger.get("positions", {})
    if not positions:
        return "No open positions."
    lines = [f"Open positions: {len(positions)}", *_positions_lines(ledger, prices)]
    return "\n".join(lines)[:MAX_MESSAGE_CHARS]


def cmd_last(journal: dict | None) -> str:
    if not journal:
        return "No wakeup journaled yet."
    reasoning = (journal.get("ai_reasoning") or "").strip()
    return (
        f"LAST AI DECISION — {journal.get('wakeup_id', '?')} "
        f"[{journal.get('status', '?')}]\n"
        f"Outlook: {journal.get('market_outlook', '?')}\n"
        f"Actions requested/approved: "
        f"{journal.get('actions_requested', 0)}/{journal.get('actions_approved', 0)}\n"
        f"Equity: ${float(journal.get('equity', 0)):,.2f}\n\n"
        f"{reasoning[:1200]}"
    )[:MAX_MESSAGE_CHARS]


HELP_TEXT = (
    "AI Trading Bot commands:\n"
    "  /status    equity, positions, last wakeup\n"
    "  /positions open-position detail\n"
    "  /last      the AI's latest decision + reasoning\n"
    "  /help      this list\n\n"
    "Answers usually arrive within a minute (always-on responder). "
    "Trade alerts and the daily/weekly reports come automatically."
)


# ---------------------------------------------------------------------------
# Command routing
# ---------------------------------------------------------------------------

def route_command(text: str, ledger: dict, journal: dict | None,
                  prices: dict[str, float]) -> str | None:
    """Map a message text to its reply. None = nothing to answer."""
    cmd = (text or "").strip().split("@", 1)[0].split(" ", 1)[0].lower()
    if cmd in ("/start", "/help"):
        return HELP_TEXT
    if cmd == "/status":
        return cmd_status(ledger, journal, prices)
    if cmd == "/positions":
        return cmd_positions(ledger, prices)
    if cmd == "/last":
        return cmd_last(journal)
    if cmd.startswith("/"):
        return f"Unknown command {cmd}.\n\n{HELP_TEXT}"
    return None


def _read_offset() -> int:
    try:
        return int(OFFSET_FILE.read_text().strip())
    except (OSError, ValueError):
        return 0


def _write_offset(value: int) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OFFSET_FILE.write_text(str(value))
    except OSError as e:
        print(f"WARNING: could not save offset: {e}", file=sys.stderr)


def main() -> int:
    # Windows consoles default to cp1252; emoji replies would crash print.
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

    # Load .env when run manually; in Actions the env is already set.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    token, chat_id = _creds()
    if not token or not chat_id:
        print(
            "Telegram commands: not configured "
            "(set AI_TELEGRAM_BOT_TOKEN / AI_TELEGRAM_CHAT_ID) — skipping.",
            file=sys.stderr,
        )
        return 0

    data = _tg(token, "getUpdates", offset=_read_offset() or None, timeout=5,
               allowed_updates=["message"])
    if data is None:
        # Not fatal: 409 (another poller) or a blip — the next wakeup retries.
        print("getUpdates unavailable — skipping command answering this run.")
        return 0

    updates = data.get("result", [])
    if not updates:
        print("No pending Telegram commands.")
        return 0

    ledger = _load_ledger()
    journal = _last_journal()
    max_update_id = 0
    answered = 0

    for update in updates:
        max_update_id = max(max_update_id, int(update.get("update_id", 0)))
        message = update.get("message") or {}
        text = message.get("text") or ""
        from_id = str((message.get("chat") or {}).get("id", ""))
        if from_id != chat_id:
            # Only the owner's chat is ever answered; everyone else is ignored.
            print(f"Ignored message from unauthorized chat {from_id}")
            continue
        prices = _current_prices(list(ledger.get("positions", {}).keys()) or
                                 ["BTC/USDT:USDT", "ETH/USDT:USDT"])
        reply = route_command(text, ledger, journal, prices)
        if reply and _tg(token, "sendMessage", chat_id=chat_id, text=reply,
                         disable_web_page_preview=True):
            answered += 1

    _write_offset(max_update_id + 1)
    print(f"Answered {answered}/{len(updates)} Telegram command(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
