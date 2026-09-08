"""
Position Tracker

Monitors open paper trading positions by fetching current prices.
Sends PRICE ALERTS when a pair moves more than 2% between runs (~2h apart
on the GitHub Actions schedule).

Does NOT send regular status updates — the bot itself handles those
(trade alerts + one daily status), so messages never duplicate.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, ".")


def _fix_windows_console() -> None:
    """Fix Windows console encoding for emoji — only when run as a script.

    Kept out of import time: rewrapping sys.stdout/stderr breaks pytest's
    capture machinery (closed-file errors) in any test module that imports
    this one.
    """
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

STATE_FILE = Path("data/results/paper_state.json")
TRACKER_FILE = Path("data/results/position_tracker.json")
LOG_DIR = Path("data/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Telegram config — credentials come ONLY from environment variables.
# Never hardcode a bot token in source: it leaks the bot to anyone with repo access.
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

# Alert thresholds
UPDATE_INTERVAL_HOURS = 4   # legacy; updates are suppressed — alerts only
PRICE_ALERT_THRESHOLD = 2.0 # Alert if price moves >2% between runs
PRICE_ALERT_COOLDOWN_HOURS = 1  # Don't spam — 1 alert per symbol per hour


def send_telegram(text: str):
    """Send a Telegram message via requests (most reliable cross-platform)."""
    if not TG_TOKEN or not TG_CHAT:
        print("  Telegram: NOT CONFIGURED — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID. Skipping send.")
        return
    try:
        import requests as _requests
        resp = _requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=20,
        )
        if resp.status_code == 200 and resp.json().get("ok"):
            print("  Telegram: sent")
            return
        print(f"  Telegram: API error - {resp.text[:100]}")
    except Exception as e:
        print(f"  Telegram: failed ({e})")


def fetch_price(symbol: str) -> float | None:
    """Fetch current price from Kraken API."""
    # Convert pair format: ETH_USDT_USDT -> ETHUSDT
    if symbol.endswith("_USDT_USDT"):
        clean = symbol[:-len("_USDT_USDT")].replace("_", "") + "USDT"
    elif symbol.endswith("_USDT"):
        clean = symbol[:-len("_USDT")].replace("_", "") + "USDT"
    else:
        clean = symbol.replace("_", "")

    kraken_map = {
        "ETHUSDT": "XETHZUSD",
        "BTCUSDT": "XBTUSD",
    }
    kraken_pair = kraken_map.get(clean, clean)

    try:
        url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_pair}"
        req = urllib.request.Request(url)
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())

        if data.get("result"):
            key = list(data["result"].keys())[0]
            price = float(data["result"][key]["c"][0])  # last trade close
            return price
    except Exception as e:
        print(f"  Kraken error for {symbol}: {e}")

    # Fallback: CoinGecko
    coin_map = {"ETHUSDT": "ethereum", "BTCUSDT": "bitcoin"}
    coin = coin_map.get(clean)
    if coin:
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read())
            return float(data[coin]["usd"])
        except Exception as e:
            print(f"  CoinGecko error for {coin}: {e}")

    return None


def load_positions() -> dict:
    """Load current positions from paper state."""
    if not STATE_FILE.exists():
        return {"positions": {}, "equity": 97.0, "cash": 97.0}

    with open(STATE_FILE) as f:
        state = json.load(f)

    return {
        "positions": state.get("positions", {}),
        "equity": state.get("equity", 97.0),
        "cash": state.get("cash", 97.0),
    }


def load_tracker_state() -> dict:
    """Load tracker history."""
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE) as f:
            return json.load(f)
    return {"last_update": None, "position_data": {}}


def save_tracker_state(state: dict):
    """Atomically persist tracker state (write temp file, then rename)."""
    tmp = TRACKER_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, TRACKER_FILE)


def should_update(tracker: dict, symbol: str, current_pnl_pct: float) -> tuple[bool, str]:
    """Determine if a regular status update should be sent for this position.

    Regular updates are suppressed by design — the bot itself sends trade
    alerts and one daily status, and duplicating them from the tracker just
    confuses. This exists only for the console log's [reason] annotation and
    any future re-enable of updates. Returns (False, reason) always.
    """
    now = datetime.now(timezone.utc)
    pos_data = tracker.get("position_data", {}).get(symbol, {})

    if not pos_data:
        return False, "new_position"

    last_update_str = pos_data.get("last_update")
    if not last_update_str:
        return False, "first_check"

    try:
        last_update = datetime.fromisoformat(last_update_str)
        hours_since = (now - last_update).total_seconds() / 3600
    except (ValueError, TypeError):
        return False, "parse_error"

    if hours_since >= UPDATE_INTERVAL_HOURS:
        return False, f"interval_{hours_since:.0f}h"

    last_pnl = pos_data.get("last_pnl_pct", 0)
    pnl_change = abs(current_pnl_pct - last_pnl)
    if pnl_change >= 1.0:
        return False, f"pnl_change_{pnl_change:.1f}%"

    return False, "no_change"


def check_price_alerts(tracker: dict, symbol: str, pair_label: str, current_price: float) -> list[str]:
    """Check if price moved more than 2% since the previous sample.

    Returns list of alert messages to send.
    """
    now = datetime.now(timezone.utc)
    alerts = []

    # Maintain price history (bounded window; samples arrive every ~2h on
    # the Actions schedule). The comparison anchor is the PREVIOUS sample:
    # the old code looked for a price "closest to 1h ago within ±30min",
    # which no sample can ever satisfy at 2h cadence — the feature silently
    # never fired. Now we measure the real elapsed window between runs.
    price_history = tracker.setdefault("price_history", {}).setdefault(symbol, [])
    price_history.append({"time": now.isoformat(), "price": current_price})

    # Keep only the last 24 hours of data (~12 samples at 2h cadence)
    cutoff = now - timedelta(hours=24)
    kept = []
    for p in price_history:
        try:
            if datetime.fromisoformat(p["time"]) > cutoff:
                kept.append(p)
        except Exception:
            continue
    price_history[:] = kept

    if len(price_history) >= 2:
        prev = price_history[-2]
        try:
            elapsed_hours = (now - datetime.fromisoformat(prev["time"])).total_seconds() / 3600.0
        except Exception:
            elapsed_hours = 0.0
        old_price = prev.get("price", 0.0)
        # Only alert on meaningful windows (>=1h) so a quick restart storm
        # of the runner can't fabricate a "2h move" from minutes of data.
        if old_price > 0 and elapsed_hours >= 1.0:
            pct_change = (current_price - old_price) / old_price * 100

            if abs(pct_change) >= PRICE_ALERT_THRESHOLD:
                # Check cooldown
                last_alert_key = f"last_price_alert_{symbol}"
                last_alert_str = tracker.get(last_alert_key)
                can_alert = True
                if last_alert_str:
                    try:
                        last_alert = datetime.fromisoformat(last_alert_str)
                        if (now - last_alert).total_seconds() < PRICE_ALERT_COOLDOWN_HOURS * 3600:
                            can_alert = False
                    except Exception:
                        pass

                if can_alert:
                    direction = "UP" if pct_change > 0 else "DOWN"
                    emoji = "📈" if pct_change > 0 else "📉"
                    alerts.append(
                        f"PRICE ALERT: {pair_label} {direction} {emoji} {pct_change:+.1f}%\n"
                        f"${old_price:,.2f} -> ${current_price:,.2f}\n"
                        f"({elapsed_hours:.1f}h change)"
                    )
                    tracker[last_alert_key] = now.isoformat()

    return alerts


def check_positions():
    """Main tracker function — called every 2 hours on the Actions schedule.

    Returns the per-position gross-mark summaries (for tests and callers).
    """
    now = datetime.now(timezone.utc)
    print(f"\n{'='*50}")
    print(f"POSITION TRACKER — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}")

    paper = load_positions()
    tracker = load_tracker_state()
    positions = paper["positions"]

    if not positions:
        print("  No open positions — nothing to track.")
        return

    updates_to_send = []
    total_unrealized = 0.0
    position_summaries = []
    price_alerts_sent = []

    for symbol, pos in positions.items():
        # Format: ETH_USDT_USDT -> ETH/USDT
        if "_USDT_USDT" in symbol:
            pair_label = symbol.replace("_USDT_USDT", "") + "/USDT"
        elif "_USDT" in symbol:
            pair_label = symbol.replace("_USDT", "") + "/USDT"
        else:
            pair_label = symbol.replace("_", "/")
        side = pos.get("side", 0)
        entry_price = pos.get("entry_price", 0)
        size_usd = pos.get("size_usd", pos.get("size", 0))
        strategy = pos.get("strategy", "Unknown")

        current_price = fetch_price(symbol)
        if current_price is None:
            print(f"  {pair_label}: Could not fetch price")
            continue

        # Check for price alerts (>2% move in 1 hour)
        price_alerts = check_price_alerts(tracker, symbol, pair_label, current_price)
        for alert in price_alerts:
            print(f"  ALERT: {alert}")
            send_telegram(alert)
            price_alerts_sent.append(alert)

        # Calculate P&L
        if side == 1:  # LONG
            qty = size_usd / entry_price
            pnl_pct = (current_price - entry_price) / entry_price * 100
            pnl_usd = qty * (current_price - entry_price)
        else:  # SHORT
            qty = size_usd / entry_price
            pnl_pct = (entry_price - current_price) / entry_price * 100
            pnl_usd = qty * (entry_price - current_price)

        # GROSS mark P&L — matches the bot's convention. Fees and funding
        # are booked into the realized P&L (total_pnl) at open/close/funding
        # time, never subtracted from the live mark, so netting them here
        # would diverge from every number the bot reports.
        total_unrealized += pnl_usd

        # Build position summary
        side_str = "LONG" if side == 1 else "SHORT"
        emoji = "+" if pnl_pct > 0 else "-" if pnl_pct < 0 else "="
        
        position_summaries.append({
            "symbol": symbol,
            "pair": pair_label,
            "side": side_str,
            "emoji": emoji,
            "entry": entry_price,
            "current": current_price,
            "size": size_usd,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "strategy": strategy,
        })

        # Check if we should send an update
        should_send, reason = should_update(tracker, symbol, pnl_pct)
        print(f"  {pair_label} {side_str}: ${entry_price:,.0f} -> ${current_price:,.0f} "
              f"({pnl_pct:+.1f}%, ${pnl_usd:+.2f}) [{reason}]")
        
        if should_send:
            updates_to_send.append((symbol, reason))
            # Update tracker data
            tracker.setdefault("position_data", {})[symbol] = {
                "last_update": now.isoformat(),
                "last_pnl_pct": pnl_pct,
                "last_price": current_price,
            }

    # Calculate live equity = cash + position market values
    # Cash was debited by (size_usd + entry_fee) on open.
    # When position closes, cash gets back size_usd + pnl_usd.
    # So position value = size_usd + unrealized_pnl.
    # Both LONG and SHORT use the same formula.
    live_equity = paper['cash']
    for pos in position_summaries:
        if pos['entry'] > 0:
            qty = pos['size'] / pos['entry']
            if pos['side'] == 'LONG':
                unrealized_pnl = qty * (pos['current'] - pos['entry'])
            else:
                unrealized_pnl = qty * (pos['entry'] - pos['current'])
            live_equity += pos['size'] + unrealized_pnl

    # Position tracker only sends PRICE ALERTS (>2% moves).
    # Regular status updates are handled by the bot (once per day or on trades).
    # This prevents duplicate/confusing messages.
    if price_alerts_sent:
        print(f"  Sent {len(price_alerts_sent)} price alert(s).")
    else:
        print("  No alerts — bot handles daily status updates.")

    # Update tracker state
    tracker["last_update"] = now.isoformat()
    save_tracker_state(tracker)
    print("\n  Check saved.")

    return position_summaries


if __name__ == "__main__":
    _fix_windows_console()
    check_positions()
