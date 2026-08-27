"""
24-Hour Position Tracker

Monitors open paper trading positions by fetching current prices every
15 minutes for 24 hours. Reports:
  - Entry vs current price
  - Unrealized P&L
  - Price change since last check
  - Session high/low

Can run as a GitHub Actions workflow (cron every 15 min) or locally.
"""

import os
import sys
import json
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, ".")

# Fix Windows console encoding for emoji
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

STATE_FILE = Path("data/results/paper_state.json")
TRACKER_FILE = Path("data/results/position_tracker.json")
LOG_DIR = Path("data/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Telegram config
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8783971913:AAH1ZdvtKvHjgVuC2c9LLebYnM-o8gBMQaY")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "5421461006")

# How often to alert on big moves (in %)
ALERT_THRESHOLD_PCT = 2.0


def send_telegram(text: str):
    """Send a Telegram message via curl (bypasses SSL issues)."""
    try:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        data = json.dumps({
            "chat_id": TG_CHAT,
            "text": text,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        print(f"  Telegram: sent")
    except Exception as e:
        print(f"  Telegram: failed ({e})")


def fetch_price(symbol: str) -> float | None:
    """Fetch current price from Kraken API."""
    # Convert pair format: ETH_USDT_USDT -> ETHUSDT
    # Remove trailing _USDT_USDT and convert underscores
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
    return {"checks": [], "alerts": []}


def save_tracker_state(state: dict):
    """Save tracker state."""
    with open(TRACKER_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def check_positions():
    """Main tracker function — called every 15 minutes."""
    now = datetime.now(timezone.utc)
    print(f"\n{'='*50}")
    print(f"POSITION TRACKER — {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*50}")

    paper = load_positions()
    tracker = load_tracker_state()
    positions = paper["positions"]

    if not positions:
        print("  No open positions.")
        # Only send alert once per day if no positions
        today = now.strftime("%Y-%m-%d")
        if not any(a.get("date") == today for a in tracker.get("alerts", [])):
            send_telegram(
                f"📊 <b>POSITION CHECK</b>\n"
                f"━━━━━━━━━━━━━━━━━━━\n"
                f"No open positions.\n"
                f"Cash: ${paper['cash']:.2f}\n"
                f"Date: {now.strftime('%b %d, %H:%M UTC')}"
            )
            tracker.setdefault("alerts", []).append({
                "date": today, "time": now.isoformat(), "type": "no_positions"
            })
            save_tracker_state(tracker)
        return

    lines = []
    total_unrealized = 0.0
    alerts_to_send = []

    for symbol, pos in positions.items():
        pair_label = symbol.replace("_USDT_USDT", "")
        side = pos.get("side", 0)
        entry_price = pos.get("entry_price", 0)
        size_usd = pos.get("size_usd", pos.get("size", 0))
        strategy = pos.get("strategy", "Unknown")

        current_price = fetch_price(symbol)
        if current_price is None:
            print(f"  {pair_label}: Could not fetch price")
            lines.append(f"  {pair_label}: ⚠️ Price unavailable")
            continue

        # Calculate P&L
        if side == 1:  # LONG
            pnl_pct = (current_price - entry_price) / entry_price * 100
            qty = size_usd / entry_price
            pnl_usd = qty * (current_price - entry_price)
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price * 100
            qty = size_usd / entry_price
            pnl_usd = qty * (entry_price - current_price)

        total_unrealized += pnl_usd

        # Price change since last check
        checks = tracker.get("checks", [])
        last_price = None
        for c in reversed(checks):
            if c.get("symbol") == symbol:
                last_price = c.get("current_price")
                break

        price_change = ""
        if last_price:
            chg = (current_price - last_price) / last_price * 100
            arrow = "📈" if chg > 0 else "📉" if chg < 0 else "➡️"
            price_change = f"\n    {arrow} {chg:+.2f}% since last check"

        # Session high/low
        symbol_checks = [c for c in checks if c.get("symbol") == symbol]
        session_high = max([c["current_price"] for c in symbol_checks] + [current_price])
        session_low = min([c["current_price"] for c in symbol_checks] + [current_price])

        emoji = "🟢" if pnl_pct > 0 else "🔴"
        side_str = "LONG" if side == 1 else "SHORT"

        line = (
            f"  {emoji} {pair_label} {side_str} ({strategy})\n"
            f"    Entry: ${entry_price:,.2f} → Current: ${current_price:,.2f}\n"
            f"    P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})\n"
            f"    Size: ${size_usd:.2f}{price_change}\n"
            f"    Session: ${session_low:,.2f} — ${session_high:,.2f}"
        )
        print(line)
        lines.append(line)

        # Save check
        tracker.setdefault("checks", []).append({
            "symbol": symbol,
            "time": now.isoformat(),
            "current_price": current_price,
            "entry_price": entry_price,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "side": side,
        })

        # Alert on big moves
        if abs(pnl_pct) >= ALERT_THRESHOLD_PCT:
            last_alert_pct = 0
            for c in reversed(symbol_checks):
                if c.get("alerted"):
                    last_alert_pct = c.get("pnl_pct", 0)
                    break
            if abs(pnl_pct - last_alert_pct) >= ALERT_THRESHOLD_PCT:
                alerts_to_send.append({
                    "symbol": symbol,
                    "pair_label": pair_label,
                    "side": side_str,
                    "pnl_pct": pnl_pct,
                    "pnl_usd": pnl_usd,
                    "current_price": current_price,
                })

    # Send summary
    summary = (
        f"📊 <b>POSITION UPDATE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        + "\n".join(lines) + "\n\n"
        f"💰 Total Unrealized: <b>${total_unrealized:+.2f}</b>\n"
        f"💵 Paper Equity: ${paper['equity']:.2f}\n"
        f"📅 {now.strftime('%b %d, %H:%M UTC')}"
    )
    send_telegram(summary)

    # Send big move alerts
    for alert in alerts_to_send:
        emoji = "🚀" if alert["pnl_pct"] > 0 else "⚠️"
        send_telegram(
            f"{emoji} <b>BIG MOVE: {alert['pair_label']} {alert['side']}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"P&L: {alert['pnl_pct']:+.2f}% (${alert['pnl_usd']:+.2f})\n"
            f"Price: ${alert['current_price']:,.2f}\n"
            f"Threshold: {ALERT_THRESHOLD_PCT}% reached"
        )
        for c in tracker.get("checks", []):
            if c.get("symbol") == alert["symbol"]:
                c["alerted"] = True

    # Clean up old checks (keep last 24 hours = ~96 checks)
    cutoff = (now - timedelta(hours=24)).isoformat()
    tracker["checks"] = [c for c in tracker.get("checks", []) if c.get("time", "") > cutoff]
    tracker["alerts"] = [a for a in tracker.get("alerts", []) if a.get("date", "") >= (now - timedelta(days=7)).strftime("%Y-%m-%d")]

    save_tracker_state(tracker)
    print(f"\n  Check saved. Total checks in buffer: {len(tracker['checks'])}")


if __name__ == "__main__":
    check_positions()
