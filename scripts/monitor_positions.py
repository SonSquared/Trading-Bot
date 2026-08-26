"""
Position Monitor - Tracks open paper positions over time.
Logs price snapshots every hour and sends Telegram updates.
Run via GitHub Actions hourly or locally.
"""
import sys
sys.path.insert(0, ".")

import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import ccxt

STATE_FILE = Path("data/results/paper_state.json")
MONITOR_LOG = Path("data/results/position_monitor.jsonl")


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def get_prices():
    exchange = ccxt.kraken({"enableRateLimit": True})
    prices = {}
    for symbol in ["ETH/USDT", "BTC/USDT"]:
        try:
            ticker = exchange.fetch_ticker(symbol)
            prices[symbol] = ticker["last"]
        except Exception as e:
            print(f"  Failed to fetch {symbol}: {e}")
    return prices


def tg_send(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    subprocess.run(
        ["curl", "-s", "-m", "15", "-X", "POST", url,
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True})],
        capture_output=True, timeout=20,
    )


def main():
    import os
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    state = load_state()
    if not state or not state.get("positions"):
        print("No open positions to monitor.")
        return

    prices = get_prices()
    now = datetime.now(timezone.utc).isoformat()

    print(f"=== POSITION MONITOR ({now[:19]}) ===")
    total_unrealized = 0
    lines = []

    for pair, pos in state["positions"].items():
        symbol = "ETH/USDT" if "ETH" in pair else "BTC/USDT"
        current_price = prices.get(symbol)
        if not current_price:
            continue

        entry = pos["entry_price"]
        side = pos["side"]
        size = pos["size_usd"]
        strategy = pos.get("strategy", "Unknown")

        pnl_pct = (current_price - entry) / entry * side
        pnl_usd = size * pnl_pct
        total_unrealized += pnl_usd

        direction = "LONG" if side == 1 else "SHORT"
        emoji = "+" if pnl_pct > 0 else ""
        status_emoji = "🟢" if pnl_usd > 0 else "🔴"

        # How long has the position been open?
        entry_time = datetime.fromisoformat(pos["entry_time"])
        hours_open = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

        line = (
            f"{status_emoji} <b>{pair.replace('_', '/')}</b> {direction}\n"
            f"  Strategy: {strategy}\n"
            f"  Entry: ${entry:,.2f} → Now: ${current_price:,.2f}\n"
            f"  P&L: <b>{emoji}{pnl_pct*100:.2f}%</b> (${pnl_usd:+.2f})\n"
            f"  Open for: {hours_open:.1f}h"
        )
        lines.append(line)
        print(f"  {pair}: {direction} entry=${entry:,.2f} now=${current_price:,.2f} P&L={emoji}{pnl_pct*100:.2f}%")

        # Log to file
        entry_log = {
            "timestamp": now,
            "pair": pair,
            "side": direction,
            "entry_price": entry,
            "current_price": current_price,
            "pnl_pct": pnl_pct * 100,
            "pnl_usd": pnl_usd,
            "hours_open": hours_open,
        }
        with open(MONITOR_LOG, "a") as f:
            f.write(json.dumps(entry_log) + "\n")

    total_emoji = "+" if total_unrealized >= 0 else ""
    print(f"\n  Total unrealized: {total_emoji}${total_unrealized:.2f}")

    # Send Telegram update
    if token and chat_id:
        msg = (
            f"📍 <b>POSITION UPDATE</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            + "\n\n".join(lines)
            + f"\n\n💰 <b>Total unrealized: {total_emoji}${total_unrealized:.2f}</b>"
            + f"\n⏰ {now[:19]} UTC"
        )
        tg_send(token, chat_id, msg)
        print("  Telegram update sent.")


if __name__ == "__main__":
    main()
