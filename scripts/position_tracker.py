"""
Position Tracker

Monitors open paper trading positions by fetching current prices.
Sends updates only when:
  - Position just opened (first check)
  - P&L changes by more than 1%
  - 4 hours since last update
  - Position was closed

Does NOT spam Telegram on every 15-minute check.
"""

import os
import sys
import json
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

# Alert thresholds
PNL_CHANGE_THRESHOLD = 1.0  # Alert if P&L changes by 1%
UPDATE_INTERVAL_HOURS = 4   # Send update at most every 4 hours
FEE_RATE = 0.0005           # Trading fee rate


def send_telegram(text: str):
    """Send a Telegram message via urllib."""
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
    """Save tracker state."""
    with open(TRACKER_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def should_update(tracker: dict, symbol: str, current_pnl_pct: float) -> tuple[bool, str]:
    """Determine if we should send an update for this position.
    
    Returns (should_send, reason).
    """
    now = datetime.now(timezone.utc)
    pos_data = tracker.get("position_data", {}).get(symbol, {})
    
    if not pos_data:
        return True, "new_position"
    
    last_update_str = pos_data.get("last_update")
    if not last_update_str:
        return True, "first_check"
    
    try:
        last_update = datetime.fromisoformat(last_update_str)
        hours_since = (now - last_update).total_seconds() / 3600
    except:
        return True, "parse_error"
    
    # Check time interval
    if hours_since >= UPDATE_INTERVAL_HOURS:
        return True, f"interval_{hours_since:.0f}h"
    
    # Check P&L change
    last_pnl = pos_data.get("last_pnl_pct", 0)
    pnl_change = abs(current_pnl_pct - last_pnl)
    if pnl_change >= PNL_CHANGE_THRESHOLD:
        return True, f"pnl_change_{pnl_change:.1f}%"
    
    return False, "no_change"


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
        # Only send "no positions" once per day
        today = now.strftime("%Y-%m-%d")
        last_msg_date = tracker.get("last_no_positions_date")
        if last_msg_date != today:
            send_telegram(
                f"📊 <b>PORTFOLIO</b>\n"
                f"No open positions\n"
                f"Cash: ${paper['cash']:.2f}\n"
                f"{now.strftime('%b %d, %H:%M UTC')}"
            )
            tracker["last_no_positions_date"] = today
            save_tracker_state(tracker)
        return

    updates_to_send = []
    total_unrealized = 0.0
    position_summaries = []

    for symbol, pos in positions.items():
        pair_label = symbol.replace("_USDT_USDT", "")
        side = pos.get("side", 0)
        entry_price = pos.get("entry_price", 0)
        size_usd = pos.get("size_usd", pos.get("size", 0))
        strategy = pos.get("strategy", "Unknown")

        current_price = fetch_price(symbol)
        if current_price is None:
            print(f"  {pair_label}: Could not fetch price")
            continue

        # Calculate P&L
        if side == 1:  # LONG
            qty = size_usd / entry_price
            pnl_pct = (current_price - entry_price) / entry_price * 100
            pnl_usd = qty * (current_price - entry_price)
        else:  # SHORT
            qty = size_usd / entry_price
            pnl_pct = (entry_price - current_price) / entry_price * 100
            pnl_usd = qty * (entry_price - current_price)

        # Deduct fees for net P&L
        total_fees = size_usd * FEE_RATE * 2  # Entry + exit fees
        net_pnl_usd = pnl_usd - total_fees
        net_pnl_pct = net_pnl_usd / size_usd * 100

        total_unrealized += net_pnl_usd

        # Build position summary
        side_str = "LONG" if side == 1 else "SHORT"
        emoji = "🟢" if net_pnl_pct > 0 else "🔴" if net_pnl_pct < 0 else "⚪"
        
        position_summaries.append({
            "symbol": symbol,
            "pair": pair_label,
            "side": side_str,
            "emoji": emoji,
            "entry": entry_price,
            "current": current_price,
            "size": size_usd,
            "pnl_pct": net_pnl_pct,
            "pnl_usd": net_pnl_usd,
            "strategy": strategy,
        })

        # Check if we should send an update
        should_send, reason = should_update(tracker, symbol, net_pnl_pct)
        print(f"  {pair_label} {side_str}: ${entry_price:,.0f} -> ${current_price:,.0f} "
              f"({net_pnl_pct:+.1f}%, ${net_pnl_usd:+.2f}) [{reason}]")
        
        if should_send:
            updates_to_send.append((symbol, reason))
            # Update tracker data
            tracker.setdefault("position_data", {})[symbol] = {
                "last_update": now.isoformat(),
                "last_pnl_pct": net_pnl_pct,
                "last_price": current_price,
            }

    # Only send Telegram if there are updates worth reporting
    if updates_to_send:
        msg = f"📊 <b>POSITION UPDATE</b>\n"
        
        for pos in position_summaries:
            msg += (
                f"\n{pos['emoji']} <b>{pos['pair']} {pos['side']}</b>\n"
                f"Entry: ${pos['entry']:,.0f} → Current: ${pos['current']:,.0f}\n"
                f"P&L: {pos['pnl_pct']:+.1f}% (${pos['pnl_usd']:+.2f})\n"
                f"Size: ${pos['size']:.2f} | {pos['strategy']}\n"
            )
        
        msg += f"\nTotal P&L: ${total_unrealized:+.2f}\n"
        msg += f"Equity: ${paper['equity']:.2f}\n"
        msg += f"{now.strftime('%b %d, %H:%M UTC')}"
        
        send_telegram(msg)
    else:
        print("  No significant changes — skipping Telegram update.")

    # Update tracker state
    tracker["last_update"] = now.isoformat()
    save_tracker_state(tracker)
    print(f"\n  Check saved.")


if __name__ == "__main__":
    check_positions()
