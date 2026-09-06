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

# Telegram config — credentials come ONLY from environment variables.
# Never hardcode a bot token in source: it leaks the bot to anyone with repo access.
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

# Alert thresholds
PNL_CHANGE_THRESHOLD = 1.0  # Alert if P&L changes by 1%
UPDATE_INTERVAL_HOURS = 4   # Send update at most every 4 hours
FEE_RATE = 0.0005           # Trading fee rate
PRICE_ALERT_THRESHOLD = 2.0 # Alert if price moves >2% in 1 hour
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


def check_price_alerts(tracker: dict, symbol: str, pair_label: str, current_price: float) -> list[str]:
    """Check if price moved more than 2% in the last hour.
    
    Returns list of alert messages to send.
    """
    now = datetime.now(timezone.utc)
    alerts = []
    
    # Maintain price history (last 4 hours, sampled every 15 min = 16 entries max)
    price_history = tracker.setdefault("price_history", {}).setdefault(symbol, [])
    price_history.append({"time": now.isoformat(), "price": current_price})
    
    # Keep only last 4 hours of data
    cutoff = now - timedelta(hours=4)
    price_history[:] = [p for p in price_history
                        if datetime.fromisoformat(p["time"]) > cutoff]
    
    # Find the price closest to 1 hour ago (dead first loop removed)
    target_time = now - timedelta(hours=1)
    best_match = None
    best_diff = timedelta(hours=99)
    for p in price_history:
        try:
            t = datetime.fromisoformat(p["time"])
            diff = abs(t - target_time)
            if diff < best_diff:
                best_diff = diff
                best_match = p
        except Exception:
            continue
    
    if best_match and best_diff < timedelta(minutes=30):
        old_price = best_match["price"]
        if old_price > 0:
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
                    emoji = "+" if pct_change > 0 else ""
                    alerts.append(
                        f"PRICE ALERT: {pair_label} {direction} {emoji}{pct_change:+.1f}%\n"
                        f"${old_price:,.2f} -> ${current_price:,.2f}\n"
                        f"(1h change)"
                    )
                    tracker[last_alert_key] = now.isoformat()
    
    return alerts


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

        # Deduct fees for net P&L
        total_fees = size_usd * FEE_RATE * 2  # Entry + exit fees
        net_pnl_usd = pnl_usd - total_fees
        net_pnl_pct = net_pnl_usd / size_usd * 100

        total_unrealized += net_pnl_usd

        # Build position summary
        side_str = "LONG" if side == 1 else "SHORT"
        emoji = "+" if net_pnl_pct > 0 else "-" if net_pnl_pct < 0 else "="
        
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


if __name__ == "__main__":
    check_positions()
