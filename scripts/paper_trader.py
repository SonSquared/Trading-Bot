"""
Paper Trading Engine

Simulates real trading with proper position tracking, P&L calculation,
and comprehensive logging. Each run fetches fresh data, runs strategies,
and simulates trades with realistic fills.

Features:
  - Retry logic with exponential backoff for API calls
  - Graceful error handling (partial failures don't crash the run)
  - Run logging for monitoring dashboard
  - State persistence between runs
"""

import os
import sys
import json
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from functools import wraps

sys.path.insert(0, ".")

import pandas as pd
import numpy as np

from trading_system.strategies import STRATEGY_REGISTRY


# --- Strategy Configs (from FULL 2022-2026 backtest optimization) ---
# MACD and ROC Momentum LOSE money on all parameters (overtrades, whipsawed)
# Bollinger+RSI and RSI_Reversion are the only profitable strategies
STRATEGIES = {
    "Bollinger+RSI ETH": {
        "strategy": "Bollinger_Reversion",
        "pair": "ETH_USDT_USDT",
        "timeframe": "4h",
        "weight": 0.40,
        "params": {"bb_period": 20, "bb_std": 2.0, "rsi_filter": True, "rsi_period": 14,
                   "rsi_oversold": 30, "rsi_overbought": 70, "exit_at_middle": True},
    },
    "Bollinger+RSI BTC": {
        "strategy": "Bollinger_Reversion",
        "pair": "BTC_USDT_USDT",
        "timeframe": "4h",
        "weight": 0.35,
        "params": {"bb_period": 20, "bb_std": 2.0, "rsi_filter": True, "rsi_period": 14,
                   "rsi_oversold": 30, "rsi_overbought": 70, "exit_at_middle": True},
    },
    "RSI_Reversion BTC": {
        "strategy": "RSI_Reversion",
        "pair": "BTC_USDT_USDT",
        "timeframe": "4h",
        "weight": 0.25,
        "params": {"rsi_period": 14, "entry_oversold": 30, "entry_overbought": 70,
                   "exit_neutral_low": 45, "exit_neutral_high": 55, "use_bb_filter": False},
    },
}

# --- Config ---
INITIAL_CAPITAL = 97.0
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
MAX_POSITION_PCT = 0.35
MIN_TRADE_USD = 5.0  # Minimum trade size (reduced from 100 for small accounts)
LOG_DIR = Path("data/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG = LOG_DIR / "paper_trades.jsonl"
STATE_FILE = LOG_DIR / "paper_state.json"
SUMMARY_FILE = LOG_DIR / "paper_summary.json"
RUN_LOG = LOG_DIR / "run_history.jsonl"


# --- Retry Decorator ---
def retry(max_attempts: int = 3, base_delay: float = 2.0, exceptions: tuple = (Exception,)):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        print(f"    Retry {attempt}/{max_attempts} in {delay:.1f}s: {e}")
                        time.sleep(delay)
                    else:
                        print(f"    Failed after {max_attempts} attempts: {e}")
            raise last_error
        return wrapper
    return decorator


# --- Run Logger ---
def log_run(status: str, duration: float, trades: int, errors: list, equity: float):
    """Log a bot run for the monitoring dashboard."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,  # "success", "partial", "failed"
        "duration_seconds": round(duration, 1),
        "trades": trades,
        "errors": errors,
        "equity": equity,
    }
    with open(RUN_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# --- Telegram ---
def load_telegram_config():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled = os.getenv("TELEGRAM_ENABLED", "").lower() == "true"
    if token and chat_id:
        return {"enabled": True, "bot_token": token, "chat_id": chat_id}
    try:
        import yaml
        cfg_path = Path("configs/bot_live.yaml")
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            tg = cfg.get("bot", {}).get("telegram", {})
            if tg.get("enabled") and tg.get("bot_token"):
                return tg
    except Exception:
        pass
    return {"enabled": False}


def get_notifier():
    from trading_system.bot.telegram_notifier import TelegramNotifier
    cfg = load_telegram_config()
    if cfg.get("enabled") and cfg.get("bot_token"):
        n = TelegramNotifier(bot_token=cfg["bot_token"], chat_id=str(cfg["chat_id"]), enabled=True)
        if n.test_connection():
            print("  Telegram: Connected")
            return n
        print("  Telegram: Connection failed")
    else:
        print("  Telegram: Not configured")
    return None


# --- Telegram Command Handling ---
def tg_api_call(token: str, method: str, params: dict = None) -> dict:
    """Make a Telegram Bot API call via curl."""
    import subprocess
    url = f"https://api.telegram.org/bot{token}/{method}"
    result = subprocess.run(
        ["curl", "-s", "-m", "15", "-X", "POST", url,
         "-H", "Content-Type: application/json",
         "-d", json.dumps(params or {})],
        capture_output=True, text=True, timeout=20,
    )
    if result.stdout:
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            pass
    return {"ok": False, "description": "No response"}


def tg_send_message(token: str, chat_id: str, text: str) -> bool:
    resp = tg_api_call(token, "sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    })
    return resp.get("ok", False)


def handle_telegram_commands(token: str, chat_id: str):
    """Check for pending Telegram commands and respond to them."""
    print("\nChecking for Telegram commands...")

    # Get recent updates
    resp = tg_api_call(token, "getUpdates", {"limit": 10, "timeout": 0})
    if not resp.get("ok") or not resp.get("result"):
        print("  No pending commands.")
        return

    state = load_state()
    equity = get_equity(state)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wr = state["wins"] / state["total_trades"] * 100 if state["total_trades"] > 0 else 0

    for update in resp["result"]:
        msg = update.get("message", {})
        text = msg.get("text", "").strip().lower()
        msg_chat_id = str(msg.get("chat", {}).get("id", ""))
        update_id = update.get("update_id", 0)

        if msg_chat_id != chat_id:
            continue

        if text == "/help":
            print("  -> /help")
            tg_send_message(token, chat_id, (
                "🤖 <b>TRADING BOT COMMANDS</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n\n"
                "/status - Full portfolio status\n"
                "/equity - Quick equity check\n"
                "/positions - Open positions\n"
                "/trades - Recent trade history\n"
                "/signals - Current strategy signals\n"
                "/restart - Run strategies now\n"
                "/dashboard - Bot monitoring dashboard\n"
                "/help - This message\n\n"
                f"Mode: <b>PAPER</b>\n"
                f"Initial: ${INITIAL_CAPITAL:,.0f}\n"
                f"Strategies: Bollinger+RSI + RSI_Reversion (3 total)\n"
                f"Backtested: +136% return, 69.7% win rate (2022-2026)\n"
                f"Note: Commands are checked every run (~15 min)."
            ))

        elif text == "/trades":
            print("  -> /trades")
            trades = []
            if TRADE_LOG.exists():
                with open(TRADE_LOG) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                trades.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            recent = trades[-10:]
            if not recent:
                tg_send_message(token, chat_id, "📋 No trades yet")
            else:
                msg_text = "📋 <b>RECENT TRADES</b>\n━━━━━━━━━━━━━━━━━━━━━\n"
                for t in reversed(recent):
                    action = t.get("action", "?")
                    pair = t.get("pair", "?").replace("_", "/")
                    if action == "CLOSE":
                        pnl = t.get("pnl_usd", 0)
                        emoji = "✅" if pnl >= 0 else "❌"
                        msg_text += f"\n{emoji} {pair} CLOSED @ ${t.get('exit_price', 0):,.2f}\n"
                        msg_text += f"  P&L: ${pnl:+.2f} ({t.get('pnl_pct', 0):+.2f}%)\n"
                    elif action.startswith("OPEN"):
                        msg_text += f"\n🟢 {pair} {action} @ ${t.get('price', 0):,.2f}\n"
                tg_send_message(token, chat_id, msg_text)

        elif text == "/equity":
            print("  -> /equity")
            tg_send_message(token, chat_id, (
                f"💰 <b>EQUITY</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Current: <b>${equity:,.2f}</b>\n"
                f"Return: <b>{total_return:+.2f}%</b>\n"
                f"Cash: ${state['cash']:,.2f}"
            ))

        elif text == "/status":
            print("  -> /status")
            positions_text = ""
            for pair, pos in state.get("positions", {}).items():
                side = "LONG" if pos.get("side") == 1 else "SHORT"
                emoji = "🟢" if pos.get("side") == 1 else "🔴"
                positions_text += f"  {emoji} {pair.replace('_', '/')} {side} @ ${pos.get('entry_price', 0):,.2f}\n"
            if not positions_text:
                positions_text = "  No open positions\n"
            tg_send_message(token, chat_id, (
                f"📋 <b>PORTFOLIO STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 Equity: <b>${equity:,.2f}</b>\n"
                f"📈 Return: <b>{total_return:+.2f}%</b>\n"
                f"💵 Cash: ${state['cash']:,.2f}\n"
                f"📊 Trades: {state['total_trades']} (W:{state['wins']} / L:{state['losses']})\n"
                f"🏆 Win Rate: {wr:.1f}%\n"
                f"📉 Realized P&L: ${state['total_pnl']:+,.2f}\n"
                f"\n📍 <b>Open Positions:</b>\n{positions_text}"
                f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            ))

        elif text == "/positions":
            print("  -> /positions")
            positions = state.get("positions", {})
            if not positions:
                tg_send_message(token, chat_id, "📍 No open positions")
            else:
                msg_text = "📍 <b>OPEN POSITIONS</b>\n━━━━━━━━━━━━━━━━━━━━━\n"
                for pair, pos in positions.items():
                    side = "LONG" if pos.get("side") == 1 else "SHORT"
                    emoji = "🟢" if pos.get("side") == 1 else "🔴"
                    msg_text += f"\n{emoji} <b>{pair.replace('_', '/')}</b> {side}\n"
                    msg_text += f"  Entry: ${pos.get('entry_price', 0):,.2f}\n"
                    msg_text += f"  Size: ${pos.get('size_usd', 0):,.2f}\n"
                    msg_text += f"  Strategy: {pos.get('strategy', 'Unknown')}\n"
                tg_send_message(token, chat_id, msg_text)

        elif text == "/signals":
            print("  -> /signals (running strategies...)")
            try:
                from trading_system.strategies import STRATEGY_REGISTRY as SR
                signals_text = "🧠 <b>CURRENT SIGNALS</b>\n━━━━━━━━━━━━━━━━━━━━━\n"
                for name, cfg in STRATEGIES.items():
                    try:
                        df = fetch_latest(cfg["pair"], cfg["timeframe"])
                        strat = SR[cfg["strategy"]]
                        sig = strat.generate_signals(df, cfg["params"])
                        latest = int(sig.iloc[-1])
                        price = float(df["close"].iloc[-1])
                        direction = "LONG" if latest == 1 else "FLAT" if latest == 0 else "SHORT"
                        emoji = "🟢" if latest == 1 else "⚪" if latest == 0 else "🔴"
                        signals_text += f"\n{emoji} <b>{name}</b>\n  Signal: {direction} (${price:,.2f})\n"
                    except Exception as e:
                        signals_text += f"\n❌ <b>{name}</b>\n  Error: {e}\n"
                tg_send_message(token, chat_id, signals_text)
            except Exception as e:
                tg_send_message(token, chat_id, f"❌ Error getting signals: {e}")

        elif text == "/restart":
            print("  -> /restart")
            tg_send_message(token, chat_id, (
                "🔄 <b>RESTARTING BOT</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━\n"
                "Strategies are running now...\n"
                "Results will appear in ~1 minute."
            ))

        elif text == "/dashboard":
            print("  -> /dashboard")
            RUN_LOG = Path("data/results/run_history.jsonl")
            runs = []
            if RUN_LOG.exists():
                with open(RUN_LOG) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                runs.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            if not runs:
                tg_send_message(token, chat_id, "No run history yet.")
            else:
                total = len(runs)
                successful = sum(1 for r in runs if r.get("status") == "success")
                success_rate = successful / total * 100
                last = runs[-1]
                status_color = {"success": "GREEN", "partial": "YELLOW", "failed": "RED"}.get(last.get("status", ""), "UNKNOWN")
                consec = 0
                for r in reversed(runs):
                    if r.get("status") == "failed":
                        consec += 1
                    else:
                        break
                tg_send_message(token, chat_id, (
                    f"📊 <b>BOT DASHBOARD</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"Status: [{status_color}] <b>{last.get('status', '?').upper()}</b>\n"
                    f"Runs: {total} ({success_rate:.0f}% success)\n"
                    f"Failed: {sum(1 for r in runs if r.get('status') == 'failed')}\n"
                    f"Consec. fails: {consec}\n\n"
                    f"💰 Equity: <b>${equity:,.2f}</b>\n"
                    f"📈 Return: <b>{total_return:+.2f}%</b>\n"
                    f"⏱ Last run: {last.get('duration_seconds', 0):.1f}s\n"
                    f"🕐 {last.get('timestamp', '?')[:19]}"
                ))

        elif text.startswith("/"):
            print(f"  -> Unknown: {text}")
            tg_send_message(token, chat_id, f"Unknown command: {text}\nType /help for available commands.")

    print("  Command check done.")


# --- State Management ---
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "cash": INITIAL_CAPITAL,
        "positions": {},
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "peak_equity": INITIAL_CAPITAL,
        "max_drawdown": 0.0,
    }


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# --- Data ---
@retry(max_attempts=3, base_delay=2.0)
def fetch_latest(pair: str, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
    """Fetch latest candles via CryptoCompare (no geo-restrictions) or cached parquet files."""
    # Try cached files first
    for pattern in [f"data/raw/{pair}/{timeframe}.parquet", f"data/raw/{pair}/klines_{timeframe}.parquet"]:
        cache_path = Path(pattern)
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            if "timestamp" in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
                df = df[df["timestamp"] > cutoff]
            return df

    # Fetch fresh data via ccxt - try Kraken first (US-friendly), then others
    import ccxt
    # ETH_USDT_USDT -> ETH/USDT
    symbol = pair.replace("_USDT_USDT", "").replace("_", "/") + "/USDT"
    
    # Try Kraken first (works from US)
    try:
        exchange = ccxt.kraken({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        print(f"    Kraken: Got {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        print(f"    Kraken failed: {e}")
    
    # Fallback to Binance spot
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        print(f"    Binance spot: Got {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        print(f"    Binance spot failed: {e}")
    
    raise Exception(f"All data sources failed for {pair}")


# --- Paper Trade Execution ---
def simulate_fills(price: float, side: int, size_usd: float) -> tuple[float, float, float]:
    slippage = price * SLIPPAGE_RATE * side
    fill_price = price + slippage
    fee = size_usd * FEE_RATE
    return fill_price, fee, abs(slippage) * size_usd / price


def open_position(state: dict, pair: str, side: int, price: float, strategy: str, size_usd: float) -> dict:
    fill_price, fee, slip = simulate_fills(price, side, size_usd)
    state["cash"] -= (fee + slip)
    state["positions"][pair] = {
        "side": side,
        "entry_price": fill_price,
        "size_usd": size_usd,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "fees_paid": fee + slip,
    }
    trade = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair": pair,
        "action": "OPEN_LONG" if side == 1 else "OPEN_SHORT",
        "price": fill_price,
        "size_usd": size_usd,
        "fee": fee,
        "slippage": slip,
        "strategy": strategy,
        "cash_after": state["cash"],
    }
    log_trade(trade)
    return trade


def close_position(state: dict, pair: str, price: float, reason: str) -> dict | None:
    pos = state["positions"].get(pair)
    if not pos:
        return None

    side = pos["side"]
    entry_price = pos["entry_price"]
    size_usd = pos["size_usd"]

    fill_price, fee, slip = simulate_fills(price, -side)
    pnl_pct = (fill_price - entry_price) / entry_price * side
    pnl_usd = size_usd * pnl_pct - fee - slip - pos["fees_paid"]

    state["cash"] += size_usd + pnl_usd
    state["total_trades"] += 1
    state["total_pnl"] += pnl_usd
    if pnl_usd > 0:
        state["wins"] += 1
    else:
        state["losses"] += 1

    del state["positions"][pair]

    trade = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair": pair,
        "action": "CLOSE",
        "entry_price": entry_price,
        "exit_price": fill_price,
        "side": "LONG" if side == 1 else "SHORT",
        "pnl_pct": pnl_pct * 100,
        "pnl_usd": pnl_usd,
        "fee": fee,
        "slippage": slip,
        "reason": reason,
        "strategy": pos["strategy"],
        "cash_after": state["cash"],
    }
    log_trade(trade)
    return trade


def log_trade(trade: dict):
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(trade, default=str) + "\n")


# --- Strategy Execution ---
def run_strategies() -> tuple[dict, list]:
    """Run all strategies. Returns (signals, errors)."""
    signals = {}
    errors = []

    for name, cfg in STRATEGIES.items():
        try:
            df = fetch_latest(cfg["pair"], cfg["timeframe"])
            if df is None or len(df) < 50:
                errors.append(f"{name}: Insufficient data ({len(df) if df is not None else 0} candles)")
                print(f"  {name}: Insufficient data ({len(df) if df is not None else 0} candles)")
                continue

            strat = STRATEGY_REGISTRY[cfg["strategy"]]
            sig = strat.generate_signals(df, cfg["params"])
            latest = int(sig.iloc[-1])
            price = float(df["close"].iloc[-1])
            atr = float(df["close"].diff().abs().rolling(14).mean().iloc[-1]) if len(df) > 14 else price * 0.01
            signals[name] = {
                "signal": latest,
                "weight": cfg["weight"],
                "pair": cfg["pair"],
                "strategy": cfg["strategy"],
                "price": price,
                "atr": atr,
            }
            direction = "LONG" if latest == 1 else "FLAT" if latest == 0 else "SHORT"
            print(f"  {name}: {direction} @ ${price:,.2f}")
        except Exception as e:
            error_msg = f"{name}: {e}"
            errors.append(error_msg)
            print(f"  {name}: ERROR - {e}")
            traceback.print_exc()

    return signals, errors


def aggregate_signals(signals: dict) -> dict:
    pair_scores = {}
    for name, s in signals.items():
        pair = s["pair"]
        if pair not in pair_scores:
            pair_scores[pair] = {"weighted_sum": 0.0, "total_weight": 0.0, "price": s["price"], "atr": s["atr"]}
        pair_scores[pair]["weighted_sum"] += s["signal"] * s["weight"]
        pair_scores[pair]["total_weight"] += s["weight"]

    results = {}
    for pair, data in pair_scores.items():
        if data["total_weight"] > 0:
            score = data["weighted_sum"] / data["total_weight"]
            if score > 0.3:
                final = 1
            elif score < -0.3:
                final = -1
            else:
                final = 0
            results[pair] = {"score": score, "signal": final, "price": data["price"], "atr": data["atr"]}
    return results


def get_equity(state: dict) -> float:
    equity = state["cash"]
    for pair, pos in state["positions"].items():
        equity += pos["size_usd"]
    return equity


# --- Main ---
def main():
    start_time = time.time()
    errors = []

    print("=" * 50)
    print("PAPER TRADING BOT")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 50)
    print()

    notifier = get_notifier()

    # Check for pending Telegram commands before running strategies
    # (uses curl directly, doesn't depend on notifier connection)
    cfg = load_telegram_config()
    if cfg.get("bot_token"):
        try:
            handle_telegram_commands(cfg["bot_token"], str(cfg["chat_id"]))
        except Exception as e:
            print(f"  Telegram command check failed: {e}")

    state = load_state()

    equity = get_equity(state)
    print(f"Starting equity: ${equity:,.2f}")
    print(f"Cash: ${state['cash']:,.2f}")
    print(f"Open positions: {len(state['positions'])}")
    print(f"Total trades: {state['total_trades']} (W:{state['wins']} / L:{state['losses']})")
    print(f"Realized P&L: ${state['total_pnl']:+,.2f}")
    print()

    # Run strategies with retry
    print("Running strategies...")
    try:
        signals, strategy_errors = run_strategies()
        errors.extend(strategy_errors)
    except Exception as e:
        errors.append(f"Strategy execution failed: {e}")
        print(f"FATAL: Strategy execution failed: {e}")
        traceback.print_exc()
        # Save run log and exit
        duration = time.time() - start_time
        log_run("failed", duration, 0, errors, get_equity(state))
        if notifier:
            try:
                notifier.notify_error(f"Bot run failed: {e}", "Strategy execution")
            except Exception:
                pass
        return
    print()

    # Aggregate
    print("Aggregating signals...")
    try:
        portfolio = aggregate_signals(signals)
        for pair, data in portfolio.items():
            direction = "LONG" if data["signal"] == 1 else "FLAT" if data["signal"] == 0 else "SHORT"
            print(f"  {pair}: {direction} (score={data['score']:.3f})")
    except Exception as e:
        errors.append(f"Signal aggregation failed: {e}")
        portfolio = {}
        print(f"ERROR: Signal aggregation failed: {e}")
    print()

    # Execute paper trades
    trades_this_run = []
    for pair, data in portfolio.items():
        try:
            current_pos = state["positions"].get(pair)
            desired = data["signal"]
            current_side = current_pos["side"] if current_pos else 0

            if desired != current_side:
                if current_side != 0:
                    t = close_position(state, pair, data["price"], "Signal reversal")
                    if t:
                        trades_this_run.append(t)
                        print(f"  CLOSED {pair}: P&L ${t['pnl_usd']:+.2f} ({t['pnl_pct']:+.2f}%)")

                if desired != 0:
                    equity_now = get_equity(state)
                    size_usd = min(equity_now * MAX_POSITION_PCT, state["cash"] * 0.95)
                    if size_usd > MIN_TRADE_USD:
                        strat_name = "Unknown"
                        for sname, sdata in signals.items():
                            if sdata["pair"] == pair:
                                strat_name = sdata["strategy"]
                                break
                        t = open_position(state, pair, desired, data["price"], strat_name, size_usd)
                        trades_this_run.append(t)
                        side = "LONG" if desired == 1 else "SHORT"
                        print(f"  OPENED {pair} {side}: ${size_usd:,.2f} @ ${data['price']:,.2f}")
                    else:
                        print(f"  {pair}: Insufficient cash (${state['cash']:.2f})")
        except Exception as e:
            error_msg = f"Trade execution {pair}: {e}"
            errors.append(error_msg)
            print(f"  ERROR: {error_msg}")
            traceback.print_exc()
    print()

    # Update peak equity and drawdown
    equity = get_equity(state)
    if equity > state.get("peak_equity", 0):
        state["peak_equity"] = equity
    dd = (state["peak_equity"] - equity) / state["peak_equity"] if state["peak_equity"] > 0 else 0
    state["max_drawdown"] = max(state.get("max_drawdown", 0), dd)

    save_state(state)

    # Build summary
    wr = state["wins"] / state["total_trades"] * 100 if state["total_trades"] > 0 else 0
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "equity": equity,
        "cash": state["cash"],
        "total_return_pct": total_return,
        "total_trades": state["total_trades"],
        "wins": state["wins"],
        "losses": state["losses"],
        "win_rate": wr,
        "realized_pnl": state["total_pnl"],
        "max_drawdown_pct": state["max_drawdown"] * 100,
        "open_positions": {p: {"side": v["side"], "entry": v["entry_price"], "size": v["size_usd"], "strategy": v["strategy"]} for p, v in state["positions"].items()},
        "trades_this_run": len(trades_this_run),
        "portfolio_signals": {p: {"direction": "LONG" if d["signal"] == 1 else "FLAT" if d["signal"] == 0 else "SHORT", "score": d["score"], "price": d["price"]} for p, d in portfolio.items()},
    }
    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    duration = time.time() - start_time
    print("=" * 50)
    print(f"Equity:     ${equity:,.2f}")
    print(f"Return:     {total_return:+.2f}%")
    print(f"Trades:     {state['total_trades']} (W:{state['wins']} / L:{state['losses']})")
    print(f"Win Rate:   {wr:.1f}%")
    print(f"P&L:        ${state['total_pnl']:+,.2f}")
    print(f"Max DD:     {state['max_drawdown']*100:.2f}%")
    print(f"Open:       {len(state['positions'])} positions")
    print(f"Duration:   {duration:.1f}s")
    if errors:
        print(f"Errors:     {len(errors)}")
        for e in errors:
            print(f"  - {e}")
    print("=" * 50)

    # Log run
    status = "success" if not errors else "partial" if trades_this_run else "failed"
    log_run(status, duration, len(trades_this_run), errors, equity)

    # Send Telegram
    if notifier:
        try:
            for t in trades_this_run:
                if t["action"].startswith("OPEN"):
                    side_str = "buy" if "LONG" in t["action"] else "sell"
                    notifier.notify_trade_open(
                        pair=t["pair"].replace("_", "/"),
                        side=side_str,
                        price=t["price"],
                        amount=t["size_usd"],
                        strategy=t["strategy"],
                        confidence=0.8,
                    )
                elif t["action"] == "CLOSE":
                    side_str = "buy" if t.get("side", "LONG") == "LONG" else "sell"
                    notifier.notify_trade_close(
                        pair=t["pair"].replace("_", "/"),
                        side=side_str,
                        entry_price=t["entry_price"],
                        exit_price=t["exit_price"],
                        pnl_pct=t["pnl_pct"],
                        pnl_usd=t["pnl_usd"],
                        reason=t["reason"],
                    )
            # Daily summary
            daily_pnl = state["total_pnl"]
            daily_pnl_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            positions_list = []
            for pair, pos in state["positions"].items():
                positions_list.append({
                    "pair": pair.replace("_", "/"),
                    "side": "long" if pos.get("side") == 1 else "short",
                    "unrealized_pnl_pct": 0.0,
                })
            notifier.notify_daily_summary(
                equity=equity,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                trades_today=len(trades_this_run),
                open_positions=positions_list,
                win_rate=wr / 100,
                total_trades=state["total_trades"],
            )
            print("Telegram notifications sent.")
        except Exception as e:
            print(f"Telegram failed: {e}")

    # Send error alert if there were critical errors
    if errors and notifier:
        try:
            notifier.notify_error(f"Bot completed with {len(errors)} error(s)", "; ".join(errors[:3]))
        except Exception:
            pass

    print("Done.")


if __name__ == "__main__":
    main()
