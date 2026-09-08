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

from trading_system.strategies import STRATEGY_REGISTRY
from trading_system.bot.risk_manager import DEFAULT_RISK_MANAGER
from trading_system.bot.candles import closed_candles
from trading_system.bot.accounting import charge_funding, FEE_RATE, SLIPPAGE_RATE


# --- Strategy Configs ---
# These hardcoded defaults are ONLY a fallback when bot_strategy_params.json
# (the walk-forward league's promoted portfolio, now committed to the repo)
# is missing or invalid. They mirror the deployed portfolio exactly so a
# fallback can never silently resurrect a strategy the league rejected.
STRATEGIES = {
    "Donchian_Breakout_ETH_USDT_USDT": {
        "strategy": "Donchian_Breakout",
        "pair": "ETH_USDT_USDT",
        "timeframe": "4h",
        "weight": 0.40,
        "params": {"channel_period": 40, "exit_period": 10, "atr_filter": False, "atr_period": 14},
    },
    "Davey_Momentum_Pullback_BTC_USDT_USDT": {
        "strategy": "Davey_Momentum_Pullback",
        "pair": "BTC_USDT_USDT",
        "timeframe": "4h",
        "weight": 0.35,
        "params": {"bar_count": 2, "pullback": 3, "exit_bars": 8, "count_higher_highs": True},
    },
    "RSI_Reversion_BTC_USDT_USDT": {
        "strategy": "RSI_Reversion",
        "pair": "BTC_USDT_USDT",
        "timeframe": "4h",
        "weight": 0.25,
        "params": {"rsi_period": 14, "entry_oversold": 25, "entry_overbought": 65,
                   "exit_neutral_low": 45, "exit_neutral_high": 55, "use_bb_filter": True,
                   "bb_period": 25, "bb_std": 2.0},
    },
}



def load_optimized_strategies() -> dict:
    """Load strategy params from optimized JSON if available.
    
    Falls back to hardcoded defaults if no optimized file exists.
    This allows monthly re-optimization to update params without code changes.
    """
    if OPTIMIZED_PARAMS_FILE.exists():
        try:
            with open(OPTIMIZED_PARAMS_FILE) as f:
                optimized = json.load(f)
            if optimized:
                print(f"  Loaded optimized params from {OPTIMIZED_PARAMS_FILE.name}")
                return optimized
        except Exception as e:
            print(f"  WARNING: Could not load optimized params: {e}")
    return None


# Active strategies — set by load_active_strategies(), used by all functions
ACTIVE_STRATEGIES: dict = STRATEGIES.copy()


def load_active_strategies() -> dict:
    """Load strategy configs, preferring optimized params if available.
    Sets the module-level ACTIVE_STRATEGIES variable.
    """
    global ACTIVE_STRATEGIES
    optimized = load_optimized_strategies()
    if optimized:
        valid = {}
        for key, cfg in optimized.items():
            required = ["strategy", "pair", "timeframe", "weight", "params"]
            if all(k in cfg for k in required) and cfg["strategy"] in STRATEGY_REGISTRY:
                valid[key] = cfg
            else:
                print(f"  WARNING: Skipping invalid optimized config '{key}'")
        if valid:
            ACTIVE_STRATEGIES = valid
            print(f"  Using {len(valid)} optimized strategy configs")
            return valid
        print("  WARNING: No valid optimized configs, using defaults")
    ACTIVE_STRATEGIES = STRATEGIES.copy()
    return ACTIVE_STRATEGIES


# --- Config ---
# Cost model is shared with the backtest engines (trading_system.bot.accounting):
# 0.05% fee per side, 0.02% slippage per side, funding at 8h UTC boundaries.
INITIAL_CAPITAL = 97.0
MAX_POSITION_PCT = 0.35
MIN_TRADE_USD = 5.0  # Minimum trade size (reduced from 100 for small accounts)
LOG_DIR = Path("data/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG = LOG_DIR / "paper_trades.jsonl"
STATE_FILE = LOG_DIR / "paper_state.json"
SUMMARY_FILE = LOG_DIR / "paper_summary.json"
RUN_LOG = LOG_DIR / "run_history.jsonl"
OPTIMIZED_PARAMS_FILE = LOG_DIR / "bot_strategy_params.json"
LOCK_FILE = LOG_DIR / "bot.lock"
_LOCK_HELD = False


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
            if last_error is None:
                last_error = RuntimeError(f"{func.__name__}: retry loop exhausted without result")
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


# --- Run Lock (prevents concurrent instances from double-opening positions) ---
def acquire_lock(max_wait_seconds: int = 180, stale_seconds: int = 600) -> bool:
    """Acquire an exclusive run lock. Returns True if acquired.

    Uses an atomic O_CREAT|O_EXCL file create. A lock older than
    ``stale_seconds`` is considered abandoned (e.g. crash) and broken.
    """
    global _LOCK_HELD
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max_wait_seconds
    while True:
        try:
            fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            _LOCK_HELD = True
            return True
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(LOCK_FILE)
                if age > stale_seconds:
                    print(f"  Run lock is stale ({age:.0f}s old) — removing and retrying.")
                    os.remove(LOCK_FILE)
                    continue
            except OSError:
                pass
            if time.time() >= deadline:
                print("  Could not acquire run lock (another instance is running).")
                return False
            time.sleep(min(5, max(0.1, deadline - time.time())))
        except OSError as e:
            print(f"  Could not acquire run lock: {e}")
            return False


def release_lock():
    """Release the run lock if this process holds it."""
    global _LOCK_HELD
    if not _LOCK_HELD:
        return
    try:
        os.remove(LOCK_FILE)
    except OSError:
        pass
    _LOCK_HELD = False


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
        print("  Telegram: Configured")
        return n
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


def tg_send_message(token: str, chat_id: str, text: str,
                    parse_mode: str | None = None) -> bool:
    """Send a Telegram message.

    parse_mode defaults to plain text: most bot messages contain dynamic
    content (strategy names, error strings, prices) that can include
    characters Telegram's HTML parser rejects (e.g. "<50 candles"), and a
    rejected parse makes Telegram return 400 — the whole message silently
    vanishes. Callers that genuinely use HTML tags (the watchdog alert)
    pass parse_mode="HTML" explicitly.
    """
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    resp = tg_api_call(token, "sendMessage", payload)
    return resp.get("ok", False)


def handle_telegram_commands(token: str, chat_id: str):
    """Check for pending Telegram commands and respond to them."""
    print("\nChecking for Telegram commands...")

    # Get recent updates
    resp = tg_api_call(token, "getUpdates", {"limit": 10, "timeout": 0})
    if not resp.get("ok") or not resp.get("result"):
        print("  No pending commands.")
        return

    # Track highest update_id so we can acknowledge processed updates
    max_update_id = 0

    state = load_state()
    # Always fetch live prices for command responses
    cmd_prices = fetch_live_prices() if state.get("positions") else {}
    equity = get_equity(state, cmd_prices)
    total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    wr = state["wins"] / state["total_trades"] * 100 if state["total_trades"] > 0 else 0

    for update in resp["result"]:
        msg = update.get("message", {})
        text = msg.get("text", "").strip().lower()
        msg_chat_id = str(msg.get("chat", {}).get("id", ""))
        update_id = update.get("update_id", 0)

        # Track highest update_id regardless of chat
        if update_id > max_update_id:
            max_update_id = update_id

        if msg_chat_id != chat_id:
            continue

        if text == "/help":
            print("  -> /help")
            tg_send_message(token, chat_id, (
                "TRADING BOT COMMANDS\n\n"
                "/status - Portfolio status\n"
                "/equity - Quick equity check\n"
                "/balance - Live equity with prices\n"
                "/positions - Open positions\n"
                "/pnl - Detailed position P&L\n"
                "/trades - Recent trade history\n"
                "/signals - Current strategy signals\n"
                "/restart - Run strategies now\n"
                "/dashboard - Bot monitoring\n"
                "/help - This message\n\n"
                f"Mode: PAPER\n"
                f"Capital: ${INITIAL_CAPITAL:,.0f}\n"
                f"Strategies: " + ", ".join(sorted(ACTIVE_STRATEGIES.keys())) + "\n"
                "Run: every ~2h (GitHub Actions)"
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
                tg_send_message(token, chat_id, "No trades yet")
            else:
                msg_text = "RECENT TRADES\n\n"
                for t in reversed(recent[-5:]):  # Last 5 trades only
                    action = t.get("action", "?")
                    pair = format_pair(t.get("pair", "?"))
                    if action == "CLOSE":
                        pnl = t.get("pnl_usd", 0)
                        emoji = "+" if pnl >= 0 else ""
                        msg_text += f"{emoji} {pair} CLOSED ${t.get('exit_price', 0):,.2f}"
                        msg_text += f" P&L: ${pnl:+.2f} ({t.get('pnl_pct', 0):+.2f}%)\n"
                    elif action.startswith("OPEN"):
                        msg_text += f"+ {pair} OPEN ${t.get('price', 0):,.2f} ({action})\n"
                tg_send_message(token, chat_id, msg_text)

        elif text in ("/equity", "/balance"):
            print(f"  -> {text}")
            tg_send_message(token, chat_id, (
                f"${equity:,.2f} ({total_return:+.1f}%)"
            ))

        elif text == "/status":
            print("  -> /status")
            positions_text = ""
            for pair, pos in state.get("positions", {}).items():
                side = "LONG" if pos.get("side") == 1 else "SHORT"
                entry = pos.get('entry_price', 0)
                current = cmd_prices.get(pair, entry)
                size = pos.get('size_usd', 0)
                side_int = pos.get('side', 0)
                if entry > 0 and size > 0:
                    qty = size / entry
                    if side_int == 1:
                        upnl = qty * (current - entry)
                    else:
                        upnl = qty * (entry - current)
                    upnl_pct = upnl / size * 100
                    positions_text += f"{'+' if upnl >= 0 else ''}{format_pair(pair)} {side} @ ${entry:,.2f} -> ${current:,.2f} ({upnl_pct:+.1f}% ${upnl:+.2f})\n"
                else:
                    positions_text += f"{'+' if side_int == 1 else ''}{format_pair(pair)} {side} @ ${entry:,.2f}\n"
            if not positions_text:
                positions_text = "No open positions\n"
            # Calculate unrealized P&L (GROSS marks — matches get_equity;
            # realized total_pnl already carries all costs, so
            # total_pnl + gross_unrealized == equity - start exactly).
            total_unrealized = 0.0
            for pair, pos in state.get("positions", {}).items():
                current = cmd_prices.get(pair, pos.get('entry_price', 0))
                entry = pos.get('entry_price', 0)
                size = pos.get('size_usd', 0)
                side_int = pos.get('side', 0)
                if entry > 0 and size > 0:
                    qty = size / entry
                    if side_int == 1:
                        total_unrealized += qty * (current - entry)
                    else:
                        total_unrealized += qty * (entry - current)
            total_pnl = state['total_pnl'] + total_unrealized
            tg_send_message(token, chat_id, (
                f"STATUS\n"
                f"{'='*28}\n"
                f"Equity: ${equity:,.2f} ({total_return:+.1f}%)\n"
                f"P&L: ${total_pnl:+.2f}"
                + (f" | {wr:.0f}% win ({state['total_trades']} trades)" if state['total_trades'] > 0 else "")
                + f"\n\n{positions_text}"
                + f"{datetime.now(timezone.utc).strftime('%b %d, %H:%M UTC')}"
            ))

        elif text == "/positions":
            print("  -> /positions")
            positions = state.get("positions", {})
            if not positions:
                tg_send_message(token, chat_id, "No open positions")
            else:
                msg_text = "OPEN POSITIONS\n\n"
                for pair, pos in positions.items():
                    side = "LONG" if pos.get("side") == 1 else "SHORT"
                    emoji = "+" if pos.get("side") == 1 else ""
                    msg_text += f"{emoji} {format_pair(pair)} {side}\n"
                    msg_text += f"  Entry: ${pos.get('entry_price', 0):,.2f}\n"
                    msg_text += f"  Size: ${pos.get('size_usd', 0):,.2f}\n"
                    msg_text += f"  Strategy: {pos.get('strategy', 'Unknown')}\n\n"
                tg_send_message(token, chat_id, msg_text)

        elif text == "/pnl":
            print("  -> /pnl")
            positions = state.get("positions", {})
            if not positions:
                tg_send_message(token, chat_id, "No open positions")
            else:
                # Fetch live prices for accurate P&L
                live_prices = {}
                for pair in positions.keys():
                    try:
                        for name, cfg in ACTIVE_STRATEGIES.items():
                            if cfg["pair"] == pair:
                                df = fetch_latest(cfg["pair"], cfg["timeframe"])
                                if df is not None and len(df) > 0:
                                    live_prices[pair] = float(df["close"].iloc[-1])
                                    break
                    except Exception:
                        pass
                
                msg_text = "POSITION P&L\n\n"
                total_pnl = 0.0
                for pair, pos in positions.items():
                    side = "LONG" if pos.get("side") == 1 else "SHORT"
                    entry = pos.get("entry_price", 0)
                    size = pos.get("size_usd", 0)
                    strategy = pos.get("strategy", "Unknown")
                    current = live_prices.get(pair, entry)
                    
                    if entry > 0:
                        qty = size / entry
                        if pos.get("side", 0) == 1:  # LONG
                            pnl_usd = qty * (current - entry)
                        else:  # SHORT
                            pnl_usd = qty * (entry - current)
                        # GROSS mark P&L — consistent with the daily message.
                        # Funding is shown separately below, not netted here.
                        pnl_pct = pnl_usd / size * 100 if size > 0 else 0
                    else:
                        pnl_usd = 0
                        pnl_pct = 0
                    
                    total_pnl += pnl_usd
                    emoji = "+" if pnl_usd >= 0 else ""
                    msg_text += f"{format_pair(pair)} {side}\n"
                    msg_text += f"  Entry: ${entry:,.2f} -> ${current:,.2f}\n"
                    msg_text += f"  P&L: {pnl_pct:+.1f}% (${pnl_usd:+.2f})\n"
                    msg_text += f"  Size: ${size:.2f} | {strategy}\n\n"
                
                msg_text += f"Total: ${total_pnl:+.2f}\n"
                funding_note = sum(
                    p.get("funding_paid", 0.0) for p in positions.values())
                if funding_note > 0:
                    msg_text += f"Funding paid to date: -${funding_note:.4f} (already in Realized)\n"
                msg_text += f"{datetime.now(timezone.utc).strftime('%b %d, %H:%M UTC')}"
                tg_send_message(token, chat_id, msg_text)

        elif text == "/signals":
            print("  -> /signals (running strategies...)")
            try:
                from trading_system.strategies import STRATEGY_REGISTRY as SR
                signals_text = "CURRENT SIGNALS\n\n"
                for name, cfg in ACTIVE_STRATEGIES.items():
                    try:
                        df = fetch_latest(cfg["pair"], cfg["timeframe"])
                        strat = SR[cfg["strategy"]]
                        sig = strat.generate_signals(df, cfg["params"])
                        latest = int(sig.iloc[-1])
                        price = float(df["close"].iloc[-1])
                        direction = "LONG" if latest == 1 else "FLAT" if latest == 0 else "SHORT"
                        emoji = "+" if latest == 1 else "=" if latest == 0 else ""
                        signals_text += f"{emoji} {name}: {direction} (${price:,.2f})\n"
                    except Exception as e:
                        signals_text += f"! {name}: Error - {e}\n"
                tg_send_message(token, chat_id, signals_text)
            except Exception as e:
                tg_send_message(token, chat_id, f"Error getting signals: {e}")

        elif text == "/restart":
            print("  -> /restart")
            # Honesty first: this runner cannot trigger a cycle on demand
            # (GitHub Actions fires on its own schedule). Saying "running
            # now... results in ~1 minute" would be a lie the user plans
            # around. Point them at /signals for an immediate read.
            tg_send_message(token, chat_id, (
                "Command acknowledged.\n"
                "This runner (GitHub Actions) cycles every ~2h on its own "
                "schedule — the next run is at most ~2h away.\n"
                "For an immediate signal read, use /signals."
            ))

        elif text == "/dashboard":
            print("  -> /dashboard")
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
                    f"BOT DASHBOARD\n\n"
                    f"Status: {last.get('status', '?').upper()}\n"
                    f"Runs: {total} ({success_rate:.0f}% success)\n"
                    f"Failed: {sum(1 for r in runs if r.get('status') == 'failed')}\n"
                    f"Consec. fails: {consec}\n\n"
                    f"Equity: ${equity:,.2f} ({total_return:+.1f}%)\n"
                    f"Last run: {last.get('duration_seconds', 0):.1f}s\n"
                    f"{last.get('timestamp', '?')[:19]}"
                ))

        elif text.startswith("/"):
            print(f"  -> Unknown: {text}")
            tg_send_message(token, chat_id, f"Unknown command: {text}\nType /help for commands.")

    # Acknowledge all processed updates so they aren't re-processed
    if max_update_id > 0:
        tg_api_call(token, "getUpdates", {"offset": max_update_id + 1, "limit": 1, "timeout": 0})

    print("  Command check done.")


# --- State Management ---
def format_pair(pair: str) -> str:
    """Convert internal pair format to display format.
    
    ETH_USDT_USDT -> ETH/USDT
    BTC_USDT_USDT -> BTC/USDT
    """
    # Extract the base asset from compound pair names
    if "_USDT_USDT" in pair:
        base = pair.replace("_USDT_USDT", "")
        return f"{base}/USDT"
    if "_USDT" in pair:
        base = pair.replace("_USDT", "")
        return f"{base}/USDT"
    return pair.replace("_", "/")


def _fresh_state() -> dict:
    """Create a clean state with correct initial capital."""
    return {
        "cash": INITIAL_CAPITAL,
        "positions": {},
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "realized_pnl": 0.0,
        "peak_equity": INITIAL_CAPITAL,
        "max_drawdown": 0.0,
        # Signal history for hysteresis — prevents overtrading
        "signal_history": {},  # {pair: [last_signal, prev_signal, ...]}
        "last_daily_alert_date": "",  # Track daily status to avoid duplicates across restarts
    }


def load_state() -> dict:
    """Load state from disk, with validation against stale cached data.
    
    GitHub Actions cache can restore old state files from previous code
    versions (e.g. when INITIAL_CAPITAL was $10,000). This detects
    mismatches and resets to avoid wildly wrong position sizes.
    """
    if not STATE_FILE.exists():
        return _fresh_state()
    
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except (json.JSONDecodeError, KeyError):
        print("  WARNING: Corrupt state file, resetting")
        return _fresh_state()
    
    # Validate equity is plausible for a $97 account
    cash = state.get("cash", 0)
    positions = state.get("positions", {})
    total_position_value = sum(p.get("size_usd", 0) for p in positions.values())
    apparent_equity = cash + total_position_value
    
    # HARD RULE: If cash or apparent equity is >5x the initial capital, 
    # it's a stale cache from a previous code version. Reset immediately.
    # This catches ALL cases: whether there are trades, positions, or not.
    MAX_PLAUSIBLE_EQUITY = INITIAL_CAPITAL * 5  # $485 max for a $97 account
    
    if cash > MAX_PLAUSIBLE_EQUITY:
        print(f"  WARNING: cash=${cash:.2f} exceeds max plausible ${MAX_PLAUSIBLE_EQUITY:.2f}")
        print("  WARNING: Stale cache from previous code version — resetting")
        return _fresh_state()
    
    if apparent_equity > MAX_PLAUSIBLE_EQUITY:
        print(f"  WARNING: equity=${apparent_equity:.2f} exceeds max plausible ${MAX_PLAUSIBLE_EQUITY:.2f}")
        print("  WARNING: Stale cache — resetting")
        return _fresh_state()
    
    if state.get("peak_equity", 0) > MAX_PLAUSIBLE_EQUITY:
        print(f"  WARNING: peak_equity=${state['peak_equity']:.2f} exceeds max plausible")
        print("  WARNING: Stale cache — resetting")
        return _fresh_state()
    
    # Also check if cash is negative (should never happen)
    if cash < 0:
        print(f"  WARNING: cash=${cash:.2f} is negative — resetting")
        return _fresh_state()
    
    return state


def save_state(state: dict):
    """Atomically persist state (write temp file, then rename)."""
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


# --- Data-freshness guard ---
# A trading decision on stale candles is a decision on a market that no
# longer exists. Every signal must come from a candle that CLOSED recently:
# at most one candle interval plus a small fetch margin ago.
MAX_CANDLE_AGE_FACTOR = 2.0  # candle may be at most 2x its interval old


def candle_age_seconds(df: pd.DataFrame, timeframe: str) -> float | None:
    """Age of the most recent CLOSED candle in seconds, or None if unknown."""
    from trading_system.bot.candles import TIMEFRAME_SECONDS
    tf_sec = TIMEFRAME_SECONDS.get(timeframe, 3600)
    closed = closed_candles(df, timeframe)
    if closed is None or len(closed) == 0 or "timestamp" not in closed.columns:
        return None
    try:
        last_ts = pd.to_datetime(closed["timestamp"].iloc[-1], utc=True)
    except Exception:
        return None
    # The candle opened at last_ts; it closed last_ts + tf_sec.
    close_ts = last_ts + pd.Timedelta(seconds=tf_sec)
    now = pd.Timestamp.now(tz="UTC")
    return max(0.0, (now - close_ts).total_seconds())


def assert_fresh_candles(df: pd.DataFrame, timeframe: str, pair: str = "") -> None:
    """Raise if the most recent closed candle is too old to trade on.

    Guards against: exchange API silently returning old data, a fallback
    cache being used without timestamps, clock issues, and any future
    regression in the data pipeline. Fail loud, never trade stale.
    """
    age = candle_age_seconds(df, timeframe)
    from trading_system.bot.candles import TIMEFRAME_SECONDS
    max_age = TIMEFRAME_SECONDS.get(timeframe, 3600) * MAX_CANDLE_AGE_FACTOR
    if age is None:
        raise RuntimeError(
            f"STALE/UNKNOWN data for {pair}: cannot determine candle age")
    if age > max_age:
        raise RuntimeError(
            f"STALE data for {pair}: latest closed candle is {age/60:.0f}min old "
            f"(max {max_age/60:.0f}min) — refusing to trade")


# --- Idempotent open guard ---
def already_open(state: dict, pair: str, side: int, max_age_seconds: int = 3600) -> bool:
    """True if an identical open was already logged recently for this pair+side.

    Defense in depth against duplicate opens: the run lock and the workflow
    concurrency group prevent concurrent runs, but a state save that fails
    after an open (crash, disk full, timeout) would lose the position from
    paper_state.json while the trade log still shows it — the next run would
    open the same position again. This cross-checks the trade log.
    """
    if not TRADE_LOG.exists():
        return False
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
    action = "OPEN_LONG" if side == 1 else "OPEN_SHORT"
    try:
        with open(TRADE_LOG) as f:
            for line in reversed(f.readlines()[-200:]):
                try:
                    t = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = t.get("timestamp", "")
                if ts < cutoff:      # ISO strings sort chronologically
                    break
                if (t.get("pair") == pair and t.get("action") == action
                        and "paper-state-consistency" not in str(t.get("note", ""))):
                    return True
    except OSError:
        return False
    return False


# --- Data ---
def fetch_fresh_data(pair: str, timeframe: str) -> pd.DataFrame:
    """Fetch fresh candles via ccxt exchanges. Returns OHLCV with timestamp column."""
    import ccxt
    # ETH_USDT_USDT -> ETH/USDT:USDT or ETH/USDT
    symbol = pair.replace("_USDT_USDT", "/USDT:USDT").replace("_", "/")
    if not "/USDT" in symbol:
        symbol = pair.replace("_", "/") + "/USDT"
    
    # Try Kraken first (works from US, no geo-blocking)
    try:
        exchange = ccxt.kraken({"enableRateLimit": True})
        spot_symbol = symbol.replace(":USDT", "")  # Kraken doesn't use perp
        ohlcv = exchange.fetch_ohlcv(spot_symbol, timeframe, limit=1000)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        print(f"    Kraken: Got {len(df)} candles for {spot_symbol}")
        return df
    except Exception as e:
        print(f"    Kraken failed: {e}")
    
    # Fallback to Binance
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        print(f"    Binance: Got {len(df)} candles for {symbol}")
        return df
    except Exception as e:
        print(f"    Binance failed: {e}")

    return None


@retry(max_attempts=3, base_delay=2.0)
def fetch_latest(pair: str, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
    """Fetch latest candles. Always prefers fresh API data over cached files.
    
    Cached parquet files from optimization runs are ONLY used if they have
    a timestamp column with recent data (< lookback_days old).
    Stale cache (no timestamps or old data) is ignored.
    """
    # ALWAYS fetch fresh data first — this is a trading bot, prices must be live
    fresh = fetch_fresh_data(pair, timeframe)
    if fresh is not None and len(fresh) > 0:
        return fresh

    # Fallback: try cached files, but ONLY if they have recent timestamps
    print("    API failed, trying cached files...")
    for pattern in [f"data/raw/{pair}/{timeframe}.parquet", f"data/raw/{pair}/klines_{timeframe}.parquet"]:
        cache_path = Path(pattern)
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            if "timestamp" in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
                df = df[df["timestamp"] > cutoff]
                if len(df) > 0:
                    print(f"    Cache hit: {len(df)} recent candles from {cache_path}")
                    return df
                else:
                    print(f"    Cache stale: no candles after {cutoff}")
            else:
                print("    Cache rejected: no timestamp column (optimization data, not trading data)")

    raise Exception(f"All data sources failed for {pair}")


# --- Paper Trade Execution ---
def simulate_fills(price: float, side: int, size_usd: float) -> tuple[float, float, float]:
    slippage = price * SLIPPAGE_RATE * side
    fill_price = price + slippage
    fee = size_usd * FEE_RATE
    return fill_price, fee, abs(slippage) * size_usd / price


def open_position(state: dict, pair: str, side: int, price: float, strategy: str, size_usd: float) -> dict:
    fill_price, fee, slip = simulate_fills(price, side, size_usd)
    # Slippage is embedded in the fill price (like the backtest engine), so
    # only the fee is an extra cash cost. Deducting slippage here as well
    # would double-charge it and make paper P&L diverge from the backtest.
    state["cash"] -= (size_usd + fee)
    state["total_pnl"] -= fee  # entry fee is an immediate P&L cost (matches cash)
    state["positions"][pair] = {
        "side": side,
        "entry_price": fill_price,
        "size_usd": size_usd,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "fees_paid": fee + slip,  # informational total cost; cash impact is the fee only
        "funding_paid": 0.0,
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

    fill_price, fee, slip = simulate_fills(price, -side, size_usd)
    pnl_pct = (fill_price - entry_price) / entry_price * side
    # Gross P&L is computed on fill prices, so slippage is already included
    # on both sides. Only the exit fee is an extra cost (the entry fee was
    # already booked into total_pnl at open). Matches the backtest engine.
    pnl_usd = size_usd * pnl_pct - fee

    state["cash"] += size_usd + pnl_usd
    state["total_trades"] += 1
    state["total_pnl"] += pnl_usd
    state["realized_pnl"] = state.get("realized_pnl", 0) + pnl_usd
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
        "size_usd": size_usd,
        "side": "LONG" if side == 1 else "SHORT",
        "pnl_pct": pnl_pct * 100,
        "pnl_usd": pnl_usd,
        "fee": fee,
        "slippage": slip,
        "funding": pos.get("funding_paid", 0.0),
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

    for name, cfg in ACTIVE_STRATEGIES.items():
        try:
            df = fetch_latest(cfg["pair"], cfg["timeframe"])
            if df is None or len(df) < 50:
                errors.append(f"{name}: Insufficient data ({len(df) if df is not None else 0} candles)")
                print(f"  {name}: Insufficient data ({len(df) if df is not None else 0} candles)")
                continue

            # Signals must come from fully-closed candles only, so live
            # behavior matches the next-open backtest model.
            closed = closed_candles(df, cfg["timeframe"])
            if closed is None or len(closed) < 50:
                errors.append(f"{name}: insufficient closed candles ({len(closed) if closed is not None else 0})")
                print(f"  {name}: Insufficient closed candles ({len(closed) if closed is not None else 0})")
                continue
            # NEVER trade on stale candles: a signal computed on hours-old
            # data describes a market that no longer exists. Fail loud.
            try:
                assert_fresh_candles(df, cfg["timeframe"], cfg["pair"])
            except RuntimeError as e:
                errors.append(f"{name}: {e}")
                print(f"  {name}: STALE — {e}")
                continue
            strat = STRATEGY_REGISTRY[cfg["strategy"]]
            sig = strat.generate_signals(closed, cfg["params"])
            latest = int(sig.iloc[-1])
            price = float(closed["close"].iloc[-1])
            candle_time = pd.to_datetime(closed["timestamp"].iloc[-1], utc=True)
            # Next-open fill price: the open of the currently forming candle is
            # exactly the "next open" after the last closed candle, which is the
            # backtest engine's fill convention. Falls back to the last closed
            # close only if the exchange returned no forming candle.
            next_open = float(df["open"].iloc[-1]) if len(df) > len(closed) else price
            atr = float(closed["close"].diff().abs().rolling(14).mean().iloc[-1]) if len(closed) > 14 else price * 0.01
            signals[name] = {
                "signal": latest,
                "weight": cfg["weight"],
                "pair": cfg["pair"],
                "strategy": cfg["strategy"],
                "price": price,
                "next_open": next_open,
                "atr": atr,
                "candle_time": candle_time,
            }
            direction = "LONG" if latest == 1 else "FLAT" if latest == 0 else "SHORT"
            print(f"  {name}: {direction} @ ${price:,.2f}")
        except Exception as e:
            error_msg = f"{name}: {e}"
            errors.append(error_msg)
            print(f"  {name}: ERROR - {e}")
            traceback.print_exc()

    return signals, errors


def aggregate_signals(signals: dict, state: dict = None) -> dict:
    """Aggregate signals with hysteresis to prevent overtrading.
    
    Hysteresis rules:
    - If we have NO current position: require score > 0.4 (LONG) or < -0.4 (SHORT)
    - If we HAVE a position: only flip if score crosses 0 (neutral) or reverses strongly
    - This widens the dead zone when entering and prevents whipsaws when holding
    """
    pair_scores = {}
    for name, s in signals.items():
        pair = s["pair"]
        if pair not in pair_scores:        pair_scores[pair] = {"weighted_sum": 0.0, "total_weight": 0.0,
                             "price": s["price"], "next_open": s.get("next_open") or s["price"],
                             "atr": s["atr"], "candle_time": None}
        pair_scores[pair]["weighted_sum"] += s["signal"] * s["weight"]
        pair_scores[pair]["total_weight"] += s["weight"]
        ct = s.get("candle_time")
        if ct is not None:
            cur = pair_scores[pair].get("candle_time")
            if cur is None or ct > cur:
                pair_scores[pair]["candle_time"] = ct
                pair_scores[pair]["next_open"] = s.get("next_open") or s["price"]

    state_positions = state.get("positions", {}) if state else {}
    
    results = {}
    for pair, data in pair_scores.items():
        if data["total_weight"] > 0:
            score = data["weighted_sum"] / data["total_weight"]
            current_pos = state_positions.get(pair)
            current_side = current_pos["side"] if current_pos else 0
            
            if current_side == 0:
                # No position — require strong signal to enter (wider dead zone)
                if score > 0.4:
                    final = 1
                elif score < -0.4:
                    final = -1
                else:
                    final = 0
            else:
                # Have a position — only flip if score clearly reverses
                # Close SHORT when score > 0.1 (slightly positive = trend reversing up)
                # Close LONG when score < -0.1 (slightly negative = trend reversing down)
                # This prevents closing on tiny score fluctuations
                if current_side == 1:  # LONG
                    final = 1 if score > -0.1 else 0
                else:  # SHORT
                    final = -1 if score < 0.1 else 0
            
            results[pair] = {"score": score, "signal": final, "price": data["price"],
                             "next_open": data.get("next_open") or data["price"],
                             "atr": data["atr"], "candle_time": data.get("candle_time")}
    return results


def fetch_live_prices(pairs: list[str] = None) -> dict[str, float]:
    """Fetch live prices for all pairs. Returns {pair: price} dict.
    
    Always fetches fresh data — never uses cached/stale prices.
    This is the SINGLE source of truth for all equity calculations.
    """
    all_pairs = set()
    # Add pairs from active strategies
    for name, cfg in ACTIVE_STRATEGIES.items():
        all_pairs.add(cfg["pair"])
    # Add any explicitly requested pairs
    if pairs:
        all_pairs.update(pairs)
    
    prices = {}
    for pair in all_pairs:
        try:
            for name, cfg in ACTIVE_STRATEGIES.items():
                if cfg["pair"] == pair:
                    df = fetch_latest(pair, cfg["timeframe"])
                    if df is not None and len(df) > 0:
                        prices[pair] = float(df["close"].iloc[-1])
                        break
        except Exception as e:
            print(f"    WARNING: Could not fetch live price for {pair}: {e}")
    return prices


def get_equity(state: dict, prices: dict = None) -> float:
    """Calculate total equity = cash + position market values.
    
    ALWAYS fetches live prices if not provided.
    This prevents stale/missing prices from showing wrong equity.
    
    Cash was debited by (size_usd + fees) on open.
    When you close, cash gets back size_usd + pnl_usd.
    So position value = size_usd + unrealized_pnl.
    
    LONG:  unrealized_pnl = qty * (current - entry)
    SHORT: unrealized_pnl = qty * (entry - current)
    
    Position value (both sides) = size_usd + unrealized_pnl
    """
    # If we have open positions and no prices provided, fetch them
    if state.get("positions") and (prices is None or len(prices) < len(state["positions"])):
        missing = [p for p in state["positions"] if p not in (prices or {})]
        if missing:
            fetched = fetch_live_prices(missing)
            if prices:
                prices = {**prices, **fetched}
            else:
                prices = fetched
    
    equity = state["cash"]
    for pair, pos in state["positions"].items():
        entry = pos.get("entry_price", 0)
        size_usd = pos.get("size_usd", 0)
        side = pos.get("side", 0)
        
        if prices and pair in prices and entry > 0:
            current = prices[pair]
            qty = size_usd / entry
            if side == 1:  # LONG
                unrealized_pnl = qty * (current - entry)
            else:  # SHORT
                unrealized_pnl = qty * (entry - current)
            equity += size_usd + unrealized_pnl
        else:
            # Last resort: use cost basis (cash was already debited)
            equity += size_usd
    return equity


# --- Main ---
def main():
    start_time = time.time()
    errors = []

    # Load strategies (optimized params if available, else defaults)
    load_active_strategies()

    print("=" * 50)
    print("PAPER TRADING BOT")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"Strategies: {', '.join(ACTIVE_STRATEGIES.keys())}")
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

    # Exclusive run lock: concurrent schedulers, manual runs, and Telegram
    # /restart must never run this bot simultaneously (observed double-open bug).
    if not acquire_lock():
        print(f"[{datetime.now(timezone.utc).isoformat()}] Another bot instance is running — skipping this run.")
        return

    state = load_state()

    # Charge 8h funding (00:00/08:00/16:00 UTC) on all open positions, so
    # P&L matches the backtest engine's cost model. Must run after the lock
    # and before any equity calculation or trade.
    funding_total, funding_per_pair = charge_funding(state)
    if funding_total > 0:
        print(f"  Funding charged: ${funding_total:.4f}")
        for pair, amt in funding_per_pair.items():
            print(f"    {format_pair(pair)}: ${amt:.4f}")
        # Every cash change must hit the ledger, or flat-moment reconciliation
        # shows an unexplained gap (observed 2026-09-07: funding debits were
        # invisible to the audit tool). Funding is a portfolio-level event,
        # so the entry carries the "*" sentinel pair — every ledger entry
        # must have a pair key so log scans never hit KeyError.
        log_trade({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "FUND",
            "pair": "*",
            "funding": funding_total,
            "cash_after": state["cash"],
        })

    # Fetch ALL live prices ONCE — single source of truth for entire run
    print("Fetching live prices...")
    current_prices = fetch_live_prices()
    print(f"  Live prices: {', '.join(f'{format_pair(pair)}=${price:,.2f}' for pair, price in current_prices.items())}")
    print()

    equity = get_equity(state, current_prices)
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
        duration = time.time() - start_time
        log_run("failed", duration, 0, errors, equity)
        if notifier:
            try:
                notifier.notify_error(f"Bot run failed: {e}", "Strategy execution")
            except Exception:
                pass
        release_lock()
        return
    print()

    # Aggregate
    print("Aggregating signals...")
    try:
        portfolio = aggregate_signals(signals, state)
        for pair, data in portfolio.items():
            direction = "LONG" if data["signal"] == 1 else "FLAT" if data["signal"] == 0 else "SHORT"
            print(f"  {pair}: {direction} (score={data['score']:.3f})")
    except Exception as e:
        errors.append(f"Signal aggregation failed: {e}")
        portfolio = {}
        print(f"ERROR: Signal aggregation failed: {e}")
    print()

    # Initialize trades tracking BEFORE risk management
    trades_this_run = []

    # --- Risk Management: check existing positions ---
    risk_manager = DEFAULT_RISK_MANAGER
    risk_closes = []
    risk_alerts = []
    
    if state["positions"]:
        # Update trailing stops
        state["positions"] = risk_manager.update_trailing_stops(state["positions"], current_prices)
        
        # Check for risk violations
        equity_now = get_equity(state, current_prices)
        risk_closes, risk_alerts = risk_manager.check_positions(
            state["positions"], current_prices, equity_now, state.get("peak_equity", equity_now)
        )
        
        # Execute risk-based closes
        for close in risk_closes:
            symbol = close["symbol"]
            if symbol in state["positions"]:
                price = current_prices.get(symbol, state["positions"][symbol]["entry_price"])
                t = close_position(state, symbol, price, close["reason"])
                if t:
                    trades_this_run.append(t)
                    print(f"  RISK CLOSE {symbol}: {close['reason']} -> P&L ${t['pnl_usd']:+.2f}")
                    if notifier:
                        try:
                            notifier.notify_trade_close(
                                pair=format_pair(symbol),
                                side="buy" if t.get("side", "LONG") == "LONG" else "sell",
                                entry_price=t["entry_price"],
                                exit_price=t["exit_price"],
                                pnl_pct=t["pnl_pct"],
                                pnl_usd=t["pnl_usd"],
                                reason=close["reason"],
                            )
                        except Exception:
                            pass
        
        # Portfolio drawdown stop: close all, then RE-ARM the peak to the
        # post-stop equity and enforce a flat cooldown. Without the re-arm a
        # flat cash account can never climb back above the old dd line (the
        # bot would stay dormant forever); without the cooldown the same run
        # would re-open and oscillate close->open->close every cycle
        # (observed in the forward replay: hundreds of dd closes).
        if any(c["reason"].startswith("Portfolio drawdown") for c in risk_closes):
            eq_after_dd = get_equity(state, current_prices)
            state["peak_equity"] = eq_after_dd
            state["dd_cooldown_until"] = (
                datetime.now(timezone.utc) + timedelta(hours=risk_manager.dd_cooldown_hours)
            ).isoformat()
            print(f"  RISK: drawdown stop -> flat {risk_manager.dd_cooldown_hours:.0f}h, "
                  f"peak re-armed at ${eq_after_dd:.2f}")

        # Incremental save: closed positions are durable immediately, so a
        # crash after a risk close can never resurrect the position next run.
        if risk_closes:
            save_state(state)

        # Send risk alerts
        for alert in risk_alerts:
            print(f"  RISK ALERT: {alert['message']}")
            if notifier:
                try:
                    msg = risk_manager.format_alert(alert)
                    cfg = load_telegram_config()
                    if cfg.get("bot_token"):
                        tg_send_message(cfg["bot_token"], str(cfg["chat_id"]), msg)
                except Exception:
                    pass

    # Open gate: always evaluated (flat accounts included) and aware of the
    # drawdown cooldown so a stopped-out bot stays flat until it expires.
    can_open, can_open_reason = risk_manager.can_open_position(
        state["positions"],
        get_equity(state, current_prices),
        state["cash"],
        current_prices,
        peak_equity=state.get("peak_equity"),
        dd_cooldown_until=state.get("dd_cooldown_until"),
        now=datetime.now(timezone.utc),
    )
    if not can_open:
        print(f"  Risk manager: Cannot open new positions ({can_open_reason})")
    
    # Execute paper trades
    last_seen = state.setdefault("last_candle_seen", {})
    for pair, data in portfolio.items():
        try:
            # Only trade once per closed candle — matches next-open backtest.
            candle_time = data.get("candle_time")
            if candle_time is not None:
                candle_key = str(candle_time)
                if last_seen.get(pair) == candle_key:
                    print(f"  {pair}: no new closed candle yet, skipping trades")
                    continue
                last_seen[pair] = candle_key

            current_pos = state["positions"].get(pair)
            desired = data["signal"]
            current_side = current_pos["side"] if current_pos else 0

            if desired != current_side:
                if current_side != 0:
                    # MINIMUM HOLD TIME CHECK: Don't close if position was opened < 2 candles ago
                    # For 4h timeframe, 2 candles = 8 hours minimum hold
                    entry_time_str = current_pos.get("entry_time", "")
                    min_hold_hours = 8  # Minimum hold time in hours
                    can_close = True
                    if entry_time_str:
                        try:
                            entry_dt = datetime.fromisoformat(entry_time_str)
                            hours_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                            if hours_held < min_hold_hours:
                                print(f"  {pair}: HOLD (open {hours_held:.1f}h ago, min {min_hold_hours}h)")
                                can_close = False
                        except (ValueError, TypeError):
                            pass
                    
                    if can_close:
                        fill = data.get("next_open") or data["price"]
                        t = close_position(state, pair, fill, "Signal reversal")
                        if t:
                            trades_this_run.append(t)
                            print(f"  CLOSED {pair}: P&L ${t['pnl_usd']:+.2f} ({t['pnl_pct']:+.2f}%)")
                    else:
                        # Position stays open — keep the current side
                        continue

                if desired != 0 and can_open:
                    # IDEMPOTENT-OPEN GUARD: if a crash/failed save lost this
                    # position from state but the trade log recorded it, do not
                    # open it a second time.
                    if already_open(state, pair, desired):
                        print(f"  {pair}: skip duplicate open (recently logged, "
                              f"position missing from state)")
                        errors.append(f"{pair}: duplicate open prevented "
                                      f"(logged open, position absent from state)")
                        continue

                    equity_now = get_equity(state)
                    size_usd = min(
                        equity_now * MAX_POSITION_PCT,
                        state["cash"] * 0.95,
                        INITIAL_CAPITAL * 0.50,  # Never risk more than 50% of initial capital per trade
                    )
                    # Per-open heat re-check: `can_open` was evaluated BEFORE
                    # this loop, so multiple opens in one run would each pass
                    # the gate while accumulating heat (0.35 + 0.35 = 70%,
                    # bypassing the 50% portfolio heat limit). Re-check with
                    # the exposure this new position would add.
                    existing_exposure = sum(
                        p.get("size_usd", 0) for p in state["positions"].values())
                    heat_pct = (existing_exposure + size_usd) / equity_now * 100                         if equity_now > 0 else 999.0
                    if heat_pct > risk_manager.max_portfolio_heat_pct:
                        print(f"  {pair}: OPEN rejected — heat {heat_pct:.0f}% would "
                              f"exceed {risk_manager.max_portfolio_heat_pct:.0f}% limit")
                        continue
                    if size_usd > MIN_TRADE_USD:
                        strat_name = "Unknown"
                        for sname, sdata in signals.items():
                            if sdata["pair"] == pair:
                                strat_name = sdata["strategy"]
                                break
                        fill = data.get("next_open") or data["price"]
                        t = open_position(state, pair, desired, fill, strat_name, size_usd)
                        trades_this_run.append(t)
                        side = "LONG" if desired == 1 else "SHORT"
                        print(f"  OPENED {pair} {side}: ${size_usd:,.2f} @ ${data['price']:,.2f}")
                        # Incremental save: the position is durable IMMEDIATELY.
                        # A crash later in this run can no longer lose it (the
                        # old end-of-run-only save is what let a crash turn a
                        # logged open into a phantom re-open on the next run).
                        save_state(state)
                    else:
                        print(f"  {pair}: Insufficient cash (${state['cash']:.2f})")
        except Exception as e:
            error_msg = f"Trade execution {pair}: {e}"
            errors.append(error_msg)
            print(f"  ERROR: {error_msg}")
            traceback.print_exc()
    print()

    # Update peak equity and drawdown (use current prices for accuracy)
    equity = get_equity(state, current_prices)
    if equity > state.get("peak_equity", 0):
        state["peak_equity"] = equity
    dd = (state["peak_equity"] - equity) / state["peak_equity"] if state["peak_equity"] > 0 else 0
    state["max_drawdown"] = max(state.get("max_drawdown", 0), dd)

    # Persist MARKED-TO-MARKET equity and last known prices. Previously the
    # state only recorded cash + entry-price sizes, so every reader
    # (run history, dashboards, /status between runs) showed equity frozen at
    # entry price — e.g. +2.9% true P&L displayed as ~0%.
    state["last_equity_marked"] = equity
    state["last_prices"] = {p: float(v) for p, v in current_prices.items()}
    state["last_run_at"] = datetime.now(timezone.utc).isoformat()

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
        "funding_charged": funding_total,
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
    print(f"Funding:    ${funding_total:,.4f} (8h UTC schedule)")
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

    # --- Telegram Notifications ---
    # Strategy: ONE message per run, ONE format, always clear.
    # Only send when: trades happened OR first run of the day.
    if notifier:
        try:
            should_notify = False
            notify_reason = ""
            
            # Always notify if trades happened
            if trades_this_run:
                should_notify = True
                notify_reason = "trade"
            
            # Daily status check: send once per day if no trades
            # Store flag IN the state file so it persists across Railway restarts
            if not should_notify:
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                last_daily = state.get("last_daily_alert_date", "")
                if last_daily != today:
                    should_notify = True
                    notify_reason = "daily"
                    state["last_daily_alert_date"] = today
                    save_state(state)  # Persist the daily flag
            
            # Persist the daily flag in one place: if today's flag wasn't
            # already set, set it and save once. (The old code had two
            # branches doing this — the second was dead by construction.)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if state.get("last_daily_alert_date") != today:
                state["last_daily_alert_date"] = today
                save_state(state)

            if should_notify:
                # Calculate unrealized P&L for each position
                total_unrealized = 0.0
                pos_lines = []
                for pair, pos in state["positions"].items():
                    current = current_prices.get(pair, pos["entry_price"])
                    entry = pos["entry_price"]
                    side = pos.get("side", 0)
                    size = pos["size_usd"]
                    if entry > 0:
                        qty = size / entry
                        if side == 1:
                            upnl = qty * (current - entry)
                        else:
                            upnl = qty * (entry - current)
                        # GROSS mark P&L. Funding/fees are already inside the
                        # Realized line (booked to cash when charged), so netting
                        # them here too would double-count and break the
                        # additive identity Equity - Start = Realized + Unrealized.
                        upnl_pct = upnl / size * 100
                        total_unrealized += upnl
                        side_str = "LONG" if side == 1 else "SHORT"
                        entry_str = f"${entry:,.2f}"
                        current_str = f"${current:,.2f}"
                        pos_lines.append(
                            f"  {side_str} {format_pair(pair)}\n"
                            f"    {entry_str} -> {current_str} ({upnl_pct:+.1f}% ${upnl:+.2f} before fees+funding)"
                        )
                # POST-TRADE equity for the message: `equity` was computed
                # before this run's trades/funding, so on a trade alert it
                # would disagree with the trade lines below it (the "wrong
                # equity number" bug). Recompute from live state + prices.
                equity_now = get_equity(state, current_prices)
                total_return_now = (equity_now - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

                # Reconciliation (exact by construction — see accounting):
                #   Equity - Start = Realized + Unrealized
                # Realized = total_pnl (all fees + funding booked at cost);
                # Unrealized = gross mark P&L of open positions, the same
                # quantity get_equity adds to cash.
                recon = equity_now - INITIAL_CAPITAL \
                    - (state["total_pnl"] + total_unrealized)
                if abs(recon) > 0.005:
                    # Should never happen; if it does, show it loudly rather
                    # than silently publishing inconsistent numbers.
                    print(f"  WARNING: message reconciliation gap ${recon:+.4f}")

                # ONE clean message format — always the same structure.
                # Every line is additive: head + body == Equity - Start.
                if notify_reason == "trade":
                    msg = "TRADE ALERT"
                else:
                    msg = "PORTFOLIO"
                msg += f"\n{'='*28}\n"
                msg += f"Equity: ${equity_now:,.2f} ({total_return_now:+.1f}%)\n"
                msg += f"P&L (realized): ${state['total_pnl']:+.2f}"
                if state['total_trades'] > 0:
                    msg += f" | {wr:.0f}% win ({state['total_trades']} trades)"
                msg += "\n"
                if state["positions"]:
                    msg += f"P&L (unrealized): ${total_unrealized:+.2f}\n"
                msg += "\n"
                
                # Trades this run (only on trade alerts)
                if trades_this_run:
                    for t in trades_this_run:
                        if t["action"].startswith("OPEN"):
                            d = "LONG" if "LONG" in t["action"] else "SHORT"
                            msg += f">> {format_pair(t['pair'])} {d} @ ${t['price']:,.2f}"
                            msg += f" (${t['size_usd']:.0f})\n"
                        elif t["action"] == "CLOSE":
                            msg += f">> {format_pair(t['pair'])} CLOSED"
                            msg += f" P&L: ${t['pnl_usd']:+.2f} ({t['pnl_pct']:+.1f}%)\n"
                    msg += "\n"
                
                # Open positions (gross marks; costs live in Realized)
                if pos_lines:
                    msg += "Positions:\n"
                    for line in pos_lines:
                        msg += f"{line}\n"
                
                msg += f"{datetime.now(timezone.utc).strftime('%b %d, %H:%M UTC')} | PAPER"
                
                notifier._send_message(msg)
                print(f"Telegram: {notify_reason} notification sent.")
        except Exception as e:
            print(f"Telegram failed: {e}")

    # Send error alert if there were critical errors
    if errors and notifier:
        try:
            notifier.notify_error(f"Bot completed with {len(errors)} error(s)", "; ".join(errors[:3]))
        except Exception:
            pass

    release_lock()
    print("Done.")


if __name__ == "__main__":
    # argparse so `--help` prints usage instead of silently running a live
    # trading cycle (observed 2026-09-07: bare script ignored unknown flags).
    import argparse
    _ap = argparse.ArgumentParser(
        description="Paper trading bot — one scheduled cycle. Takes no flags;"
                    " configuration comes from the environment and data files.")
    _ap.parse_args()
    main()
