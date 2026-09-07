"""
DEPRECATED — DO NOT RUN.

This is the OLD legacy runner, superseded by scripts/paper_trader.py. It is
kept only so old references don't break. It has its OWN separate state file,
state format, equity basis, and Telegram summary — running it produces
numbers that will NOT match the real bot's messages (this was the source of
conflicting equity figures in Telegram).

The scheduled bot is scripts/paper_trader.py (.github/workflows/bot.yml).
"""

import sys

if __name__ == "__main__":
    sys.stderr.write(
        "ERROR: scripts/run_github_bot.py is DEPRECATED and disabled.\n"
        "It keeps its own state and sends Telegram numbers that do not match "
        "the real bot. Use scripts/paper_trader.py (the scheduled runner).\n"
    )
    sys.exit(2)

# Unreachable below — legacy code intentionally left inert for reference.
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, ".")

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

from trading_system.strategies import STRATEGY_REGISTRY


# Strategy configurations (from optimization results)
STRATEGIES = {
    "MACD ETH": {
        "strategy": "MACD",
        "pair": "ETH_USDT_USDT",
        "timeframe": "4h",
        "weight": 0.41,
        "params": {
            "fast_period": 8,
            "slow_period": 21,
            "signal_period": 5,
            "use_ema": True
        }
    },
    "ROC_Momentum ETH": {
        "strategy": "ROC_Momentum",
        "pair": "ETH_USDT_USDT",
        "timeframe": "4h",
        "weight": 0.17,
        "params": {
            "roc_period": 10,
            "signal_period": 5,
            "use_ema": True,
            "ema_period": 12
        }
    },
    "MACD BTC": {
        "strategy": "MACD",
        "pair": "BTC_USDT_USDT",
        "timeframe": "4h",
        "weight": 0.43,
        "params": {
            "fast_period": 8,
            "slow_period": 21,
            "signal_period": 5,
            "use_ema": True
        }
    }
}

BOT_MODE = os.getenv("BOT_MODE", "paper")
LOG_DIR = Path("data/results")
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG = LOG_DIR / "github_bot_log.jsonl"
STATE_FILE = LOG_DIR / "github_bot_state.json"

# Telegram config — read from env (GitHub Actions secrets) or config file
def load_telegram_config() -> dict:
    """Load Telegram config, preferring environment variables."""
    import yaml
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled = os.getenv("TELEGRAM_ENABLED", "").lower() == "true"

    if token and chat_id:
        return {"enabled": True, "bot_token": token, "chat_id": chat_id}

    # Fall back to config file
    try:
        config_path = Path("configs/bot_live.yaml")
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            tg = cfg.get("bot", {}).get("telegram", {})
            if tg.get("enabled") and tg.get("bot_token"):
                return tg
    except Exception:
        pass

    return {"enabled": False}


def get_notifier():
    """Create Telegram notifier if configured."""
    from trading_system.bot.telegram_notifier import TelegramNotifier
    cfg = load_telegram_config()
    if cfg.get("enabled") and cfg.get("bot_token"):
        notifier = TelegramNotifier(
            bot_token=cfg["bot_token"],
            chat_id=str(cfg["chat_id"])
        )
        if notifier.test_connection():
            print("  Telegram: Connected")
            return notifier
        else:
            print("  Telegram: Connection failed, notifications disabled")
    else:
        print("  Telegram: Not configured, notifications disabled")
    return None


def load_state() -> dict:
    """Load last known position state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"positions": {}, "cash": 10000.0, "equity": 10000.0}


def save_state(state: dict):
    """Save current position state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def fetch_latest_data(pair: str, timeframe: str, lookback_days: int = 30) -> pd.DataFrame:
    """Fetch latest candles for signal generation."""
    # Try cached files first
    for pattern in [f"data/raw/{pair}/{timeframe}.parquet", f"data/raw/{pair}/klines_{timeframe}.parquet"]:
        cache_path = Path(pattern)
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            if "timestamp" in df.columns:
                cutoff = datetime.utcnow() - timedelta(days=lookback_days)
                df = df[df["timestamp"] > cutoff]
            return df

    # Fetch fresh data via ccxt
    try:
        import ccxt
        pair_sym = pair.replace("_", "/").replace("USDT_USDT", "USDT:USDT")
        exchange = ccxt.binanceusdm({"enableRateLimit": True, "options": {"defaultType": "future"}})
        since = int((datetime.utcnow() - timedelta(days=lookback_days)).timestamp() * 1000)
        ohlcv = exchange.fetch_ohlcv(pair_sym, timeframe, since=since, limit=1000)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"  Failed to fetch {pair} {timeframe}: {e}")
        return pd.DataFrame()


def run_strategies() -> dict:
    """Run all strategies and return aggregated signal."""
    signals = {}

    for name, config in STRATEGIES.items():
        try:
            df = fetch_latest_data(config["pair"], config["timeframe"])
            if df is None or len(df) < 50:
                print(f"  {name}: Insufficient data ({len(df) if df is not None else 0} candles)")
                continue

            # Run strategy
            strat = STRATEGY_REGISTRY[config["strategy"]]
            signal = strat.generate_signals(df, config["params"])

            latest_signal = signal.iloc[-1]
            signals[name] = {
                "signal": int(latest_signal),
                "weight": config["weight"],
                "pair": config["pair"],
                "timeframe": config["timeframe"],
                "strategy": config["strategy"]
            }

            direction = "LONG" if latest_signal == 1 else "FLAT" if latest_signal == 0 else "SHORT"
            print(f"  {name}: {direction} (signal={latest_signal})")

        except Exception as e:
            print(f"  {name}: Error - {e}")
            signals[name] = {"signal": 0, "weight": config["weight"], "error": str(e)}

    return signals


def aggregate_signals(signals: dict) -> dict:
    """Weighted signal aggregation per pair."""
    pair_signals = {}

    for name, sig_data in signals.items():
        pair = sig_data["pair"]
        if pair not in pair_signals:
            pair_signals[pair] = {"weighted_sum": 0.0, "total_weight": 0.0, "signals": []}

        pair_signals[pair]["weighted_sum"] += sig_data["signal"] * sig_data["weight"]
        pair_signals[pair]["total_weight"] += sig_data["weight"]
        pair_signals[pair]["signals"].append(name)

    # Normalize to get final signal
    results = {}
    for pair, data in pair_signals.items():
        if data["total_weight"] > 0:
            normalized = data["weighted_sum"] / data["total_weight"]
            # Threshold: |score| > 0.3 triggers trade
            if normalized > 0.3:
                final_signal = 1  # Long
            elif normalized < -0.3:
                final_signal = -1  # Short
            else:
                final_signal = 0  # Flat
            results[pair] = {"score": normalized, "signal": final_signal}

    return results


def log_trade(state: dict, pair: str, action: str, price: float, reason: str):
    """Append trade to JSONL log."""
    trade = {
        "timestamp": datetime.utcnow().isoformat(),
        "pair": pair,
        "action": action,
        "price": price,
        "reason": reason,
        "cash_before": state["cash"],
        "equity_before": state["equity"]
    }
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(trade) + "\n")


def execute_trades(state: dict, portfolio_signals: dict):
    """Execute trades based on signals."""
    for pair, sig_data in portfolio_signals.items():
        current_position = state["positions"].get(pair, 0)
        desired_signal = sig_data["signal"]

        if desired_signal != current_position:
            # Need to trade
            price = get_current_price(pair)

            if current_position != 0:
                # Close existing position
                action = "close"
                state["positions"][pair] = 0
                log_trade(state, pair, action, price, f"Signal change: {current_position} -> {desired_signal}")

            if desired_signal != 0:
                # Open new position
                action = "open_long" if desired_signal == 1 else "open_short"
                state["positions"][pair] = desired_signal
                log_trade(state, pair, action, price, f"New position: {desired_signal}")

    # Update equity (simplified — actual P&L would need price tracking)
    state["equity"] = state["cash"]
    for pair, pos in state["positions"].items():
        if pos != 0:
            # Simplified equity update
            pass

    save_state(state)


def get_current_price(pair: str) -> float:
    """Get latest price for a pair."""
    try:
        df = fetch_latest_data(pair, "4h", lookback_days=1)
        if df is not None and len(df) > 0:
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def send_trade_alert(notifier, pair: str, action: str, state: dict, sig_data: dict):
    """Send Telegram notification for a trade."""
    if notifier is None:
        return
    try:
        if action.startswith("open"):
            side = "LONG" if "long" in action else "SHORT"
            price = get_current_price(pair)
            notifier.notify_trade_opened(
                pair=pair.replace("_", "/"),
                side=side,
                entry_price=price,
                size=0.0,  # Not tracked in simplified mode
                strategy=sig_data.get("strategy", "Unknown"),
                confidence=abs(sig_data.get("score", 0.0)),
            )
        elif action == "close":
            notifier.notify_trade_closed(
                pair=pair.replace("_", "/"),
                side="LONG",
                entry_price=0.0,
                exit_price=get_current_price(pair),
                pnl_pct=0.0,
                pnl_usd=0.0,
                reason="Signal reversal",
            )
    except Exception as e:
        print(f"  Telegram notify failed: {e}")


def main():
    print("=== GitHub Bot Runner ===")
    print(f"Time: {datetime.utcnow().isoformat()}")
    print(f"Mode: {BOT_MODE}")
    print()

    # Initialize Telegram notifier
    notifier = get_notifier()
    print()

    # Load state
    state = load_state()
    print(f"Cash: ${state['cash']:.2f}")
    print(f"Positions: {state['positions']}")
    print()

    # Run strategies
    print("Running strategies...")
    signals = run_strategies()
    print()

    # Aggregate signals
    print("Aggregating signals...")
    portfolio_signals = aggregate_signals(signals)

    for pair, data in portfolio_signals.items():
        direction = "LONG" if data["signal"] == 1 else "FLAT" if data["signal"] == 0 else "SHORT"
        print(f"  {pair}: {direction} (score={data['score']:.3f})")
    print()

    # Execute trades
    trades_executed = []
    if BOT_MODE == "live":
        print("Executing live trades...")
        # Track what trades happen
        for pair, sig_data in portfolio_signals.items():
            current_position = state["positions"].get(pair, 0)
            desired_signal = sig_data["signal"]
            if desired_signal != current_position:
                action = "close" if current_position != 0 else ("open_long" if desired_signal == 1 else "open_short")
                trades_executed.append((pair, action, sig_data))
        execute_trades(state, portfolio_signals)
    else:
        print("Paper mode — no trades executed")
        # Track signal changes for paper notifications
        for pair, sig_data in portfolio_signals.items():
            current_position = state["positions"].get(pair, 0)
            desired_signal = sig_data["signal"]
            if desired_signal != current_position:
                action = "signal_change"
                trades_executed.append((pair, action, sig_data))
        log_trade(state, "PAPER", "signal_check", 0.0,
                  json.dumps({p: d["score"] for p, d in portfolio_signals.items()}))

    # Send Telegram notifications
    if notifier and trades_executed:
        print(f"Sending {len(trades_executed)} trade alert(s)...")
        for pair, action, sig_data in trades_executed:
            send_trade_alert(notifier, pair, action, state, sig_data)

    # Send daily summary
    if notifier:
        try:
            notifier.notify_daily_summary(
                equity=state["equity"],
                pnl=0.0,
                positions=state["positions"],
                win_rate=0.0,
            )
        except Exception as e:
            print(f"  Daily summary notify failed: {e}")

    # Summary
    print()
    print(f"State saved to {STATE_FILE}")
    print(f"Log appended to {TRADE_LOG}")
    if trades_executed:
        print(f"{len(trades_executed)} trade alert(s) sent via Telegram")
    print("Done.")


if __name__ == "__main__":
    main()
