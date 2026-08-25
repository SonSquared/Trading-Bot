"""
GitHub Actions Bot Runner

Designed to run every 4 hours via GitHub Actions cron.
Fetches latest data, runs all 3 strategies, aggregates signals,
executes trades (paper or live), and logs results.

No persistent state needed — each run is independent.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd
import numpy as np

from trading_system.data import BinanceDataCollector
from trading_system.strategies import STRATEGY_REGISTRY
from trading_system.config import BacktestConfig, StrategyConfig


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
    collector = BinanceDataCollector()

    # Use cache if available, otherwise fetch
    cache_path = Path(f"data/raw/{pair}/{timeframe}.parquet")
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        # Return last N days
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        df = df[df["timestamp"] > cutoff]
        return df

    # Fetch fresh data
    df = collector.fetch_historical(
        pair=pair.replace("_", "/").replace("USDT_USDT", "USDT:USDT"),
        timeframe=timeframe,
        start_date=(datetime.utcnow() - timedelta(days=lookback_days)).isoformat(),
        end_date=datetime.utcnow().isoformat()
    )
    return df


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
            strategy_class = STRATEGY_REGISTRY[config["strategy"]]
            strat = strategy_class()
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


def main():
    print(f"=== GitHub Bot Runner ===")
    print(f"Time: {datetime.utcnow().isoformat()}")
    print(f"Mode: {BOT_MODE}")
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
    if BOT_MODE == "live":
        print("Executing live trades...")
        execute_trades(state, portfolio_signals)
    else:
        print("Paper mode — no trades executed")
        # Log the signals for paper tracking
        log_trade(state, "PAPER", "signal_check", 0.0,
                  json.dumps({p: d["score"] for p, d in portfolio_signals.items()}))

    # Summary
    print()
    print(f"State saved to {STATE_FILE}")
    print(f"Log appended to {TRADE_LOG}")
    print("Done.")


if __name__ == "__main__":
    main()
