#!/usr/bin/env python3
"""
Monthly Walk-Forward Re-Optimization

Fetches fresh data from Kraken, runs walk-forward optimization
on all active strategies, and saves updated parameters.

This script is designed to run via GitHub Actions once per month.
Results are saved to data/results/optimized_params.json and
data/results/bot_strategy_params.json which paper_trader.py loads.

If Telegram is configured, sends a summary notification.
"""

import sys
import os
import json
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.strategies import STRATEGY_REGISTRY

# --- Config ---
INITIAL_CAPITAL = 97.0
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
MAX_POS_PCT = 0.35
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Parameter grids for each strategy
PARAM_GRIDS = {
    "Bollinger_Reversion": {
        "grid": {
            "bb_period": [10, 15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
            "rsi_filter": [False],
            "exit_at_middle": [False],
        },
        "pairs": ["ETH_USDT_USDT", "BTC_USDT_USDT"],
    },
    "RSI_Reversion": {
        "grid": {
            "rsi_period": [10, 14, 21],
            "entry_oversold": [25, 30, 35],
            "entry_overbought": [60, 65, 70, 75],
            "exit_neutral_low": [40, 45, 50],
            "exit_neutral_high": [50, 55, 60],
            "use_bb_filter": [False],
        },
        "pairs": ["BTC_USDT_USDT"],
    },
}

# Strategy weights for the portfolio
WEIGHTS = {
    "ETH_USDT_USDT": 0.40,
    "BTC_USDT_USDT": 0.35,
}

# Telegram config
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8783971913:AAH1ZdvtKvHjgVuC2c9LLebYnM-o8gBMQaY")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "5421461006")


# --- Telegram ---
def send_telegram(text):
    """Send a Telegram notification."""
    try:
        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=20,
        )
        if resp.status_code == 200 and resp.json().get("ok"):
            print("  Telegram: sent")
        else:
            print(f"  Telegram: failed - {resp.text[:100]}")
    except Exception as e:
        print(f"  Telegram: error - {e}")


# --- Data ---
def fetch_data(pair: str, timeframe: str = "4h", days: int = 730) -> pd.DataFrame:
    """Fetch OHLCV data from Kraken API."""
    import ccxt

    symbol_map = {
        "ETH_USDT_USDT": "ETH/USDT",
        "BTC_USDT_USDT": "BTC/USDT",
    }
    symbol = symbol_map.get(pair, pair.replace("_", "/"))

    try:
        exchange = ccxt.kraken({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        print(f"  Kraken: {len(df)} candles for {symbol} ({timeframe})")
        return df
    except Exception as e:
        print(f"  Kraken failed for {symbol}: {e}")

    # Fallback to Binance
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        print(f"  Binance: {len(df)} candles for {symbol} ({timeframe})")
        return df
    except Exception as e:
        print(f"  Binance failed for {symbol}: {e}")

    return None


# --- Backtest ---
def backtest_single(df: pd.DataFrame, strat_name: str, params: dict) -> dict:
    """Run a single strategy backtest. Returns performance metrics."""
    sig = STRATEGY_REGISTRY[strat_name].generate_signals(df, params)
    cash = INITIAL_CAPITAL
    position = None  # (side, entry_price, qty, cost)
    trades = []
    equity_curve = []

    for i in range(len(df)):
        price = float(df["close"].iloc[i])
        signal = int(sig.iloc[i])

        # Close on signal reversal
        if position and signal != position[0]:
            side, entry, qty, cost = position
            exit_p = price * (1 - SLIPPAGE_RATE * side)
            if side == 1:
                pnl = qty * (exit_p - entry) - cost
            else:
                pnl = qty * (entry - exit_p) - cost
            cash += qty * entry + pnl
            trades.append({"pnl": pnl, "side": side})
            position = None

        # Open new position
        if not position and signal != 0:
            size = min(cash * MAX_POS_PCT, cash * 0.95)
            if size > 5:
                entry_p = price * (1 + SLIPPAGE_RATE * signal)
                qty = size / entry_p
                fee = size * FEE_RATE
                cash -= (size + fee)
                position = (signal, entry_p, qty, fee)

        # Track equity
        eq = cash
        if position:
            side, entry, qty, cost = position
            if side == 1:
                eq += qty * price
            else:
                eq += qty * entry + qty * (entry - price)
        equity_curve.append(eq)

    # Close remaining position
    if position:
        side, entry, qty, cost = position
        price = float(df["close"].iloc[-1])
        exit_p = price * (1 - SLIPPAGE_RATE * side)
        if side == 1:
            pnl = qty * (exit_p - entry) - cost
        else:
            pnl = qty * (entry - exit_p) - cost
        cash += qty * entry + pnl
        trades.append({"pnl": pnl, "side": side})

    final = cash
    wins = sum(1 for t in trades if t["pnl"] > 0)
    total_pnl = sum(t["pnl"] for t in trades)
    n_trades = len(trades)

    # Max drawdown
    eq_arr = np.array(equity_curve) if equity_curve else np.array([INITIAL_CAPITAL])
    peaks = np.maximum.accumulate(eq_arr)
    dd = (peaks - eq_arr) / np.where(peaks > 0, peaks, 1)
    max_dd = float(np.max(dd)) * 100

    # Sharpe (annualized, 4h bars)
    if len(eq_arr) > 1:
        returns = np.diff(eq_arr) / np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1)
        if np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(6 * 365))
        else:
            sharpe = 0
    else:
        sharpe = 0

    ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    score = ret - max_dd * 0.5  # penalize drawdown

    return {
        "final": final,
        "return_pct": ret,
        "trades": n_trades,
        "wins": wins,
        "win_rate": wins / n_trades * 100 if n_trades > 0 else 0,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total_pnl": total_pnl,
        "score": score,
    }


# --- Grid Search ---
def optimize_strategy(strat_name, pair, data, grid, max_combos=500):
    """Grid search with scoring. Returns top results sorted by score."""
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    if len(combos) > max_combos:
        np.random.seed(42)
        indices = np.random.choice(len(combos), max_combos, replace=False)
        combos = [combos[i] for i in indices]

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            r = backtest_single(data, strat_name, params)
            r["params"] = params
            results.append(r)
        except Exception:
            pass

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# --- Walk-Forward Split ---
def walk_forward_windows(data, train_months=12, test_months=6):
    """Split data into rolling train/test windows."""
    ts = data["timestamp"]
    windows = []
    start_idx = 0

    while True:
        train_end_date = ts.iloc[start_idx] + pd.DateOffset(months=train_months)
        train_end_idx = ts.searchsorted(train_end_date)

        test_end_date = ts.iloc[start_idx] + pd.DateOffset(months=train_months + test_months)
        test_end_idx = min(ts.searchsorted(test_end_date), len(ts) - 1)

        if train_end_idx >= len(ts) or train_end_idx <= start_idx:
            break
        if test_end_idx <= train_end_idx:
            break

        windows.append({
            "train_start": start_idx,
            "train_end": train_end_idx,
            "test_start": train_end_idx,
            "test_end": test_end_idx,
            "train_start_date": str(ts.iloc[start_idx])[:10],
            "train_end_date": str(ts.iloc[train_end_idx - 1])[:10],
            "test_start_date": str(ts.iloc[train_end_idx])[:10],
            "test_end_date": str(ts.iloc[test_end_idx])[:10],
        })

        start_idx = train_end_idx

    return windows


# --- Main ---
def main():
    t0 = time.time()
    print("=" * 70)
    print("MONTHLY WALK-FORWARD RE-OPTIMIZATION")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # Fetch fresh data for each pair
    all_data = {}
    for pair in ["ETH_USDT_USDT", "BTC_USDT_USDT"]:
        print(f"\nFetching {pair}...")
        df = fetch_data(pair, "4h")
        if df is not None and len(df) > 100:
            all_data[pair] = df
            print(f"  Data range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        else:
            print(f"  SKIP: insufficient data")

    if not all_data:
        print("\nERROR: No data available")
        send_telegram("Re-optimization FAILED: No data available from exchanges")
        return

    # Align all data to same length
    min_len = min(len(d) for d in all_data.values())
    for pair in all_data:
        all_data[pair] = all_data[pair].tail(min_len).reset_index(drop=True)

    print(f"\nUsing {min_len} candles per pair")

    # Run optimization for each strategy+pair
    results = {}
    errors = []

    for strat_name, config in PARAM_GRIDS.items():
        print(f"\n{'='*70}")
        print(f"OPTIMIZING: {strat_name}")
        print(f"{'='*70}")

        for pair in config["pairs"]:
            if pair not in all_data:
                print(f"  SKIP {pair}: no data")
                continue

            data = all_data[pair]
            pair_label = pair.replace("_USDT_USDT", "")
            windows = walk_forward_windows(data)

            if not windows:
                print(f"  SKIP {pair_label}: not enough data for walk-forward")
                continue

            print(f"\n  {pair_label}: {len(windows)} walk-forward windows")

            window_results = []
            for w_idx, w in enumerate(windows):
                train = data.iloc[w["train_start"]:w["train_end"]].reset_index(drop=True)
                test = data.iloc[w["test_start"]:w["test_end"]].reset_index(drop=True)

                if len(train) < 50 or len(test) < 20:
                    continue

                # Optimize on train
                train_results = optimize_strategy(strat_name, pair, train, config["grid"], max_combos=300)

                if not train_results:
                    continue

                # Validate top 3 on test
                best_test_return = -999
                best_params = None
                for tr in train_results[:3]:
                    test_r = backtest_single(test, strat_name, tr["params"])
                    if test_r["return_pct"] > best_test_return:
                        best_test_return = test_r["return_pct"]
                        best_params = tr["params"]

                if best_params:
                    # Re-run best on full test for final metrics
                    final_r = backtest_single(test, strat_name, best_params)
                    window_results.append({
                        "window": w_idx + 1,
                        "train_dates": f"{w['train_start_date']} to {w['train_end_date']}",
                        "test_dates": f"{w['test_start_date']} to {w['test_end_date']}",
                        "params": best_params,
                        "train_return": train_results[0]["return_pct"],
                        "test_return": final_r["return_pct"],
                        "test_win_rate": final_r["win_rate"],
                        "test_max_dd": final_r["max_dd"],
                        "test_trades": final_r["trades"],
                    })
                    print(f"    Window {w_idx+1}: train={train_results[0]['return_pct']:+.1f}% test={final_r['return_pct']:+.1f}% (WR={final_r['win_rate']:.0f}%, trades={final_r['trades']})")
                else:
                    print(f"    Window {w_idx+1}: no valid params")

            # Select best params from most recent profitable window
            profitable = [r for r in window_results if r["test_return"] > 0]
            if profitable:
                best_window = profitable[-1]  # Most recent profitable
                best_params = best_window["params"]
            elif window_results:
                best_window = window_results[-1]  # Most recent (even if loss)
                best_params = best_window["params"]
            else:
                best_params = None
                print(f"    WARNING: No valid params found for {strat_name} on {pair}")

            if best_params:
                # Run full backtest with best params
                full_r = backtest_single(data, strat_name, best_params)
                avg_test_return = np.mean([r["test_return"] for r in window_results]) if window_results else 0
                profitable_windows = sum(1 for r in window_results if r["test_return"] > 0)

                key = f"{strat_name}_{pair}"
                results[key] = {
                    "strategy": strat_name,
                    "pair": pair,
                    "params": best_params,
                    "full_return": full_r["return_pct"],
                    "full_trades": full_r["trades"],
                    "full_win_rate": full_r["win_rate"],
                    "full_max_dd": full_r["max_dd"],
                    "avg_test_return": avg_test_return,
                    "profitable_windows": profitable_windows,
                    "total_windows": len(window_results),
                    "window_results": window_results,
                }

                print(f"\n  BEST: {pair_label} {strat_name}")
                print(f"    Params: {json.dumps(best_params)}")
                print(f"    Full backtest: {full_r['return_pct']:+.1f}% return, {full_r['win_rate']:.0f}% WR, {full_r['max_dd']:.1f}% DD")
                print(f"    Walk-forward: {profitable_windows}/{len(window_results)} profitable windows, avg test return: {avg_test_return:+.1f}%")
            else:
                errors.append(f"{strat_name} on {pair}: no valid params")

    # --- Save results ---
    elapsed = time.time() - t0

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_candles": min_len,
        "elapsed_seconds": round(elapsed, 1),
        "strategies": results,
        "errors": errors,
    }

    # Save full report
    report_path = RESULTS_DIR / "optimized_params.json"
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    # Save bot-compatible params
    bot_params = {}
    for key, data in results.items():
        pair = data["pair"]
        weight = WEIGHTS.get(pair, 0.30)
        bot_params[key] = {
            "strategy": data["strategy"],
            "pair": data["pair"],
            "timeframe": "4h",
            "weight": weight,
            "params": data["params"],
        }

    bot_params_path = RESULTS_DIR / "bot_strategy_params.json"
    with open(bot_params_path, "w") as f:
        json.dump(bot_params, f, indent=2)
    print(f"  Bot params saved: {bot_params_path}")

    # --- Telegram summary ---
    summary_lines = ["MONTHLY RE-OPTIMIZATION COMPLETE\n"]
    summary_lines.append(f"Data: {min_len} candles | Time: {elapsed:.0f}s\n")

    for key, data in results.items():
        pair_label = data["pair"].replace("_USDT_USDT", "")
        summary_lines.append(
            f"{data['strategy']} {pair_label}\n"
            f"  Full: {data['full_return']:+.1f}% | WR: {data['full_win_rate']:.0f}% | DD: {data['full_max_dd']:.1f}%\n"
            f"  Walk-forward: {data['profitable_windows']}/{data['total_windows']} profitable\n"
            f"  Params: {json.dumps(data['params'])}\n"
        )

    if errors:
        summary_lines.append(f"\nErrors: {'; '.join(errors)}")

    summary_lines.append(f"\n{datetime.now(timezone.utc).strftime('%b %d, %H:%M UTC')}")
    send_telegram("\n".join(summary_lines))

    # Print final summary
    print(f"\n{'='*70}")
    print(f"RE-OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Strategies optimized: {len(results)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    for key, data in results.items():
        pair_label = data["pair"].replace("_USDT_USDT", "")
        print(f"\n  {data['strategy']} {pair_label}:")
        print(f"    Return: {data['full_return']:+.1f}% | Win Rate: {data['full_win_rate']:.0f}% | Max DD: {data['full_max_dd']:.1f}%")
        print(f"    Walk-forward: {data['profitable_windows']}/{data['total_windows']} profitable windows")
        print(f"    Params: {json.dumps(data['params'])}")

    print(f"\n{'='*70}")
    return results


if __name__ == "__main__":
    main()
