"""
Walk-Forward Optimization Engine

Splits historical data into rolling windows:
  - In-sample (training): 12 months of data to optimize parameters
  - Out-of-sample (validation): 6 months to test robustness

For each window:
  1. Grid search over parameter combinations on in-sample data
  2. Rank by risk-adjusted return (Sharpe-like) on in-sample
  3. Validate top candidates on out-of-sample data
  4. Select the most robust parameter set

The final "current regime" params come from the most recent window.

Strategy families optimized:
  - Bollinger_Reversion (ETH + BTC)
  - RSI_Reversion (BTC)
"""
import sys
sys.path.insert(0, ".")

import json
import time
from pathlib import Path
from datetime import datetime
from itertools import product

import pandas as pd
import numpy as np

from trading_system.strategies import STRATEGY_REGISTRY


# --- Config ---
INITIAL_CAPITAL = 97.0
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
MAX_POS_PCT = 0.35
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Parameter Grids ---
BOLLINGER_GRID = {
    "bb_period": [10, 15, 20],
    "bb_std": [1.5, 2.0],
    "rsi_filter": [False],
    "exit_at_middle": [False],
}

RSI_GRID = {
    "rsi_period": [10, 14, 21],
    "entry_oversold": [25, 30, 35],
    "entry_overbought": [65, 70, 75],
    "exit_neutral_low": [45, 50],
    "exit_neutral_high": [50, 55],
    "use_bb_filter": [False],
    "bb_period": [20],
    "bb_std": [2.0],
}

STRATEGY_CONFIGS = {
    "Bollinger_Reversion": {
        "grid": BOLLINGER_GRID,
        "pairs": ["ETH_USDT_USDT", "BTC_USDT_USDT"],
    },
    "RSI_Reversion": {
        "grid": RSI_GRID,
        "pairs": ["BTC_USDT_USDT"],
    },
}


# --- Data Loading ---
def load_parquet(pair: str, tf: str = "4h") -> pd.DataFrame:
    df = pd.read_parquet(f"data/raw/{pair}/klines_{tf}.parquet")
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dtype.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df.sort_values("timestamp").reset_index(drop=True)


# --- Backtest Engine ---
def backtest_single(df: pd.DataFrame, strat_name: str, params: dict) -> dict:
    """Backtest a single strategy on a single pair."""
    sig = STRATEGY_REGISTRY[strat_name].generate_signals(df, params)
    cash = INITIAL_CAPITAL
    position = None  # (side, entry_price, qty, cost)
    trades = []
    equity_curve = []

    for i in range(len(df)):
        price = float(df["close"].iloc[i])
        signal = int(sig.iloc[i])

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

        if not position and signal != 0:
            size = min(cash * MAX_POS_PCT, cash * 0.95)
            if size > 5:
                entry_p = price * (1 + SLIPPAGE_RATE * signal)
                qty = size / entry_p
                fee = size * FEE_RATE
                cash -= (size + fee)
                position = (signal, entry_p, qty, fee)

        eq = cash
        if position:
            side, entry, qty, cost = position
            if side == 1:
                eq += qty * price
            else:
                eq += qty * entry + qty * (entry - price)
        equity_curve.append(eq)

    # Close remaining
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

    # Composite score: prioritize return, penalize drawdown, reward consistency
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


def backtest_portfolio(strategies: dict, eth_data: pd.DataFrame, btc_data: pd.DataFrame) -> dict:
    """Portfolio-level backtest with multiple strategies."""
    sig_data = {}
    for name, cfg in strategies.items():
        df = eth_data if "ETH" in cfg["pair"] else btc_data
        sig = STRATEGY_REGISTRY[cfg["strategy"]].generate_signals(df, cfg["params"])
        sig_data[name] = {"signal": sig, "weight": cfg["weight"], "pair": cfg["pair"]}

    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    equity_curve = []

    for i in range(len(eth_data)):
        eth_price = float(eth_data["close"].iloc[i])
        btc_price = float(btc_data["close"].iloc[i])

        pair_scores = {}
        for name, s in sig_data.items():
            pair = s["pair"]
            sv = int(s["signal"].iloc[i])
            price = eth_price if "ETH" in pair else btc_price
            if pair not in pair_scores:
                pair_scores[pair] = {"ws": 0.0, "tw": 0.0, "price": price}
            pair_scores[pair]["ws"] += sv * s["weight"]
            pair_scores[pair]["tw"] += s["weight"]

        for pair, data in pair_scores.items():
            if data["tw"] <= 0:
                continue
            score = data["ws"] / data["tw"]
            fs = 1 if score > 0.3 else (-1 if score < -0.3 else 0)
            price = data["price"]
            cur = positions.get(pair)
            cs = cur["side"] if cur else 0

            if fs != cs:
                if cs != 0:
                    ep = price * (1 - SLIPPAGE_RATE * cs)
                    if cs == 1:
                        pnl = cur["qty"] * (ep - cur["entry"]) - cur["cost"]
                    else:
                        pnl = cur["qty"] * (cur["entry"] - ep) - cur["cost"]
                    cash += cur["qty"] * cur["entry"] + pnl
                    trades.append({"pnl": pnl, "pair": pair})
                    del positions[pair]
                if fs != 0:
                    sz = min(cash * MAX_POS_PCT, cash * 0.95)
                    if sz > 5:
                        ep = price * (1 + SLIPPAGE_RATE * fs)
                        qty = sz / ep
                        fee = sz * FEE_RATE
                        cash -= (sz + fee)
                        positions[pair] = {"side": fs, "entry": ep, "qty": qty, "cost": fee}

        eq = cash
        for pair, pos in positions.items():
            p = float(eth_data["close"].iloc[i]) if "ETH" in pair else float(btc_data["close"].iloc[i])
            if pos["side"] == 1:
                eq += pos["qty"] * p
            else:
                eq += pos["qty"] * pos["entry"] + pos["qty"] * (pos["entry"] - p)
        equity_curve.append(eq)

    for pair, pos in list(positions.items()):
        p = float(eth_data["close"].iloc[-1]) if "ETH" in pair else float(btc_data["close"].iloc[-1])
        ep = p * (1 - SLIPPAGE_RATE * pos["side"])
        if pos["side"] == 1:
            pnl = pos["qty"] * (ep - pos["entry"]) - pos["cost"]
        else:
            pnl = pos["qty"] * (pos["entry"] - ep) - pos["cost"]
        cash += pos["qty"] * pos["entry"] + pnl
        trades.append({"pnl": pnl, "pair": pair})

    final = cash
    wins = sum(1 for t in trades if t["pnl"] > 0)
    eq_arr = np.array(equity_curve) if equity_curve else np.array([INITIAL_CAPITAL])
    peaks = np.maximum.accumulate(eq_arr)
    dd = (peaks - eq_arr) / np.where(peaks > 0, peaks, 1)
    max_dd = float(np.max(dd)) * 100
    total_pnl = sum(t["pnl"] for t in trades)
    ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    return {
        "final": final, "return_pct": ret, "trades": len(trades),
        "wins": wins, "win_rate": wins / len(trades) * 100 if trades else 0,
        "max_dd": max_dd, "total_pnl": total_pnl, "score": ret - max_dd * 0.5,
    }


# --- Walk-Forward Engine ---
def optimize_strategy(
    strat_name: str,
    pair: str,
    data: pd.DataFrame,
    grid: dict,
    max_combos: int = 500,
    label: str = "",
) -> list[dict]:
    """Grid search optimization with scoring. Returns sorted results."""
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))

    # Sample if too many combos
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


def walk_forward_split(eth_data: pd.DataFrame, btc_data: pd.DataFrame, train_months: int = 12, test_months: int = 6):
    """Split data into rolling train/test windows."""
    # Get timestamp column
    ts = eth_data["timestamp"]

    # Calculate boundaries (in months)
    windows = []
    start_idx = 0

    while True:
        # Find train end index (train_months from start)
        train_end_date = ts.iloc[start_idx] + pd.DateOffset(months=train_months)
        train_end_idx = ts.searchsorted(train_end_date)

        # Find test end index (test_months from train end)
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

        # Slide forward by test_months
        start_idx = train_end_idx

    return windows


def run_walk_forward():
    """Main walk-forward optimization loop."""
    print("=" * 70)
    print("WALK-FORWARD OPTIMIZATION")
    print("=" * 70)
    print()

    # Load data
    eth_data = load_parquet("ETH_USDT_USDT")
    btc_data = load_parquet("BTC_USDT_USDT")
    min_len = min(len(eth_data), len(btc_data))
    eth_data = eth_data.tail(min_len).reset_index(drop=True)
    btc_data = btc_data.tail(min_len).reset_index(drop=True)
    print(f"Data: {min_len} candles ({eth_data['timestamp'].iloc[0]} to {eth_data['timestamp'].iloc[-1]})")
    print()

    # Split into windows
    windows = walk_forward_split(eth_data, btc_data, train_months=12, test_months=6)
    print(f"Found {len(windows)} walk-forward windows:")
    for i, w in enumerate(windows):
        print(f"  Window {i+1}: Train {w['train_start_date']} to {w['train_end_date']} | Test {w['test_start_date']} to {w['test_end_date']}")
    print()

    # For each strategy, optimize across all windows
    regime_params = {}  # Will hold the latest optimized params

    for strat_name, config in STRATEGY_CONFIGS.items():
        print(f"{'='*70}")
        print(f"OPTIMIZING: {strat_name}")
        print(f"{'='*70}")

        all_window_results = []

        for w_idx, window in enumerate(windows):
            print(f"\n  Window {w_idx+1}/{len(windows)}:")

            for pair in config["pairs"]:
                pair_label = pair.replace("_USDT_USDT", "")
                data_full = eth_data if "ETH" in pair else btc_data

                # Slice data
                train_data = data_full.iloc[window["train_start"]:window["train_end"]].reset_index(drop=True)
                test_data = data_full.iloc[window["test_start"]:window["test_end"]].reset_index(drop=True)

                if len(train_data) < 50 or len(test_data) < 20:
                    print(f"    {pair_label}: Insufficient data (train={len(train_data)}, test={len(test_data)})")
                    continue

                # Optimize on train
                print(f"    {pair_label}: Optimizing on {len(train_data)} train candles...", end="", flush=True)
                train_results = optimize_strategy(strat_name, pair, train_data, config["grid"], max_combos=300)

                if not train_results:
                    print(" No valid results")
                    continue

                # Validate top 5 on test
                top_5 = train_results[:5]
                test_results = []
                for tr in top_5:
                    test_r = backtest_single(test_data, strat_name, tr["params"])
                    test_results.append({
                        "params": tr["params"],
                        "train_score": tr["score"],
                        "train_return": tr["return_pct"],
                        "test_score": test_r["score"],
                        "test_return": test_r["return_pct"],
                        "test_win_rate": test_r["win_rate"],
                        "test_max_dd": test_r["max_dd"],
                    })

                # Select best by test score
                best = max(test_results, key=lambda x: x["test_score"])
                print(f" Best: train={best['train_return']:+.1f}% test={best['test_return']:+.1f}% (WR={best['test_win_rate']:.0f}%, DD={best['test_max_dd']:.1f}%)")

                all_window_results.append({
                    "window": w_idx + 1,
                    "strategy": strat_name,
                    "pair": pair,
                    "best_params": best["params"],
                    "train_return": best["train_return"],
                    "test_return": best["test_return"],
                    "test_win_rate": best["test_win_rate"],
                    "test_max_dd": best["test_max_dd"],
                })

                # The latest window's best params become the "current regime" params
                regime_params[f"{strat_name}_{pair}"] = best["params"]

            time.sleep(0.3)

        # Summary for this strategy
        if all_window_results:
            print(f"\n  --- {strat_name} Walk-Forward Summary ---")
            avg_test_return = np.mean([r["test_return"] for r in all_window_results])
            avg_test_wr = np.mean([r["test_win_rate"] for r in all_window_results])
            avg_test_dd = np.mean([r["test_max_dd"] for r in all_window_results])
            profitable_windows = sum(1 for r in all_window_results if r["test_return"] > 0)
            print(f"  Avg test return: {avg_test_return:+.1f}%")
            print(f"  Avg test win rate: {avg_test_wr:.0f}%")
            print(f"  Avg test max DD: {avg_test_dd:.1f}%")
            print(f"  Profitable windows: {profitable_windows}/{len(all_window_results)}")

    # --- Build current regime portfolio ---
    print(f"\n{'='*70}")
    print("CURRENT REGIME PARAMETERS (from latest window)")
    print(f"{'='*70}")

    portfolio = {}
    for key, params in regime_params.items():
        # Parse key: "Bollinger_Reversion_ETH_USDT_USDT" -> strat="Bollinger_Reversion", pair="ETH_USDT_USDT"
        for strat_name in STRATEGY_CONFIGS:
            if key.startswith(strat_name + "_"):
                pair_full = key[len(strat_name) + 1:]
                break
        else:
            print(f"  WARNING: Could not parse key {key}")
            continue
        # Assign weights
        if "Bollinger" in strat_name:
            weight = 0.40 if "ETH" in pair_full else 0.35
        else:
            weight = 0.25
        portfolio[key] = {
            "strategy": strat_name,
            "pair": pair_full,
            "weight": weight,
            "params": params,
        }
        print(f"  {key}: {params}")

    # Validate current regime on full data
    print(f"\n  Validating current regime on full dataset...")
    full_result = backtest_portfolio(portfolio, eth_data, btc_data)
    print(f"  Full dataset result: {full_result['return_pct']:+.1f}% return, {full_result['win_rate']:.0f}% win rate, {full_result['max_dd']:.1f}% max DD")

    # Save optimized params
    output = {
        "timestamp": datetime.now().isoformat(),
        "data_range": f"{eth_data['timestamp'].iloc[0]} to {eth_data['timestamp'].iloc[-1]}",
        "windows": len(windows),
        "strategies": {},
        "portfolio": {k: {"strategy": v["strategy"], "pair": v["pair"], "weight": v["weight"], "params": v["params"]} for k, v in portfolio.items()},
        "full_backtest": {k: v for k, v in full_result.items() if k != "equity_curve"},
    }

    # Add per-strategy window results
    for strat_name in STRATEGY_CONFIGS:
        output["strategies"][strat_name] = {
            "windows": [r for r in all_window_results if r.get("strategy") == strat_name],
        }

    out_path = RESULTS_DIR / "optimized_params.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    # Also save a separate file with just the params for the bot to load
    bot_params = {}
    for key, data in portfolio.items():
        bot_params[key] = {
            "strategy": data["strategy"],
            "pair": data["pair"],
            "timeframe": "4h",
            "weight": data["weight"],
            "params": data["params"],
        }

    bot_params_path = RESULTS_DIR / "bot_strategy_params.json"
    with open(bot_params_path, "w") as f:
        json.dump(bot_params, f, indent=2)
    print(f"  Bot params saved to {bot_params_path}")

    return portfolio


if __name__ == "__main__":
    run_walk_forward()
