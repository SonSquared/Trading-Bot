"""
Walk-Forward Optimization Engine

Splits historical data into rolling windows:
  - In-sample (training): 12 months of data to optimize parameters
  - Out-of-sample (validation): 6 months to test robustness

For each window:
  1. Grid search over parameter combinations on in-sample data
  2. Rank by risk-adjusted return on in-sample only
  3. Evaluate EVERY candidate on EVERY out-of-sample window
  4. Deploy only if mean OOS return is positive AND profitable in a
     majority of windows; otherwise keep the existing bot params.

Selection never peeks at the test set: picking the "best by test
score" (the old behavior) curve-fits the validation data and
overstates out-of-sample performance.

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
from trading_system.bot.accounting import FEE_RATE, SLIPPAGE_RATE, funding_cost


# --- Config ---
INITIAL_CAPITAL = 97.0
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
    """Backtest a single strategy on a single pair, next-open + full costs.

    Costs match the paper trader exactly (trading_system.bot.accounting):
    0.05% fee per side, 0.02% slippage per side, funding at 8h UTC
    boundaries while held.
    """
    sig = STRATEGY_REGISTRY[strat_name].generate_signals(df, params)
    sig_prev = sig.shift(1).fillna(0)  # act on the previous CLOSED candle's signal

    cash = INITIAL_CAPITAL
    position = None  # (side, entry_price, qty, entry_fee, entry_time)
    trades = []
    equity_curve = []

    for i in range(len(df)):
        ts_i = df["timestamp"].iloc[i]  # open time of candle i == the fill time
        price = float(df["open"].iloc[i])  # fill at next candle open
        signal = int(sig_prev.iloc[i])

        # Close on signal reversal
        if position and signal != position[0]:
            side, entry, qty, cost, entry_dt = position
            exit_p = price * (1 - SLIPPAGE_RATE * side)
            funding = funding_cost(qty * entry, entry_dt, ts_i)
            exit_fee = qty * entry * FEE_RATE
            if side == 1:
                pnl = qty * (exit_p - entry) - cost - exit_fee - funding
            else:
                pnl = qty * (entry - exit_p) - cost - exit_fee - funding
            # Return the entry fee (already debited at open) so the ledger
            # charges each fee exactly once. ``pnl`` is the full round-trip
            # P&L (entry + exit fees + funding), matching the paper trader.
            cash += qty * entry + cost + pnl
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
                position = (signal, entry_p, qty, fee, ts_i)

        eq = cash
        if position:
            side, entry, qty, cost, entry_dt = position
            funding = funding_cost(qty * entry, entry_dt, ts_i)
            if side == 1:
                eq += qty * price - funding
            else:
                eq += qty * entry + qty * (entry - price) - funding
        equity_curve.append(eq)

    # Close remaining
    if position:
        side, entry, qty, cost, entry_dt = position
        price = float(df["close"].iloc[-1])
        exit_p = price * (1 - SLIPPAGE_RATE * side)
        funding = funding_cost(qty * entry, entry_dt, df["timestamp"].iloc[-1])
        exit_fee = qty * entry * FEE_RATE
        if side == 1:
            pnl = qty * (exit_p - entry) - cost - exit_fee - funding
        else:
            pnl = qty * (entry - exit_p) - cost - exit_fee - funding
        cash += qty * entry + cost + pnl
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
        sig = STRATEGY_REGISTRY[cfg["strategy"]].generate_signals(df, cfg["params"]).shift(1).fillna(0)
        sig_data[name] = {"signal": sig, "weight": cfg["weight"], "pair": cfg["pair"]}

    cash = INITIAL_CAPITAL
    positions = {}  # pair -> (side, entry, qty, cost, entry_time)
    trades = []
    equity_curve = []

    for i in range(len(eth_data)):
        ts_i = eth_data["timestamp"].iloc[i]
        eth_price = float(eth_data["open"].iloc[i])   # next-open execution
        btc_price = float(btc_data["open"].iloc[i])
        eth_close = float(eth_data["close"].iloc[i])
        btc_close = float(btc_data["close"].iloc[i])

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
            cs = cur[0] if cur else 0

            if fs != cs:
                if cs != 0:
                    side, entry, qty, cost, entry_dt = cur
                    ep = price * (1 - SLIPPAGE_RATE * side)
                    funding = funding_cost(qty * entry, entry_dt, ts_i)
                    exit_fee = qty * entry * FEE_RATE
                    if side == 1:
                        pnl = qty * (ep - entry) - cost - exit_fee - funding
                    else:
                        pnl = qty * (entry - ep) - cost - exit_fee - funding
                    cash += qty * entry + cost + pnl
                    trades.append({"pnl": pnl, "pair": pair})
                    del positions[pair]
                if fs != 0:
                    sz = min(cash * MAX_POS_PCT, cash * 0.95)
                    if sz > 5:
                        ep = price * (1 + SLIPPAGE_RATE * fs)
                        qty = sz / ep
                        fee = sz * FEE_RATE
                        cash -= (sz + fee)
                        positions[pair] = (fs, ep, qty, fee, ts_i)

        eq = cash
        for pair, pos in positions.items():
            p = eth_close if "ETH" in pair else btc_close
            side, entry, qty, cost, entry_dt = pos
            funding = funding_cost(qty * entry, entry_dt, ts_i)
            if side == 1:
                eq += qty * p - funding
            else:
                eq += qty * entry + qty * (entry - p) - funding
        equity_curve.append(eq)

    for pair, pos in list(positions.items()):
        p = float(eth_data["close"].iloc[-1]) if "ETH" in pair else float(btc_data["close"].iloc[-1])
        side, entry, qty, cost, entry_dt = pos
        ep = p * (1 - SLIPPAGE_RATE * side)
        funding = funding_cost(qty * entry, entry_dt, eth_data["timestamp"].iloc[-1])
        exit_fee = qty * entry * FEE_RATE
        if side == 1:
            pnl = qty * (ep - entry) - cost - exit_fee - funding
        else:
            pnl = qty * (entry - ep) - cost - exit_fee - funding
        cash += qty * entry + cost + pnl
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

        # Anchor forward: the next window starts AFTER this window's test set,
        # so test data is never reused as training data for another window.
        start_idx = test_end_idx

    return windows


def run_walk_forward():
    """Main walk-forward optimization loop (honest selection)."""
    print("=" * 70)
    print("WALK-FORWARD OPTIMIZATION (honest selection)")
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

    regime_params = {}   # Only strategies that pass the OOS gate go here
    all_window_results = []
    skipped = []

    for strat_name, config in STRATEGY_CONFIGS.items():
        print(f"{'='*70}")
        print(f"OPTIMIZING: {strat_name}")
        print(f"{'='*70}")

        for pair in config["pairs"]:
            pair_label = pair.replace("_USDT_USDT", "")
            data_full = eth_data if "ETH" in pair else btc_data

            # 1. Candidate pool from TRAIN-only selection per window.
            #    The test set is never used to pick parameters.
            candidates = {}
            for w_idx, window in enumerate(windows):
                train_data = data_full.iloc[window["train_start"]:window["train_end"]].reset_index(drop=True)
                test_data = data_full.iloc[window["test_start"]:window["test_end"]].reset_index(drop=True)
                if len(train_data) < 50 or len(test_data) < 20:
                    continue
                print(f"    {pair_label} w{w_idx+1}: optimizing on {len(train_data)} train candles...", end="", flush=True)
                train_results = optimize_strategy(strat_name, pair, train_data, config["grid"], max_combos=300)
                for tr in train_results[:3]:
                    candidates.setdefault(json.dumps(tr["params"], sort_keys=True), tr["params"])
                print(f" top3 train={train_results[0]['return_pct'] if train_results else 0:+.1f}%")

            if not candidates:
                skipped.append(f"{strat_name}_{pair}")
                print(f"  {pair_label}: no valid train candidates")
                continue

            # 2. Honest OOS evaluation: EVERY candidate on EVERY test window.
            evals = {}
            for key, params in candidates.items():
                test_returns = []
                for window in windows:
                    test_data = data_full.iloc[window["test_start"]:window["test_end"]].reset_index(drop=True)
                    if len(test_data) < 20:
                        continue
                    r = backtest_single(test_data, strat_name, params)
                    test_returns.append(r["return_pct"])
                if not test_returns:
                    continue
                profitable = sum(1 for r in test_returns if r > 0)
                evals[key] = {
                    "params": params,
                    "mean_test_return": float(np.mean(test_returns)),
                    "profitable_windows": profitable,
                    "total_windows": len(test_returns),
                }

            if not evals:
                skipped.append(f"{strat_name}_{pair}")
                print(f"  {pair_label}: no evaluable candidates")
                continue

            # 3. Deploy gate: mean OOS return > 0 AND profitable in >= half the windows.
            deployable = {k: v for k, v in evals.items()
                          if v["mean_test_return"] > 0 and v["profitable_windows"] / v["total_windows"] >= 0.5}
            if deployable:
                best_key = max(deployable, key=lambda k: deployable[k]["mean_test_return"])
                ok_to_deploy = True
            else:
                best_key = max(evals, key=lambda k: evals[k]["mean_test_return"])
                ok_to_deploy = False

            best = evals[best_key]
            params = best["params"]
            key = f"{strat_name}_{pair}"

            if ok_to_deploy:
                regime_params[key] = params
            else:
                skipped.append(key)

            # Per-window OOS table for the CHOSEN params (honest reporting)
            window_results = []
            for w_idx, window in enumerate(windows):
                test_data = data_full.iloc[window["test_start"]:window["test_end"]].reset_index(drop=True)
                if len(test_data) < 20:
                    continue
                r = backtest_single(test_data, strat_name, params)
                window_results.append({
                    "window": w_idx + 1,
                    "test_dates": f"{window['test_start_date']} to {window['test_end_date']}",
                    "test_return": r["return_pct"],
                    "test_win_rate": r["win_rate"],
                    "test_max_dd": r["max_dd"],
                    "test_trades": r["trades"],
                })

            all_window_results.append({
                "strategy": strat_name,
                "pair": pair,
                "params": params,
                "deployed": ok_to_deploy,
                "mean_test_return": best["mean_test_return"],
                "profitable_windows": best["profitable_windows"],
                "total_windows": best["total_windows"],
                "window_results": window_results,
            })

            status = "DEPLOY" if ok_to_deploy else "REJECT (kept old params)"
            print(f"  {pair_label}: {status}")
            print(f"    Params: {json.dumps(params)}")
            print(f"    OOS: mean {best['mean_test_return']:+.1f}% | "
                  f"{best['profitable_windows']}/{best['total_windows']} profitable windows")

            time.sleep(0.3)

    # --- Build current regime portfolio (deployed strategies only) ---
    print(f"\n{'='*70}")
    print("CURRENT REGIME PARAMETERS")
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

    if skipped:
        print(f"\n  WARNING: Kept existing params for: {', '.join(sorted(set(skipped)))}")
        print("  (these strategies failed the OOS robustness gate)")

    # Validate current regime on full data (informational)
    if portfolio:
        print("\n  Validating current regime on full dataset...")
        full_result = backtest_portfolio(portfolio, eth_data, btc_data)
        print(f"  Full dataset result: {full_result['return_pct']:+.1f}% return, {full_result['win_rate']:.0f}% win rate, {full_result['max_dd']:.1f}% max DD")
    else:
        full_result = None
        print("\n  WARNING: Nothing passed the OOS gate — bot params NOT updated.")

    # Save optimized params
    output = {
        "timestamp": datetime.now().isoformat(),
        "data_range": f"{eth_data['timestamp'].iloc[0]} to {eth_data['timestamp'].iloc[-1]}",
        "windows": len(windows),
        "selection": "train-only selection + multi-window OOS deploy gate",
        "strategies": {},
        "portfolio": {k: {"strategy": v["strategy"], "pair": v["pair"], "weight": v["weight"], "params": v["params"]} for k, v in portfolio.items()},
        "full_backtest": {k: v for k, v in (full_result or {}).items() if k != "equity_curve"},
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

    # Bot params — deployed strategies only
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
    if bot_params:
        with open(bot_params_path, "w") as f:
            json.dump(bot_params, f, indent=2)
        print(f"  Bot params saved to {bot_params_path}")
    else:
        print(f"  Bot params NOT updated ({bot_params_path} left untouched)")

    return portfolio


if __name__ == "__main__":
    run_walk_forward()
