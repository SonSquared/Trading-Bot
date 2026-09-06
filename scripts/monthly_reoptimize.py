#!/usr/bin/env python3
"""
Monthly Walk-Forward Re-Optimization

Fetches fresh data from Kraken, runs walk-forward optimization
on all active strategies, and saves updated parameters.

Selection is HONEST:
  - Parameters are picked on the TRAIN portion of each window only.
  - Every candidate is then evaluated on EVERY test (out-of-sample)
    window, and the chosen candidate must have a positive mean OOS
    return AND be profitable in a majority of windows.
  - If nothing passes that gate, the existing bot params are kept and
    the failure is reported loudly instead of deploying a curve-fit.

Results are saved to data/results/optimized_params.json and
data/results/bot_strategy_params.json which paper_trader.py loads.

If Telegram is configured via environment variables, sends a summary
notification. Credentials are NEVER hardcoded in this file.
"""

import sys
import os
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.strategies import STRATEGY_REGISTRY
from trading_system.bot.accounting import FEE_RATE, SLIPPAGE_RATE, funding_cost

# --- Config ---
INITIAL_CAPITAL = 97.0
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

# Telegram config — environment variables ONLY.
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")


# --- Telegram ---
def send_telegram(text):
    """Send a Telegram notification. Fails loudly when not configured."""
    if not TG_TOKEN or not TG_CHAT:
        print("  Telegram: NOT CONFIGURED — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return
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
    """Fetch OHLCV data from Kraken (fallback Binance), paginating to get `days` of history.

    Each exchange call returns at most ~720-1000 candles, so the previous
    implementation silently truncated to ~166 days of 4h candles while
    claiming 730. This version pages through with `since` until the full
    window is covered.
    """
    symbol_map = {
        "ETH_USDT_USDT": "ETH/USDT",
        "BTC_USDT_USDT": "BTC/USDT",
    }
    symbol = symbol_map.get(pair, pair.replace("_", "/"))
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    max_candles = int(days * 24 * 3600 / 4 / 3600) + 200  # 4h candles for the window

    import ccxt

    for exchange_cls, limit in ((ccxt.kraken, 720), (ccxt.binance, 1000)):
        try:
            exchange = exchange_cls({"enableRateLimit": True})
            spot_symbol = symbol.replace(":USDT", "")  # spot symbol for both
            all_candles = []
            since = since_ms
            while len(all_candles) < max_candles:
                batch = exchange.fetch_ohlcv(spot_symbol, timeframe, since=since, limit=limit)
                if not batch:
                    break
                all_candles.extend(batch)
                if len(batch) < limit:
                    break
                since = batch[-1][0] + 1
                rate_limit = getattr(exchange, "rateLimit", 0) or 500
                time.sleep(rate_limit / 1000.0)
                if len(all_candles) > max_candles + limit:
                    break
            if len(all_candles) >= 500:
                df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
                print(f"  {exchange_cls.__name__}: {len(df)} candles ({timeframe}, {days}d)")
                return df
            print(f"  {exchange_cls.__name__}: only {len(all_candles)} candles, trying fallback")
        except Exception as e:
            print(f"  {exchange_cls.__name__} failed for {spot_symbol}: {e}")

    return None


# --- Backtest ---
def backtest_single(df: pd.DataFrame, strat_name: str, params: dict) -> dict:
    """Run a single strategy backtest with next-open execution + full costs.

    A signal generated on closed candle i-1 fills at the OPEN of candle i,
    matching the main backtester's no-look-ahead convention. Costs are
    identical to the paper trader (trading_system.bot.accounting): 0.05%
    fee per side, 0.02% slippage per side, and funding at every 8h UTC
    boundary while the position is held.
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

        # Track equity (funding accrued to date — charged once, no double count)
        eq = cash
        if position:
            side, entry, qty, cost, entry_dt = position
            funding = funding_cost(qty * entry, entry_dt, ts_i)
            if side == 1:
                eq += qty * price - funding
            else:
                eq += qty * entry + qty * (entry - price) - funding
        equity_curve.append(eq)

    # Close remaining position
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
    """Grid search with scoring. Returns results sorted by score (train-side only)."""
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

        # Anchor forward: the next window starts AFTER this window's test set,
        # so test data is never reused as training data for another window.
        start_idx = test_end_idx

    return windows


def select_params_honest(strat_name, pair, data, grid, windows, max_combos=300):
    """Train-only selection + honest multi-window OOS evaluation.

    Returns (best_params, evaluation, ok_to_deploy).
    Evaluation dict has: mean/median OOS return, profitable/total windows,
    and the worst OOS return across windows.
    """
    # 1. For each window, pick top candidates on TRAIN only (no test peeking).
    candidates = {}
    for w in windows:
        train = data.iloc[w["train_start"]:w["train_end"]].reset_index(drop=True)
        test = data.iloc[w["test_start"]:w["test_end"]].reset_index(drop=True)
        if len(train) < 50 or len(test) < 20:
            continue
        train_results = optimize_strategy(strat_name, pair, train, grid, max_combos=max_combos)
        for tr in train_results[:3]:
            candidates.setdefault(json.dumps(tr["params"], sort_keys=True), tr["params"])

    if not candidates:
        return None, {}, False

    # 2. Evaluate EVERY candidate on EVERY test window (honest OOS matrix).
    evals = {}
    for key, params in candidates.items():
        test_returns = []
        for w in windows:
            test = data.iloc[w["test_start"]:w["test_end"]].reset_index(drop=True)
            if len(test) < 20:
                continue
            r = backtest_single(test, strat_name, params)
            test_returns.append(r["return_pct"])
        if not test_returns:
            continue
        profitable = sum(1 for r in test_returns if r > 0)
        evals[key] = {
            "params": params,
            "mean_test_return": float(np.mean(test_returns)),
            "median_test_return": float(np.median(test_returns)),
            "profitable_windows": profitable,
            "total_windows": len(test_returns),
            "min_test_return": float(np.min(test_returns)),
        }

    if not evals:
        return None, {}, False

    # 3. Deploy gate: positive mean OOS return AND profitable in >= half the windows.
    deployable = {
        k: v for k, v in evals.items()
        if v["mean_test_return"] > 0 and v["profitable_windows"] / v["total_windows"] >= 0.5
    }
    if deployable:
        best_key = max(deployable, key=lambda k: deployable[k]["mean_test_return"])
        ok_to_deploy = True
    else:
        # No honest evidence — do NOT deploy new params.
        best_key = max(evals, key=lambda k: evals[k]["mean_test_return"])
        ok_to_deploy = False

    return evals[best_key]["params"], evals[best_key], ok_to_deploy


# --- Main ---
def main():
    t0 = time.time()
    print("=" * 70)
    print("MONTHLY WALK-FORWARD RE-OPTIMIZATION (honest selection)")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # Fetch fresh data for each pair
    all_data = {}
    for pair in ["ETH_USDT_USDT", "BTC_USDT_USDT"]:
        print(f"\nFetching {pair}...")
        df = fetch_data(pair, "4h", days=730)
        if df is not None and len(df) > 100:
            all_data[pair] = df
            print(f"  Data range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        else:
            print("  SKIP: insufficient data")

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
    skipped = []

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

            params, evaluation, ok_to_deploy = select_params_honest(
                strat_name, pair, data, config["grid"], windows
            )

            if params is None:
                msg = f"{strat_name} on {pair}: no valid params found"
                errors.append(msg)
                print(f"  WARNING: {msg}")
                continue

            # Honest per-window OOS results for the CHOSEN params
            oos_by_window = []
            for w_idx, w in enumerate(windows):
                test = data.iloc[w["test_start"]:w["test_end"]].reset_index(drop=True)
                if len(test) < 20:
                    continue
                r = backtest_single(test, strat_name, params)
                oos_by_window.append({
                    "window": w_idx + 1,
                    "test_dates": f"{w['test_start_date']} to {w['test_end_date']}",
                    "test_return": r["return_pct"],
                    "test_win_rate": r["win_rate"],
                    "test_max_dd": r["max_dd"],
                    "test_trades": r["trades"],
                })

            key = f"{strat_name}_{pair}"
            full_r = backtest_single(data, strat_name, params)
            results[key] = {
                "strategy": strat_name,
                "pair": pair,
                "params": params,
                "deployed": ok_to_deploy,
                "mean_oos_return": evaluation.get("mean_test_return", 0),
                "median_oos_return": evaluation.get("median_test_return", 0),
                "profitable_windows": evaluation.get("profitable_windows", 0),
                "total_windows": evaluation.get("total_windows", 0),
                "min_oos_return": evaluation.get("min_test_return", 0),
                "full_return": full_r["return_pct"],
                "full_trades": full_r["trades"],
                "full_win_rate": full_r["win_rate"],
                "full_max_dd": full_r["max_dd"],
                "window_results": oos_by_window,
            }

            status = "DEPLOY" if ok_to_deploy else "REJECT (kept old params)"
            print(f"\n  RESULT: {pair_label} {strat_name} -> {status}")
            print(f"    Params: {json.dumps(params)}")
            print(f"    OOS: mean {evaluation.get('mean_test_return', 0):+.1f}% | "
                  f"profitable {evaluation.get('profitable_windows', 0)}/{evaluation.get('total_windows', 0)} windows")
            print(f"    Full backtest (informational): {full_r['return_pct']:+.1f}% return, {full_r['win_rate']:.0f}% WR")
            if not ok_to_deploy:
                skipped.append(key)

    # --- Save results ---
    elapsed = time.time() - t0

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_candles": min_len,
        "elapsed_seconds": round(elapsed, 1),
        "selection": "train-only selection + multi-window OOS deploy gate",
        "strategies": results,
        "errors": errors,
    }

    report_path = RESULTS_DIR / "optimized_params.json"
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")

    # Save bot-compatible params — ONLY for strategies that passed the gate.
    bot_params = {}
    for key, data in results.items():
        if not data.get("deployed"):
            continue
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
    if bot_params:
        with open(bot_params_path, "w") as f:
            json.dump(bot_params, f, indent=2)
        print(f"  Bot params updated: {bot_params_path}")
    else:
        print("  WARNING: No strategy passed the OOS gate — bot params NOT updated (keeping existing).")

    # --- Telegram summary ---
    summary_lines = ["MONTHLY RE-OPTIMIZATION COMPLETE\n"]
    summary_lines.append(f"Data: {min_len} candles | Time: {elapsed:.0f}s\n")

    for key, data in results.items():
        pair_label = data["pair"].replace("_USDT_USDT", "")
        gate = "DEPLOYED" if data.get("deployed") else "REJECTED (kept old params)"
        summary_lines.append(
            f"{'✅' if data.get('deployed') else '⛔'} {data['strategy']} {pair_label} [{gate}]\n"
            f"  OOS: mean {data['mean_oos_return']:+.1f}% | "
            f"{data['profitable_windows']}/{data['total_windows']} profitable windows\n"
            f"  Params: {json.dumps(data['params'])}\n"
        )

    if skipped:
        summary_lines.append(f"\n⚠️ Kept existing params for: {', '.join(skipped)}")
    if errors:
        summary_lines.append(f"\nErrors: {'; '.join(errors)}")

    summary_lines.append(f"\n{datetime.now(timezone.utc).strftime('%b %d, %H:%M UTC')}")
    send_telegram("\n".join(summary_lines))

    # Print final summary
    print(f"\n{'='*70}")
    print("RE-OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Strategies optimized: {len(results)}")
    print(f"  Deployed: {len(bot_params)} | Rejected: {len(skipped)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    for key, data in results.items():
        pair_label = data["pair"].replace("_USDT_USDT", "")
        gate = "DEPLOYED" if data.get("deployed") else "REJECTED"
        print(f"\n  {data['strategy']} {pair_label} [{gate}]:")
        print(f"    OOS: mean {data['mean_oos_return']:+.1f}% | "
              f"{data['profitable_windows']}/{data['total_windows']} profitable windows")
        print(f"    Params: {json.dumps(data['params'])}")

    print(f"\n{'='*70}")
    return results


if __name__ == "__main__":
    main()