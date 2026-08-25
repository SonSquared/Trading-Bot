#!/usr/bin/env python3
"""
Test ZScore_Reversion with different SL distances.
Uses the compare_sltp_impact.py pattern: run base backtest, then replay trades with SL.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy
from trading_system.indicators import atr

INITIAL_CAPITAL = 10000.0
TRADING_HOURS_PER_YEAR = 365 * 24
RISK_FREE_RATE = 0.04


def calc_metrics(equity_arr, n_hours):
    """Calculate metrics from numpy equity array."""
    if len(equity_arr) < 2:
        return {}
    total_return = (equity_arr[-1] / equity_arr[0]) - 1
    n_years = n_hours / TRADING_HOURS_PER_YEAR if n_hours > 0 else 1
    cagr = (equity_arr[-1] / equity_arr[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    returns = np.diff(equity_arr) / np.where(equity_arr[:-1] != 0, equity_arr[:-1], 1.0)
    returns = np.where(np.isfinite(returns), returns, 0.0)

    vol = float(np.std(returns, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR)) if len(returns) > 1 else 0
    excess = returns - RISK_FREE_RATE / TRADING_HOURS_PER_YEAR
    sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR)) if len(excess) > 1 and np.std(excess) > 0 else 0

    neg_rets = returns[returns < 0]
    downside_dev = float(np.std(neg_rets, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR)) if len(neg_rets) > 1 else 0
    sortino = (cagr - RISK_FREE_RATE) / downside_dev if downside_dev > 0 else 0

    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / np.where(running_max > 0, running_max, 1.0)
    max_dd = abs(float(drawdown.min()))

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
    }


def replay_with_sl(trades, df, atr_series, sl_mult):
    """Replay trades with stop-loss only (no TP, no trailing)."""
    if not trades or sl_mult == 0:
        return None, trades

    simulated = []
    equity = INITIAL_CAPITAL

    for trade in trades:
        entry_price = trade["entry_price"]
        position_size = trade["position_size"]
        is_long = position_size > 0
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]

        # Get ATR at entry
        entry_idx = df.index.get_indexer([entry_time], method="nearest")[0]
        entry_atr = float(atr_series.iloc[entry_idx]) if entry_idx < len(atr_series) and np.isfinite(atr_series.iloc[entry_idx]) else entry_price * 0.02

        # Set SL level
        if is_long:
            sl_price = entry_price - entry_atr * sl_mult
        else:
            sl_price = entry_price + entry_atr * sl_mult

        # Get trade window candles
        trade_mask = (df.index >= entry_time) & (df.index <= exit_time)
        trade_candles = df[trade_mask]

        if trade_candles.empty:
            equity += trade["pnl"]
            simulated.append(trade)
            continue

        # Check each candle for SL hit
        exit_price = None
        exit_reason = "signal"

        for _, candle in trade_candles.iterrows():
            high = candle["high"]
            low = candle["low"]

            if is_long and low <= sl_price:
                exit_price = sl_price
                exit_reason = "stop_loss"
                break
            elif not is_long and high >= sl_price:
                exit_price = sl_price
                exit_reason = "stop_loss"
                break

        if exit_price is None:
            exit_price = trade["exit_price"]
            exit_reason = "signal"

        # P&L
        pnl_raw = position_size * (exit_price - entry_price)
        notional = abs(position_size * entry_price)
        fee = notional * 0.0005 * 2  # taker fee both sides
        pnl_net = pnl_raw - fee

        equity += pnl_net
        simulated.append({
            **trade,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl": pnl_net,
        })

    return equity, simulated


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    bt_config = cfg.backtest
    engine = BacktestEngine(bt_config)

    # ZScore parameter combos to test
    param_combos = [
        {"lookback": 20, "entry_threshold": 2.0, "exit_threshold": 0.5},  # default
        {"lookback": 15, "entry_threshold": 1.8, "exit_threshold": 0.3},  # tighter
        {"lookback": 25, "entry_threshold": 2.0, "exit_threshold": 0.5},  # wider lookback
        {"lookback": 30, "entry_threshold": 1.5, "exit_threshold": 0.25}, # more extreme entry
    ]

    # SL distances to test
    sl_configs = [0, 3.0, 5.0, 8.0, 10.0, 15.0]

    pairs = [
        ("ETH/USDT:USDT", "4h"),
        ("BTC/USDT:USDT", "4h"),
    ]

    all_results = []

    print("=" * 100)
    print("  ZSCORE_REVERSION STOP-LOSS DISTANCE TEST")
    print("=" * 100)

    for pair, tf in pairs:
        print(f"\n{'=' * 100}")
        print(f"  {pair} {tf}")
        print(f"{'=' * 100}")

        strategy = get_strategy("ZScore_Reversion")
        df = loader.load(pair, tf)
        atr_series = atr(df, period=14)
        n_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600

        for params in param_combos:
            # Generate signals once
            signals = strategy.generate_signals(df, params)

            # Run base backtest (no SL)
            base_config = SystemConfig.default()
            base_config.backtest.execution.stop_loss_atr_mult = 0
            base_engine = BacktestEngine(base_config.backtest)
            base_result = base_engine.run(df, signals, "ZScore_Reversion", params, pair, tf)

            if base_result.total_trades == 0:
                continue

            base_trades = base_result.trades
            print(f"\n  Params: lb={params['lookback']}, entry_z={params['entry_threshold']}, exit_z={params['exit_threshold']}")
            print(f"  Base: {base_result.total_trades} trades, Sharpe={base_result.sharpe:.2f}, "
                  f"Return={base_result.total_return*100:.1f}%, WR={base_result.win_rate*100:.0f}%, "
                  f"MaxDD={base_result.max_drawdown*100:.2f}%")

            for sl_mult in sl_configs:
                if sl_mult == 0:
                    # Use base results
                    sl_exits = 0
                    signal_exits = base_result.total_trades
                    metrics = {
                        "total_return": base_result.total_return,
                        "sharpe": base_result.sharpe,
                        "sortino": base_result.sortino,
                        "max_drawdown": base_result.max_drawdown,
                    }
                else:
                    final_eq, sim_trades = replay_with_sl(base_trades, df, atr_series, sl_mult)
                    if final_eq is None:
                        continue

                    # Build equity curve
                    sim_equity = np.full(len(df), INITIAL_CAPITAL, dtype=np.float64)
                    running = INITIAL_CAPITAL
                    for t in sim_trades:
                        running += t["pnl"]
                        try:
                            exit_idx = df.index.get_indexer([t["exit_time"]], method="nearest")[0]
                            sim_equity[exit_idx:] = running
                        except:
                            pass

                    metrics = calc_metrics(sim_equity, n_hours)
                    sl_exits = sum(1 for t in sim_trades if t.get("exit_reason") == "stop_loss")
                    signal_exits = len(sim_trades) - sl_exits

                sl_label = f"{sl_mult}x ATR" if sl_mult > 0 else "None  "
                sl_pct = (sl_exits / base_result.total_trades * 100) if base_result.total_trades > 0 else 0
                print(f"    SL={sl_label}: Ret={metrics['total_return']*100:+7.1f}%  "
                      f"Sharpe={metrics['sharpe']:6.2f}  MaxDD={metrics['max_drawdown']*100:5.2f}%  "
                      f"SL_hit={sl_exits:4d} ({sl_pct:4.1f}%)  Signal={signal_exits:4d}")

                all_results.append({
                    "pair": pair,
                    "tf": tf,
                    "params": params,
                    "sl_mult": sl_mult,
                    "return_pct": metrics["total_return"] * 100,
                    "sharpe": metrics["sharpe"],
                    "max_dd_pct": metrics["max_drawdown"] * 100,
                    "sl_exits": sl_exits,
                    "signal_exits": signal_exits,
                    "total_trades": base_result.total_trades,
                    "win_rate": base_result.win_rate,
                })

    # Find the best configuration
    print(f"\n\n{'=' * 100}")
    print("  BEST CONFIGURATION BY SHARPE (SL > 0)")
    print(f"{'=' * 100}")

    sl_results = [r for r in all_results if r["sl_mult"] > 0]
    sl_results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n  {'Pair':<20} {'Params':<40} {'SL':>6} {'Ret%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'SL_hit':>7} {'Total':>6}")
    print(f"  {'-' * 110}")
    for r in sl_results[:10]:
        p = r["params"]
        param_str = f"lb={p['lookback']},ez={p['entry_threshold']},xz={p['exit_threshold']}"
        print(f"  {r['pair']:<20} {param_str:<40} {r['sl_mult']:>5.1f}x {r['return_pct']:>+7.1f}% "
              f"{r['sharpe']:>8.2f} {r['max_dd_pct']:>7.2f}% {r['sl_exits']:>7} {r['total_trades']:>6}")

    # Compare no SL vs best SL
    print(f"\n\n{'=' * 100}")
    print("  NO SL vs BEST SL COMPARISON")
    print(f"{'=' * 100}")

    for pair, tf in pairs:
        pair_results = [r for r in all_results if r["pair"] == pair]
        no_sl = [r for r in pair_results if r["sl_mult"] == 0]
        with_sl = [r for r in pair_results if r["sl_mult"] > 0]

        if not no_sl or not with_sl:
            continue

        # Best no-SL by Sharpe
        best_no_sl = max(no_sl, key=lambda x: x["sharpe"])
        # Best with-SL by Sharpe
        best_with_sl = max(with_sl, key=lambda x: x["sharpe"])

        p_ns = best_no_sl["params"]
        p_ws = best_with_sl["params"]

        print(f"\n  {pair} {tf}:")
        print(f"    No SL:     lb={p_ns['lookback']}, ez={p_ns['entry_threshold']}, xz={p_ns['exit_threshold']}")
        print(f"               Ret={best_no_sl['return_pct']:+.1f}%  Sharpe={best_no_sl['sharpe']:.2f}  "
              f"MaxDD={best_no_sl['max_dd_pct']:.2f}%  Trades={best_no_sl['total_trades']}")
        print(f"    Best SL:   lb={p_ws['lookback']}, ez={p_ws['entry_threshold']}, xz={p_ws['exit_threshold']}  "
              f"SL={best_with_sl['sl_mult']}x ATR")
        print(f"               Ret={best_with_sl['return_pct']:+.1f}%  Sharpe={best_with_sl['sharpe']:.2f}  "
              f"MaxDD={best_with_sl['max_dd_pct']:.2f}%  SL_hit={best_with_sl['sl_exits']}/{best_with_sl['total_trades']}")

        ret_diff = best_with_sl["return_pct"] - best_no_sl["return_pct"]
        sharpe_diff = best_with_sl["sharpe"] - best_no_sl["sharpe"]
        print(f"    Impact:    Return {ret_diff:+.1f}%, Sharpe {sharpe_diff:+.2f}")

    # Save results
    out_path = Path("data/results/zscore_sl_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
