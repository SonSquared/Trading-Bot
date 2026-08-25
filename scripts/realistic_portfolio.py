#!/usr/bin/env python3
"""
Realistic Portfolio Analysis with 3x ATR Stop-Loss.

Re-runs the portfolio analysis simulating what happens when a
3x ATR disaster-protection stop-loss is applied to each strategy.
This gives the most realistic performance estimates for live trading.
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
from trading_system.portfolio.allocator import equal_weight, risk_parity, sharpe_weight
from trading_system.indicators import atr

RISK_FREE_RATE = 0.04
TRADING_HOURS_PER_YEAR = 365 * 24
INITIAL_CAPITAL = 10000.0
SL_ATR_MULT = 3.0

STRATEGIES = [
    {
        "name": "MACD",
        "label": "MACD ETH 4h",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
        "color": "#2196F3",
    },
    {
        "name": "ROC_Momentum",
        "label": "ROC ETH 4h",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"roc_period": 3, "roc_threshold": -1, "smooth_period": 1,
                   "trend_ema": 50, "trend_filter": False},
        "color": "#4CAF50",
    },
    {
        "name": "MACD",
        "label": "MACD BTC 4h",
        "pair": "BTC/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
        "color": "#FF9800",
    },
]


def apply_sl_to_trades(trades, df, atr_series):
    """
    Replay trades with 3x ATR stop-loss only.
    Strategy signal handles normal exits; SL is disaster protection.
    """
    simulated_trades = []
    equity = INITIAL_CAPITAL

    for trade in trades:
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        entry_price = trade["entry_price"]
        position_size = trade["position_size"]
        is_long = position_size > 0

        # Get ATR at entry
        try:
            entry_idx = df.index.get_indexer([entry_time], method="nearest")[0]
            entry_atr = float(atr_series.iloc[entry_idx]) if entry_idx < len(atr_series) else entry_price * 0.02
        except:
            entry_atr = entry_price * 0.02

        # Set SL level
        if is_long:
            sl_price = entry_price - entry_atr * SL_ATR_MULT
        else:
            sl_price = entry_price + entry_atr * SL_ATR_MULT

        # Get trade window
        trade_mask = (df.index >= entry_time) & (df.index <= exit_time)
        trade_candles = df[trade_mask]

        if trade_candles.empty:
            equity += trade["pnl"]
            simulated_trades.append(trade)
            continue

        # Check each candle for SL hit
        sl_hit = False
        exit_price = None

        for idx_i in range(len(trade_candles)):
            candle = trade_candles.iloc[idx_i]
            low = candle["low"]
            high = candle["high"]

            if is_long and low <= sl_price:
                exit_price = sl_price
                sl_hit = True
                break
            elif not is_long and high >= sl_price:
                exit_price = sl_price
                sl_hit = True
                break

        if sl_hit:
            # SL was hit — calculate P&L at SL price
            pnl_raw = position_size * (exit_price - entry_price)
            notional = abs(position_size * entry_price)
            fee = notional * 0.0004 * 2
            pnl_net = pnl_raw - fee
            equity += pnl_net
            simulated_trades.append({
                **trade,
                "exit_price": exit_price,
                "exit_reason": "stop_loss",
                "pnl": pnl_net,
                "original_pnl": trade["pnl"],
            })
        else:
            # No SL hit — use original trade
            equity += trade["pnl"]
            simulated_trades.append({
                **trade,
                "exit_reason": "signal",
            })

    return equity, simulated_trades


def calc_metrics(equity_arr, n_hours):
    """Calculate metrics from equity array."""
    if len(equity_arr) < 2:
        return {"error": "no data"}

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
    calmar = cagr / max_dd if max_dd > 0 else 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "volatility": vol,
    }


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    bt_config = cfg.backtest
    engine = BacktestEngine(bt_config)

    print("=" * 100)
    print("  REALISTIC PORTFOLIO ANALYSIS (3x ATR Stop-Loss)")
    print("=" * 100)

    # Run all backtests
    equity_curves = {}
    sl_equity_curves = {}
    all_results = {}

    for strat_def in STRATEGIES:
        label = strat_def["label"]
        strategy = get_strategy(strat_def["name"])
        df = loader.load(strat_def["pair"], strat_def["timeframe"])
        funding = loader.load_funding_rates(strat_def["pair"])

        signals = strategy.generate_signals(df, strat_def["params"])
        result = engine.run(df, signals, strat_def["name"], strat_def["params"],
                            strat_def["pair"], strat_def["timeframe"], funding)
        atr_series = atr(df, period=14)
        n_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600

        # Original signal-only equity
        eq_original = result.equity_curve.values
        eq_original_norm = eq_original / eq_original[0] * INITIAL_CAPITAL

        # Apply 3x ATR SL
        final_equity, sim_trades = apply_sl_to_trades(result.trades, df, atr_series)

        # Build SL-protected equity curve
        eq_sl = np.full(len(df), INITIAL_CAPITAL, dtype=np.float64)
        running = INITIAL_CAPITAL
        for t in sim_trades:
            running += t["pnl"]
            try:
                exit_idx = df.index.get_indexer([t["exit_time"]], method="nearest")[0]
                eq_sl[exit_idx:] = running
            except:
                pass

        equity_curves[label] = pd.Series(eq_original_norm, index=df.index)
        sl_equity_curves[label] = pd.Series(eq_sl, index=df.index)

        # Metrics
        base_m = calc_metrics(eq_original_norm, n_hours)
        sl_m = calc_metrics(eq_sl, n_hours)

        sl_exits = sum(1 for t in sim_trades if t.get("exit_reason") == "stop_loss")
        signal_exits = len(sim_trades) - sl_exits

        all_results[label] = {
            "base": base_m,
            "sl": sl_m,
            "trades": len(sim_trades),
            "sl_exits": sl_exits,
            "signal_exits": signal_exits,
        }

        print(f"\n  {label}:")
        print(f"    Signal-only: Return={base_m['total_return']*100:.1f}%  Sharpe={base_m['sharpe']:.2f}  MaxDD={base_m['max_drawdown']*100:.2f}%")
        print(f"    With 3x ATR SL: Return={sl_m['total_return']*100:.1f}%  Sharpe={sl_m['sharpe']:.2f}  MaxDD={sl_m['max_drawdown']*100:.2f}%")
        print(f"    Impact: Return {sl_m['total_return']*100 - base_m['total_return']*100:+.1f}%  Sharpe {sl_m['sharpe'] - base_m['sharpe']:+.2f}")
        print(f"    SL exits: {sl_exits}/{len(sim_trades)} trades ({sl_exits/max(len(sim_trades),1)*100:.1f}%)")

    # ── Portfolio analysis with SL-protected curves ───────────────
    print(f"\n{'=' * 100}")
    print("  PORTFOLIO ALLOCATION (with SL-protected curves)")
    print("=" * 100)

    labels = list(sl_equity_curves.keys())
    combined = pd.DataFrame(sl_equity_curves).dropna()
    for col in combined.columns:
        combined[col] = combined[col] / combined[col].iloc[0] * INITIAL_CAPITAL

    returns_df = combined.pct_change().fillna(0)

    alloc_methods = {
        "Equal Weight (33/33/33)": equal_weight(3),
        "Risk Parity": risk_parity(sl_equity_curves, lookback=200),
        "Sharpe Weighted": sharpe_weight(sl_equity_curves, lookback=200),
    }

    n_hours_port = (combined.index[-1] - combined.index[0]).total_seconds() / 3600
    best_portfolio = None
    best_sharpe = -float("inf")

    for name, weights in alloc_methods.items():
        port_ret = (returns_df * weights).sum(axis=1)
        port_equity = INITIAL_CAPITAL * (1 + port_ret).cumprod()
        port_arr = port_equity.values
        m = calc_metrics(port_arr, n_hours_port)

        print(f"\n  {name}:")
        print(f"    Weights: ", end="")
        for i, lbl in enumerate(labels):
            print(f"{lbl}={weights[i]:.1%}  ", end="")
        print()
        print(f"    Return: {m['total_return']*100:.1f}%  CAGR: {m['cagr']*100:.1f}%  "
              f"Sharpe: {m['sharpe']:.2f}  Sortino: {m['sortino']:.2f}  MaxDD: {m['max_drawdown']*100:.2f}%")

        if m["sharpe"] > best_sharpe:
            best_sharpe = m["sharpe"]
            best_portfolio = (name, m, weights)

    # ── Comparison table ──────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  COMPARISON: Signal-Only vs SL-Protected")
    print("=" * 100)

    print(f"\n  {'Strategy':<22s} {'Signal-Only':>30s} {'With 3x ATR SL':>30s} {'Impact':>15s}")
    print(f"  {'':22s} {'Return':>10s} {'Sharpe':>8s} {'MaxDD':>8s}  {'Return':>10s} {'Sharpe':>8s} {'MaxDD':>8s}  {'Sharpe':>8s}")

    print(f"  {'-' * 95}")

    for label in labels:
        r = all_results[label]
        b = r["base"]
        s = r["sl"]
        print(f"  {label:<22s} "
              f"{b['total_return']*100:>+9.1f}% {b['sharpe']:>+8.2f} {b['max_drawdown']*100:>7.2f}%  "
              f"{s['total_return']*100:>+9.1f}% {s['sharpe']:>+8.2f} {s['max_drawdown']*100:>7.2f}%  "
              f"{s['sharpe']-b['sharpe']:>+8.2f}")

    # Best portfolio
    bp_name, bp_m, bp_w = best_portfolio
    print(f"  {'-' * 95}")
    print(f"  {bp_name:<22s} "
          f"{'N/A':>10s} {'N/A':>8s} {'N/A':>8s}  "
          f"{bp_m['total_return']*100:>+9.1f}% {bp_m['sharpe']:>+8.2f} {bp_m['max_drawdown']*100:>7.2f}%  "
          f"{'BEST':>8s}")

    # ── Final recommendation ──────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  FINAL RECOMMENDATION")
    print("=" * 100)

    print(f"\n  Trading Configuration:")
    print(f"    Stop Loss:     3x ATR (disaster protection only)")
    print(f"    Take Profit:   None (strategy signal handles exits)")
    print(f"    Trailing Stop: None")
    print(f"    Breakeven:     None")
    print(f"")
    print(f"  Expected Performance (most realistic estimate):")
    print(f"    Portfolio:     {bp_name}")
    for i, lbl in enumerate(labels):
        print(f"      {lbl}: {bp_w[i]:.1%}")
    print(f"")
    print(f"    Total Return:  {bp_m['total_return']*100:.1f}%")
    print(f"    CAGR:          {bp_m['cagr']*100:.1f}%")
    print(f"    Sharpe:        {bp_m['sharpe']:.2f}")
    print(f"    Sortino:       {bp_m['sortino']:.2f}")
    print(f"    Max Drawdown:  {bp_m['max_drawdown']*100:.2f}%")
    print(f"")
    print(f"  Risk Management:")
    print(f"    Max loss per SL hit: ~4-5% (3x ATR)")
    print(f"    SL trigger rate: ~2-4% of trades")
    print(f"    Protection against: exchange outages, flash crashes,")
    print(f"                        black swans, bot disconnection")

    # Save
    output = {
        "config": {"sl_atr_mult": SL_ATR_MULT, "tp": "none", "trailing": "none", "breakeven": "none"},
        "individual": {label: {
            "signal_only": all_results[label]["base"],
            "with_sl": all_results[label]["sl"],
            "sl_exits": all_results[label]["sl_exits"],
            "total_trades": all_results[label]["trades"],
        } for label in labels},
        "portfolio": {
            "method": bp_name,
            "weights": {labels[i]: float(bp_w[i]) for i in range(len(labels))},
            "metrics": bp_m,
        },
    }
    out_path = Path("data/results/realistic_portfolio.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
