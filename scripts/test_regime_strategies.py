#!/usr/bin/env python3
"""
Test Strategy Performance by Market Regime.

Runs all strategy families during trending vs choppy periods
to identify which strategies work best in each regime.
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
from trading_system.bot.regime_detector import RegimeDetector, MarketRegime
from trading_system.indicators import atr

RISK_FREE_RATE = 0.04
TRADING_HOURS_PER_YEAR = 365 * 24
INITIAL_CAPITAL = 10000.0
SL_ATR_MULT = 3.0

# All candidate strategies
STRATEGIES = [
    # Trend strategies
    {"name": "MACD", "label": "MACD", "family": "trend",
     "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False}},
    {"name": "ROC_Momentum", "label": "ROC_Mom", "family": "momentum",
     "params": {"roc_period": 3, "roc_threshold": -1, "smooth_period": 1, "trend_ema": 50, "trend_filter": False}},
    # Mean reversion strategies
    {"name": "RSI_Reversion", "label": "RSI_Rev", "family": "mean_reversion",
     "params": {"rsi_period": 14, "entry_oversold": 25, "entry_overbought": 75,
                "exit_neutral_low": 40, "exit_neutral_high": 50,
                "use_bb_filter": True, "bb_period": 20, "bb_std": 2.0}},
    {"name": "Bollinger_Reversion", "label": "BB_Rev", "family": "mean_reversion",
     "params": {"bb_period": 20, "bb_std": 1.5, "rsi_filter": True, "rsi_period": 14,
                "rsi_oversold": 25, "rsi_overbought": 75, "exit_at_middle": True}},
    {"name": "ZScore_Reversion", "label": "ZScore", "family": "mean_reversion",
     "params": {"lookback": 20, "entry_threshold": 2.0, "exit_threshold": 0.5, "use_sma_baseline": True}},
]

# Params grids for MR strategies (wider search)
MR_GRIDS = {
    "RSI_Reversion": {
        "rsi_period": [7, 10, 14, 21],
        "entry_oversold": [20, 25, 30],
        "entry_overbought": [70, 75, 80],
        "exit_neutral_low": [35, 40, 45],
        "exit_neutral_high": [55, 60, 65],
        "use_bb_filter": [False, True],
        "bb_period": [15, 20],
        "bb_std": [1.5, 2.0, 2.5],
    },
    "Bollinger_Reversion": {
        "bb_period": [15, 20, 25],
        "bb_std": [1.5, 2.0, 2.5],
        "rsi_filter": [False, True],
        "rsi_period": [10, 14],
        "rsi_oversold": [20, 25, 30],
        "rsi_overbought": [70, 75, 80],
        "exit_at_middle": [True, False],
    },
    "ZScore_Reversion": {
        "lookback": [10, 15, 20, 30],
        "entry_threshold": [1.0, 1.5, 2.0, 2.5],
        "exit_threshold": [0.0, 0.25, 0.5, 0.75],
        "use_sma_baseline": [True, False],
    },
}


def calc_sharpe(equity_arr):
    """Quick Sharpe from equity array."""
    if len(equity_arr) < 10:
        return 0
    returns = np.diff(equity_arr) / np.where(equity_arr[:-1] != 0, equity_arr[:-1], 1.0)
    returns = np.where(np.isfinite(returns), returns, 0.0)
    active = returns[returns != 0]
    if len(active) < 5 or np.std(active) == 0:
        return 0
    return float(np.mean(active) / np.std(active, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR))


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    bt_config = cfg.backtest
    bt_config.execution.stop_loss_atr_mult = 0  # No SL for regime testing
    engine = BacktestEngine(bt_config)

    print("=" * 100)
    print("  REGIME-CONDITIONAL STRATEGY TESTING")
    print("=" * 100)

    # Load data
    pair = "ETH/USDT:USDT"
    tf = "4h"
    df = loader.load(pair, tf)
    funding = loader.load_funding_rates(pair)

    # Detect regimes
    detector = RegimeDetector()
    print(f"\n  Detecting regimes for {len(df)} candles...")

    # Classify each candle's regime
    regimes = []
    for i in range(200, len(df)):  # Skip first 200 for indicator warmup
        sub_df = df.iloc[:i + 1]
        state = detector.detect(sub_df)
        regimes.append({
            "idx": i,
            "regime": state.regime.value,
            "adx": state.adx_value,
        })

    regime_series = pd.Series([r["regime"] for r in regimes], index=df.index[200:])

    # Count regimes
    for reg in ["trending", "choppy", "transitional"]:
        count = sum(1 for r in regimes if r["regime"] == reg)
        print(f"  {reg.capitalize():>15s}: {count}/{len(regimes)} candles ({count/len(regimes)*100:.0f}%)")

    # ── Test each strategy in each regime ─────────────────────────
    print(f"\n{'=' * 100}")
    print("  STRATEGY PERFORMANCE BY REGIME (ETH/USDT 4h)")
    print("=" * 100)

    results_table = []

    for s_def in STRATEGIES:
        strategy = get_strategy(s_def["name"])
        signals = strategy.generate_signals(df, s_def["params"])

        # Calculate equity curve
        position = signals.replace(0, np.nan).ffill().fillna(0)
        strat_returns = df["close"].pct_change().fillna(0) * position.shift(1).fillna(0)
        equity = INITIAL_CAPITAL * (1 + strat_returns).cumprod()

        # Split by regime (align indices)
        for regime_name in ["trending", "choppy", "transitional"]:
            mask = regime_series.reindex(equity.index, method="ffill") == regime_name
            if mask.sum() < 20:
                continue

            regime_eq = equity[mask]
            regime_rets = strat_returns[mask]

            total_ret = (regime_eq.iloc[-1] / regime_eq.iloc[0]) - 1 if len(regime_eq) > 1 else 0
            sharpe = calc_sharpe(regime_eq.values)
            n_trades = int((position[mask].diff().abs() > 0).sum())

            results_table.append({
                "strategy": s_def["label"],
                "family": s_def["family"],
                "regime": regime_name,
                "return": total_ret,
                "sharpe": sharpe,
                "n_candles": mask.sum(),
            })

    # Print results
    print(f"\n  {'Strategy':<15s} {'Family':<16s} {'Regime':<15s} {'Return':>8s} {'Sharpe':>8s} {'Candles':>8s}")
    print(f"  {'-' * 75}")

    for r in results_table:
        print(f"  {r['strategy']:<15s} {r['family']:<16s} {r['regime']:<15s} "
              f"{r['return']*100:>+7.1f}% {r['sharpe']:>+8.2f} {r['n_candles']:>8d}")

    # ── Best strategy per regime ──────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  BEST STRATEGY PER REGIME")
    print("=" * 100)

    for regime_name in ["trending", "choppy", "transitional"]:
        regime_results = [r for r in results_table if r["regime"] == regime_name]
        if not regime_results:
            continue
        best = max(regime_results, key=lambda x: x["sharpe"])
        worst = min(regime_results, key=lambda x: x["sharpe"])

        print(f"\n  {regime_name.upper()}:")
        print(f"    Best:  {best['strategy']:<15s} Sharpe={best['sharpe']:>+.2f}  Return={best['return']*100:>+.1f}%")
        print(f"    Worst: {worst['strategy']:<15s} Sharpe={worst['sharpe']:>+.2f}  Return={worst['return']*100:>+.1f}%")

    # ── Regime-aware vs fixed allocation backtest ─────────────────
    print(f"\n{'=' * 100}")
    print("  REGIME-AWARE vs FIXED ALLOCATION BACKTEST")
    print("=" * 100)

    # Trend strategies: MACD + ROC_Momentum
    # MR strategies: RSI_Reversion (best performer in choppy)
    trend_strats = [s for s in STRATEGIES if s["family"] in ("trend", "momentum")]
    mr_strats = [s for s in STRATEGIES if s["family"] == "mean_reversion"]

    # Generate all signals
    all_signals = {}
    for s_def in trend_strats + mr_strats:
        strategy = get_strategy(s_def["name"])
        sigs = strategy.generate_signals(df, s_def["params"])
        all_signals[s_def["label"]] = sigs

    # Fixed allocation: 50% trend, 50% MR (equal split)
    fixed_trend_weight = 0.5
    fixed_mr_weight = 0.5

    # Regime-aware allocation
    regime_aware_equity = np.full(len(df), INITIAL_CAPITAL, dtype=np.float64)
    fixed_equity = np.full(len(df), INITIAL_CAPITAL, dtype=np.float64)
    cash_ra = INITIAL_CAPITAL
    cash_fixed = INITIAL_CAPITAL

    for i in range(201, len(df)):
        ts = df.index[i]

        # Get regime
        if ts in regime_series.index:
            regime = regime_series.loc[ts]
        else:
            regime = "transitional"

        # Regime weights
        if regime == "trending":
            ra_trend_w = 0.85
            ra_mr_w = 0.15
        elif regime == "choppy":
            ra_trend_w = 0.15
            ra_mr_w = 0.85
        else:
            ra_trend_w = 0.50
            ra_mr_w = 0.50

        # Calculate portfolio return for this candle
        price_change = df["close"].iloc[i] / df["close"].iloc[i - 1] - 1

        # Trend component return
        trend_ret = 0
        for s_def in trend_strats:
            sig = all_signals[s_def["label"]].iloc[i - 1] if i - 1 < len(all_signals[s_def["label"]]) else 0
            trend_ret += sig * price_change * (1.0 / len(trend_strats))

        # MR component return
        mr_ret = 0
        for s_def in mr_strats:
            sig = all_signals[s_def["label"]].iloc[i - 1] if i - 1 < len(all_signals[s_def["label"]]) else 0
            mr_ret += sig * price_change * (1.0 / len(mr_strats))

        # Regime-aware
        ra_ret = ra_trend_w * trend_ret + ra_mr_w * mr_ret
        cash_ra *= (1 + ra_ret)
        regime_aware_equity[i] = cash_ra

        # Fixed
        fix_ret = fixed_trend_weight * trend_ret + fixed_mr_weight * mr_ret
        cash_fixed *= (1 + fix_ret)
        fixed_equity[i] = cash_fixed

    # Calculate metrics
    ra_eq = regime_aware_equity[201:]
    fix_eq = fixed_equity[201:]
    n_hours = (df.index[-1] - df.index[201]).total_seconds() / 3600

    ra_m = {
        "return": (ra_eq[-1] / ra_eq[0]) - 1,
        "sharpe": calc_sharpe(ra_eq),
        "max_dd": float(((np.maximum.accumulate(ra_eq) - ra_eq) / np.maximum.accumulate(ra_eq)).max()),
    }
    fix_m = {
        "return": (fix_eq[-1] / fix_eq[0]) - 1,
        "sharpe": calc_sharpe(fix_eq),
        "max_dd": float(((np.maximum.accumulate(fix_eq) - fix_eq) / np.maximum.accumulate(fix_eq)).max()),
    }

    print(f"\n  {'Method':<25s} {'Return':>8s} {'Sharpe':>8s} {'MaxDD':>8s}")
    print(f"  {'-' * 55}")
    print(f"  {'Fixed (50/50)':<25s} {fix_m['return']*100:>+7.1f}% {fix_m['sharpe']:>+8.2f} {fix_m['max_dd']*100:>7.2f}%")
    print(f"  {'Regime-Aware':<25s} {ra_m['return']*100:>+7.1f}% {ra_m['sharpe']:>+8.2f} {ra_m['max_dd']*100:>7.2f}%")

    # Improvement
    if fix_m["sharpe"] != 0:
        sharpe_improvement = (ra_m["sharpe"] - fix_m["sharpe"]) / abs(fix_m["sharpe"]) * 100
        print(f"\n  Sharpe improvement: {sharpe_improvement:>+.1f}%")
        print(f"  Return improvement: {(ra_m['return']-fix_m['return'])*100:>+.1f}%")

    print(f"\n  Time: {time.time()-t0:.0f}s")
    print("=" * 100)


if __name__ == "__main__":
    main()
