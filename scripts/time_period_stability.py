#!/usr/bin/env python3
"""
Time-Period Stability Analysis.

Splits ETH 4h data into 4 equal periods and tests each strategy
to verify they remain profitable across different market regimes.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy

RISK_FREE_RATE = 0.04
TRADING_HOURS_PER_YEAR = 365 * 24

STRATEGIES = [
    {
        "name": "MACD",
        "label": "MACD ETH 4h",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
    },
    {
        "name": "ROC_Momentum",
        "label": "ROC ETH 4h",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"roc_period": 3, "roc_threshold": -1, "smooth_period": 1,
                   "trend_ema": 50, "trend_filter": False},
    },
    {
        "name": "MACD",
        "label": "MACD BTC 4h",
        "pair": "BTC/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
    },
]


def calc_metrics(equity_series, n_hours):
    """Calculate metrics from an equity curve."""
    if equity_series is None or len(equity_series) < 2:
        return {"error": "no data"}
    eq = equity_series.values
    total_return = (eq[-1] / eq[0]) - 1
    n_years = n_hours / TRADING_HOURS_PER_YEAR if n_hours > 0 else 1
    cagr = (eq[-1] / eq[0]) ** (1 / n_years) - 1 if n_years > 0 and eq[0] > 0 else 0

    returns = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1.0)
    returns = np.where(np.isfinite(returns), returns, 0.0)

    vol = float(np.std(returns, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR)) if len(returns) > 1 else 0
    excess = returns - RISK_FREE_RATE / TRADING_HOURS_PER_YEAR
    sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR)) if len(excess) > 1 and np.std(excess) > 0 else 0

    neg_rets = returns[returns < 0]
    downside_dev = float(np.std(neg_rets, ddof=1) * np.sqrt(TRADING_HOURS_PER_YEAR)) if len(neg_rets) > 1 else 0
    sortino = (cagr - RISK_FREE_RATE) / downside_dev if downside_dev > 0 else 0

    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / np.where(running_max > 0, running_max, 1.0)
    max_dd = abs(float(drawdown.min()))
    calmar = cagr / max_dd if max_dd > 0 else 0

    win_rate = np.mean(returns[1:] > 0) if len(returns) > 1 else 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "volatility": vol,
        "win_rate": win_rate,
        "candles": len(eq),
    }


def run_strategy_on_slice(strategy, df, funding, pair, tf, name, params, engine):
    """Run a strategy on a data slice and return metrics."""
    if df is None or len(df) < 100:
        return {"error": "too few candles"}
    try:
        signals = strategy.generate_signals(df, params)
        result = engine.run(df, signals, name, params, pair, tf, funding)
        n_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
        return calc_metrics(result.equity_curve, n_hours)
    except Exception as e:
        return {"error": str(e)}


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    bt_config = cfg.backtest
    engine = BacktestEngine(bt_config)

    # Load full data
    eth_4h = loader.load("ETH/USDT:USDT", "4h")
    btc_4h = loader.load("BTC/USDT:USDT", "4h")
    eth_funding = loader.load_funding_rates("ETH/USDT:USDT")
    btc_funding = loader.load_funding_rates("BTC/USDT:USDT")

    # Split into 4 equal periods
    n_eth = len(eth_4h)
    n_btc = len(btc_4h)
    q_eth = n_eth // 4
    q_btc = n_btc // 4

    eth_periods = [
        ("Q1: Jan-Jun 2022 (Bear Onset)", eth_4h.iloc[:q_eth]),
        ("Q2: Jul 2022-Dec 2023 (Crypto Winter)", eth_4h.iloc[q_eth:2*q_eth]),
        ("Q3: Jan-Jun 2024 (Early Recovery)", eth_4h.iloc[2*q_eth:3*q_eth]),
        ("Q4: Jul 2024-Aug 2026 (Bull+Consol)", eth_4h.iloc[3*q_eth:]),
    ]

    btc_periods = [
        ("Q1: Jan-Jun 2022 (Bear Onset)", btc_4h.iloc[:q_btc]),
        ("Q2: Jul 2022-Dec 2023 (Crypto Winter)", btc_4h.iloc[q_btc:2*q_btc]),
        ("Q3: Jan-Jun 2024 (Early Recovery)", btc_4h.iloc[2*q_btc:3*q_btc]),
        ("Q4: Jul 2024-Aug 2026 (Bull+Consol)", btc_4h.iloc[3*q_btc:]),
    ]

    print("=" * 100)
    print("  TIME-PERIOD STABILITY ANALYSIS")
    print("=" * 100)

    print("\n  ETH/USDT 4h: {} candles, {} per quarter".format(n_eth, q_eth))
    for name, df in eth_periods:
        print("    {}: {} to {} ({} candles)".format(
            name, df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d'), len(df)))

    print("\n  BTC/USDT 4h: {} candles, {} per quarter".format(n_btc, q_btc))
    for name, df in btc_periods:
        print("    {}: {} to {} ({} candles)".format(
            name, df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d'), len(df)))

    # Run each strategy on each period
    all_results = {}

    for strat_def in STRATEGIES:
        strat_name = strat_def["name"]
        label = strat_def["label"]
        pair = strat_def["pair"]
        tf = strat_def["timeframe"]
        params = strat_def["params"]
        strategy = get_strategy(strat_name)

        if "ETH" in pair:
            periods = eth_periods
            funding = eth_funding
        else:
            periods = btc_periods
            funding = btc_funding

        print("\n" + "-" * 100)
        print("  {} ({} on {} {})".format(label, strat_name, pair, tf))
        print("  Params: {}".format(json.dumps(params)))
        print("-" * 100)

        period_results = []
        for period_name, df_slice in periods:
            metrics = run_strategy_on_slice(strategy, df_slice, funding, pair, tf, strat_name, params, engine)
            metrics["period"] = period_name
            period_results.append(metrics)

            if "error" in metrics:
                print("  {:<35s} ERROR: {}".format(period_name, metrics["error"]))
            else:
                print("  {:<35s} Ret={:>+7.1f}%  CAGR={:>+6.1f}%  Sharpe={:>+7.2f}  "
                      "Sortino={:>+7.2f}  MaxDD={:>6.2f}%  WR={:>5.1%}".format(
                    period_name, metrics["total_return"]*100, metrics["cagr"]*100,
                    metrics["sharpe"], metrics["sortino"],
                    metrics["max_drawdown"]*100, metrics.get("win_rate", 0)))

        all_results[label] = period_results

        # Stability assessment
        valid = [r for r in period_results if "error" not in r]
        if valid:
            sharpes = [r["sharpe"] for r in valid]
            returns = [r["total_return"] for r in valid]
            profitable_periods = sum(1 for r in returns if r > 0)

            print("\n  Stability Summary:")
            print("    Profitable periods: {}/{}".format(profitable_periods, len(valid)))
            print("    Sharpe range: [{:.2f}, {:.2f}]".format(min(sharpes), max(sharpes)))
            print("    Sharpe std:   {:.2f}".format(np.std(sharpes)))
            print("    Return range: [{:.1f}%, {:.1f}%]".format(min(returns)*100, max(returns)*100))

            if profitable_periods == len(valid):
                print("    => ALL periods profitable -- HIGHLY STABLE")
            elif profitable_periods >= len(valid) * 0.75:
                print("    => Most periods profitable -- STABLE")
            elif profitable_periods >= len(valid) * 0.5:
                print("    => Half periods profitable -- MODERATE")
            else:
                print("    => Few periods profitable -- UNSTABLE")

    # Cross-strategy comparison
    print("\n" + "=" * 100)
    print("  CROSS-STRATEGY STABILITY COMPARISON")
    print("=" * 100)

    period_names = [p[0] for p in eth_periods]

    # Sharpe table
    print("\n  Sharpe by Period:")
    print("  {:<35s}".format("Period"), end="")
    for label in all_results:
        print(" {:>16s}".format(label), end="")
    print()
    print("  " + "-" * (35 + 17 * len(all_results)))
    for i, pname in enumerate(period_names):
        print("  {:<35s}".format(pname), end="")
        for label in all_results:
            r = all_results[label][i]
            if "error" in r:
                print(" {:>16s}".format("N/A"), end="")
            else:
                print(" {:>+15.2f}".format(r["sharpe"]), end="")
        print()

    # Return table
    print("\n  Total Return by Period:")
    print("  {:<35s}".format("Period"), end="")
    for label in all_results:
        print(" {:>16s}".format(label), end="")
    print()
    print("  " + "-" * (35 + 17 * len(all_results)))
    for i, pname in enumerate(period_names):
        print("  {:<35s}".format(pname), end="")
        for label in all_results:
            r = all_results[label][i]
            if "error" in r:
                print(" {:>16s}".format("N/A"), end="")
            else:
                print(" {:>+14.1f}%".format(r["total_return"]*100), end="")
        print()

    # MaxDD table
    print("\n  Max Drawdown by Period:")
    print("  {:<35s}".format("Period"), end="")
    for label in all_results:
        print(" {:>16s}".format(label), end="")
    print()
    print("  " + "-" * (35 + 17 * len(all_results)))
    for i, pname in enumerate(period_names):
        print("  {:<35s}".format(pname), end="")
        for label in all_results:
            r = all_results[label][i]
            if "error" in r:
                print(" {:>16s}".format("N/A"), end="")
            else:
                print(" {:>14.2f}%".format(r["max_drawdown"]*100), end="")
        print()

    # Overall stability scores
    print("\n" + "=" * 100)
    print("  OVERALL STABILITY SCORES")
    print("=" * 100)

    for label in all_results:
        valid = [r for r in all_results[label] if "error" not in r]
        if not valid:
            print("\n  {}: NO VALID DATA".format(label))
            continue

        sharpes = [r["sharpe"] for r in valid]
        returns = [r["total_return"] for r in valid]
        max_dds = [r["max_drawdown"] for r in valid]

        profitable = sum(1 for r in returns if r > 0)
        profitable_pct = profitable / len(valid)

        mean_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        cv_sharpe = std_sharpe / abs(mean_sharpe) if mean_sharpe != 0 else float("inf")

        consistency = (profitable_pct * 0.4 +
                       max(0, 1 - cv_sharpe * 0.3) * 0.3 +
                       max(0, 1 - max(max_dds) * 10) * 0.3)
        consistency = max(0, min(1, consistency))

        print("\n  {}:".format(label))
        print("    Profitable periods:     {}/{} ({:.0%})".format(profitable, len(valid), profitable_pct))
        print("    Mean Sharpe:            {:.2f}".format(mean_sharpe))
        print("    Sharpe std:             {:.2f}".format(std_sharpe))
        print("    Sharpe CV:              {:.2f}".format(cv_sharpe))
        print("    Worst MaxDD:            {:.2f}%".format(max(max_dds)*100))
        print("    Consistency score:      {:.2f}/1.00".format(consistency))
        if consistency >= 0.8:
            print("    => HIGHLY STABLE across all regimes")
        elif consistency >= 0.6:
            print("    => STABLE across most regimes")
        elif consistency >= 0.4:
            print("    => MODERATE stability")
        else:
            print("    => UNSTABLE -- regime-dependent")

    # Save
    output = {
        "period_names": period_names,
        "results": {label: [{"period": r.get("period", ""),
                             "sharpe": r.get("sharpe"),
                             "total_return": r.get("total_return"),
                             "max_drawdown": r.get("max_drawdown"),
                             "sortino": r.get("sortino"),
                             "error": r.get("error")}
                            for r in periods]
                    for label, periods in all_results.items()},
    }
    out_path = Path("data/results/time_period_stability.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print("\n  Total time: {:.0f}s".format(elapsed))
    print("  Results saved to: {}".format(out_path))
    print("=" * 100)


if __name__ == "__main__":
    main()
