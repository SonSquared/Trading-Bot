#!/usr/bin/env python3
"""
Portfolio Allocation Model.

Combines the 3 selected strategies:
  1. MACD ETH/USDT 4h (fast=4, slow=10, signal=2, no histogram)
  2. ROC_Momentum ETH/USDT 4h (roc_period=3, threshold=-1, smooth=1, no trend)
  3. MACD BTC/USDT 4h (fast=4, slow=10, signal=2, no histogram)

Allocation methods: equal-weight, risk-parity, Sharpe-weighted.
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
from trading_system.ranking.correlation import StrategyCorrelationAnalyzer

# ── Strategy definitions ──────────────────────────────────────────

STRATEGIES = [
    {
        "name": "MACD",
        "label": "MACD_ETH_4h",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
    },
    {
        "name": "ROC_Momentum",
        "label": "ROC_ETH_4h",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {"roc_period": 3, "roc_threshold": -1, "smooth_period": 1,
                   "trend_ema": 50, "trend_filter": False},
    },
    {
        "name": "MACD",
        "label": "MACD_BTC_4h",
        "pair": "BTC/USDT:USDT",
        "timeframe": "4h",
        "params": {"fast": 4, "slow": 10, "signal": 2, "use_histogram": False},
    },
]

RISK_FREE_RATE = 0.04
TRADING_HOURS_PER_YEAR = 365 * 24


def calc_portfolio_metrics(equity: pd.Series, name: str) -> dict:
    """Calculate metrics from a combined portfolio equity curve."""
    if equity.empty or len(equity) < 2:
        return {"name": name, "error": "insufficient data"}

    eq = equity.values
    total_return = (eq[-1] / eq[0]) - 1

    n_hours = (equity.index[-1] - equity.index[0]).total_seconds() / 3600
    n_years = n_hours / TRADING_HOURS_PER_YEAR
    cagr = (eq[-1] / eq[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

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

    return {
        "name": name,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "volatility": vol,
        "final_equity": float(eq[-1]),
    }


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    bt_config = cfg.backtest
    engine = BacktestEngine(bt_config)

    # ── Phase 1: Run individual backtests ─────────────────────────
    print("=" * 80)
    print("  PHASE 1: Individual Strategy Backtests")
    print("=" * 80)

    equity_curves = {}
    individual_results = {}
    strategy_data = {}  # label -> (pair, tf, df, funding)

    for strat_def in STRATEGIES:
        strat_name = strat_def["name"]
        label = strat_def["label"]
        pair = strat_def["pair"]
        tf = strat_def["timeframe"]
        params = strat_def["params"]

        print(f"\n  {label}: {strat_name} on {pair} {tf}")
        strategy = get_strategy(strat_name)
        df = loader.load(pair, tf)
        funding = loader.load_funding_rates(pair)

        signals = strategy.generate_signals(df, params)
        result = engine.run(df, signals, strat_name, params, pair, tf, funding)

        equity_curves[label] = result.equity_curve
        individual_results[label] = result
        strategy_data[label] = (pair, tf, df, funding)

        print(f"    Return: {result.total_return*100:.1f}%  Sharpe: {result.sharpe:.2f}  "
              f"MaxDD: {result.max_drawdown*100:.2f}%  Trades: {result.total_trades}  "
              f"WR: {result.win_rate:.0%}  PF: {result.profit_factor:.2f}")

    # ── Phase 2: Correlation analysis ─────────────────────────────
    print(f"\n{'=' * 80}")
    print("  PHASE 2: Strategy Correlation Analysis")
    print("=" * 80)

    corr_analyzer = StrategyCorrelationAnalyzer()
    corr_matrix = corr_analyzer.calculate_returns_matrix(equity_curves)
    corr_summary = corr_analyzer.get_correlation_summary(equity_curves)

    print(f"\n  Correlation Matrix:")
    labels = list(equity_curves.keys())
    print(f"  {'':>18s}", end="")
    for lbl in labels:
        print(f" {lbl:>16s}", end="")
    print()
    for i, lbl_i in enumerate(labels):
        print(f"  {lbl_i:>18s}", end="")
        for j, lbl_j in enumerate(labels):
            val = corr_matrix.iloc[i, j]
            print(f" {val:>16.3f}", end="")
        print()

    print(f"\n  Avg pairwise correlation: {corr_summary['avg_correlation']:.3f}")
    print(f"  Max pairwise correlation: {corr_summary['max_correlation']:.3f}")
    for pair_info in corr_summary["pairs"]:
        print(f"    {pair_info['strategy_a']} <-> {pair_info['strategy_b']}: {pair_info['correlation']:.3f}")

    if corr_summary['avg_correlation'] < 0.5:
        print(f"  => LOW correlation — good diversification")
    elif corr_summary['avg_correlation'] < 0.7:
        print(f"  => MODERATE correlation — acceptable diversification")
    else:
        print(f"  => HIGH correlation — limited diversification benefit")

    # ── Phase 3: Portfolio allocation ─────────────────────────────
    print(f"\n{'=' * 80}")
    print("  PHASE 3: Portfolio Allocation Methods")
    print("=" * 80)

    # Align all equity curves to common index
    combined = pd.DataFrame(equity_curves)
    combined = combined.dropna()

    # Normalize each equity to start at initial_capital
    initial = cfg.backtest.execution.initial_capital
    for col in combined.columns:
        combined[col] = combined[col] / combined[col].iloc[0] * initial

    print(f"\n  Aligned data: {len(combined)} candles ({combined.index[0]} to {combined.index[-1]})")
    print(f"  Initial capital: ${initial:,.0f}")

    # Calculate allocations
    allocation_methods = {
        "Equal Weight": equal_weight(len(STRATEGIES)),
        "Risk Parity": risk_parity(equity_curves, lookback=200),
        "Sharpe Weighted": sharpe_weight(equity_curves, lookback=200),
    }

    portfolio_results = {}

    for method_name, weights in allocation_methods.items():
        print(f"\n  --- {method_name} ---")
        print(f"  Weights: ", end="")
        for i, lbl in enumerate(labels):
            print(f"{lbl}={weights[i]:.1%}  ", end="")
        print()

        # Build combined equity: weighted sum of returns
        returns_df = combined.pct_change().fillna(0)
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Convert returns back to equity curve
        portfolio_equity = initial * (1 + portfolio_returns).cumprod()
        portfolio_equity.name = method_name

        metrics = calc_portfolio_metrics(portfolio_equity, method_name)
        portfolio_results[method_name] = metrics

        print(f"    Return: {metrics['total_return']*100:.1f}%  CAGR: {metrics['cagr']*100:.1f}%  "
              f"Sharpe: {metrics['sharpe']:.2f}  Sortino: {metrics['sortino']:.2f}  "
              f"MaxDD: {metrics['max_drawdown']*100:.2f}%  Vol: {metrics['volatility']*100:.1f}%")
        print(f"    Final Equity: ${metrics['final_equity']:,.0f}")

    # ── Phase 4: Comparison summary ───────────────────────────────
    print(f"\n{'=' * 80}")
    print("  PHASE 4: Individual vs Portfolio Comparison")
    print("=" * 80)

    all_results = {}
    for lbl in labels:
        r = individual_results[lbl]
        all_results[lbl] = {
            "name": lbl,
            "total_return": r.total_return,
            "cagr": r.cagr,
            "sharpe": r.sharpe,
            "sortino": r.sortino,
            "calmar": r.calmar,
            "max_drawdown": r.max_drawdown,
            "volatility": getattr(r, 'volatility', 0),
        }
    for name, m in portfolio_results.items():
        all_results[name] = m

    # Print table
    header = f"  {'Strategy':<22s} {'Return':>8s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'Vol':>8s}"
    print(f"\n{header}")
    print(f"  {'-' * 70}")

    for lbl in labels:
        m = all_results[lbl]
        print(f"  {lbl:<22s} {m['total_return']*100:>7.1f}% {m['cagr']*100:>7.1f}% "
              f"{m['sharpe']:>8.2f} {m['sortino']:>8.2f} {m['max_drawdown']*100:>7.2f}% "
              f"{m.get('volatility',0)*100:>7.1f}%")

    print(f"  {'-' * 70}")
    for name in portfolio_results:
        m = all_results[name]
        print(f"  {name:<22s} {m['total_return']*100:>7.1f}% {m['cagr']*100:>7.1f}% "
              f"{m['sharpe']:>8.2f} {m['sortino']:>8.2f} {m['max_drawdown']*100:>7.2f}% "
              f"{m.get('volatility',0)*100:>7.1f}%")

    # ── Phase 5: Diversification benefit ──────────────────────────
    print(f"\n{'=' * 80}")
    print("  PHASE 5: Diversification Benefit Analysis")
    print("=" * 80)

    best_individual = max(
        [(lbl, all_results[lbl]) for lbl in labels],
        key=lambda x: x[1]["sharpe"]
    )
    best_portfolio = max(
        [(name, all_results[name]) for name in portfolio_results],
        key=lambda x: x[1]["sharpe"]
    )

    print(f"\n  Best individual: {best_individual[0]} (Sharpe={best_individual[1]['sharpe']:.2f}, MaxDD={best_individual[1]['max_drawdown']*100:.2f}%)")
    print(f"  Best portfolio:  {best_portfolio[0]} (Sharpe={best_portfolio[1]['sharpe']:.2f}, MaxDD={best_portfolio[1]['max_drawdown']*100:.2f}%)")

    sharpe_benefit = best_portfolio[1]["sharpe"] - best_individual[1]["sharpe"]
    dd_benefit = best_individual[1]["max_drawdown"] - best_portfolio[1]["max_drawdown"]

    print(f"\n  Sharpe improvement:  {sharpe_benefit:+.2f}")
    print(f"  MaxDD reduction:     {dd_benefit*100:+.2f}%")

    if sharpe_benefit > 0:
        print(f"  => Portfolio ADDS value (higher Sharpe)")
    else:
        print(f"  => Portfolio REDUCES Sharpe (concentration wins)")

    if dd_benefit > 0:
        print(f"  => Portfolio REDUCES drawdown (smoother equity)")
    else:
        print(f"  => Portfolio has HIGHER drawdown")

    # ── Recommendation ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  RECOMMENDATION")
    print("=" * 80)

    # Pick the best portfolio method
    best_method_name = best_portfolio[0]
    best_method_metrics = best_portfolio[1]
    best_weights = allocation_methods[best_method_name]

    print(f"\n  Recommended allocation method: {best_method_name}")
    print(f"  Weights:")
    for i, lbl in enumerate(labels):
        print(f"    {lbl}: {best_weights[i]:.1%}")
    print(f"\n  Portfolio metrics:")
    print(f"    Total Return: {best_method_metrics['total_return']*100:.1f}%")
    print(f"    CAGR:         {best_method_metrics['cagr']*100:.1f}%")
    print(f"    Sharpe:       {best_method_metrics['sharpe']:.2f}")
    print(f"    Sortino:      {best_method_metrics['sortino']:.2f}")
    print(f"    Max Drawdown: {best_method_metrics['max_drawdown']*100:.2f}%")

    # Save results
    output = {
        "strategies": STRATEGIES,
        "individual_metrics": {lbl: {k: v for k, v in m.items() if k != "name"}
                               for lbl, m in all_results.items() if lbl in labels},
        "portfolio_metrics": {name: {k: v for k, v in m.items() if k != "name"}
                              for name, m in portfolio_results.items()},
        "correlation_matrix": corr_matrix.to_dict(),
        "correlation_summary": corr_summary,
        "allocation_weights": {name: w.tolist() for name, w in allocation_methods.items()},
        "recommendation": {
            "method": best_method_name,
            "weights": {labels[i]: float(best_weights[i]) for i in range(len(labels))},
            "metrics": {k: v for k, v in best_method_metrics.items() if k != "name"},
        },
    }

    out_path = Path("data/results/portfolio_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
