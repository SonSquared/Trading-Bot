#!/usr/bin/env python3
"""
Test Dynamic Risk Manager: Compare regime-aware position sizing vs fixed sizing.

Runs the same strategies with:
  1. Fixed risk_per_trade = 2% (current default)
  2. Regime-adaptive risk_per_trade (0.6% choppy, 1.4% transitional, 2% trending)

Shows whether reducing position size during choppy markets improves risk-adjusted returns.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy
from trading_system.bot.dynamic_risk import DynamicRiskManager, RegimeRiskConfig


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
        "label": "ROC_Momentum ETH 4h",
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


def run_backtest(engine, df, strategy_name, params, pair, tf, risk_mults=None):
    """Run a single backtest and return results dict."""
    strategy = get_strategy(strategy_name)
    signals = strategy.generate_signals(df, params)
    result = engine.run(df, signals, strategy_name, params, pair, tf,
                        risk_multipliers=risk_mults)
    return {
        "return": result.total_return * 100,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "max_dd": result.max_drawdown * 100,
        "trades": result.total_trades,
        "win_rate": result.win_rate * 100,
        "profit_factor": result.profit_factor,
        "avg_holding": result.avg_holding_period,
        "calmar": result.calmar,
    }


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)

    # Create DynamicRiskManager with different configs
    configs = {
        "Fixed 2% (baseline)": None,  # No regime scaling

        "Conservative (0.6/1.4/2.0)": RegimeRiskConfig(
            trending_mult=1.0,      # 2% in trends
            transitional_mult=0.7,  # 1.4% in transitions
            choppy_mult=0.3,        # 0.6% in choppy
        ),

        "Aggressive reduction (0.2/0.6/1.0)": RegimeRiskConfig(
            trending_mult=1.0,      # 2% in trends
            transitional_mult=0.5,  # 1.0% in transitions
            choppy_mult=0.1,        # 0.2% in choppy
        ),

        "Moderate (0.5/0.8/1.0)": RegimeRiskConfig(
            trending_mult=1.0,      # 2% in trends
            transitional_mult=0.8,  # 1.6% in transitions
            choppy_mult=0.5,        # 1.0% in choppy
        ),
    }

    print("=" * 110)
    print("  DYNAMIC RISK MANAGER: Regime-Aware Position Sizing Test")
    print("=" * 110)

    all_results = []

    for strat_def in STRATEGIES:
        print(f"\n{'=' * 110}")
        print(f"  {strat_def['label']} ({strat_def['name']} on {strat_def['pair']} {strat_def['timeframe']})")
        print(f"{'=' * 110}")

        strategy = get_strategy(strat_def["name"])
        df = loader.load(strat_def["pair"], strat_def["timeframe"])

        for config_name, risk_config in configs.items():
            if risk_config is None:
                # Baseline: no regime scaling
                bt_config = SystemConfig.default()
                bt_config.backtest.execution.stop_loss_atr_mult = 3.0
                engine = BacktestEngine(bt_config.backtest)
                result = run_backtest(engine, df, strat_def["name"], strat_def["params"],
                                      strat_def["pair"], strat_def["timeframe"])
                risk_mults = None
            else:
                # Regime-adaptive: compute multipliers first, then run backtest
                drm = DynamicRiskManager(risk_config)
                risk_mults = drm.compute_multipliers(df)

                bt_config = SystemConfig.default()
                bt_config.backtest.execution.stop_loss_atr_mult = 3.0
                engine = BacktestEngine(bt_config.backtest)
                result = run_backtest(engine, df, strat_def["name"], strat_def["params"],
                                      strat_def["pair"], strat_def["timeframe"],
                                      risk_mults=risk_mults)

            print(f"\n  {config_name}:")
            print(f"    Return:  {result['return']:+7.1f}%   "
                  f"Sharpe: {result['sharpe']:6.2f}   "
                  f"Sortino: {result['sortino']:6.2f}   "
                  f"MaxDD: {result['max_dd']:5.2f}%")
            print(f"    Trades:  {result['trades']:>5d}   "
                  f"WR: {result['win_rate']:5.1f}%    "
                  f"PF: {result['profit_factor']:5.2f}   "
                  f"Avg Hold: {result['avg_holding']:5.1f} candles")

            all_results.append({
                "strategy": strat_def["label"],
                "config": config_name,
                **result,
            })

    # ── Regime Distribution Analysis ───────────────────────────────
    print(f"\n\n{'=' * 110}")
    print("  REGIME DISTRIBUTION (ETH 4h, 2022-2026)")
    print(f"{'=' * 110}")

    eth_4h = loader.load("ETH/USDT:USDT", "4h")
    drm = DynamicRiskManager(RegimeRiskConfig())
    summary = drm.get_regime_summary(eth_4h)

    print(f"\n  {'Regime':<15} {'Count':>8} {'% Time':>8} {'Avg Mult':>10}")
    print(f"  {'-' * 45}")
    for regime in ["trending", "transitional", "choppy"]:
        s = summary[regime]
        print(f"  {regime.capitalize():<15} {s['count']:>8} {s['pct']:>7.1f}% "
              f"{s['avg_multiplier']:>10.3f}")
    o = summary["overall"]
    print(f"  {'-' * 45}")
    print(f"  {'Overall':<15} {'':>8} {'':>8} {o['mean_multiplier']:>10.3f}")

    # ── Summary Comparison Table ───────────────────────────────────
    print(f"\n\n{'=' * 110}")
    print("  SUMMARY: Dynamic Risk vs Fixed Risk")
    print(f"{'=' * 110}")

    print(f"\n  {'Strategy':<25} {'Config':<35} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Sortino':>8}")
    print(f"  {'-' * 95}")
    for r in all_results:
        print(f"  {r['strategy']:<25} {r['config']:<35} "
              f"{r['return']:>+7.1f}% {r['sharpe']:>8.2f} {r['max_dd']:>7.2f}% {r['sortino']:>8.2f}")

    # ── Improvement analysis ───────────────────────────────────────
    print(f"\n\n{'=' * 110}")
    print("  IMPROVEMENT vs BASELINE (Fixed 2%)")
    print(f"{'=' * 110}")

    for strat_def in STRATEGIES:
        label = strat_def["label"]
        baseline = next((r for r in all_results
                         if r["strategy"] == label and r["config"] == "Fixed 2% (baseline)"), None)
        if not baseline:
            continue

        print(f"\n  {label}:")
        for r in all_results:
            if r["strategy"] != label or r["config"] == "Fixed 2% (baseline)":
                continue
            ret_diff = r["return"] - baseline["return"]
            sharpe_diff = r["sharpe"] - baseline["sharpe"]
            dd_diff = r["max_dd"] - baseline["max_dd"]

            verdict = "BETTER" if r["sharpe"] > baseline["sharpe"] else "WORSE"
            print(f"    {r['config']:<35} "
                  f"Sharpe {sharpe_diff:+.2f}  MaxDD {dd_diff:+.2f}%  "
                  f"Return {ret_diff:+.1f}%  [{verdict}]")

    # Save results
    out_path = Path("data/results/dynamic_risk_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "regime_summary": summary}, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print("=" * 110)


if __name__ == "__main__":
    main()
