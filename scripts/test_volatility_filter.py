#!/usr/bin/env python3
"""
Test Volatility Filter: Compare different ATR percentile thresholds.

Tests whether filtering out low-volatility candles improves strategy
performance by avoiding noisy signals where fees eat profits.
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
from trading_system.bot.volatility_filter import VolatilityFilter, VolatilityFilterConfig


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


def run_backtest_with_filter(engine, df, strategy_name, params, pair, tf, vol_filter=None):
    """Run backtest, optionally filtering signals through volatility gate."""
    strategy = get_strategy(strategy_name)
    signals = strategy.generate_signals(df, params)

    if vol_filter is not None:
        # Zero out signals during low-vol periods
        allow = vol_filter.compute_filter(df)
        signals = signals.copy()
        signals[~allow] = 0  # Set filtered candles to flat

    result = engine.run(df, signals, strategy_name, params, pair, tf)
    return {
        "return": result.total_return * 100,
        "sharpe": result.sharpe,
        "sortino": result.sortino,
        "max_dd": result.max_drawdown * 100,
        "trades": result.total_trades,
        "win_rate": result.win_rate * 100,
        "profit_factor": result.profit_factor,
        "avg_holding": result.avg_holding_period,
    }


def main():
    t0 = time.time()
    cfg = SystemConfig.default()
    loader = DataLoader(cfg)

    # Filter thresholds to test (percentile of ATR to allow trading)
    filter_configs = {
        "No filter (baseline)": None,
        "ATR > 5th pctl": VolatilityFilterConfig(min_percentile=5.0),
        "ATR > 10th pctl": VolatilityFilterConfig(min_percentile=10.0),
        "ATR > 15th pctl": VolatilityFilterConfig(min_percentile=15.0),
        "ATR > 20th pctl": VolatilityFilterConfig(min_percentile=20.0),
        "ATR > 25th pctl": VolatilityFilterConfig(min_percentile=25.0),
        "ATR > 0.3% abs": VolatilityFilterConfig(min_percentile=0, min_atr_pct=0.3),
        "ATR > 0.5% abs": VolatilityFilterConfig(min_percentile=0, min_atr_pct=0.5),
        "ATR > 0.8% abs": VolatilityFilterConfig(min_percentile=0, min_atr_pct=0.8),
    }

    print("=" * 115)
    print("  VOLATILITY FILTER: Low-Vol Period Avoidance Test")
    print("=" * 115)

    # First: analyze ATR distribution for context
    print("\n  ATR Distribution (as % of price):")
    for pair_name, tf in [("ETH/USDT:USDT", "4h"), ("BTC/USDT:USDT", "4h")]:
        df = loader.load(pair_name, tf)
        from trading_system.indicators import atr as atr_func
        atr_series = atr_func(df, period=14)
        atr_pct = (atr_series / df["close"] * 100).dropna()
        print(f"    {pair_name} {tf}: mean={atr_pct.mean():.3f}%  "
              f"median={atr_pct.median():.3f}%  "
              f"p5={atr_pct.quantile(0.05):.3f}%  "
              f"p10={atr_pct.quantile(0.10):.3f}%  "
              f"p25={atr_pct.quantile(0.25):.3f}%  "
              f"p50={atr_pct.quantile(0.50):.3f}%  "
              f"p90={atr_pct.quantile(0.90):.3f}%")

    all_results = []

    for strat_def in STRATEGIES:
        print(f"\n{'=' * 115}")
        print(f"  {strat_def['label']} ({strat_def['name']} on {strat_def['pair']})")
        print(f"{'=' * 115}")

        strategy = get_strategy(strat_def["name"])
        df = loader.load(strat_def["pair"], strat_def["timeframe"])

        for config_name, filter_config in filter_configs.items():
            if filter_config is None:
                vf = None
            else:
                vf = VolatilityFilter(filter_config)

            bt_config = SystemConfig.default()
            bt_config.backtest.execution.stop_loss_atr_mult = 3.0
            engine = BacktestEngine(bt_config.backtest)

            result = run_backtest_with_filter(
                engine, df, strat_def["name"], strat_def["params"],
                strat_def["pair"], strat_def["timeframe"], vf
            )

            # Get filter stats if applicable
            if vf is not None:
                stats = vf.get_filter_stats(df)
                filtered_info = f"  (filtered {stats['filtered_pct']:.1f}% candles)"
            else:
                stats = None
                filtered_info = ""

            print(f"\n  {config_name}{filtered_info}:")
            print(f"    Ret={result['return']:+7.1f}%  "
                  f"Sharpe={result['sharpe']:6.2f}  "
                  f"MaxDD={result['max_dd']:5.2f}%  "
                  f"Sortino={result['sortino']:6.2f}  "
                  f"Trades={result['trades']:>5d}  "
                  f"WR={result['win_rate']:5.1f}%  "
                  f"PF={result['profit_factor']:5.2f}")

            all_results.append({
                "strategy": strat_def["label"],
                "config": config_name,
                "filtered_pct": stats["filtered_pct"] if stats else 0,
                **result,
            })

    # ── Summary: Best config per strategy ──────────────────────────
    print(f"\n\n{'=' * 115}")
    print("  BEST CONFIGURATION PER STRATEGY (by Sharpe)")
    print(f"{'=' * 115}")

    for strat_def in STRATEGIES:
        label = strat_def["label"]
        strat_results = [r for r in all_results if r["strategy"] == label]
        best = max(strat_results, key=lambda x: x["sharpe"])
        baseline = next(r for r in strat_results if r["config"] == "No filter (baseline)")

        sharpe_diff = best["sharpe"] - baseline["sharpe"]
        ret_diff = best["return"] - baseline["return"]
        dd_diff = best["max_dd"] - baseline["max_dd"]

        print(f"\n  {label}:")
        print(f"    Baseline:  Sharpe={baseline['sharpe']:.2f}  "
              f"Return={baseline['return']:+.1f}%  MaxDD={baseline['max_dd']:.2f}%")
        print(f"    Best:      {best['config']}  "
              f"Sharpe={best['sharpe']:.2f} ({sharpe_diff:+.2f})  "
              f"Return={best['return']:+.1f}% ({ret_diff:+.1f}%)  "
              f"MaxDD={best['max_dd']:.2f}% ({dd_diff:+.2f}%)")
        if best["filtered_pct"] > 0:
            print(f"    Filtered:  {best['filtered_pct']:.1f}% of candles excluded")

    # ── Impact analysis ────────────────────────────────────────────
    print(f"\n\n{'=' * 115}")
    print("  FULL COMPARISON TABLE")
    print(f"{'=' * 115}")

    print(f"\n  {'Strategy':<25} {'Config':<22} {'Filt%':>6} {'Return':>8} {'Sharpe':>8} "
          f"{'MaxDD':>8} {'Sortino':>8} {'Trades':>7} {'WR':>6} {'PF':>6}")
    print(f"  {'-' * 108}")

    for r in all_results:
        print(f"  {r['strategy']:<25} {r['config']:<22} {r['filtered_pct']:>5.1f}% "
              f"{r['return']:>+7.1f}% {r['sharpe']:>8.2f} {r['max_dd']:>7.2f}% "
              f"{r['sortino']:>8.2f} {r['trades']:>7} {r['win_rate']:>5.1f}% "
              f"{r['profit_factor']:>6.2f}")

    # Save
    out_path = Path("data/results/volatility_filter_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()
