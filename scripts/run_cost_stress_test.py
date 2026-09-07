#!/usr/bin/env python3
"""
Cost Stress Testing for the 3 selected strategies.

Tests strategy robustness under increased transaction costs:
1x (baseline), 1.5x, 2x, 3x fee multipliers.

A robust strategy should remain profitable even at 2-3x normal fees.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.validation.robustness import RobustnessAnalyzer


# The 3 selected strategies from full optimization + WF + MC validation
SELECTED = [
    {
        "strategy": "ROC_Momentum",
        "pair": "ETH/USDT:USDT",
        "timeframe": "1h",
        "params": {"roc_period": 6, "roc_threshold": -1, "smooth_period": 1, "trend_filter": False},
        "family": "Momentum",
    },
    {
        "strategy": "MACD",
        "pair": "ETH/USDT:USDT",
        "timeframe": "1h",
        "params": {"fast": 8, "slow": 24, "signal": 5, "use_histogram": False},
        "family": "Trend Following",
    },
    {
        "strategy": "BB_Squeeze",
        "pair": "ETH/USDT:USDT",
        "timeframe": "4h",
        "params": {
            "bb_period": 15, "bb_std": 2.0,
            "kc_period": 15, "kc_atr_mult": 1.0,
            "squeeze_lookback": 6, "momentum_period": 8,
        },
        "family": "Mean Reversion",
    },
]


def main():
    print("=" * 80)
    print("COST STRESS TESTING — 3 Selected Strategies")
    print("Fee multipliers: 1.0x (baseline), 1.5x, 2.0x, 3.0x")
    print("=" * 80)

    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    robustness = RobustnessAnalyzer(cfg.backtest)

    cost_multipliers = [1.0, 1.5, 2.0, 3.0]
    all_results = []

    for s in SELECTED:
        strat_name = s["strategy"]
        pair = s["pair"]
        tf = s["timeframe"]
        params = s["params"]
        family = s["family"]

        print(f"\n{'-' * 80}")
        print(f"  {strat_name} ({family})  |  {pair}  |  {tf}")
        print(f"  Params: {json.dumps(params)}")
        print(f"{'-' * 80}")

        # Load data
        df = loader.load(pair, tf)
        if df is None or df.empty:
            print(f"  ERROR: No data for {pair} {tf}")
            continue

        funding = loader.load_funding_rates(pair)
        print(f"  Data: {len(df):,} candles, {df.index[0]} to {df.index[-1]}")

        # Run cost stress test
        t0 = time.time()
        result = robustness.test_cost_robustness(
            df=df,
            strategy_name=strat_name,
            params=params,
            pair=pair,
            timeframe=tf,
            cost_multipliers=cost_multipliers,
            funding_rates=funding,
        )
        elapsed = time.time() - t0

        # Print detailed results
        print(f"\n  {'Cost Mult':>10s}  {'Sharpe':>8s}  {'Return':>10s}  {'Net Profit':>12s}  {'Fees':>10s}  {'Profitable':>10s}")
        print(f"  {'-' * 65}")

        for cr in result["results"]:
            mult = cr["cost_multiplier"]
            sharpe = cr["sharpe"]
            ret = cr["total_return"]
            net = cr["net_profit"]
            fees = cr["total_fees"]
            profitable = cr["profitable"]

            ret_str = f"{ret * 100:+.2f}%"
            profit_str = f"${net:+,.2f}"
            fee_str = f"${fees:,.2f}"
            prof_str = "  YES" if profitable else "  NO"

            # Flag problematic values
            marker = ""
            if mult > 1.0 and profitable:
                marker = " <-- still profitable"
            elif mult > 1.0 and not profitable:
                marker = " <-- NOT profitable"

            print(f"  {mult:>8.1f}x  {sharpe:>8.2f}  {ret_str:>10s}  {profit_str:>12s}  {fee_str:>10s}  {prof_str}{marker}")

        # Summary
        base_sharpe = result["results"][0]["sharpe"] if result["results"] else 0
        worst_sharpe = result["results"][-1]["sharpe"] if result["results"] else 0
        base_ret = result["results"][0]["total_return"] if result["results"] else 0
        worst_ret = result["results"][-1]["total_return"] if result["results"] else 0

        print("\n  Summary:")
        print(f"    Cost robustness score:    {result['cost_robustness_score']:.0%}")
        print(f"    Profitable at all levels: {'YES' if result['profitable_at_all_levels'] else 'NO'}")
        print(f"    Profitable at 1.5x fees:  {'YES' if result['profitable_at_1_5x'] else 'NO'}")
        print(f"    Sharpe degradation:        {result['sharpe_degradation']:.1%}")
        print(f"    Return degradation:        {abs((base_ret - worst_ret) / base_ret) * 100:.1f}% absolute" if base_ret != 0 else "    Return degradation: N/A (0% base return)")
        print(f"    Time: {elapsed:.1f}s")

        # Verdict
        if result["cost_robustness_score"] >= 0.75:
            verdict = "PASS — Highly robust to cost increases"
        elif result["cost_robustness_score"] >= 0.50:
            verdict = "PASS — Moderately robust, but monitor at 3x"
        else:
            verdict = "FAIL — Not robust to increased costs"

        print(f"    Verdict: {verdict}")

        all_results.append({
            "strategy": strat_name,
            "pair": pair,
            "timeframe": tf,
            "family": family,
            "result": result,
            "verdict": verdict,
        })

    # ── Final Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY — Cost Stress Test Results")
    print(f"{'=' * 80}")
    print(f"\n  {'#':>3s}  {'Strategy':<20s} {'Family':<18s} {'Score':>6s} {'All Levels':>11s} {'1.5x':>6s} {'Degrad':>8s}  Verdict")
    print(f"  {'-' * 95}")

    for i, r in enumerate(all_results):
        res = r["result"]
        print(f"  {i+1:>3d}  {r['strategy']:<20s} {r['family']:<18s} "
              f"{res['cost_robustness_score']:>5.0%} "
              f"{'YES' if res['profitable_at_all_levels'] else 'NO':>11s} "
              f"{'YES' if res['profitable_at_1_5x'] else 'NO':>6s} "
              f"{res['sharpe_degradation']:>7.1%}  {r['verdict']}")

    # Check if all pass
    all_pass = all(r["result"]["cost_robustness_score"] >= 0.50 for r in all_results)
    print(f"\n  Overall: {'ALL 3 STRATEGIES PASS cost stress testing' if all_pass else 'WARNING: Some strategies failed cost stress testing'}")
    print(f"\n{'=' * 80}")

    # Save results to JSON for downstream use
    out_path = Path("data/results/cost_stress_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = []
    for r in all_results:
        save_data.append({
            "strategy": r["strategy"],
            "pair": r["pair"],
            "timeframe": r["timeframe"],
            "family": r["family"],
            "cost_robustness_score": r["result"]["cost_robustness_score"],
            "profitable_at_all_levels": r["result"]["profitable_at_all_levels"],
            "profitable_at_1_5x": r["result"]["profitable_at_1_5x"],
            "sharpe_degradation": r["result"]["sharpe_degradation"],
            "results": r["result"]["results"],
            "verdict": r["verdict"],
        })
    out_path.write_text(json.dumps(save_data, indent=2, default=str), encoding="utf-8")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
