#!/usr/bin/env python3
"""
Parameter Perturbation Sensitivity Testing for the 3 selected strategies.

Tests how robust each strategy is to ±30% changes in each parameter individually.
A robust strategy should maintain positive performance across nearby parameter values.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy
from trading_system.optimization.param_space import perturb_params


# The 3 selected strategies
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


def run_single_perturbation(engine, strategy, strat_name, df, base_params, override_params, pair, tf, funding):
    """Run a single perturbed parameter set and return key metrics."""
    params = base_params.copy()
    params.update(override_params)
    try:
        signals = strategy.generate_signals(df, params)
        result = engine.run(df, signals, strat_name, params, pair, tf, funding)
        return {
            "sharpe": result.sharpe,
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
            "trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "net_profit": result.net_profit,
            "params": params,
        }
    except Exception as e:
        return {"sharpe": 0, "total_return": 0, "max_drawdown": 0, "trades": 0,
                "win_rate": 0, "profit_factor": 0, "net_profit": 0, "params": params, "error": str(e)}


def test_single_param(engine, strategy, strat_name, df, base_params, param_name, pair, tf, funding, base_result):
    """Test sensitivity to a single parameter by sweeping through its perturbed values."""
    base_val = base_params.get(param_name)

    # Skip categorical params (bools, strings)
    if not isinstance(base_val, (int, float)):
        return None

    # Generate perturbed values for this single param
    delta = abs(base_val) * 0.3
    if isinstance(base_val, int):
        delta = max(1, int(delta))
        low = max(1, base_val - delta * 3)
        high = base_val + delta * 3
        values = list(range(low, high + 1, max(1, delta)))
        values = [v for v in values if v > 0]
    else:
        low = max(0.001, base_val - delta * 3)
        high = base_val + delta * 3
        values = np.linspace(low, high, 13).tolist()

    if len(values) < 3:
        return None

    results = []
    for val in values:
        p = base_params.copy()
        if isinstance(base_val, int):
            p[param_name] = int(round(val))
        else:
            p[param_name] = round(val, 4)

        try:
            r = run_single_perturbation(engine, strategy, strat_name, df, base_params, {param_name: p[param_name]}, pair, tf, funding)
            results.append({
                "value": p[param_name],
                "sharpe": r["sharpe"],
                "total_return": r["total_return"],
                "trades": r["trades"],
            })
        except Exception:
            results.append({"value": p[param_name], "sharpe": 0, "total_return": 0, "trades": 0})

    if not results:
        return None

    sharpes = [r["sharpe"] for r in results]
    returns = [r["total_return"] for r in results]

    profitable_count = sum(1 for s in sharpes if s > 0)
    base_sharpe = base_result["sharpe"]

    # Sensitivity: std of Sharpe across perturbations (lower = more robust)
    sharpe_std = float(np.std(sharpes))
    sharpe_range = float(max(sharpes) - min(sharpes))
    avg_sharpe = float(np.mean([s for s in sharpes if s > 0])) if any(s > 0 for s in sharpes) else 0

    return {
        "param_name": param_name,
        "base_value": base_val,
        "n_values_tested": len(values),
        "profitable_pct": profitable_count / len(results),
        "sharpe_mean": avg_sharpe,
        "sharpe_std": sharpe_std,
        "sharpe_range": sharpe_range,
        "sharpe_min": float(min(sharpes)),
        "sharpe_max": float(max(sharpes)),
        "return_mean": float(np.mean(returns)),
        "return_min": float(min(returns)),
        "return_max": float(max(returns)),
        "detail": results,
    }


def main():
    print("=" * 80)
    print("PARAMETER PERTURBATION SENSITIVITY TESTING")
    print("Testing: +/- 30% per parameter, 13 values each")
    print("=" * 80)

    cfg = SystemConfig.default()
    loader = DataLoader(cfg)
    engine = BacktestEngine(cfg.backtest)

    all_results = []

    for s in SELECTED:
        strat_name = s["strategy"]
        pair = s["pair"]
        tf = s["timeframe"]
        base_params = s["params"]
        family = s["family"]

        print(f"\n{'=' * 80}")
        print(f"  {strat_name} ({family})  |  {pair}  |  {tf}")
        print(f"  Base params: {json.dumps(base_params)}")
        print(f"{'=' * 80}")

        # Load data
        df = loader.load(pair, tf)
        if df is None or df.empty:
            print(f"  ERROR: No data for {pair} {tf}")
            continue

        funding = loader.load_funding_rates(pair)
        strategy = get_strategy(strat_name)

        # Get base result
        t0 = time.time()
        base_r = run_single_perturbation(engine, strategy, strat_name, df, base_params, {}, pair, tf, funding)
        print(f"  Base performance: Sharpe={base_r['sharpe']:.2f}  Return={base_r['total_return']*100:.1f}%  "
              f"Trades={base_r['trades']}  MaxDD={base_r['max_drawdown']*100:.1f}%")

        # Test each numeric parameter individually
        param_results = []
        for param_name in base_params:
            result = test_single_param(engine, strategy, strat_name, df, base_params, param_name, pair, tf, funding, base_r)
            if result is not None:
                param_results.append(result)

        # Print per-parameter sensitivity
        print(f"\n  Parameter Sensitivity (per-param sweep, +/-30%):")
        print(f"  {'Param':<20s} {'Base':>8s} {'Sharpe Avg':>10s} {'Sharpe Std':>10s} {'Range':>8s} {'Ret Avg':>8s} {'Ret Min':>8s} {'Prof%':>6s}  Sensitivity")
        print(f"  {'-' * 105}")

        for pr in param_results:
            # Classify sensitivity
            if pr["sharpe_std"] < 0.3:
                sensitivity = "LOW (robust)"
            elif pr["sharpe_std"] < 0.8:
                sensitivity = "MEDIUM"
            else:
                sensitivity = "HIGH (fragile)"

            base_display = f"{pr['base_value']}" if isinstance(pr['base_value'], int) else f"{pr['base_value']:.2f}"
            print(f"  {pr['param_name']:<20s} {base_display:>8s} "
                  f"{pr['sharpe_mean']:>10.2f} {pr['sharpe_std']:>10.3f} "
                  f"{pr['sharpe_range']:>8.2f} "
                  f"{pr['return_mean']*100:>7.1f}% {pr['return_min']*100:>7.1f}% "
                  f"{pr['profitable_pct']:>5.0%}  {sensitivity}")

            # Print the sweep detail
            vals_str = " ".join(
                f"{d['value']}->S{d['sharpe']:.1f}/R{d['total_return']*100:.0f}%"
                for d in pr["detail"]
            )
            print(f"    Sweep: {vals_str}")

        # Overall parameter stability
        if param_results:
            avg_std = float(np.mean([pr["sharpe_std"] for pr in param_results]))
            avg_profitable = float(np.mean([pr["profitable_pct"] for pr in param_results]))
            worst_param = max(param_results, key=lambda x: x["sharpe_std"])
            best_param = min(param_results, key=lambda x: x["sharpe_std"])

            print(f"\n  Overall Parameter Stability:")
            print(f"    Avg Sharpe std across params: {avg_std:.3f}")
            print(f"    Avg profitable rate: {avg_profitable:.0%}")
            print(f"    Most sensitive param: {worst_param['param_name']} (std={worst_param['sharpe_std']:.3f})")
            print(f"    Most stable param:     {best_param['param_name']} (std={best_param['sharpe_std']:.3f})")

            if avg_std < 0.3:
                overall = "HIGHLY ROBUST -- Performance stable across parameter changes"
            elif avg_std < 0.6:
                overall = "ROBUST -- Some sensitivity but generally stable"
            elif avg_std < 1.0:
                overall = "MODERATE -- Notable sensitivity in some parameters"
            else:
                overall = "FRAGILE -- Performance varies significantly with parameters"

            print(f"    Verdict: {overall}")
        else:
            avg_std = 0
            avg_profitable = 0
            overall = "N/A (no numeric params)"

        elapsed = time.time() - t0
        print(f"\n  Time: {elapsed:.1f}s")

        all_results.append({
            "strategy": strat_name,
            "pair": pair,
            "timeframe": tf,
            "family": family,
            "base_sharpe": base_r["sharpe"],
            "base_return": base_r["total_return"],
            "base_trades": base_r["trades"],
            "avg_sharpe_std": avg_std,
            "avg_profitable_pct": avg_profitable,
            "param_results": param_results,
            "overall_verdict": overall,
        })

    # ── Final Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY -- Parameter Perturbation Sensitivity")
    print(f"{'=' * 80}")
    print(f"\n  {'#':>3s}  {'Strategy':<20s} {'Family':<18s} {'Base Sh':>8s} {'Avg Std':>8s} {'Prof%':>6s}  Verdict")
    print(f"  {'-' * 90}")

    for i, r in enumerate(all_results):
        print(f"  {i+1:>3d}  {r['strategy']:<20s} {r['family']:<18s} "
              f"{r['base_sharpe']:>8.2f} {r['avg_sharpe_std']:>8.3f} "
              f"{r['avg_profitable_pct']:>5.0%}  {r['overall_verdict']}")

    all_robust = all(r["avg_sharpe_std"] < 1.0 for r in all_results)
    print(f"\n  Overall: {'ALL 3 STRATEGIES ARE PARAMETER-ROBUST' if all_robust else 'WARNING: Some strategies show parameter fragility'}")
    print(f"{'=' * 80}")

    # Save results
    out_path = Path("data/results/param_sensitivity_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = []
    for r in all_results:
        save_entry = {
            "strategy": r["strategy"],
            "pair": r["pair"],
            "timeframe": r["timeframe"],
            "family": r["family"],
            "base_sharpe": r["base_sharpe"],
            "base_return": r["base_return"],
            "avg_sharpe_std": r["avg_sharpe_std"],
            "avg_profitable_pct": r["avg_profitable_pct"],
            "overall_verdict": r["overall_verdict"],
            "param_sensitivity": [],
        }
        for pr in r["param_results"]:
            save_entry["param_sensitivity"].append({
                "param": pr["param_name"],
                "base": pr["base_value"],
                "sharpe_std": pr["sharpe_std"],
                "sharpe_range": pr["sharpe_range"],
                "profitable_pct": pr["profitable_pct"],
                "return_min": pr["return_min"],
                "return_max": pr["return_max"],
            })
        save_data.append(save_entry)
    out_path.write_text(json.dumps(save_data, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
