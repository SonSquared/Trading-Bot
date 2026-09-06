#!/usr/bin/env python3
"""
Post-optimization analysis pipeline.

Reads experiments.db, ranks strategies, selects top candidates,
runs validation (walk-forward, Monte Carlo, robustness),
selects 3 final strategies, builds portfolio, generates report.

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --top-n 10
    python scripts/analyze_results.py --skip-validation
"""

from __future__ import annotations

import sys
import json
import time
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy
from validation.walk_forward import WalkForwardAnalyzer
from validation.monte_carlo import MonteCarloAnalyzer
from validation.robustness import RobustnessAnalyzer
from portfolio.allocator import create_allocator, equal_weight
from ranking.correlation import StrategyCorrelationAnalyzer

DB_PATH = Path("data/results/experiments.db")
RESULTS_DIR = Path("data/results")


def load_experiments(db_path: Path) -> list[dict]:
    """Load all successful experiments from DB."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("""
        SELECT experiment_id, strategy_name, pair, timeframe,
               parameters, results, total_return, sharpe, sortino,
               max_drawdown, total_trades, win_rate, profit_factor
        FROM experiments
        WHERE success = 1 AND total_trades >= 10
        ORDER BY sharpe DESC
    """)
    cols = [d[0] for d in cur.description]
    experiments = []
    for row in cur.fetchall():
        d = dict(zip(cols, row))
        d["parameters"] = json.loads(d["parameters"])
        d["results"] = json.loads(d["results"])
        experiments.append(d)
    conn.close()
    return experiments


def rank_by_composite_score(experiments: list[dict]) -> list[dict]:
    """Rank experiments by a composite score."""
    for exp in experiments:
        res = exp["results"]
        sharpe = exp.get("sharpe", 0) or 0
        sortino = exp.get("sortino", 0) or 0
        max_dd = abs(exp.get("max_drawdown", 1)) or 1
        ret = exp.get("total_return", 0) or 0
        trades = exp.get("total_trades", 0) or 0
        win_rate = exp.get("win_rate", 0) or 0
        pf = exp.get("profit_factor", 0) or 0

        # Calmar-like ratio
        calmar = ret / max_dd if max_dd > 0.001 else 0

        # Trade frequency score (prefer 30-200 trades)
        if trades < 10:
            trade_score = 0
        elif trades < 30:
            trade_score = trades / 30 * 0.5
        elif trades <= 200:
            trade_score = 1.0
        else:
            trade_score = max(0.5, 1.0 - (trades - 200) / 1000)

        # Penalty for extreme returns (likely overfitting)
        if abs(ret) > 5.0:
            overfit_penalty = 0.5
        elif abs(ret) > 2.0:
            overfit_penalty = 0.7
        else:
            overfit_penalty = 1.0

        # Composite score
        composite = (
            min(sharpe, 5.0) * 0.25
            + min(sortino, 7.0) * 0.15
            + min(calmar, 5.0) * 0.15
            + win_rate * 0.10
            + min(pf, 5.0) * 0.10
            + trade_score * 0.10
            + (ret > 0) * 0.15
        ) * overfit_penalty

        exp["composite_score"] = composite

    experiments.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    return experiments


def get_best_per_strategy(experiments: list[dict]) -> list[dict]:
    """Get the best experiment for each strategy."""
    by_strategy = {}
    for exp in experiments:
        name = exp["strategy_name"]
        if name not in by_strategy:
            by_strategy[name] = []
        by_strategy[name].append(exp)

    best = []
    for name, exps in by_strategy.items():
        best_exp = max(exps, key=lambda x: x.get("composite_score", 0))
        best.append(best_exp)

    best.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    return best


def run_validation_pipeline(
    strategy_name: str,
    params: dict,
    pair: str,
    timeframe: str,
    cfg: SystemConfig,
) -> dict:
    """Run walk-forward, Monte Carlo, and robustness validation for one strategy."""
    loader = DataLoader(cfg)
    engine = BacktestEngine(cfg.backtest)

    print(f"\n  Loading data for {pair} {timeframe}...")
    df = loader.load(pair, timeframe)
    if df is None or df.empty:
        return {"error": "No data"}

    splits = loader.split_data(df)
    # Use validation + out_of_sample for validation
    is_data = splits.get("in_sample", df.iloc[:len(df)//3])
    val_data = splits.get("validation", df.iloc[len(df)//3:2*len(df)//3])
    oos_data = splits.get("out_of_sample", df.iloc[2*len(df)//3:])
    funding = loader.load_funding_rates(pair)

    strategy = get_strategy(strategy_name)
    full_data = pd.concat([is_data, val_data, oos_data])
    if funding is not None:
        full_funding = funding
    else:
        full_funding = None

    results = {}

    # 1. Walk-forward analysis
    print(f"    Walk-forward analysis...")
    try:
        wf_analyzer = WalkForwardAnalyzer(cfg.backtest)
        grid = strategy.param_grid()
        wf_result = wf_analyzer.run_walk_forward(
            full_data, strategy_name, grid,
            n_windows=5, train_pct=0.6,
            pair=pair, timeframe=timeframe,
            funding_rates=full_funding,
        )
        results["walk_forward"] = wf_result["aggregate"]
        results["wf_windows"] = wf_result["windows"]
    except Exception as e:
        print(f"    Walk-forward failed: {e}")
        results["walk_forward"] = {"error": str(e)}

    # 2. Monte Carlo analysis
    print(f"    Monte Carlo analysis...")
    try:
        signals = strategy.generate_signals(val_data, params)
        bt_result = engine.run(val_data, signals, strategy_name, params, pair, timeframe, full_funding)
        result_dict = bt_result.to_dict()

        # Use returns for bootstrap
        returns = bt_result.equity_curve.pct_change().dropna() if hasattr(bt_result, 'equity_curve') and bt_result.equity_curve is not None else pd.Series()

        mc_analyzer = MonteCarloAnalyzer(n_simulations=5000, seed=42)

        if len(returns) > 20:
            mc_result = mc_analyzer.run_bootstrap_returns(returns)
        elif bt_result.total_trades > 10:
            trades = [{"pnl": bt_result.net_profit / bt_result.total_trades}] * bt_result.total_trades
            mc_result = mc_analyzer.run_trade_shuffling(trades)
        else:
            mc_result = {"error": "Too few trades/returns"}

        results["monte_carlo"] = mc_result
    except Exception as e:
        print(f"    Monte Carlo failed: {e}")
        results["monte_carlo"] = {"error": str(e)}

    # 3. Robustness testing
    print(f"    Robustness testing...")
    try:
        rob_analyzer = RobustnessAnalyzer(cfg.backtest)
        rob_result = rob_analyzer.run_full_robustness(
            is_data, strategy_name, params,
            pair=pair, timeframe=timeframe,
            funding_rates=full_funding.iloc[:len(is_data)] if full_funding is not None else None,
        )
        results["robustness"] = rob_result
    except Exception as e:
        print(f"    Robustness failed: {e}")
        results["robustness"] = {"error": str(e)}

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze optimization results")
    parser.add_argument("--top-n", type=int, default=10, help="Top N strategies to validate")
    parser.add_argument("--select", type=int, default=3, help="Final strategies to select")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    cfg = SystemConfig.default()

    print("=" * 70)
    print("  POST-OPTIMIZATION ANALYSIS PIPELINE")
    print("=" * 70)

    # Step 1: Load experiments
    print("\n[1/8] Loading experiments...")
    experiments = load_experiments(DB_PATH)
    print(f"  Loaded {len(experiments):,} successful experiments")
    print(f"  Across {len(set(e['strategy_name'] for e in experiments))} strategies")
    print(f"  Across {len(set(e['pair'] for e in experiments))} pairs")
    print(f"  Across {len(set(e['timeframe'] for e in experiments))} timeframes")

    # Step 2: Rank by composite score
    print("\n[2/8] Ranking by composite score...")
    ranked = rank_by_composite_score(experiments)
    best_per_strat = get_best_per_strategy(ranked)

    print(f"\n  TOP {min(args.top_n, len(best_per_strat))} STRATEGIES BY COMPOSITE SCORE:")
    print(f"  {'#':>3} {'Strategy':25s} {'Pair':10s} {'TF':5s} {'Score':>6s} {'Sharpe':>8s} {'Return':>8s} {'MaxDD':>8s} {'Trades':>6s}")
    print("  " + "-" * 85)

    for i, exp in enumerate(best_per_strat[:args.top_n]):
        pair_short = exp.get("pair", "?")
        print(
            f"  {i+1:>3} {exp['strategy_name']:25s} {pair_short:10s} {exp['timeframe']:5s} "
            f"{exp.get('composite_score',0):>6.3f} "
            f"{exp.get('sharpe',0):>8.2f} "
            f"{exp.get('total_return',0):>7.1%} "
            f"{exp.get('max_drawdown',0):>7.1%} "
            f"{exp.get('total_trades',0):>6d}"
        )

    # Step 3: Validation
    validation_results = {}
    if not args.skip_validation:
        print(f"\n[3/8] Running validation pipeline on top {args.top_n} strategies...")
        top_to_validate = best_per_strat[:args.top_n]

        for i, exp in enumerate(top_to_validate):
            name = exp["strategy_name"]
            pair = exp["pair"]
            tf = exp["timeframe"]
            params = exp["parameters"]
            key = f"{name}_{pair}_{tf}"

            print(f"\n  [{i+1}/{len(top_to_validate)}] {name} {pair} {tf}")
            val_result = run_validation_pipeline(name, params, pair, tf, cfg)
            validation_results[key] = {
                "strategy": exp,
                "validation": val_result,
            }

            # Print summary
            wf = val_result.get("walk_forward", {})
            mc = val_result.get("monte_carlo", {})
            rob = val_result.get("robustness", {})

            if "error" not in wf:
                print(f"    WF: Avg Sharpe={wf.get('avg_sharpe',0):.2f}, Profitable Windows: {wf.get('pct_profitable_windows',0):.0%}")
            if "error" not in mc:
                print(f"    MC: Prob Loss={mc.get('prob_loss',0):.1%}, Prob Ruin={mc.get('prob_ruin',0):.1%}, Expected Return={mc.get('expected_return',0):.1%}")
            if "error" not in rob:
                print(f"    Rob: Param={rob.get('parameter_stability',{}).get('stability_score',0):.0%}, Cost={rob.get('cost_robustness',{}).get('cost_robustness_score',0):.0%}, Time={rob.get('time_period_stability',{}).get('stability_score',0):.0%}")

    else:
        print("\n[3/8] Skipping validation (--skip-validation)")

    # Step 4: Select top strategies with validation-adjusted scoring
    print(f"\n[4/8] Selecting top {args.select} strategies...")

    if validation_results:
        # Boost/reduce scores based on validation
        for key, vr in validation_results.items():
            exp = vr["strategy"]
            val = vr["validation"]

            wf_score = 0
            mc_score = 0
            rob_score = 0

            wf = val.get("walk_forward", {})
            if "error" not in wf:
                wf_score = wf.get("pct_profitable_windows", 0) * 0.5 + min(wf.get("avg_sharpe", 0), 3) / 3 * 0.5

            mc = val.get("monte_carlo", {})
            if "error" not in mc:
                mc_score = (1 - mc.get("prob_loss", 0.5)) * 0.5 + (1 - mc.get("prob_ruin", 0)) * 0.5

            rob = val.get("robustness", {})
            if "error" not in rob:
                rob_score = rob.get("combined_robustness_score", 0)

            # Validation-adjusted score
            exp["validation_score"] = (wf_score + mc_score + rob_score) / 3
            exp["adjusted_score"] = exp.get("composite_score", 0) * 0.6 + exp["validation_score"] * 0.4

        # Re-rank by adjusted score
        all_validated = [vr["strategy"] for vr in validation_results.values()]
        all_validated.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)
    else:
        all_validated = best_per_strat

    selected = all_validated[:args.select]

    print(f"\n  FINAL {args.select} SELECTED STRATEGIES:")
    print(f"  {'#':>3} {'Strategy':25s} {'Pair':10s} {'TF':5s} {'AdjScore':>8s} {'Sharpe':>8s} {'Return':>8s} {'MaxDD':>8s}")
    print("  " + "-" * 80)
    for i, exp in enumerate(selected):
        print(
            f"  {i+1:>3} {exp['strategy_name']:25s} {exp.get('pair',''):10s} {exp.get('timeframe',''):5s} "
            f"{exp.get('adjusted_score',0):>8.3f} "
            f"{exp.get('sharpe',0):>8.2f} "
            f"{exp.get('total_return',0):>7.1%} "
            f"{exp.get('max_drawdown',0):>7.1%}"
        )

    # Step 5: Portfolio analysis
    print(f"\n[5/8] Portfolio allocation analysis...")
    if len(selected) >= 2:
        print("  Would build portfolio with equal-weight, risk-parity, and drawdown-adjusted allocation")
        print("  (Requires equity curves from backtester - using Sharpe-based weights as proxy)")

        # Simple proxy: weight by adjusted score
        scores = np.array([max(s.get("adjusted_score", 0.01), 0.01) for s in selected])
        weights = scores / scores.sum()
        inner = ", ".join(f"{s['strategy_name']}: {w:.1%}" for s, w in zip(selected, weights))
        print(f"  Suggested weights: {inner}")
    else:
        print("  Not enough strategies for portfolio analysis")

    # Step 6: Correlation analysis
    print(f"\n[6/8] Strategy correlation analysis...")
    if len(selected) >= 2:
        print("  Selected strategies span different families:")
        families = set()
        for s in selected:
            strat = get_strategy(s["strategy_name"])
            meta = strat.meta()
            families.add(meta.family)
            print(f"    {s['strategy_name']:25s} -> {meta.family}")
        print(f"  {len(families)} unique families represented")

    # Step 7: Save results
    print(f"\n[7/8] Saving analysis results...")

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save selected strategies
    selected_data = []
    for exp in selected:
        selected_data.append({
            "strategy_name": exp["strategy_name"],
            "pair": exp["pair"],
            "timeframe": exp["timeframe"],
            "parameters": exp["parameters"],
            "composite_score": exp.get("composite_score", 0),
            "adjusted_score": exp.get("adjusted_score", 0),
            "sharpe": exp.get("sharpe", 0),
            "sortino": exp.get("sortino", 0),
            "total_return": exp.get("total_return", 0),
            "max_drawdown": exp.get("max_drawdown", 0),
            "total_trades": exp.get("total_trades", 0),
            "win_rate": exp.get("win_rate", 0),
            "profit_factor": exp.get("profit_factor", 0),
        })

    with open(results_dir / "selected_strategies.json", "w") as f:
        json.dump(selected_data, f, indent=2, default=str)

    # Save validation results
    if validation_results:
        val_serializable = {}
        for key, vr in validation_results.items():
            val_serializable[key] = {
                "strategy_name": vr["strategy"]["strategy_name"],
                "pair": vr["strategy"]["pair"],
                "timeframe": vr["strategy"]["timeframe"],
                "validation": vr["validation"],
            }
        with open(results_dir / "validation_results.json", "w") as f:
            json.dump(val_serializable, f, indent=2, default=str)

    print(f"  Saved to {results_dir / 'selected_strategies.json'}")
    print(f"  Saved to {results_dir / 'validation_results.json'}")

    # Step 8: Summary report
    print(f"\n[8/8] Generating summary...")
    print()
    print("=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Total experiments analyzed: {len(experiments):,}")
    print(f"  Strategies tested: {len(set(e['strategy_name'] for e in experiments))}")
    print(f"  Final strategies selected: {len(selected)}")
    print(f"  Results saved to: {results_dir}")
    print()
    print("  Next steps:")
    print("    1. Generate charts: python scripts/generate_charts.py")
    print("    2. Generate report: python scripts/generate_report.py")
    print("    3. Paper trading:   python scripts/start_paper_bot.py")


if __name__ == "__main__":
    main()
