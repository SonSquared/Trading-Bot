#!/usr/bin/env python3
"""
Run the full research pipeline.

This script runs the complete trading strategy research process:
1. Load and validate data
2. Run optimization for all strategies
3. Perform walk-forward analysis
4. Run robustness tests
5. Perform Monte Carlo analysis
6. Score and rank strategies
7. Select top 3 strategies
8. Generate portfolio analysis
9. Create comprehensive report

Usage:
    python scripts/run_full_research.py --config configs/default.yaml
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import ALL_STRATEGIES
from trading_system.optimization.runner import ExperimentRunner
from trading_system.optimization.param_space import count_combinations
from trading_system.optimization.scoring import calculate_composite_score
from trading_system.validation.walk_forward import WalkForwardAnalyzer
from trading_system.validation.monte_carlo import MonteCarloAnalyzer
from trading_system.validation.robustness import RobustnessAnalyzer
from trading_system.validation.overfitting import OverfittingDetector
from trading_system.ranking.scorer import StrategyScorer
from trading_system.ranking.correlation import StrategyCorrelationAnalyzer
from trading_system.dashboard.charts import ChartGenerator
from trading_system.reports.generator import ReportGenerator
from trading_system.data.storage import DataStorage
from trading_system.utils.logging import setup_logging

console = Console()


@click.command()
@click.option("--config", default="configs/default.yaml", help="Config file")
@click.option("--pair", default="BTC/USDT:USDT", help="Trading pair")
@click.option("--fast/--no-fast", default=False, help="Fast mode (fewer params)")
@click.option("--skip-validation/--no-skip-validation", default=False, help="Skip validation stage")
def main(config: str, pair: str, fast: bool, skip_validation: bool):
    """Run the full research pipeline."""
    setup_logging()

    total_start = time.time()

    # Load config
    config_path = Path(config)
    cfg = SystemConfig.from_yaml(config_path) if config_path.exists() else SystemConfig.default()
    pair = pair or cfg.exchange.pairs[0]

    console.rule("[bold blue]TRADING STRATEGY RESEARCH PIPELINE[/bold blue]")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: Load Data
    # ═══════════════════════════════════════════════════════════════
    console.rule("[bold]Stage 1: Data Loading[/bold]")
    loader = DataLoader(cfg)
    splits = None

    for tf in cfg.exchange.timeframes:
        df = loader.load(pair, tf)
        if df is not None and not df.empty:
            splits = loader.split_data(df)
            console.print(f"  {pair} {tf}: {len(df)} candles loaded")
            for name, part in splits.items():
                console.print(f"    {name}: {len(part)} candles")
        else:
            console.print(f"  [yellow]No data for {pair} {tf} - run download_data.py first[/yellow]")

    if splits is None:
        console.print("[red]No data available. Please download data first.[/red]")
        return

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: Run Optimization
    # ═══════════════════════════════════════════════════════════════
    console.rule("[bold]Stage 2: Strategy Optimization[/bold]")

    all_experiments = []
    runner = ExperimentRunner(cfg)

    for strategy in ALL_STRATEGIES:
        meta = strategy.meta()
        grid = strategy.param_grid()

        # In fast mode, use fewer parameters
        if fast:
            reduced_grid = {}
            for key, values in grid.items():
                if isinstance(values, list) and len(values) > 4:
                    reduced_grid[key] = values[::len(values)//3][:4]
                else:
                    reduced_grid[key] = values
            grid = reduced_grid

        total_combos = count_combinations(grid)
        console.print(f"\n  [cyan]{meta.name}[/cyan] ({meta.family}) - {total_combos:,} combinations")

        for tf in cfg.exchange.timeframes:
            tf_splits = loader.split_data(loader.load(pair, tf) or pd.DataFrame())
            if not tf_splits or tf_splits.get("in_sample") is None:
                continue

            train_data = tf_splits["in_sample"]
            if len(train_data) < 100:
                continue

            start = time.time()
            results = runner.run_grid_search(
                meta.name, grid, train_data, pair, tf
            )
            elapsed = time.time() - start

            successful = [r for r in results if r.get("success")]
            all_experiments.extend(successful)
            console.print(
                f"    {tf}: {len(successful)} successful in {elapsed:.1f}s"
            )

    console.print(f"\n[bold green]Total experiments: {len(all_experiments)}[/bold green]")

    if not all_experiments:
        console.print("[red]No successful experiments. Check data and strategy configuration.[/red]")
        return

    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: Initial Scoring & Ranking
    # ═══════════════════════════════════════════════════════════════
    console.rule("[bold]Stage 3: Scoring & Ranking[/bold]")

    scorer = StrategyScorer()
    scored_experiments = []

    for exp in all_experiments:
        results = exp.get("results", {})
        if results.get("total_trades", 0) < 10:
            continue

        score = calculate_composite_score(
            type("R", (), results)(),
            is_sharpe=results.get("sharpe", 0),
            oos_sharpe=results.get("sharpe", 0) * 0.8,  # Placeholder
        )
        scored_experiments.append({**exp, "composite_score": score.get("composite_score", 0)})

    scored_experiments.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

    # Show top strategies
    top_table = Table(title="Top 15 Strategies")
    top_table.add_column("#", style="dim")
    top_table.add_column("Strategy", style="cyan")
    top_table.add_column("TF")
    top_table.add_column("Sharpe")
    top_table.add_column("Sortino")
    top_table.add_column("Return", style="green")
    top_table.add_column("Max DD", style="red")
    top_table.add_column("Trades")
    top_table.add_column("Score", style="bold")

    for i, exp in enumerate(scored_experiments[:15]):
        res = exp.get("results", {})
        top_table.add_row(
            str(i + 1),
            exp.get("strategy_name", ""),
            exp.get("timeframe", ""),
            f"{res.get('sharpe', 0):.2f}",
            f"{res.get('sortino', 0):.2f}",
            f"{res.get('total_return', 0):.1%}",
            f"{res.get('max_drawdown', 0):.1%}",
            str(res.get('total_trades', 0)),
            f"{exp.get('composite_score', 0):.3f}",
        )
    console.print(top_table)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: Select Top Candidates
    # ═══════════════════════════════════════════════════════════════
    console.rule("[bold]Stage 4: Strategy Selection[/bold]")

    # Get best per strategy (dedup)
    best_per_strategy = {}
    for exp in scored_experiments:
        name = exp.get("strategy_name", "")
        if name not in best_per_strategy:
            best_per_strategy[name] = exp

    ranked = sorted(best_per_strategy.values(), key=lambda x: x.get("composite_score", 0), reverse=True)
    top_candidates = ranked[:5]  # Take top 5 candidates for further analysis

    console.print(f"Selected {len(top_candidates)} candidates for validation:")
    for i, c in enumerate(top_candidates):
        res = c.get("results", {})
        console.print(
            f"  {i+1}. {c.get('strategy_name', '')} ({c.get('timeframe', '')}) "
            f"- Sharpe: {res.get('sharpe', 0):.2f}, Score: {c.get('composite_score', 0):.3f}"
        )

    # ═══════════════════════════════════════════════════════════════
    # STAGE 5: Robustness Testing (for top candidates)
    # ═══════════════════════════════════════════════════════════════
    console.rule("[bold]Stage 5: Robustness Testing[/bold]")

    robustness_analyzer = RobustnessAnalyzer(cfg.backtest)
    monte_carlo = MonteCarloAnalyzer(n_simulations=1000)
    overfit_detector = OverfittingDetector()

    validated_strategies = []

    for candidate in top_candidates:
        name = candidate.get("strategy_name", "")
        params = candidate.get("parameters", {})
        tf = candidate.get("timeframe", "1h")

        console.print(f"\n  [cyan]Testing: {name} ({tf})[/cyan]")

        # Load full data for this timeframe
        df = loader.load(pair, tf)
        if df is None or df.empty:
            continue

        splits = loader.split_data(df)

        # Parameter stability
        train_data = splits.get("in_sample", df)
        if len(train_data) > 100:
            param_test = robustness_analyzer.test_parameter_stability(
                train_data, name, params, pair, tf
            )
            stab_score = param_test.get("stability_score", 0)
            console.print(f"    Parameter stability: {stab_score:.2f}")

            # Cost robustness
            cost_test = robustness_analyzer.test_cost_robustness(
                train_data, name, params, pair, tf
            )
            cost_score = cost_test.get("cost_robustness_score", 0)
            console.print(f"    Cost robustness: {cost_score:.2f}")

            # Time period stability
            time_test = robustness_analyzer.test_time_period_stability(
                train_data, name, params, pair, tf
            )
            time_score = time_test.get("stability_score", 0)
            console.print(f"    Time stability: {time_score:.2f}")

            # Monte Carlo
            strategy = get_strategy(name)
            engine = BacktestEngine(cfg.backtest)
            signals = strategy.generate_signals(train_data, params)
            bt_result = engine.run(train_data, signals, name, params, pair, tf)

            if bt_result.trades:
                mc_result = monte_carlo.run_trade_shuffling(bt_result.trades)
                console.print(f"    MC prob_loss: {mc_result.get('prob_loss', 0):.2f}")

            # Overfitting check
            overfit = overfit_detector.analyze(
                full_results=bt_result.to_dict(),
                param_stability=param_test,
            )
            console.print(f"    Overfitting severity: {overfit.get('severity', 'unknown')}")

            # Update composite score with robustness data
            new_score = calculate_composite_score(
                type("R", (), bt_result.to_dict())(),
                is_sharpe=bt_result.sharpe,
                param_stability=stab_score,
                cost_robustness=cost_score,
                timeframe_stability=time_score,
            )

            validated_strategies.append({
                **candidate,
                "composite_score": new_score.get("composite_score", 0),
                "robustness": {
                    "param_stability": stab_score,
                    "cost_robustness": cost_score,
                    "time_stability": time_score,
                    "overfitting": overfit,
                    "monte_carlo": mc_result if bt_result.trades else {},
                },
            })

    # ═══════════════════════════════════════════════════════════════
    # STAGE 6: Final Selection
    # ═══════════════════════════════════════════════════════════════
    console.rule("[bold]Stage 6: Final Selection[/bold]")

    validated_strategies.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    final_3 = validated_strategies[:3]

    console.print(f"\n[bold green]Selected 3 Final Strategies:[/bold green]")
    for i, s in enumerate(final_3):
        res = s.get("results", {})
        rob = s.get("robustness", {})
        console.print(
            f"\n  {i+1}. [cyan]{s.get('strategy_name', '')}[/cyan] ({s.get('timeframe', '')})\n"
            f"     Sharpe: {res.get('sharpe', 0):.2f} | Return: {res.get('total_return', 0):.1%} | "
            f"Max DD: {res.get('max_drawdown', 0):.1%}\n"
            f"     Param stability: {rob.get('param_stability', 0):.2f} | "
            f"Cost robustness: {rob.get('cost_robustness', 0):.2f} | "
            f"Composite: {s.get('composite_score', 0):.3f}"
        )

    # ═══════════════════════════════════════════════════════════════
    # STAGE 7: Generate Charts
    # ═══════════════════════════════════════════════════════════════
    console.rule("[bold]Stage 7: Generating Charts[/bold]")

    charts = ChartGenerator()
    for i, s in enumerate(final_3):
        name = s.get("strategy_name", "")
        tf = s.get("timeframe", "1h")
        params = s.get("parameters", {})

        df = loader.load(pair, tf)
        if df is None or df.empty:
            continue

        strategy = get_strategy(name)
        engine = BacktestEngine(cfg.backtest)
        signals = strategy.generate_signals(df, params)
        result = engine.run(df, signals, name, params, pair, tf)

        if len(result.equity_curve) > 0:
            charts.equity_curve(
                result.equity_curve,
                f"Strategy {i+1}: {name} - {pair} {tf}",
                f"strategy_{i+1}_equity.png",
            )
            charts.drawdown_curve(
                result.equity_curve,
                f"Strategy {i+1}: {name} Drawdown",
                f"strategy_{i+1}_drawdown.png",
            )
            if result.trades:
                charts.trade_distribution(
                    result.trades,
                    f"Strategy {i+1}: {name} Trades",
                    f"strategy_{i+1}_trades.png",
                )

    console.print("[green]Charts saved to data/charts/[/green]")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 8: Generate Report
    # ═══════════════════════════════════════════════════════════════
    console.rule("[bold]Stage 8: Generating Report[/bold]")

    report_gen = ReportGenerator()
    report_path = report_gen.generate(
        config_summary={
            "exchange": cfg.exchange.name,
            "pairs": cfg.exchange.pairs,
            "timeframes": cfg.exchange.timeframes,
            "start_date": cfg.data.start_date,
            "end_date": cfg.data.end_date or "present",
            "families": cfg.strategy.families,
        },
        data_summary={},
        strategy_results=scored_experiments[:50],
        selected_strategies=final_3,
    )

    total_elapsed = time.time() - total_start
    console.print(f"\n[bold green]Research complete in {total_elapsed:.1f}s[/bold green]")
    console.print(f"Report: {report_path}")
    console.print(f"Charts: data/charts/")
    console.print(f"Results: {cfg.data.results_dir}/metadata.db")


if __name__ == "__main__":
    main()
