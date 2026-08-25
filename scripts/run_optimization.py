#!/usr/bin/env python3
"""
Run parameter optimization for a strategy.

Usage:
    python scripts/run_optimization.py --strategy MA_Crossover --pair "BTC/USDT:USDT" --timeframe 1h --method grid
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.optimization.runner import ExperimentRunner
from trading_system.optimization.param_space import count_combinations
from trading_system.strategies import get_strategy
from trading_system.data.storage import DataStorage
from trading_system.utils.logging import setup_logging

console = Console()


@click.command()
@click.option("--strategy", default="MA_Crossover", help="Strategy name")
@click.option("--pair", default="BTC/USDT:USDT", help="Trading pair")
@click.option("--timeframe", default="1h", help="Timeframe")
@click.option("--config", default="configs/default.yaml", help="Config file")
@click.option("--method", default="grid", type=click.Choice(["grid", "random"]), help="Search method")
@click.option("--n-samples", default=1000, help="Number of random samples (for random search)")
@click.option("--split", default="in_sample", help="Data split to use for optimization")
def main(strategy: str, pair: str, timeframe: str, config: str, method: str, n_samples: int, split: str):
    """Run parameter optimization."""
    setup_logging()

    # Load config
    config_path = Path(config)
    cfg = SystemConfig.from_yaml(config_path) if config_path.exists() else SystemConfig.default()

    # Load data
    loader = DataLoader(cfg)
    df = loader.load(pair, timeframe)

    if df is None or df.empty:
        console.print(f"[red]No data found. Run download_data.py first.[/red]")
        return

    # Split data
    splits = loader.split_data(df)
    data = splits.get(split, df)

    console.print(f"[bold blue]Optimization: {strategy}[/bold blue]")
    console.print(f"  Pair: {pair}, Timeframe: {timeframe}")
    console.print(f"  Data split: {split} ({len(data)} candles)")
    console.print(f"  Method: {method}")

    # Get strategy and count combinations
    strat = get_strategy(strategy)
    grid = strat.param_grid()
    total = count_combinations(grid)
    console.print(f"  Total combinations: {total:,}")

    # Run optimization
    runner = ExperimentRunner(cfg)
    start_time = time.time()

    if method == "grid":
        results = runner.run_grid_search(
            strategy, grid, data, pair, timeframe
        )
    else:
        results = runner.run_random_search(
            strategy, grid, data, pair, timeframe, n_samples=n_samples
        )

    elapsed = time.time() - start_time

    # Filter successful results
    successful = [r for r in results if r.get("success", False)]
    console.print(f"\n[green]Completed {len(successful)}/{len(results)} experiments in {elapsed:.1f}s[/green]")

    # Sort by Sharpe
    successful.sort(key=lambda r: r.get("results", {}).get("sharpe", -999), reverse=True)

    # Show top 10
    if successful:
        from rich.table import Table
        table = Table(title=f"Top 10 Results - {strategy}")
        table.add_column("#", style="dim")
        table.add_column("Sharpe", style="cyan")
        table.add_column("Sortino")
        table.add_column("Return", style="green")
        table.add_column("Max DD", style="red")
        table.add_column("Trades")
        table.add_column("Win Rate")
        table.add_column("Parameters")

        for i, r in enumerate(successful[:10]):
            res = r["results"]
            table.add_row(
                str(i + 1),
                f"{res.get('sharpe', 0):.2f}",
                f"{res.get('sortino', 0):.2f}",
                f"{res.get('total_return', 0):.1%}",
                f"{res.get('max_drawdown', 0):.1%}",
                str(res.get('total_trades', 0)),
                f"{res.get('win_rate', 0):.1%}",
                str(r.get("parameters", {}))[:60],
            )
        console.print(table)

    # Save results to database
    storage = DataStorage(cfg.data.results_dir, cfg.data.results_dir)
    storage.save_batch_experiments(successful)
    console.print(f"\n[green]Results saved to {cfg.data.results_dir}/metadata.db[/green]")
    storage.close()


if __name__ == "__main__":
    main()
