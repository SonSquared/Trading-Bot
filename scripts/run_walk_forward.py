#!/usr/bin/env python3
"""Run walk-forward analysis for a strategy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.validation.walk_forward import WalkForwardAnalyzer
from trading_system.strategies import get_strategy
from trading_system.utils.logging import setup_logging

console = Console()

@click.command()
@click.option("--strategy", required=True, help="Strategy name")
@click.option("--pair", default="BTC/USDT:USDT")
@click.option("--timeframe", default="1h")
@click.option("--windows", default=10, help="Number of walk-forward windows")
@click.option("--config", default="configs/default.yaml")
def main(strategy: str, pair: str, timeframe: str, windows: int, config: str):
    setup_logging()
    cfg = SystemConfig.from_yaml(config) if Path(config).exists() else SystemConfig.default()
    loader = DataLoader(cfg)
    df = loader.load(pair, timeframe)
    if df is None or df.empty:
        console.print("[red]No data found. Run download_data.py first.[/red]")
        return

    strat = get_strategy(strategy)
    analyzer = WalkForwardAnalyzer(cfg.backtest)

    console.print(f"[bold blue]Walk-Forward Analysis: {strategy}[/bold blue]")
    result = analyzer.run_walk_forward(
        df, strategy, strat.param_grid(), n_windows=windows,
        pair=pair, timeframe=timeframe,
    )

    agg = result.get("aggregate", {})
    console.print("\n[bold green]Results:[/bold green]")
    console.print(f"  Windows: {result.get('n_windows', 0)}")
    console.print(f"  Avg Sharpe: {agg.get('avg_sharpe', 0):.2f}")
    console.print(f"  Median Sharpe: {agg.get('median_sharpe', 0):.2f}")
    console.print(f"  Profitable windows: {agg.get('pct_profitable_windows', 0):.1%}")
    console.print(f"  Avg return: {agg.get('avg_return', 0):.1%}")
    console.print(f"  Worst drawdown: {agg.get('worst_max_drawdown', 0):.1%}")

if __name__ == "__main__":
    main()
