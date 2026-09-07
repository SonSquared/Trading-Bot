#!/usr/bin/env python3
"""
Run a single backtest for a strategy.

Usage:
    python scripts/run_backtest.py --strategy MA_Crossover --pair "BTC/USDT:USDT" --timeframe 1h
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy, ALL_STRATEGIES
from trading_system.utils.logging import setup_logging

console = Console()


@click.command()
@click.option("--strategy", default="MA_Crossover", help="Strategy name")
@click.option("--pair", default="BTC/USDT:USDT", help="Trading pair")
@click.option("--timeframe", default="1h", help="Timeframe")
@click.option("--config", default="configs/default.yaml", help="Config file")
@click.option("--list-strategies", is_flag=True, help="List all available strategies")
def main(strategy: str, pair: str, timeframe: str, config: str, list_strategies: bool):
    """Run a single backtest."""
    setup_logging()

    if list_strategies:
        table = Table(title="Available Strategies")
        table.add_column("Name", style="cyan")
        table.add_column("Family", style="green")
        table.add_column("Description")
        for s in ALL_STRATEGIES:
            m = s.meta()
            table.add_row(m.name, m.family, m.description)
        console.print(table)
        return

    # Load config
    config_path = Path(config)
    cfg = SystemConfig.from_yaml(config_path) if config_path.exists() else SystemConfig.default()

    # Load data
    loader = DataLoader(cfg)
    df = loader.load(pair, timeframe)

    if df is None or df.empty:
        console.print(f"[red]No data found for {pair} {timeframe}[/red]")
        console.print("Run 'python scripts/download_data.py' first to download data.")
        return

    # Get strategy
    strat = get_strategy(strategy)
    params = strat.default_params()

    console.print(f"[bold blue]Running backtest: {strategy}[/bold blue]")
    console.print(f"  Pair: {pair}")
    console.print(f"  Timeframe: {timeframe}")
    console.print(f"  Data: {len(df)} candles ({df.index[0]} to {df.index[-1]})")
    console.print(f"  Parameters: {params}")

    # Run backtest
    engine = BacktestEngine(cfg.backtest)
    signals = strat.generate_signals(df, params)

    result = engine.run(
        df, signals,
        strategy_name=strategy,
        params=params,
        pair=pair,
        timeframe=timeframe,
    )

    # Print results
    console.print()
    console.print("[bold green]Backtest Results[/bold green]")
    console.print(result.summary())

    # Print trade details
    if result.trades:
        table = Table(title=f"Trades ({len(result.trades)} total)")
        table.add_column("Entry Time")
        table.add_column("Exit Time")
        table.add_column("Side")
        table.add_column("Entry Price")
        table.add_column("Exit Price")
        table.add_column("P&L")
        table.add_column("Reason")

        for t in result.trades[:20]:  # Show first 20
            side = "LONG" if t.get("position_size", 0) > 0 else "SHORT"
            pnl = t.get("pnl", 0)
            style = "green" if pnl > 0 else "red"
            table.add_row(
                str(t.get("entry_time", "")),
                str(t.get("exit_time", "")),
                side,
                f"${t.get('entry_price', 0):.2f}",
                f"${t.get('exit_price', 0):.2f}",
                f"[{style}]${pnl:.2f}[/{style}]",
                t.get("exit_reason", ""),
            )
        console.print(table)

    # Generate chart
    from trading_system.dashboard.charts import ChartGenerator
    charts = ChartGenerator()
    if len(result.equity_curve) > 0:
        charts.equity_curve(result.equity_curve, f"{strategy} - {pair} {timeframe}")
        charts.drawdown_curve(result.equity_curve, f"{strategy} Drawdown")
        if result.trades:
            charts.trade_distribution(result.trades, f"{strategy} Trade Distribution")
        console.print("\n[green]Charts saved to data/charts/[/green]")


if __name__ == "__main__":
    main()
