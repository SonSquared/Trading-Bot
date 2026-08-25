#!/usr/bin/env python3
"""Run Monte Carlo analysis for a strategy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.validation.monte_carlo import MonteCarloAnalyzer
from trading_system.backtester.engine import BacktestEngine
from trading_system.strategies import get_strategy
from trading_system.utils.logging import setup_logging

console = Console()

@click.command()
@click.option("--strategy", required=True)
@click.option("--pair", default="BTC/USDT:USDT")
@click.option("--timeframe", default="1h")
@click.option("--simulations", default=10000)
@click.option("--config", default="configs/default.yaml")
def main(strategy: str, pair: str, timeframe: str, simulations: int, config: str):
    setup_logging()
    cfg = SystemConfig.from_yaml(config) if Path(config).exists() else SystemConfig.default()
    loader = DataLoader(cfg)
    df = loader.load(pair, timeframe)
    if df is None or df.empty:
        console.print("[red]No data found.[/red]")
        return

    strat = get_strategy(strategy)
    params = strat.default_params()
    engine = BacktestEngine(cfg.backtest)
    signals = strat.generate_signals(df, params)
    result = engine.run(df, signals, strategy, params, pair, timeframe)

    mc = MonteCarloAnalyzer(n_simulations=simulations)
    mc_result = mc.run_trade_shuffling(result.trades)

    console.print(f"\n[bold green]Monte Carlo Results ({simulations} simulations):[/bold green]")
    for k, v in mc_result.items():
        if isinstance(v, float):
            console.print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            console.print(f"  {k}: {[f'{x:.2f}' for x in v]}")
        else:
            console.print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
