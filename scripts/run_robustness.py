#!/usr/bin/env python3
"""Run robustness testing for a strategy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console

from trading_system.config import SystemConfig
from trading_system.data.loader import DataLoader
from trading_system.validation.robustness import RobustnessAnalyzer
from trading_system.strategies import get_strategy
from trading_system.utils.logging import setup_logging

console = Console()

@click.command()
@click.option("--strategy", required=True)
@click.option("--pair", default="BTC/USDT:USDT")
@click.option("--timeframe", default="1h")
@click.option("--config", default="configs/default.yaml")
def main(strategy: str, pair: str, timeframe: str, config: str):
    setup_logging()
    cfg = SystemConfig.from_yaml(config) if Path(config).exists() else SystemConfig.default()
    loader = DataLoader(cfg)
    df = loader.load(pair, timeframe)
    if df is None or df.empty:
        console.print("[red]No data found.[/red]")
        return

    strat = get_strategy(strategy)
    params = strat.default_params()
    analyzer = RobustnessAnalyzer(cfg.backtest)

    console.print(f"[bold blue]Robustness Testing: {strategy}[/bold blue]")
    result = analyzer.run_full_robustness(df, strategy, params, pair, timeframe)

    console.print(f"\n[bold green]Combined Robustness Score: {result.get('combined_robustness_score', 0):.3f}[/bold green]")
    ps = result.get("parameter_stability", {})
    cr = result.get("cost_robustness", {})
    ts = result.get("time_period_stability", {})
    console.print(f"  Parameter stability: {ps.get('stability_score', 0):.2f} ({ps.get('profitable_variations', 0)}/{ps.get('total_tested', 0)} profitable)")
    console.print(f"  Cost robustness: {cr.get('cost_robustness_score', 0):.2f}")
    console.print(f"  Time stability: {ts.get('stability_score', 0):.2f} ({ts.get('profitable_periods', 0)}/{ts.get('n_periods', 0)} profitable)")

if __name__ == "__main__":
    main()
