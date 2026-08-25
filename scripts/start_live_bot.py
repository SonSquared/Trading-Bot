#!/usr/bin/env python3
"""Start the live trading bot. Requires API keys and explicit confirmation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console

from trading_system.config import SystemConfig
from trading_system.bot.core import TradingBot
from trading_system.utils.logging import setup_logging

console = Console()

@click.command()
@click.option("--config", default="configs/default.yaml")
@click.option("--strategies", default="MA_Crossover")
@click.option("--mode", type=click.Choice(["paper", "dry_run", "live"]), default="dry_run")
@click.option("--confirm", is_flag=True, help="Confirm live trading (required for live mode)")
def main(config: str, strategies: str, mode: str, confirm: bool):
    setup_logging()
    cfg = SystemConfig.from_yaml(config) if Path(config).exists() else SystemConfig.default()

    if mode == "live" and not confirm:
        console.print("[red]Live trading requires --confirm flag.[/red]")
        console.print("This will use real money. Please confirm you understand the risks.")
        return

    cfg.bot.mode = mode
    strategy_names = [s.strip() for s in strategies.split(",")]
    strategy_params = {name: {} for name in strategy_names}

    console.print(f"[bold {'red' if mode == 'live' else 'blue'}]Starting {mode.upper()} Trading Bot[/bold]")
    console.print(f"  Strategies: {strategy_names}")
    console.print(f"  Pairs: {cfg.exchange.pairs}")
    console.print(f"  Mode: {mode}")
    if mode == "live":
        console.print("[bold red]WARNING: This will trade real money![/bold red]")
    console.print(f"  Press Ctrl+C to stop\n")

    bot = TradingBot(cfg, strategy_names, strategy_params)
    bot.start()

if __name__ == "__main__":
    main()
