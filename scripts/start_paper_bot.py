#!/usr/bin/env python3
"""Start the paper trading bot."""
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
@click.option("--strategies", default="MA_Crossover", help="Comma-separated strategy names")
@click.option("--params", default="", help="JSON string of strategy parameters")
def main(config: str, strategies: str, params: str):
    setup_logging()
    cfg = SystemConfig.from_yaml(config) if Path(config).exists() else SystemConfig.default()
    cfg.bot.mode = "paper"

    import json
    strategy_names = [s.strip() for s in strategies.split(",")]
    strategy_params = json.loads(params) if params else {name: {} for name in strategy_names}

    console.print("[bold blue]Starting Paper Trading Bot[/bold blue]")
    console.print(f"  Strategies: {strategy_names}")
    console.print(f"  Pairs: {cfg.exchange.pairs}")
    console.print("  Mode: paper")
    console.print("  Press Ctrl+C to stop\n")

    bot = TradingBot(cfg, strategy_names, strategy_params)
    bot.start()

if __name__ == "__main__":
    main()
