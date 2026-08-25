#!/usr/bin/env python3
"""
Download historical data from Binance.

Usage:
    python scripts/download_data.py [--config configs/default.yaml] [--start 2022-01-01] [--end 2024-12-31]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.table import Table

from trading_system.config import SystemConfig
from trading_system.data.downloader import BinanceDataDownloader
from trading_system.data.validator import DataValidator
from trading_system.data.cleaner import DataCleaner
from trading_system.data.storage import DataStorage
from trading_system.utils.logging import setup_logging

console = Console()


@click.command()
@click.option("--config", default="configs/default.yaml", help="Config file path")
@click.option("--start", default="", help="Start date (YYYY-MM-DD)")
@click.option("--end", default="", help="End date (YYYY-MM-DD)")
@click.option("--pairs", default="", help="Comma-separated pairs to download")
@click.option("--timeframes", default="", help="Comma-separated timeframes")
@click.option("--validate/--no-validate", default=True, help="Validate downloaded data")
def main(
    config: str,
    start: str,
    end: str,
    pairs: str,
    timeframes: str,
    validate: bool,
):
    """Download historical data from Binance Futures."""
    setup_logging()

    # Load config
    config_path = Path(config)
    if config_path.exists():
        cfg = SystemConfig.from_yaml(config_path)
    else:
        console.print(f"[yellow]Config not found at {config}, using defaults[/yellow]")
        cfg = SystemConfig.default()

    # Override with CLI args
    if start:
        cfg.data.start_date = start
    if end:
        cfg.data.end_date = end
    if pairs:
        cfg.exchange.pairs = [p.strip() for p in pairs.split(",")]
    if timeframes:
        cfg.exchange.timeframes = [t.strip() for t in timeframes.split(",")]

    console.print("[bold blue]Downloading historical data from Binance[/bold blue]")
    console.print(f"  Pairs: {cfg.exchange.pairs}")
    console.print(f"  Timeframes: {cfg.exchange.timeframes}")
    console.print(f"  Start: {cfg.data.start_date}")
    console.print(f"  End: {cfg.data.end_date or 'latest'}")
    console.print()

    # Initialize components
    downloader = BinanceDataDownloader(cfg)
    cleaner = DataCleaner()
    storage = DataStorage(cfg.data.data_dir, cfg.data.results_dir)
    validator = DataValidator()

    # Download and process each pair/timeframe
    for pair in cfg.exchange.pairs:
        console.print(f"[bold green]Pair: {pair}[/bold green]")

        # Download funding rates
        console.print("  Downloading funding rates...")
        funding_df = downloader.download_funding_rates(
            pair, cfg.data.start_date, cfg.data.end_date, save=False
        )
        if not funding_df.empty:
            funding_df = cleaner.clean_funding_rates(funding_df)
            storage.save_data(funding_df, pair, "1h", "funding_rates")
            console.print(f"    Saved {len(funding_df)} funding rate records")

        # Download klines for each timeframe
        for tf in cfg.exchange.timeframes:
            console.print(f"  Downloading {tf} klines...")

            df = downloader.download_klines(
                pair, tf, cfg.data.start_date, cfg.data.end_date, save=False
            )

            if df.empty:
                console.print(f"    [red]No data received for {pair} {tf}[/red]")
                continue

            # Validate
            if validate:
                is_valid, report = validator.validate_and_report(df, pair, tf)
                console.print(f"    Validation: {report.error_count} errors, {report.warning_count} warnings")
                if report.issues:
                    for issue in report.issues[:5]:  # Show first 5
                        console.print(f"      [{issue.severity}] {issue.description}")

            # Clean
            df = cleaner.clean(df, tf)
            console.print(f"    After cleaning: {len(df)} candles")

            # Save
            storage.save_data(df, pair, tf, "klines")
            console.print(f"    [green]Saved {pair} {tf}: {len(df)} candles[/green]")

        console.print()

    # Print summary
    console.print("[bold blue]Download Summary[/bold blue]")
    table = Table()
    table.add_column("Pair")
    table.add_column("Timeframe")
    table.add_column("Candles")
    table.add_column("Start")
    table.add_column("End")

    for pair in cfg.exchange.pairs:
        for tf in cfg.exchange.timeframes:
            df = storage.load_data(pair, tf, "klines")
            if df is not None and not df.empty:
                table.add_row(
                    pair, tf, str(len(df)),
                    str(df.index[0].date()), str(df.index[-1].date())
                )

    console.print(table)
    storage.close()

    console.print("[bold green]Download complete![/bold green]")


if __name__ == "__main__":
    main()
