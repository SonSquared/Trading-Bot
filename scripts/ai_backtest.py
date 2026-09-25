#!/usr/bin/env python3
"""AI Trading Bot — mechanical backtest of the execution skeleton.

Replays the bot's own wakeup loop (the production ``AIAgent`` in paper mode) over
historical candles: the six schedule slots a day, the venue's min-notional and
lot-step filters, taker fees, the risk/exposure caps, stop/target geometry, and
the drawdown halt — on a $97 account.

    python scripts/ai_backtest.py                    # default suite, real data
    python scripts/ai_backtest.py --synthetic        # no data files needed
    python scripts/ai_backtest.py --list-sources
    python scripts/ai_backtest.py --source random --seed 7 --quick
    python scripts/ai_backtest.py --source trend --trigger-model intrabar
    python scripts/ai_backtest.py --source journal --journal decisions.jsonl
    python scripts/ai_backtest.py --json results.json

WHAT THE DECISION SOURCE IS, AND IS NOT
---------------------------------------
The strategy's judgement is supplied by a pluggable *decision source*. The
shipped sources are mechanical rules and coin flips, clearly labelled as such,
plus a replay of decisions the model actually made. An LLM cannot be backtested
without look-ahead bias — its output depends on everything it believes at the
moment it is asked — so this tool deliberately does not pretend to test it. It
tests the machinery around the decision, which is the part that can be measured
and the part that loses money when it is wrong.

Zero network, zero LLM calls, no orders. Read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from trading_system.bot.backtest import BacktestConfig, run_suite
from trading_system.bot.backtest.report import render
from trading_system.bot.backtest.sources import build_source, source_catalog

DEFAULT_SOURCES = "null,random,trend,rsi-reversion"


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@click.command()
@click.option("--config", default="configs/ai_bot.yaml",
              help="AI bot config (account size, risk rules, pairs, schedule)")
@click.option("--data-dir", default="data/raw",
              help="Directory holding <SYMBOL>/klines_<timeframe>.parquet")
@click.option("--pairs", default="", help="Comma-separated pairs (default: config)")
@click.option("--timeframe", default="", help="Candle size (default: config)")
@click.option("--start", default="", help="First slot (YYYY-MM-DD, default: data start)")
@click.option("--end", default="", help="Last slot (YYYY-MM-DD, default: data end)")
@click.option("--days", default=0, show_default=True,
              help="Trim the span to the last N days of the data (0 = all of it)")
@click.option("--source", "sources", default=DEFAULT_SOURCES, show_default=True,
              help="Comma-separated decision sources (see --list-sources)")
@click.option("--seed", default=7, show_default=True, help="Seed for the random source")
@click.option("--trigger-model", type=click.Choice(["slot", "intrabar"]),
              default="slot", show_default=True,
              help="How stops/targets fill (slot = the bot's paper behaviour)")
@click.option("--journal", default="", help="Decisions JSONL for --source journal")
@click.option("--no-folds", is_flag=True, help="Skip the walk-forward folds")
@click.option("--no-regimes", is_flag=True, help="Skip the per-regime windows")
@click.option("--folds", default=4, show_default=True,
              help="Walk-forward windows across the span")
@click.option("--quick", is_flag=True,
              help="Short run: last 365 days, at most 2 folds, no regimes")
@click.option("--synthetic", is_flag=True,
              help="Generate deterministic candles instead of reading data files")
@click.option("--synthetic-days", default=400, show_default=True)
@click.option("--forming-candle", is_flag=True,
              help="Include a synthetic current candle (as production's fetch does)")
@click.option("--json", "json_out", default="",
              help="Write JSON to this path ('-' = stdout instead of text)")
@click.option("--with-curve", is_flag=True, help="Include equity curves in JSON")
@click.option("--keep-workdir", is_flag=True, help="Keep the replay's data dirs")
@click.option("--list-sources", is_flag=True, help="Describe the decision sources")
def main(**opts) -> None:
    """Replay the AI bot's execution mechanics over historical candles."""
    _configure_stdout()

    if opts["list_sources"]:
        click.echo("Decision sources (provenance is printed with every result):\n")
        for entry in source_catalog():
            click.echo(f"  {entry['name']:<15} [{entry['label']}]")
            click.echo(f"      {entry['description']}")
            click.echo(f"      fitted to data: {'YES' if entry['fitted'] else 'no'}")
            click.echo(f"      -> {entry['caveat']}\n")
        return

    start = opts["start"] or ""
    end = opts["end"] or ""
    folds = opts["folds"]
    days = opts["days"]
    regimes = not opts["no_regimes"]
    if opts["quick"]:
        # A short run has to be short in the dimension that costs time: the
        # span. Fewer folds alone still replays all 4.6 years (and an explicit
        # --days or --start/--end still wins over the 365-day default).
        folds = min(folds, 2) if not opts["no_folds"] else 0
        regimes = False
        if not days and not start:
            days = 365

    journal_path = Path(opts["journal"]) if opts["journal"] else None
    shared = dict(
        data_dir=Path(opts["data_dir"]),
        config_path=Path(opts["config"]),
        pairs=[p.strip() for p in opts["pairs"].split(",") if p.strip()],
        timeframe=opts["timeframe"],
        start=start or None,
        end=end or None,
        days=days or None,
        folds=folds,
        trigger_model=opts["trigger_model"],
        regimes=regimes,
        folds_enabled=not opts["no_folds"] and folds > 0,
        forming_candle=opts["forming_candle"],
        journal_path=journal_path,
        keep_workdir=opts["keep_workdir"],
        synthetic=opts["synthetic"],
        synthetic_days=opts["synthetic_days"],
    )

    names = [s.strip() for s in opts["sources"].split(",") if s.strip()]
    configs = []
    for name in names:
        params = {"seed": opts["seed"]} if name == "random" else {}
        # Fail early and clearly on a typo, before any data is loaded.
        try:
            build_source(name, **({} if name == "journal" else params))
        except ValueError as e:
            raise click.ClickException(str(e)) from e
        configs.append(BacktestConfig(source=name, source_params=params, **shared))

    command = "python scripts/ai_backtest.py " + " ".join(sys.argv[1:])
    results = run_suite(configs)

    if opts["json_out"]:
        payload = [r.as_dict(with_curve=opts["with_curve"]) for r in results]
        text = json.dumps(payload, indent=2, default=str)
        if opts["json_out"] == "-":
            click.echo(text)
            return
        Path(opts["json_out"]).write_text(text, encoding="utf-8")
        click.echo(f"JSON written to {opts['json_out']}")

    click.echo(render(results, command=command))


if __name__ == "__main__":
    main()
