"""Replay the AI bot's real mechanics on historical data.

The AI bot's own paper mode is already a simulator — of the account. What was
missing is a clock: a way to run that same code against the past. This package
supplies exactly that and nothing more:

  historical.py  the ExchangeInterface surface, served from stored OHLCV at a
                 moving cursor (strictly closed candles, no look-ahead)
  sources.py     pluggable decision sources, each labelled mechanical / null /
                 recorded-LLM / test
  engine.py      drives the REAL AIAgent.run_wakeup() once per schedule slot on
                 a real PaperLedger, and aggregates the result
  report.py      per-window/per-regime numbers plus a plain-English verdict

The production classes are used as-is (AIAgent, PaperLedger, _validate_decision,
venue_limits), so validation, sizing, venue filters, fees and stop geometry
cannot drift from what the bot actually does. What is *not* modelled — and is
listed in every report — is everything about judgement: the LLM's calls, real
fill slippage, funding and liquidity.
"""

from trading_system.bot.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    WindowResult,
    run_backtest,
    run_suite,
)
from trading_system.bot.backtest.historical import (
    HistoricalExchange,
    OrderPathTouched,
    load_ohlcv,
    symbol_for,
)
from trading_system.bot.backtest.sources import (
    DecisionSource,
    JournalSource,
    NullSource,
    RandomSource,
    RsiReversionSource,
    ScriptedSource,
    SlotContext,
    TrendSource,
    build_source,
    source_catalog,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "DecisionSource",
    "HistoricalExchange",
    "JournalSource",
    "NullSource",
    "OrderPathTouched",
    "RandomSource",
    "RsiReversionSource",
    "ScriptedSource",
    "SlotContext",
    "TrendSource",
    "WindowResult",
    "build_source",
    "load_ohlcv",
    "run_backtest",
    "run_suite",
    "source_catalog",
    "symbol_for",
]
