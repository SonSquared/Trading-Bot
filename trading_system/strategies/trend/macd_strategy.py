"""
MACD Strategy.

Generates signals based on MACD line crossing the signal line,
with optional histogram momentum confirmation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import macd
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class MACDStrategy(BaseStrategy):
    """MACD crossover with optional histogram filter."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="MACD",
            family="trend",
            description="MACD line/signal crossover with optional histogram confirmation",
            param_ranges={
                "fast": list(range(8, 21, 2)),
                "slow": list(range(20, 35, 2)),
                "signal": list(range(5, 13, 2)),
                "use_histogram": [True, False],
            },
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "fast": 12,
            "slow": 26,
            "signal": 9,
            "use_histogram": False,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "fast": [8, 10, 12, 15, 18],
            "slow": [20, 24, 26, 30, 34],
            "signal": [5, 7, 9, 11],
            "use_histogram": [False, True],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]

        macd_df = macd(
            close,
            fast=params.get("fast", 12),
            slow=params.get("slow", 26),
            signal=params.get("signal", 9),
        )

        macd_line = macd_df["macd"]
        signal_line = macd_df["signal"]
        histogram = macd_df["histogram"]

        signals = pd.Series(0, index=df.index, dtype=int)

        # Long: MACD crosses above signal line
        long_cross = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        signals[long_cross] = 1

        # Short: MACD crosses below signal line
        short_cross = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        signals[short_cross] = -1

        # Optional histogram confirmation
        if params.get("use_histogram", False):
            signals[(signals == 1) & (histogram <= 0)] = 0
            signals[(signals == -1) & (histogram >= 0)] = 0

        return signals
