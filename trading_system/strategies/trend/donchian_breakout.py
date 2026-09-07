"""
Donchian Channel Breakout Strategy.

Enters long when price breaks above the upper channel,
enters short when price breaks below the lower channel.
Classic trend-following turtle system variant.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import donchian_channel
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian channel breakout with ATR-based exits."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Donchian_Breakout",
            family="trend",
            description="Breakout above/below Donchian channel with ATR-based exits",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "channel_period": 20,
            "exit_period": 10,
            "atr_filter": True,
            "atr_period": 14,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "channel_period": [10, 15, 20, 25, 30, 40, 50, 60],
            "exit_period": [5, 7, 10, 15, 20],
            "atr_filter": [False, True],
            "atr_period": [10, 14, 20],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        entry_period = params.get("channel_period", 20)
        exit_period = params.get("exit_period", 10)

        # Entry channel (longer period)
        entry_channel = donchian_channel(df, entry_period)
        # Exit channel (shorter period)
        exit_df = pd.DataFrame({"high": high, "low": low})
        exit_channel = donchian_channel(exit_df, exit_period)

        signals = pd.Series(0, index=df.index, dtype=int)

        # Long: close breaks above upper channel
        long_entry = close > entry_channel["upper"].shift(1)
        # Short: close breaks below lower channel
        short_entry = close < entry_channel["lower"].shift(1)

        signals[long_entry] = 1
        signals[short_entry] = -1

        # Exit long when close breaks below exit lower channel
        exit_long = close < exit_channel["lower"].shift(1)
        exit_short = close > exit_channel["upper"].shift(1)

        # Apply exits
        in_long = signals == 1
        in_short = signals == -1

        signals[exit_long & in_long.shift(1).fillna(False)] = 0
        signals[exit_short & in_short.shift(1).fillna(False)] = 0

        # Carry position (maintain signal until exit)
        position = signals.replace(0, pd.NA).ffill().fillna(0).astype(int)

        return position
