"""
Supertrend Strategy.

Uses the Supertrend indicator for direction changes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import supertrend as supertrend_indicator, atr
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class SupertrendStrategy(BaseStrategy):
    """Supertrend direction-based trend following."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Supertrend",
            family="trend",
            description="Trend following based on Supertrend direction changes",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "period": 10,
            "multiplier": 3.0,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "period": [7, 10, 12, 14, 18, 20, 25],
            "multiplier": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        st_df = supertrend_indicator(
            df,
            period=params.get("period", 10),
            multiplier=params.get("multiplier", 3.0),
        )

        direction = st_df["direction"]

        signals = pd.Series(0, index=df.index, dtype=int)

        # Direction change from -1 to 1 = long entry
        long_entry = (direction == 1) & (direction.shift(1) == -1)
        # Direction change from 1 to -1 = short entry
        short_entry = (direction == -1) & (direction.shift(1) == 1)

        signals[long_entry] = 1
        signals[short_entry] = -1

        # Carry position
        position = signals.replace(0, pd.NA).ffill().fillna(0).astype(int)

        return position
