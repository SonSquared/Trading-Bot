"""
Moving Average Crossover Strategy.

Generates long signals when fast MA crosses above slow MA,
and short signals when fast MA crosses below slow MA.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import sma, ema
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover — configurable SMA/EMA, multiple timeframe."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="MA_Crossover",
            family="trend",
            description="Fast/slow moving average crossover with optional trend filter",
            param_ranges={
                "ma_type": ["sma", "ema"],
                "fast_period": list(range(5, 51, 5)),
                "slow_period": list(range(20, 201, 10)),
                "trend_filter": [True, False],
                "trend_filter_period": [100, 200],
            },
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "ma_type": "ema",
            "fast_period": 12,
            "slow_period": 26,
            "trend_filter": True,
            "trend_filter_period": 200,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "ma_type": ["sma", "ema"],
            "fast_period": [5, 8, 10, 12, 15, 20, 25, 30],
            "slow_period": [20, 30, 40, 50, 60, 80, 100, 150, 200],
            "trend_filter": [False, True],
            "trend_filter_period": [100, 150, 200],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]
        ma_type = params.get("ma_type", "ema")
        fast = params.get("fast_period", 12)
        slow = params.get("slow_period", 26)

        ma_fn = ema if ma_type == "ema" else sma
        fast_ma = ma_fn(close, fast)
        slow_ma = ma_fn(close, slow)

        signals = pd.Series(0, index=df.index, dtype=int)

        # Long: fast crosses above slow
        long_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        signals[long_cross] = 1

        # Short: fast crosses below slow
        short_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        signals[short_cross] = -1

        # Trend filter: only trade in direction of longer MA
        if params.get("trend_filter", False):
            trend_period = params.get("trend_filter_period", 200)
            trend_ma = ma_fn(close, trend_period)
            # Block counter-trend signals
            signals[(signals == 1) & (close < trend_ma)] = 0
            signals[(signals == -1) & (close > trend_ma)] = 0

        return signals
