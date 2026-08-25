"""
ADX Trend Following Strategy.

Uses ADX to determine trend strength, and +DI/-DI for direction.
Only enters trades when a clear trend is present.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import adx as adx_indicator, ema
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class ADXTrendStrategy(BaseStrategy):
    """ADX-based trend following with directional filters."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ADX_Trend",
            family="trend",
            description="Trend following using ADX strength and DI direction",
            param_ranges={
                "adx_period": list(range(10, 25, 2)),
                "adx_threshold": list(range(20, 36, 5)),
                "use_ema_filter": [True, False],
                "ema_period": [50, 100, 200],
            },
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "adx_period": 14,
            "adx_threshold": 25,
            "use_ema_filter": True,
            "ema_period": 100,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "adx_period": [10, 12, 14, 16, 18, 20],
            "adx_threshold": [20, 22, 25, 28, 30, 35],
            "use_ema_filter": [False, True],
            "ema_period": [50, 100, 150, 200],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]
        adx_df = adx_indicator(df, period=params.get("adx_period", 14))

        adx_val = adx_df["adx"]
        plus_di = adx_df["plus_di"]
        minus_di = adx_df["minus_di"]
        threshold = params.get("adx_threshold", 25)

        signals = pd.Series(0, index=df.index, dtype=int)

        # Trending conditions
        trending = adx_val > threshold

        # Long: +DI > -DI and trending up
        long_cond = trending & (plus_di > minus_di)
        # Short: -DI > +DI and trending down
        short_cond = trending & (minus_di > plus_di)

        # Generate crossover signals (enter when trend starts)
        long_start = long_cond & (~long_cond.shift(1).fillna(False))
        short_start = short_cond & (~short_cond.shift(1).fillna(False))

        signals[long_start] = 1
        signals[short_start] = -1

        # Exit when trend ends
        signals[(~long_cond) & (signals.shift(1).fillna(0).isin([1]))] = 0
        signals[(~short_cond) & (signals.shift(1).fillna(0).isin([-1]))] = 0

        # Optional EMA filter
        if params.get("use_ema_filter", True):
            ema_period = params.get("ema_period", 100)
            ema_val = ema(close, ema_period)
            signals[(signals == 1) & (close < ema_val)] = 0
            signals[(signals == -1) & (close > ema_val)] = 0

        return signals
