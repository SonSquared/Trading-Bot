"""
RSI Momentum Strategy.

Enters long when RSI crosses above oversold threshold with momentum,
enters short when RSI crosses below overbought threshold with momentum.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import rsi, ema
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class RSIMomentumStrategy(BaseStrategy):
    """RSI-based momentum strategy with trend filter."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="RSI_Momentum",
            family="momentum",
            description="RSI momentum with trend filter and configurable thresholds",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "use_trend_filter": True,
            "trend_ema": 200,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "rsi_period": [7, 10, 14, 18, 21],
            "oversold": [20, 25, 30, 35, 40],
            "overbought": [60, 65, 70, 75, 80],
            "use_trend_filter": [False, True],
            "trend_ema": [100, 150, 200],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]
        rsi_val = rsi(close, params.get("rsi_period", 14))

        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)

        signals = pd.Series(0, index=df.index, dtype=int)

        # Long: RSI crosses above oversold
        long_cross = (rsi_val > oversold) & (rsi_val.shift(1) <= oversold)
        signals[long_cross] = 1

        # Short: RSI crosses below overbought
        short_cross = (rsi_val < overbought) & (rsi_val.shift(1) >= overbought)
        signals[short_cross] = -1

        # Trend filter
        if params.get("use_trend_filter", True):
            ema_period = params.get("trend_ema", 200)
            trend = ema(close, ema_period)
            signals[(signals == 1) & (close < trend)] = 0
            signals[(signals == -1) & (close > trend)] = 0

        return signals
