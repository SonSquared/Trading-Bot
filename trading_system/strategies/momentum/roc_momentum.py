"""
Rate of Change (ROC) Momentum Strategy.

Enters based on ROC crossing zero with confirmation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import roc, ema, atr
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class ROCMomentumStrategy(BaseStrategy):
    """Rate of change momentum with trend and volatility filters."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ROC_Momentum",
            family="momentum",
            description="Momentum based on rate of change crossing zero",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "roc_period": 12,
            "roc_threshold": 0,
            "smooth_period": 3,
            "trend_filter": True,
            "trend_ema": 100,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "roc_period": [6, 8, 10, 12, 15, 20, 25],
            "roc_threshold": [-1, 0, 1],
            "smooth_period": [1, 3, 5],
            "trend_filter": [False, True],
            "trend_ema": [50, 100, 150],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]
        roc_val = roc(close, params.get("roc_period", 12))

        # Smooth the ROC
        smooth = params.get("smooth_period", 1)
        if smooth > 1:
            roc_smooth = roc_val.rolling(window=smooth, min_periods=1).mean()
        else:
            roc_smooth = roc_val

        threshold = params.get("roc_threshold", 0)

        signals = pd.Series(0, index=df.index, dtype=int)

        # Long: ROC crosses above threshold
        long_cross = (roc_smooth > threshold) & (roc_smooth.shift(1) <= threshold)
        signals[long_cross] = 1

        # Short: ROC crosses below threshold
        short_cross = (roc_smooth < threshold) & (roc_smooth.shift(1) >= threshold)
        signals[short_cross] = -1

        # Trend filter
        if params.get("trend_filter", True):
            ema_period = params.get("trend_ema", 100)
            trend = ema(close, ema_period)
            signals[(signals == 1) & (close < trend)] = 0
            signals[(signals == -1) & (close > trend)] = 0

        return signals
