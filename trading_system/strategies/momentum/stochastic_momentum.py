"""
Stochastic Momentum Strategy.

Uses Stochastic Oscillator with %K/%D crossover signals.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import stochastic, ema
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class StochasticMomentumStrategy(BaseStrategy):
    """Stochastic oscillator momentum strategy."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Stochastic_Momentum",
            family="momentum",
            description="Stochastic %K/%D crossover with zone filters",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "k_period": 14,
            "d_period": 3,
            "smooth_k": 3,
            "oversold": 20,
            "overbought": 80,
            "trend_filter": True,
            "trend_ema": 200,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "k_period": [9, 14, 21],
            "d_period": [3, 5, 7],
            "smooth_k": [3, 5],
            "oversold": [15, 20, 25],
            "overbought": [75, 80, 85],
            "trend_filter": [False, True],
            "trend_ema": [100, 200],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]
        stoch_df = stochastic(
            df,
            k_period=params.get("k_period", 14),
            d_period=params.get("d_period", 3),
            smooth_k=params.get("smooth_k", 3),
        )

        k = stoch_df["k"]
        d = stoch_df["d"]
        oversold = params.get("oversold", 20)
        overbought = params.get("overbought", 80)

        signals = pd.Series(0, index=df.index, dtype=int)

        # Long: %K crosses above %D in oversold zone
        long_cross = (k > d) & (k.shift(1) <= d.shift(1)) & (k < oversold + 10)
        signals[long_cross] = 1

        # Short: %K crosses below %D in overbought zone
        short_cross = (k < d) & (k.shift(1) >= d.shift(1)) & (k > overbought - 10)
        signals[short_cross] = -1

        # Trend filter
        if params.get("trend_filter", True):
            ema_val = ema(close, params.get("trend_ema", 200))
            signals[(signals == 1) & (close < ema_val)] = 0
            signals[(signals == -1) & (close > ema_val)] = 0

        return signals
