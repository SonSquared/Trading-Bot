"""
Bollinger Band Mean Reversion Strategy.

Buys when price touches or breaks below the lower band,
sells when price returns to the middle band.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_system.indicators import bollinger_bands, rsi
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class BollingerReversionStrategy(BaseStrategy):
    """Bollinger Band mean reversion with RSI confirmation."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Bollinger_Reversion",
            family="mean_reversion",
            description="Buy at lower BB, sell at middle BB, with RSI confirmation",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_filter": True,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "exit_at_middle": True,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "bb_period": [15, 20, 25, 30],
            "bb_std": [1.5, 2.0, 2.5, 3.0],
            "rsi_filter": [False, True],
            "rsi_period": [10, 14, 21],
            "rsi_oversold": [25, 30, 35],
            "rsi_overbought": [65, 70, 75],
            "exit_at_middle": [True, False],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]
        bb = bollinger_bands(
            close,
            period=params.get("bb_period", 20),
            std_dev=params.get("bb_std", 2.0),
        )

        rsi_filter = params.get("rsi_filter", True)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_overbought = params.get("rsi_overbought", 70)

        # Pre-compute all values as numpy arrays (much faster than .iloc)
        close_arr = close.values.astype(np.float64)
        lower_arr = bb["lower"].values.astype(np.float64)
        upper_arr = bb["upper"].values.astype(np.float64)
        middle_arr = bb["middle"].values.astype(np.float64)

        if rsi_filter:
            rsi_val = rsi(close, params.get("rsi_period", 14)).values.astype(np.float64)

        exit_at_middle = params.get("exit_at_middle", True)
        n = len(df)
        pos_arr = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(1, n):
            if position == 0:
                # Long: price touches lower band
                if close_arr[i] <= lower_arr[i]:
                    if rsi_filter and rsi_val[i] > rsi_oversold + 10:
                        pass  # RSI not oversold enough
                    else:
                        position = 1

                # Short: price touches upper band
                elif close_arr[i] >= upper_arr[i]:
                    if rsi_filter and rsi_val[i] < rsi_overbought - 10:
                        pass
                    else:
                        position = -1

            elif position == 1:
                if exit_at_middle:
                    if close_arr[i] >= middle_arr[i]:
                        position = 0
                else:
                    if rsi_filter and rsi_val[i] > rsi_overbought:
                        position = 0

            elif position == -1:
                if exit_at_middle:
                    if close_arr[i] <= middle_arr[i]:
                        position = 0
                else:
                    if rsi_filter and rsi_val[i] < rsi_oversold:
                        position = 0

            pos_arr[i] = position

        return pd.Series(pos_arr, index=df.index, dtype=int)
