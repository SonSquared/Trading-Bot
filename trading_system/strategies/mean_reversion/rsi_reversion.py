"""
RSI Mean Reversion Strategy.

Buys when RSI is deeply oversold, sells when RSI returns to neutral.
The inverse for shorts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_system.indicators import rsi, atr, bollinger_bands
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class RSIReversionStrategy(BaseStrategy):
    """RSI mean reversion with oversold/overbought entries."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="RSI_Reversion",
            family="mean_reversion",
            description="Buy oversold, sell overbought RSI with mean-reversion exit",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "rsi_period": 14,
            "entry_oversold": 30,
            "entry_overbought": 70,
            "exit_neutral_low": 45,
            "exit_neutral_high": 55,
            "use_bb_filter": True,
            "bb_period": 20,
            "bb_std": 2.0,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "rsi_period": [7, 10, 14, 21],
            "entry_oversold": [20, 25, 30, 35],
            "entry_overbought": [65, 70, 75, 80],
            "exit_neutral_low": [40, 45, 50],
            "exit_neutral_high": [50, 55, 60],
            "use_bb_filter": [False, True],
            "bb_period": [15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]
        rsi_val = rsi(close, params.get("rsi_period", 14)).values.astype(np.float64)

        entry_oversold = params.get("entry_oversold", 30)
        entry_overbought = params.get("entry_overbought", 70)
        exit_low = params.get("exit_neutral_low", 45)
        exit_high = params.get("exit_neutral_high", 55)

        bb_filter = params.get("use_bb_filter", True)
        close_arr = close.values.astype(np.float64)
        if bb_filter:
            bb = bollinger_bands(close, params.get("bb_period", 20), params.get("bb_std", 2.0))
            bb_lower = bb["lower"].values.astype(np.float64)
            bb_upper = bb["upper"].values.astype(np.float64)

        n = len(df)
        pos_arr = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(1, n):
            r = rsi_val[i]
            if np.isnan(r):
                pos_arr[i] = position
                continue

            if position == 0:
                if r < entry_oversold:
                    if bb_filter:
                        if close_arr[i] <= bb_lower[i] * 1.02:
                            position = 1
                    else:
                        position = 1
                elif r > entry_overbought:
                    if bb_filter:
                        if close_arr[i] >= bb_upper[i] * 0.98:
                            position = -1
                    else:
                        position = -1

            elif position == 1:
                if r > exit_high:
                    position = 0

            elif position == -1:
                if r < exit_low:
                    position = 0

            pos_arr[i] = position

        return pd.Series(pos_arr, index=df.index, dtype=int)
