"""
ATR Breakout Strategy.

Enters when price breaks out beyond ATR-based bands from the recent range.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_system.indicators import atr, ema
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class ATRBreakoutStrategy(BaseStrategy):
    """ATR-based channel breakout."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ATR_Breakout",
            family="volatility",
            description="Breakout beyond ATR-adjusted price channels",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "lookback": 20,
            "atr_period": 14,
            "atr_mult": 1.5,
            "use_close_only": True,
            "exit_atr_mult": 2.0,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "lookback": [10, 15, 20, 30, 40],
            "atr_period": [10, 14, 20],
            "atr_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
            "use_close_only": [True, False],
            "exit_atr_mult": [1.5, 2.0, 2.5, 3.0],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        lookback = params.get("lookback", 20)
        atr_period = params.get("atr_period", 14)
        atr_mult = params.get("atr_mult", 1.5)

        atr_val = atr(df, atr_period)

        if params.get("use_close_only", True):
            channel_high = close.rolling(window=lookback, min_periods=lookback).max()
            channel_low = close.rolling(window=lookback, min_periods=lookback).min()
        else:
            channel_high = high.rolling(window=lookback, min_periods=lookback).max()
            channel_low = low.rolling(window=lookback, min_periods=lookback).min()

        upper_break = (channel_high + atr_mult * atr_val).values
        lower_break = (channel_low - atr_mult * atr_val).values

        exit_mult = params.get("exit_atr_mult", 2.0)
        mid = ((channel_high + channel_low) / 2).values
        atr_arr = atr_val.values.astype(np.float64)
        close_arr = close.values.astype(np.float64)

        n = len(df)
        pos_arr = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(1, n):
            if position == 0:
                if close_arr[i] > upper_break[i - 1]:
                    position = 1
                elif close_arr[i] < lower_break[i - 1]:
                    position = -1

            elif position == 1:
                atr_i = atr_arr[i] if np.isfinite(atr_arr[i]) else 0
                if close_arr[i] < mid[i] - exit_mult * atr_i:
                    position = 0

            elif position == -1:
                atr_i = atr_arr[i] if np.isfinite(atr_arr[i]) else 0
                if close_arr[i] > mid[i] + exit_mult * atr_i:
                    position = 0

            pos_arr[i] = position

        return pd.Series(pos_arr, index=df.index, dtype=int)
