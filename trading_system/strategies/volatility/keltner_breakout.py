"""
Keltner Channel Breakout Strategy.

Enters when price breaks out of the Keltner Channel,
indicating a strong directional move beyond normal volatility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_system.indicators import keltner_channel, atr
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class KeltnerBreakoutStrategy(BaseStrategy):
    """Keltner Channel breakout with ATR-based entries and exits."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Keltner_Breakout",
            family="volatility",
            description="Breakout above/below Keltner Channel",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "ema_period": 20,
            "atr_period": 10,
            "multiplier": 2.0,
            "exit_multiplier": 1.0,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "ema_period": [15, 20, 25, 30],
            "atr_period": [10, 14, 20],
            "multiplier": [1.5, 2.0, 2.5, 3.0],
            "exit_multiplier": [0.5, 1.0, 1.5],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]

        kc = keltner_channel(
            df,
            ema_period=params.get("ema_period", 20),
            atr_period=params.get("atr_period", 10),
            multiplier=params.get("multiplier", 2.0),
        )

        exit_mult = params.get("exit_multiplier", 1.0)
        atr_val = atr(df, params.get("atr_period", 10))

        # Pre-compute as numpy arrays
        close_arr = close.values.astype(np.float64)
        upper_arr = kc["upper"].values.astype(np.float64)
        lower_arr = kc["lower"].values.astype(np.float64)
        middle_arr = kc["middle"].values.astype(np.float64)
        atr_arr = atr_val.values.astype(np.float64)

        n = len(df)
        pos_arr = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(1, n):
            if position == 0:
                if close_arr[i] > upper_arr[i - 1]:
                    position = 1
                elif close_arr[i] < lower_arr[i - 1]:
                    position = -1

            elif position == 1:
                atr_i = atr_arr[i] if np.isfinite(atr_arr[i]) else 0
                exit_level = middle_arr[i] - exit_mult * atr_i
                if close_arr[i] < exit_level:
                    position = 0

            elif position == -1:
                atr_i = atr_arr[i] if np.isfinite(atr_arr[i]) else 0
                exit_level = middle_arr[i] + exit_mult * atr_i
                if close_arr[i] > exit_level:
                    position = 0

            pos_arr[i] = position

        return pd.Series(pos_arr, index=df.index, dtype=int)
