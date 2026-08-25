"""
Volatility Expansion Strategy.

Enters when volatility expands from a low-volatility regime,
trading in the direction of the initial move.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import atr, bollinger_bands, ema
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class VolExpansionStrategy(BaseStrategy):
    """Trade breakouts from volatility contraction."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Vol_Expansion",
            family="volatility",
            description="Enter on volatility expansion from contraction regime",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "squeeze_threshold": 0.03,
            "expansion_mult": 1.5,
            "atr_period": 14,
            "trend_ema": 50,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "bb_period": [15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
            "squeeze_threshold": [0.02, 0.03, 0.04, 0.05],
            "expansion_mult": [1.2, 1.5, 2.0],
            "atr_period": [10, 14, 20],
            "trend_ema": [30, 50, 100],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        import numpy as np
        close = df["close"]

        # Bollinger Band width as volatility measure
        bb = bollinger_bands(
            close,
            period=params.get("bb_period", 20),
            std_dev=params.get("bb_std", 2.0),
        )
        bb_width = bb["bandwidth"].values

        # ATR for direction and exits
        atr_val = atr(df, params.get("atr_period", 14)).values
        close_arr = close.values
        atr_pct = np.where(close_arr > 0, atr_val / close_arr, 0.0)

        # Pre-compute EMA (was previously computed inside the loop — O(n²) bug)
        ema_val = ema(close, params.get("trend_ema", 50)).values

        # Volatility state
        squeeze_thresh = params.get("squeeze_threshold", 0.03)
        exp_mult = params.get("expansion_mult", 1.5)

        n = len(df)
        pos_arr = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(2, n):
            was_squeezed = bb_width[i - 2] < squeeze_thresh if np.isfinite(bb_width[i - 2]) else False
            is_expanding = atr_pct[i] > atr_pct[i - 1] * exp_mult if np.isfinite(atr_pct[i]) and np.isfinite(atr_pct[i-1]) else False
            close_val = close_arr[i]

            if position == 0:
                if was_squeezed and is_expanding:
                    if close_val > close_arr[i - 1]:
                        position = 1
                    elif close_val < close_arr[i - 1]:
                        position = -1

            elif position == 1:
                ema_i = ema_val[i] if np.isfinite(ema_val[i]) else close_val
                if close_val < ema_i or (np.isfinite(atr_pct[i]) and np.isfinite(atr_pct[i-1]) and atr_pct[i] < atr_pct[i - 1] * 0.8):
                    position = 0

            elif position == -1:
                ema_i = ema_val[i] if np.isfinite(ema_val[i]) else close_val
                if close_val > ema_i or (np.isfinite(atr_pct[i]) and np.isfinite(atr_pct[i-1]) and atr_pct[i] < atr_pct[i - 1] * 0.8):
                    position = 0

            pos_arr[i] = position

        return pd.Series(pos_arr, index=df.index, dtype=int)
