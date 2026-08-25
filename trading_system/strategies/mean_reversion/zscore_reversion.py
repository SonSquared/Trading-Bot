"""
Z-Score Mean Reversion Strategy.

Enters when the price Z-score deviates significantly from the mean,
exits when it reverts to the mean.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_system.indicators.utils import zscore
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class ZScoreReversionStrategy(BaseStrategy):
    """Z-score based mean reversion."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="ZScore_Reversion",
            family="mean_reversion",
            description="Enter when Z-score is extreme, exit at mean reversion",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "lookback": 20,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "use_sma_baseline": True,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "lookback": [15, 20, 30, 40, 50],
            "entry_threshold": [1.5, 2.0, 2.5, 3.0],
            "exit_threshold": [0.0, 0.25, 0.5, 0.75],
            "use_sma_baseline": [True, False],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]
        lookback = params.get("lookback", 20)
        zs = zscore(close, lookback).values.astype(np.float64)

        entry_thresh = params.get("entry_threshold", 2.0)
        exit_thresh = params.get("exit_threshold", 0.5)

        n = len(df)
        pos_arr = np.zeros(n, dtype=np.int32)
        position = 0

        for i in range(1, n):
            z = zs[i]
            if np.isnan(z):
                pos_arr[i] = position
                continue

            if position == 0:
                if z < -entry_thresh:
                    position = 1
                elif z > entry_thresh:
                    position = -1
            elif position == 1:
                if z > -exit_thresh:
                    position = 0
            elif position == -1:
                if z < exit_thresh:
                    position = 0

            pos_arr[i] = position

        return pd.Series(pos_arr, index=df.index, dtype=int)
