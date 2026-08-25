"""
Volatility Regime Strategy.

Adapts trading behavior based on the current volatility regime.
Trending in high volatility, mean-reverting in low volatility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_system.indicators import (
    atr, bollinger_bands, rsi, volatility_regime, ema, historical_volatility,
)
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class RegimeVolatilityStrategy(BaseStrategy):
    """Adaptive strategy based on volatility regime detection."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Regime_Volatility",
            family="volatility",
            description="Trend-following in high vol, mean-reversion in low vol",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "short_atr": 10,
            "long_atr": 50,
            "high_vol_threshold": 1.2,
            "low_vol_threshold": 0.8,
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "ema_fast": 10,
            "ema_slow": 30,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "short_atr": [5, 10, 15],
            "long_atr": [30, 50, 100],
            "high_vol_threshold": [1.0, 1.2, 1.5],
            "low_vol_threshold": [0.5, 0.7, 0.8],
            "bb_period": [15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
            "rsi_period": [10, 14, 21],
            "ema_fast": [5, 10, 15],
            "ema_slow": [20, 30, 50],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]

        vol_ratio = volatility_regime(
            df,
            short_period=params.get("short_atr", 10),
            long_period=params.get("long_atr", 50),
        )

        high_thresh = params.get("high_vol_threshold", 1.2)
        low_thresh = params.get("low_vol_threshold", 0.8)

        bb = bollinger_bands(
            close,
            period=params.get("bb_period", 20),
            std_dev=params.get("bb_std", 2.0),
        )
        rsi_val = rsi(close, params.get("rsi_period", 14))
        ema_fast_val = ema(close, params.get("ema_fast", 10))
        ema_slow_val = ema(close, params.get("ema_slow", 30))

        # Convert to numpy arrays for speed
        close_arr = close.values.astype(np.float64)
        vr_arr = vol_ratio.values.astype(np.float64)
        bb_lower_arr = bb["lower"].values.astype(np.float64)
        bb_upper_arr = bb["upper"].values.astype(np.float64)
        rsi_arr = rsi_val.values.astype(np.float64)
        ema_fast_arr = ema_fast_val.values.astype(np.float64)
        ema_slow_arr = ema_slow_val.values.astype(np.float64)

        n = len(df)
        signals_arr = np.zeros(n, dtype=np.int32)

        for i in range(1, n):
            vr = vr_arr[i]
            if np.isnan(vr):
                continue

            if vr > high_thresh:
                # HIGH VOLATILITY: EMA crossover
                if ema_fast_arr[i] > ema_slow_arr[i] and ema_fast_arr[i - 1] <= ema_slow_arr[i - 1]:
                    signals_arr[i] = 1
                elif ema_fast_arr[i] < ema_slow_arr[i] and ema_fast_arr[i - 1] >= ema_slow_arr[i - 1]:
                    signals_arr[i] = -1

            elif vr < low_thresh:
                # LOW VOLATILITY: BB + RSI mean reversion
                rsi_i = rsi_arr[i] if np.isfinite(rsi_arr[i]) else 50
                if close_arr[i] < bb_lower_arr[i] and rsi_i < 35:
                    signals_arr[i] = 1
                elif close_arr[i] > bb_upper_arr[i] and rsi_i > 65:
                    signals_arr[i] = -1

        return pd.Series(signals_arr, index=df.index, dtype=int)
