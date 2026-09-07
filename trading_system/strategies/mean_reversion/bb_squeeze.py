"""
Bollinger Band Squeeze Strategy.

Detects volatility contraction (squeeze) followed by expansion,
then trades in the breakout direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_system.indicators import bollinger_bands, keltner_channel
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class BBSqueezeStrategy(BaseStrategy):
    """Bollinger Band squeeze detection and breakout."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="BB_Squeeze",
            family="mean_reversion",
            description="Volatility squeeze (BB inside KC) followed by breakout direction",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "kc_period": 20,
            "kc_atr_mult": 1.5,
            "squeeze_lookback": 6,
            "momentum_period": 12,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "bb_period": [15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
            "kc_period": [15, 20, 25],
            "kc_atr_mult": [1.0, 1.5, 2.0],
            "squeeze_lookback": [3, 6, 9, 12],
            "momentum_period": [8, 12, 16],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]

        bb = bollinger_bands(
            close,
            period=params.get("bb_period", 20),
            std_dev=params.get("bb_std", 2.0),
        )

        kc = keltner_channel(
            df,
            ema_period=params.get("kc_period", 20),
            multiplier=params.get("kc_atr_mult", 1.5),
        )

        # Squeeze detection: BB inside KC
        squeeze = ((bb["lower"] > kc["lower"]) & (bb["upper"] < kc["upper"])).values

        # Momentum for direction
        mom_period = params.get("momentum_period", 12)
        momentum = (close - close.rolling(window=mom_period, min_periods=mom_period).mean()).values

        lookback = params.get("squeeze_lookback", 6)

        n = len(df)
        pos_arr = np.zeros(n, dtype=np.int32)
        position = 0
        in_squeeze = False
        squeeze_bars = 0

        for i in range(1, n):
            if squeeze[i]:
                squeeze_bars += 1
                in_squeeze = True
            else:
                if in_squeeze and squeeze_bars >= lookback:
                    mom_i = momentum[i] if np.isfinite(momentum[i]) else 0
                    if mom_i > 0:
                        position = 1
                    elif mom_i < 0:
                        position = -1
                in_squeeze = False
                squeeze_bars = 0

            if position != 0:
                mom_i = momentum[i] if np.isfinite(momentum[i]) else 0
                if position == 1 and mom_i < 0:
                    position = 0
                elif position == -1 and mom_i > 0:
                    position = 0

            pos_arr[i] = position

        return pd.Series(pos_arr, index=df.index, dtype=int)
