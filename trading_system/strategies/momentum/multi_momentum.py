"""
Multi-Momentum Composite Strategy.

Combines RSI, ROC, and CCI momentum signals.
Requires majority vote for entry.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import rsi, roc, cci, ema
from trading_system.strategies.base import BaseStrategy, StrategyMeta


class MultiMomentumStrategy(BaseStrategy):
    """Multi-indicator momentum with voting."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Multi_Momentum",
            family="momentum",
            description="Composite momentum using RSI, ROC, and CCI with majority vote",
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "rsi_period": 14,
            "rsi_threshold": 50,
            "roc_period": 12,
            "cci_period": 20,
            "cci_threshold": 0,
            "min_votes": 2,
            "trend_ema": 200,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "rsi_period": [10, 14, 21],
            "rsi_threshold": [45, 50, 55],
            "roc_period": [8, 12, 16],
            "cci_period": [14, 20, 28],
            "cci_threshold": [-50, 0, 50],
            "min_votes": [1, 2, 3],
            "trend_ema": [100, 200],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        close = df["close"]

        # Individual signals
        rsi_val = rsi(close, params.get("rsi_period", 14))
        roc_val = roc(close, params.get("roc_period", 12))
        cci_val = cci(df, params.get("cci_period", 20))

        rsi_thresh = params.get("rsi_threshold", 50)
        cci_thresh = params.get("cci_threshold", 0)

        # RSI vote: 1 if above threshold, -1 if below
        rsi_vote = pd.Series(0, index=df.index, dtype=int)
        rsi_vote[rsi_val > rsi_thresh] = 1
        rsi_vote[rsi_val < (100 - rsi_thresh)] = -1

        # ROC vote: 1 if positive, -1 if negative
        roc_vote = pd.Series(0, index=df.index, dtype=int)
        roc_vote[roc_val > 0] = 1
        roc_vote[roc_val < 0] = -1

        # CCI vote
        cci_vote = pd.Series(0, index=df.index, dtype=int)
        cci_vote[cci_val > cci_thresh] = 1
        cci_vote[cci_val < -cci_thresh] = -1

        # Majority vote
        total_votes = rsi_vote + roc_vote + cci_vote
        min_votes = params.get("min_votes", 2)

        signals = pd.Series(0, index=df.index, dtype=int)
        signals[total_votes >= min_votes] = 1
        signals[total_votes <= -min_votes] = -1

        # Trend filter
        ema_val = ema(close, params.get("trend_ema", 200))
        signals[(signals == 1) & (close < ema_val)] = 0
        signals[(signals == -1) & (close > ema_val)] = 0

        return signals
