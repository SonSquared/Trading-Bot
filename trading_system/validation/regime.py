"""
Market regime detection and strategy performance analysis by regime.

Identifies: bull, bear, sideways, high-vol, low-vol regimes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from trading_system.indicators import atr, historical_volatility, ema

logger = structlog.get_logger(__name__)


class RegimeAnalyzer:
    """Market regime detection and strategy analysis by regime."""

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def detect_regimes(
        self,
        df: pd.DataFrame,
        vol_short: int = 20,
        vol_long: int = 100,
    ) -> pd.DataFrame:
        """
        Detect market regimes based on trend and volatility.

        Regimes: "bull", "bear", "sideways", "high_vol", "low_vol"
        """
        close = df["close"]

        # Trend detection using EMAs
        ema_short = ema(close, 20)
        ema_long = ema(close, 50)

        # Volatility regime
        atr_short = atr(df, vol_short)
        atr_long = atr(df, vol_long)
        vol_ratio = atr_short / atr_long.replace(0, np.nan)

        # Trend strength using returns
        returns_20d = close.pct_change(20)

        regimes = pd.Series("sideways", index=df.index)

        # Bull: EMA short > EMA long, positive returns
        bull = (ema_short > ema_long) & (returns_20d > 0.02)
        # Bear: EMA short < EMA long, negative returns
        bear = (ema_short < ema_long) & (returns_20d < -0.02)

        regimes[bull] = "bull"
        regimes[bear] = "bear"

        # Override with volatility regimes if extreme
        high_vol = vol_ratio > 1.5
        low_vol = vol_ratio < 0.7

        # Add vol suffix
        regimes[high_vol] = regimes[high_vol] + "_high_vol"
        regimes[low_vol] = regimes[low_vol] + "_low_vol"

        return regimes

    def analyze_by_regime(
        self,
        signals: pd.Series,
        df: pd.DataFrame,
        regime_df: pd.Series,
    ) -> dict[str, Any]:
        """Analyze strategy performance broken down by market regime."""
        regimes = regime_df.unique()

        regime_stats = {}
        for regime in regimes:
            mask = regime_df == regime
            regime_signals = signals[mask]

            n_signals = (regime_signals != 0).sum()
            long_signals = (regime_signals > 0).sum()
            short_signals = (regime_signals < 0).sum()

            regime_stats[regime] = {
                "n_candles": mask.sum(),
                "n_signals": int(n_signals),
                "long_signals": int(long_signals),
                "short_signals": int(short_signals),
                "signal_rate": float(n_signals / mask.sum()) if mask.sum() > 0 else 0,
            }

        return regime_stats

    def calculate_regime_stability(
        self,
        backtest_results_by_regime: dict[str, dict],
    ) -> float:
        """
        Calculate regime stability score.

        A strategy that works across all regimes is more stable.
        """
        if not backtest_results_by_regime:
            return 0.0

        profitable_regimes = sum(
            1 for r in backtest_results_by_regime.values()
            if r.get("sharpe", 0) > 0 or r.get("total_return", 0) > 0
        )

        return profitable_regimes / len(backtest_results_by_regime)
