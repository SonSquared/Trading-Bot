"""
Volatility Filter for Trading Strategies.

Prevents trading during extremely low-volatility periods where:
- Strategy signals are likely noise (no real price movement)
- Fees and slippage eat any small profits
- Win rates drop because price doesn't move enough to hit targets

The filter is simple: only trade when ATR is above a minimum
percentile of its recent history. This is a binary gate — either
the market has enough volatility to trade, or it doesn't.

Unlike regime detection (which tries to classify trending vs choppy),
this just answers: "Is there enough movement to justify a trade?"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from trading_system.indicators import atr

logger = structlog.get_logger(__name__)


@dataclass
class VolatilityFilterConfig:
    """Configuration for the volatility filter."""
    atr_period: int = 14            # ATR calculation period
    lookback: int = 100             # Rolling window for percentile calculation
    min_percentile: float = 10.0    # Minimum ATR percentile to allow trades (0-100)
    # e.g., 10.0 means "only trade when ATR is above the 10th percentile"

    # Alternative: absolute ATR threshold (as % of price)
    min_atr_pct: float = 0.0       # 0 = disabled; e.g., 0.5 = min 0.5% ATR


class VolatilityFilter:
    """
    Filters out low-volatility candles where trading is unprofitable.

    Computes ATR and its rolling percentile. Produces a boolean mask:
    - True = enough volatility to trade
    - False = too quiet, skip this candle

    The filter can be applied to signals before running the backtester:
        signals_filtered = signals.where(vol_filter.allow_trading)

    Or integrated directly into the backtester via the `risk_multipliers`
    parameter (0.0 = no trade, 1.0 = full trade).
    """

    def __init__(self, config: VolatilityFilterConfig | None = None):
        self.config = config or VolatilityFilterConfig()

    def compute_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a boolean volatility filter for each candle.

        Returns:
            pd.Series[bool]: True where trading is allowed, False to skip
        """
        atr_series = atr(df, self.config.atr_period)
        n = len(df)
        allow = pd.Series(True, index=df.index)

        # Need enough history for percentile calculation
        warmup = self.config.lookback

        if n < warmup + self.config.atr_period:
            return allow  # Not enough data, allow all

        for i in range(warmup, n):
            window = atr_series.iloc[max(0, i - self.config.lookback):i + 1].dropna()
            if len(window) < 10:
                allow.iloc[i] = True
                continue

            current_atr = atr_series.iloc[i]
            if np.isnan(current_atr):
                allow.iloc[i] = False
                continue

            # Percentile of current ATR within the lookback window
            pct = (window < current_atr).sum() / len(window) * 100

            # Check percentile threshold
            above_percentile = pct >= self.config.min_percentile

            # Check absolute threshold (if set)
            if self.config.min_atr_pct > 0:
                atr_pct = current_atr / df["close"].iloc[i] * 100
                above_absolute = atr_pct >= self.config.min_atr_pct
                allow.iloc[i] = above_percentile and above_absolute
            else:
                allow.iloc[i] = above_percentile

        return allow

    def compute_filter_fast(self, df: pd.DataFrame) -> pd.Series:
        """
        Vectorized version for performance.

        Uses rolling percentile instead of per-candle loop.
        Less accurate for edge cases but ~100x faster.
        """
        atr_series = atr(df, self.config.atr_period)
        n = len(df)

        # Rolling percentile: fraction of lookback values below current
        atr_rolling = atr_series.rolling(window=self.config.lookback, min_periods=10)
        # Use rolling rank (percentile) — fraction of values <= current
        rolling_pct = atr_series.rolling(window=self.config.lookback, min_periods=10).apply(
            lambda x: (x[:-1] < x.iloc[-1]).sum() / (len(x) - 1) * 100,
            raw=False,
        )

        allow = rolling_pct >= self.config.min_percentile

        # Apply absolute threshold if set
        if self.config.min_atr_pct > 0:
            atr_pct = atr_series / df["close"] * 100
            allow = allow & (atr_pct >= self.config.min_atr_pct)

        # Fill NaN (warmup period) with True
        allow = allow.fillna(True)

        return allow

    def compute_risk_multipliers(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute risk multipliers based on volatility.

        Returns:
            pd.Series: 1.0 when volatility is sufficient, 0.0 when filtered out
        """
        allow = self.compute_filter(df)
        return allow.astype(float)

    def get_filter_stats(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about the filter's behavior.

        Returns dict with filtered percentage, ATR distribution, etc.
        """
        allow = self.compute_filter(df)
        atr_series = atr(df, self.config.atr_period)

        filtered_pct = (~allow).sum() / len(allow) * 100
        allowed_pct = allow.sum() / len(allow) * 100

        # ATR statistics
        atr_clean = atr_series.dropna()
        atr_pct_series = (atr_clean / df["close"].iloc[:len(atr_clean)] * 100)

        return {
            "total_candles": len(df),
            "allowed_candles": int(allow.sum()),
            "filtered_candles": int((~allow).sum()),
            "filtered_pct": filtered_pct,
            "allowed_pct": allowed_pct,
            "atr_mean_pct": float(atr_pct_series.mean()),
            "atr_median_pct": float(atr_pct_series.median()),
            "atr_min_pct": float(atr_pct_series.min()),
            "atr_max_pct": float(atr_pct_series.max()),
        }
