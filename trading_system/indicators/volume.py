"""
Volume indicators.

All functions are strictly causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV).

    Requires columns: close, volume
    """
    direction = np.sign(df["close"].diff())
    direction.iloc[0] = 0
    return (direction * df["volume"]).cumsum()


def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Volume-Weighted Average Price (rolling).

    Requires columns: high, low, close, volume
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).rolling(window=period, min_periods=1).sum() / \
           df["volume"].rolling(window=period, min_periods=1).sum()


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    return df["volume"].rolling(window=period, min_periods=period).mean()


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume relative to its moving average."""
    vol_avg = volume_sma(df, period)
    return df["volume"] / vol_avg.replace(0, np.nan)


def accumulation_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Accumulation/Distribution Line.

    Requires columns: high, low, close, volume
    """
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
          (df["high"] - df["low"]).replace(0, np.nan)
    return (clv * df["volume"]).cumsum()


def chaikin_money_flow(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """
    Chaikin Money Flow.

    Requires columns: high, low, close, volume
    """
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
          (df["high"] - df["low"]).replace(0, np.nan)
    mfv = clv * df["volume"]
    return mfv.rolling(window=period, min_periods=period).sum() / \
           df["volume"].rolling(window=period, min_periods=period).sum()


def volume_profile(
    df: pd.DataFrame,
    bins: int = 50,
) -> pd.DataFrame:
    """
    Volume profile — volume at each price level.

    Requires columns: high, low, close, volume
    Returns DataFrame with price_level and volume_at_level.
    """
    price_min = df["low"].min()
    price_max = df["high"].max()
    levels = np.linspace(price_min, price_max, bins)
    bin_size = levels[1] - levels[0]

    vol_at_level = np.zeros(bins)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3

    for i in range(len(df)):
        # Approximate which bins this candle's volume falls into
        candle_low = df["low"].iloc[i]
        candle_high = df["high"].iloc[i]
        candle_vol = df["volume"].iloc[i]

        for j, level in enumerate(levels):
            level_low = level - bin_size / 2
            level_high = level + bin_size / 2
            if candle_high >= level_low and candle_low <= level_high:
                vol_at_level[j] += candle_vol / max(1, bins // 3)

    return pd.DataFrame({
        "price_level": levels,
        "volume_at_level": vol_at_level,
    })
