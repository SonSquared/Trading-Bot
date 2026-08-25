"""
Indicator utility functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crossover(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Returns True where series_a crosses above series_b."""
    return (series_a > series_b) & (series_a.shift(1) <= series_b.shift(1))


def crossunder(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Returns True where series_a crosses below series_b."""
    return (series_a < series_b) & (series_a.shift(1) >= series_b.shift(1))


def above(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Returns True where series_a is above series_b."""
    return series_a > series_b


def below(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
    """Returns True where series_a is below series_b."""
    return series_a < series_b


def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling Z-score."""
    mean = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    return (series - mean) / std.replace(0, np.nan)


def percentile_rank(series: pd.Series, period: int = 100) -> pd.Series:
    """Rolling percentile rank of the current value."""
    def _rank(x: np.ndarray) -> float:
        return (x[-1] > x[:-1]).sum() / (len(x) - 1) * 100
    return series.rolling(window=period, min_periods=period).apply(_rank, raw=True)


def returns(series: pd.Series, period: int = 1) -> pd.Series:
    """Percentage returns over N periods."""
    return series.pct_change(periods=period)


def log_returns(series: pd.Series, period: int = 1) -> pd.Series:
    """Log returns over N periods."""
    return np.log(series / series.shift(period))


def rolling_max(series: pd.Series, period: int) -> pd.Series:
    """Rolling maximum."""
    return series.rolling(window=period, min_periods=period).max()


def rolling_min(series: pd.Series, period: int) -> pd.Series:
    """rolling minimum."""
    return series.rolling(window=period, min_periods=period).min()


def rolling_rank(series: pd.Series, period: int) -> pd.Series:
    """Rolling percentile rank (0 to 1)."""
    def _pct_rank(x: np.ndarray) -> float:
        return (x[-1] >= x[:-1]).sum() / (len(x) - 1)
    return series.rolling(window=period, min_periods=period).apply(_pct_rank, raw=True)


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heikin Ashi candles.

    Requires columns: open, high, low, close
    Returns DataFrame with ha_open, ha_high, ha_low, ha_close
    """
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    ha_open = pd.Series(np.nan, index=df.index)
    ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2

    ha_high = pd.concat([df["high"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([df["low"], ha_open, ha_close], axis=1).min(axis=1)

    return pd.DataFrame({
        "ha_open": ha_open,
        "ha_high": ha_high,
        "ha_low": ha_low,
        "ha_close": ha_close,
    })
