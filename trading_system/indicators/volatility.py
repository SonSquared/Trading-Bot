"""
Volatility indicators.

All functions are strictly causal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR).

    Requires columns: high, low, close
    Uses Wilder's smoothing (EMA with alpha = 1/period).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    close_prev = close.shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range."""
    close_prev = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - close_prev).abs()
    tr3 = (df["low"] - close_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Returns DataFrame with columns: upper, middle, lower, bandwidth, pct_b
    """
    middle = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()

    upper = middle + std_dev * std
    lower = middle - std_dev * std

    # Bandwidth: (upper - lower) / middle
    bandwidth = (upper - lower) / middle.replace(0, np.nan)

    # %B: (price - lower) / (upper - lower)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "pct_b": pct_b,
    })


def keltner_channel(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Keltner Channel.

    Requires columns: high, low, close
    Returns DataFrame with columns: upper, middle, lower
    """
    close = df["close"]
    middle = close.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    atr_val = atr(df, atr_period)

    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val

    return pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower,
    })


def historical_volatility(
    series: pd.Series,
    period: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """
    Historical volatility (standard deviation of log returns).
    """
    log_returns = np.log(series / series.shift(1))
    vol = log_returns.rolling(window=period, min_periods=period).std()

    if annualize:
        # For crypto: 365 * 24 hours per day
        vol = vol * np.sqrt(365 * 24)

    return vol


def normalized_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as percentage of price."""
    atr_val = atr(df, period)
    return atr_val / df["close"].replace(0, np.nan) * 100


def chandelier_exit(
    df: pd.DataFrame,
    period: int = 22,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Chandelier Exit.

    Requires columns: high, low, close
    Returns DataFrame with columns: long_exit, short_exit
    """
    atr_val = atr(df, period)
    highest = df["high"].rolling(window=period, min_periods=period).max()
    lowest = df["low"].rolling(window=period, min_periods=period).min()

    long_exit = highest - multiplier * atr_val
    short_exit = lowest + multiplier * atr_val

    return pd.DataFrame({
        "long_exit": long_exit,
        "short_exit": short_exit,
    })


def volatility_regime(
    df: pd.DataFrame,
    short_period: int = 10,
    long_period: int = 50,
) -> pd.Series:
    """
    Volatility regime detection.

    Returns ratio of short-term to long-term ATR.
    > 1 means high volatility regime, < 1 means low volatility regime.
    """
    atr_short = atr(df, short_period)
    atr_long = atr(df, long_period)
    return atr_short / atr_long.replace(0, np.nan)
