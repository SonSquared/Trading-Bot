"""
Trend indicators.

All functions take a pandas DataFrame with OHLCV columns and return pd.Series or pd.DataFrame.
No future data is ever used — strict causality is enforced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def dema(series: pd.Series, period: int) -> pd.Series:
    """Double Exponential Moving Average."""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * e1 - e2


def wma(series: pd.Series, period: int) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).

    Returns DataFrame with columns: macd, signal, histogram
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with +DI and -DI.

    Requires columns: high, low, close
    Returns DataFrame with columns: adx, plus_di, minus_di
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)

    # Smoothed averages (Wilder's smoothing = EMA with alpha = 1/period)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr)

    # ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({
        "adx": adx_val,
        "plus_di": plus_di,
        "minus_di": minus_di,
    })


def donchian_channel(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    """
    Donchian Channel.

    Requires columns: high, low
    Returns DataFrame with columns: upper, lower, middle
    """
    upper = df["high"].rolling(window=period, min_periods=period).max()
    lower = df["low"].rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2

    return pd.DataFrame({
        "upper": upper,
        "lower": lower,
        "middle": middle,
    })


def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Supertrend indicator.

    Requires columns: high, low, close
    Returns DataFrame with columns: supertrend, direction
    Direction: 1 = uptrend (long), -1 = downtrend (short)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Basic bands
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    close_arr = close.to_numpy(dtype=float)
    upper_band = basic_upper.to_numpy(dtype=float).copy()
    lower_band = basic_lower.to_numpy(dtype=float).copy()
    atr_arr = atr.to_numpy(dtype=float)

    n = len(df)
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    for i in range(n):
        # Warm-up: ATR (and thus the bands) is NaN for the first `period`
        # rows. Any value carried through NaN comparisons would stay NaN
        # forever, so skip until the bands are valid.
        if np.isnan(atr_arr[i]):
            continue

        if i > 0 and np.isfinite(upper_band[i - 1]):
            # Final upper band: keep the basic band only if it ratchets
            # down or price closed above the previous final band.
            if not (upper_band[i] < upper_band[i - 1] or close_arr[i - 1] > upper_band[i - 1]):
                upper_band[i] = upper_band[i - 1]

            # Final lower band: ratchet up or price closed below.
            if not (lower_band[i] > lower_band[i - 1] or close_arr[i - 1] < lower_band[i - 1]):
                lower_band[i] = lower_band[i - 1]
            prev_dir = direction[i - 1]
        else:
            # First valid row: bands start fresh; canonical start in a
            # downtrend (the first close above the upper band flips us).
            prev_dir = -1.0

        # Direction
        if prev_dir == 1:  # Was uptrend
            if close_arr[i] < lower_band[i]:
                direction[i] = -1
                st[i] = upper_band[i]
            else:
                direction[i] = 1
                st[i] = lower_band[i]
        else:  # Was downtrend
            if close_arr[i] > upper_band[i]:
                direction[i] = 1
                st[i] = lower_band[i]
            else:
                direction[i] = -1
                st[i] = upper_band[i]

    return pd.DataFrame({
        "supertrend": pd.Series(st, index=df.index),
        "direction": pd.Series(direction, index=df.index),
    })


def parabolic_sar(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """
    Parabolic SAR.

    Requires columns: high, low, close
    Returns Series with SAR values.
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    sar = np.zeros(n)
    af = af_start
    uptrend = True
    ep = high[0]
    sar[0] = low[0]

    for i in range(1, n):
        sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

        if uptrend:
            sar[i] = min(sar[i], low[i - 1])
            if i >= 2:
                sar[i] = min(sar[i], low[i - 2])
            if low[i] < sar[i]:
                uptrend = False
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_increment, af_max)
        else:
            sar[i] = max(sar[i], high[i - 1])
            if i >= 2:
                sar[i] = max(sar[i], high[i - 2])
            if high[i] > sar[i]:
                uptrend = True
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_increment, af_max)

    return pd.Series(sar, index=df.index, name="parabolic_sar")
