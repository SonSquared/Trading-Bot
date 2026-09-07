"""
Momentum indicators.

All functions are strictly causal — only use past data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Uses Wilder's smoothing (EMA with alpha = 1/period).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))

    # Degenerate denominators (Wilder conventions):
    #   avg_loss == 0 with gains  -> RSI 100 (pure strength; rs = inf)
    #   avg_loss == 0, no gains   -> RSI 50 (no movement = neutral, NOT 0)
    #   avg_gain == 0 with losses -> RSI 0 via the formula (rs = 0)
    # Warm-up rows (NaN) must stay NaN.
    rsi_val = rsi_val.where(avg_loss > 0, 100.0)
    rsi_val = rsi_val.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi_val.where(avg_loss.notna())


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (ROC) as percentage."""
    return (series / series.shift(period) - 1) * 100


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K and %D).

    Requires columns: high, low, close
    Returns DataFrame with columns: k, d
    """
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()

    fast_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = fast_k.rolling(window=smooth_k, min_periods=1).mean()
    d = k.rolling(window=d_period, min_periods=1).mean()

    return pd.DataFrame({
        "k": k,
        "d": d,
    })


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R.

    Requires columns: high, low, close
    Returns Series with values from -100 to 0.
    """
    high_max = df["high"].rolling(window=period, min_periods=period).max()
    low_min = df["low"].rolling(window=period, min_periods=period).min()

    wr = -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)
    return wr


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI).

    Requires columns: high, low, close
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci_val = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)
    return cci_val


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI).

    Requires columns: high, low, close, volume
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = tp * df["volume"]

    delta = tp.diff()
    pos_mf = raw_mf.where(delta > 0, 0.0)
    neg_mf = raw_mf.where(delta < 0, 0.0)

    pos_sum = pos_mf.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(window=period, min_periods=period).sum()

    mf_ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mf_ratio))


def stoch_rsi(
    series: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> pd.DataFrame:
    """
    Stochastic RSI.

    Applies Stochastic oscillator to RSI values.
    Returns DataFrame with columns: k, d
    """
    rsi_vals = rsi(series, rsi_period)
    rsi_min = rsi_vals.rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max = rsi_vals.rolling(window=stoch_period, min_periods=stoch_period).max()

    stoch_rsi_k = (rsi_vals - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k = stoch_rsi_k.rolling(window=k_smooth, min_periods=1).mean() * 100
    d = k.rolling(window=d_smooth, min_periods=1).mean()

    return pd.DataFrame({
        "k": k,
        "d": d,
    })


def awesome_oscillator(
    df: pd.DataFrame,
    fast_period: int = 5,
    slow_period: int = 34,
) -> pd.Series:
    """
    Awesome Oscillator.

    Requires columns: high, low, close
    Difference of 5-period and 34-period SMA of the midpoint.
    """
    midpoint = (df["high"] + df["low"]) / 2
    fast_sma = midpoint.rolling(window=fast_period, min_periods=fast_period).mean()
    slow_sma = midpoint.rolling(window=slow_period, min_periods=slow_period).mean()
    return fast_sma - slow_sma


def ultimate_oscillator(
    df: pd.DataFrame,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> pd.Series:
    """
    Ultimate Oscillator.

    Requires columns: high, low, close
    """
    close_prev = df["close"].shift(1)
    true_low = pd.concat([df["low"], close_prev], axis=1).min(axis=1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - close_prev).abs(),
        (df["low"] - close_prev).abs(),
    ], axis=1).max(axis=1)

    buying_pressure = df["close"] - true_low

    def _avg(bp: pd.Series, tr: pd.Series, period: int) -> pd.Series:
        bp_sum = bp.rolling(window=period, min_periods=period).sum()
        tr_sum = tr.rolling(window=period, min_periods=period).sum()
        return bp_sum / tr_sum.replace(0, np.nan)

    avg1 = _avg(buying_pressure, true_range, period1)
    avg2 = _avg(buying_pressure, true_range, period2)
    avg3 = _avg(buying_pressure, true_range, period3)

    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
