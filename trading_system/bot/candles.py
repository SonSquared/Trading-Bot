"""
Closed-candle helpers shared by all bot entry points.

Live/paper bots must only generate signals and trade on fully-closed
candles. A candle with open time `ts` is complete when `ts + timeframe
<= now`. Acting on the forming (partial) candle makes live behavior
diverge from the next-open backtest model and causes intra-candle
whipsaw churn, so every signal must come from closed candles only.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

# Seconds per candle for each supported timeframe
TIMEFRAME_SECONDS = {
    "4h": 4 * 3600,
    "1h": 3600,
    "30m": 1800,
    "15m": 900,
}


def closed_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Drop the currently forming candle; return only fully-closed candles.

    Accepts either a DataFrame with a ``timestamp`` column (paper trader
    data pipeline) or a DataFrame indexed by tz-aware UTC timestamps
    (``ExchangeInterface.get_ohlcv``). If the timestamp/index is naive it
    is treated as UTC.
    """
    tf_sec = TIMEFRAME_SECONDS.get(timeframe, 3600)
    cutoff = datetime.now(timezone.utc) - pd.Timedelta(seconds=tf_sec)

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        return df[ts <= cutoff]

    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        return df[idx <= cutoff]

    return df