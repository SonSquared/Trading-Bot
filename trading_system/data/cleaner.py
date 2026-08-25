"""
Data cleaning and preprocessing.

Handles: forward-fill for small gaps, removal of duplicates,
interpolation for minor issues, and data type enforcement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DataCleaner:
    """Cleans and preprocesses OHLCV data."""

    def __init__(
        self,
        max_forward_fill_candles: int = 3,
        remove_zero_volume: bool = False,
    ):
        """
        Args:
            max_forward_fill_candles: Maximum consecutive candles to forward-fill
            remove_zero_volume: Whether to remove zero-volume candles
        """
        self.max_forward_fill_candles = max_forward_fill_candles
        self.remove_zero_volume = remove_zero_volume

    def clean(self, df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
        """Apply all cleaning steps to a DataFrame."""
        if df.empty:
            return df

        df = df.copy()
        initial_len = len(df)

        # Step 1: Remove exact duplicates
        df = self._remove_duplicates(df)

        # Step 2: Fix phantom candles (exchange glitches / fat-finger wicks)
        df = self._fix_phantom_candles(df)

        # Step 3: Fix OHLC integrity
        df = self._fix_ohlc_integrity(df)

        # Step 4: Remove zero/negative prices
        df = self._remove_bad_prices(df)

        # Step 5: Forward-fill small gaps
        df = self._forward_fill_gaps(df, timeframe)

        # Step 6: Optionally remove zero-volume candles
        if self.remove_zero_volume:
            df = df[df["volume"] > 0]

        # Step 7: Ensure correct data types
        df = self._enforce_types(df)

        # Step 8: Sort by timestamp
        df = df.sort_index()

        logger.info(
            "data_cleaned",
            initial=initial_len,
            final=len(df),
            removed=initial_len - len(df),
        )

        return df

    def clean_funding_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean funding rate data."""
        if df.empty:
            return df

        df = df.copy()

        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # Clip extreme funding rates (Binance max is ±0.3%, but data errors can be larger)
        if "funding_rate" in df.columns:
            df["funding_rate"] = df["funding_rate"].clip(-0.05, 0.05)

        # Remove NaN
        df = df.dropna(subset=["funding_rate"])

        # Sort
        df = df.sort_index()

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping the first occurrence."""
        n_before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        if len(df) < n_before:
            logger.debug("removed_duplicates", count=n_before - len(df))
        return df

    def _fix_ohlc_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix impossible OHLC relationships."""
        # Ensure high >= max(open, close) and low <= min(open, close)
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)

        # Ensure high >= low
        mask = df["high"] < df["low"]
        if mask.any():
            # Swap them
            df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values
            logger.debug("fixed_ohlc_swap", count=mask.sum())

        return df

    def _fix_phantom_candles(self, df: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
        """Detect and fix phantom candles caused by exchange glitches.

        A phantom candle is one where high or low deviates more than
        `threshold` (20%) from the rolling median of the surrounding
        candles, while the open and close remain normal.  This catches
        fat-finger wicks and index-price manipulation candles common on
        crypto exchanges.

        Fix: replace the extreme wick with the candle's open/close median
        (which is almost always correct).
        """
        if len(df) < 11:
            return df

        df = df.copy()
        window = 11  # 5 candles before + self + 5 after
        median_prices = (df["open"] + df["close"]) / 2.0
        rolling_median = median_prices.rolling(window, center=True, min_periods=3).median()

        n_fixed = 0

        # Check for phantom high wicks
        high_deviation = (df["high"] - rolling_median).abs() / rolling_median
        high_mask = high_deviation > threshold
        if high_mask.any():
            # Replace extreme highs with the candle's own open/close median
            df.loc[high_mask, "high"] = median_prices[high_mask]
            n_fixed += high_mask.sum()

        # Check for phantom low wicks
        low_deviation = (df["low"] - rolling_median).abs() / rolling_median
        low_mask = low_deviation > threshold
        if low_mask.any():
            df.loc[low_mask, "low"] = median_prices[low_mask]
            n_fixed += low_mask.sum()

        if n_fixed > 0:
            logger.info("fixed_phantom_candles", count=int(n_fixed))

        return df

    def _remove_bad_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with zero or negative prices."""
        price_cols = ["open", "high", "low", "close"]
        mask = pd.Series(True, index=df.index)
        for col in price_cols:
            if col in df.columns:
                mask &= df[col] > 0

        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.debug("removed_bad_prices", count=n_removed)
        return df[mask]

    def _forward_fill_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Forward-fill small gaps in the data."""
        TIMEFRAME_MINUTES = {"15m": 15, "30m": 30, "1h": 60, "4h": 240}
        expected_min = TIMEFRAME_MINUTES.get(timeframe, 60)

        if len(df) < 2:
            return df

        # Find gaps
        gaps = df.index.to_series().diff()
        expected_delta = pd.Timedelta(minutes=expected_min)
        max_fill_delta = expected_delta * self.max_forward_fill_candles

        # Identify candles right after a gap that should be filled
        gap_mask = gaps <= max_fill_delta
        gap_mask.iloc[0] = True  # Keep first row

        # Forward fill only small gaps
        filled = df.copy()
        small_gap_mask = (gaps > expected_delta) & (gaps <= max_fill_delta)

        if small_gap_mask.any():
            # Reindex with the expected frequency, then forward fill
            full_idx = pd.date_range(
                start=df.index[0], end=df.index[-1], freq=expected_delta, tz=df.index.tz
            )
            filled = df.reindex(full_idx)
            filled = filled.ffill(limit=self.max_forward_fill_candles)

            # Drop rows that are still NaN in close (couldn't be filled)
            before_fill = len(filled)
            filled = filled.dropna(subset=["close"])
            if len(filled) < before_fill:
                logger.debug("dropped_unfillable", count=before_fill - len(filled))

        return filled

    def _enforce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types."""
        float_cols = ["open", "high", "low", "close", "volume"]
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)

        # Ensure timezone-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        return df
