"""
Tests for technical indicators.

Verifies correctness of all indicator calculations.
"""

import numpy as np
import pandas as pd
import pytest

from trading_system.indicators.trend import (
    sma, ema, macd, adx, donchian_channel, supertrend,
)
from trading_system.indicators.momentum import rsi, roc, stochastic, williams_r, cci
from trading_system.indicators.volatility import atr, bollinger_bands, keltner_channel
from trading_system.indicators.volume import obv, vwap, relative_volume
from trading_system.indicators.utils import crossover, crossunder, zscore, heikin_ashi


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    # Simulate a random walk
    returns = np.random.randn(n) * 0.01
    close = 100 * np.cumprod(1 + returns)

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_price = close * (1 + np.random.randn(n) * 0.003)
    volume = np.random.uniform(1000, 10000, n)

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


class TestSMA:
    def test_basic(self, sample_ohlcv):
        result = sma(sample_ohlcv["close"], 20)
        assert len(result) == len(sample_ohlcv)
        assert result.isna().sum() == 19  # First 19 should be NaN
        assert not result.iloc[20:].isna().any()

    def test_known_values(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = sma(series, 5)
        assert result.iloc[4] == pytest.approx(3.0)
        assert result.iloc[5] == pytest.approx(4.0)
        assert result.iloc[9] == pytest.approx(8.0)

    def test_period_1(self):
        series = pd.Series([1.0, 2.0, 3.0])
        result = sma(series, 1)
        pd.testing.assert_series_equal(result, series)


class TestEMA:
    def test_basic(self, sample_ohlcv):
        result = ema(sample_ohlcv["close"], 20)
        assert len(result) == len(sample_ohlcv)
        assert not result.isna().all()

    def test_faster_than_sma(self, sample_ohlcv):
        ema_result = ema(sample_ohlcv["close"], 20)
        sma_result = sma(sample_ohlcv["close"], 20)
        # EMA should respond faster to recent price changes
        assert ema_result.iloc[-1] != sma_result.iloc[-1]


class TestMACD:
    def test_basic(self, sample_ohlcv):
        result = macd(sample_ohlcv["close"])
        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_histogram(self, sample_ohlcv):
        result = macd(sample_ohlcv["close"])
        # Histogram should be macd - signal
        np.testing.assert_allclose(
            result["histogram"].dropna().values,
            (result["macd"] - result["signal"]).dropna().values,
        )


class TestRSI:
    def test_bounds(self, sample_ohlcv):
        result = rsi(sample_ohlcv["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_known_values(self):
        # RSI of a series that goes up then down
        series = pd.Series([10 + i * 0.5 for i in range(50)] + [35 - i * 0.3 for i in range(20)], dtype=float)
        result = rsi(series, 14)
        valid = result.dropna()
        # After the uptrend, RSI should be high
        assert len(valid) > 0
        assert valid.max() > 60  # Should reach high RSI during uptrend

    def test_length(self, sample_ohlcv):
        result = rsi(sample_ohlcv["close"], 14)
        assert len(result) == len(sample_ohlcv)


class TestATR:
    def test_positive(self, sample_ohlcv):
        result = atr(sample_ohlcv, 14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_length(self, sample_ohlcv):
        result = atr(sample_ohlcv, 14)
        assert len(result) == len(sample_ohlcv)


class TestBollingerBands:
    def test_upper_above_lower(self, sample_ohlcv):
        result = bollinger_bands(sample_ohlcv["close"], 20, 2.0)
        valid = result.dropna()
        assert (valid["upper"] >= valid["lower"]).all()

    def test_middle_is_sma(self, sample_ohlcv):
        result = bollinger_bands(sample_ohlcv["close"], 20, 2.0)
        sma_val = sma(sample_ohlcv["close"], 20)
        np.testing.assert_allclose(
            result["middle"].dropna().values,
            sma_val.dropna().values,
            rtol=1e-10,
        )

    def test_pct_b(self, sample_ohlcv):
        result = bollinger_bands(sample_ohlcv["close"], 20, 2.0)
        assert "pct_b" in result.columns
        assert "bandwidth" in result.columns


class TestADX:
    def test_positive(self, sample_ohlcv):
        result = adx(sample_ohlcv, 14)
        valid = result["adx"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_di_positive(self, sample_ohlcv):
        result = adx(sample_ohlcv, 14)
        valid = result.dropna()
        assert (valid["plus_di"] >= 0).all()
        assert (valid["minus_di"] >= 0).all()


class TestDonchianChannel:
    def test_upper_above_lower(self, sample_ohlcv):
        result = donchian_channel(sample_ohlcv, 20)
        valid = result.dropna()
        assert (valid["upper"] >= valid["lower"]).all()

    def test_middle_between(self, sample_ohlcv):
        result = donchian_channel(sample_ohlcv, 20)
        valid = result.dropna()
        assert (valid["middle"] >= valid["lower"]).all()
        assert (valid["middle"] <= valid["upper"]).all()


class TestCrossover:
    def test_basic(self):
        a = pd.Series([1.0, 3.0, 2.0])
        b = pd.Series([2.0, 2.0, 2.0])
        result = crossover(a, b)
        assert result.iloc[1] == True  # 3 crosses above 2 at index 1

    def test_crossunder(self):
        a = pd.Series([3, 2, 1, 2, 3])
        b = pd.Series([2, 2, 2, 2, 2])
        result = crossunder(a, b)
        assert result.iloc[2] == True  # 1 crosses below 2


class TestOBV:
    def test_basic(self, sample_ohlcv):
        result = obv(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)
        assert result.iloc[0] == 0  # First value is 0


class TestHeikinAshi:
    def test_basic(self, sample_ohlcv):
        result = heikin_ashi(sample_ohlcv)
        assert "ha_open" in result.columns
        assert "ha_close" in result.columns
        assert "ha_high" in result.columns
        assert "ha_low" in result.columns
        assert len(result) == len(sample_ohlcv)


class TestCausrality:
    """Verify that indicators do not use future data."""

    def test_sma_causality(self):
        """SMA at time T should not change if we add future data."""
        np.random.seed(42)
        data = pd.Series(np.random.randn(100).cumsum() + 100, dtype=float)

        # SMA with first 50 points
        sma_50 = sma(data[:50], 20)
        # SMA with all 100 points
        sma_100 = sma(data, 20)

        # First 50 points should be identical
        np.testing.assert_array_equal(
            sma_50.dropna().values,
            sma_100.iloc[:50].dropna().values,
        )

    def test_rsi_causality(self):
        """RSI at time T should not change if we add future data."""
        np.random.seed(42)
        data = pd.Series(np.abs(np.random.randn(100)).cumsum() + 50, dtype=float)

        rsi_50 = rsi(data[:50], 14)
        rsi_100 = rsi(data, 14)

        np.testing.assert_array_equal(
            rsi_50.dropna().values,
            rsi_100.iloc[:50].dropna().values,
        )
