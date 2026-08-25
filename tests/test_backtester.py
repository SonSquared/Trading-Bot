"""
Tests for the backtesting engine.

Verifies correct P&L accounting, position tracking, and metric calculations.
"""

import numpy as np
import pandas as pd
import pytest

from trading_system.backtester.engine import BacktestEngine
from trading_system.backtester.results import BacktestResults
from trading_system.backtester.fees import FeeCalculator
from trading_system.backtester.slippage import FixedSlippage, ATRAdaptiveSlippage, NoSlippage
from trading_system.config import (
    BacktestConfig, FeeConfig, SlippageConfig, ExecutionConfig,
)


@pytest.fixture
def config():
    return BacktestConfig(
        fees=FeeConfig(taker_fee=0.0004, maker_fee=0.0002),
        slippage=SlippageConfig(model="none"),
        execution=ExecutionConfig(
            initial_capital=10000,
            leverage=1.0,
            risk_per_trade=0.02,
        ),
    )


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_price = close + np.random.randn(n) * 0.2
    volume = np.ones(n) * 1000

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)


class TestFeeCalculator:
    def test_entry_fee(self):
        calc = FeeCalculator(FeeConfig(taker_fee=0.0004))
        fee = calc.calculate_entry_fee(100.0, 10.0)
        assert fee == pytest.approx(0.4)  # 10 * 100 * 0.0004

    def test_maker_fee_lower(self):
        calc = FeeCalculator(FeeConfig(maker_fee=0.0002, taker_fee=0.0004))
        taker = calc.calculate_entry_fee(100.0, 10.0, is_maker=False)
        maker = calc.calculate_entry_fee(100.0, 10.0, is_maker=True)
        assert maker < taker

    def test_negative_position(self):
        calc = FeeCalculator(FeeConfig(taker_fee=0.0004))
        fee = calc.calculate_entry_fee(100.0, -10.0)
        assert fee == pytest.approx(0.4)  # abs(-10) * 100 * 0.0004


class TestSlippage:
    def test_no_slippage(self):
        model = NoSlippage(SlippageConfig())
        result = model.calculate_slippage(100.0, 1.0, True)
        assert result == 100.0

    def test_fixed_slippage_long(self):
        model = FixedSlippage(SlippageConfig(base_slippage=0.001))
        result = model.calculate_slippage(100.0, 1.0, True)
        assert result == pytest.approx(100.1)  # Long pays more

    def test_fixed_slippage_short(self):
        model = FixedSlippage(SlippageConfig(base_slippage=0.001))
        result = model.calculate_slippage(100.0, 1.0, False)
        assert result == pytest.approx(99.9)  # Short gets less


class TestBacktestEngine:
    def test_empty_data(self, config):
        engine = BacktestEngine(config)
        df = pd.DataFrame()
        signals = pd.Series(dtype=int)
        result = engine.run(df, signals)
        assert isinstance(result, BacktestResults)
        assert result.total_trades == 0

    def test_all_flat_signals(self, config, sample_data):
        engine = BacktestEngine(config)
        signals = pd.Series(0, index=sample_data.index, dtype=int)
        result = engine.run(sample_data, signals)
        assert result.total_trades == 0
        assert result.total_return == 0

    def test_all_long_signals(self, config, sample_data):
        engine = BacktestEngine(config)
        signals = pd.Series(1, index=sample_data.index, dtype=int)
        result = engine.run(sample_data, signals)
        # Should have at least one trade
        assert result.total_trades >= 1
        # Should have equity curve
        assert len(result.equity_curve) == len(sample_data)

    def test_total_return_consistency(self, config, sample_data):
        engine = BacktestEngine(config)
        signals = pd.Series(1, index=sample_data.index, dtype=int)
        result = engine.run(sample_data, signals)

        if len(result.equity_curve) > 0:
            expected_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
            assert result.total_return == pytest.approx(expected_return, abs=0.01)

    def test_no_negative_equity(self, config, sample_data):
        engine = BacktestEngine(config)
        # Alternate between long and flat
        signals = pd.Series(0, index=sample_data.index, dtype=int)
        signals.iloc[::10] = 1  # Go long every 10 candles
        result = engine.run(sample_data, signals)

        # Equity should never be negative with 1x leverage
        if len(result.equity_curve) > 0:
            assert (result.equity_curve >= 0).all()

    def test_fees_positive(self, config, sample_data):
        engine = BacktestEngine(config)
        signals = pd.Series(0, index=sample_data.index, dtype=int)
        signals.iloc[::20] = 1  # Trade every 20 candles
        result = engine.run(sample_data, signals)

        if result.total_trades > 0:
            assert result.total_fees >= 0


class TestBacktestResults:
    def test_to_dict(self):
        result = BacktestResults(
            strategy_name="test",
            sharpe=1.5,
            total_return=0.15,
            max_drawdown=0.1,
        )
        d = result.to_dict()
        assert d["strategy_name"] == "test"
        assert d["sharpe"] == 1.5

    def test_summary(self):
        result = BacktestResults(
            strategy_name="test",
            sharpe=1.5,
            total_return=0.15,
        )
        s = result.summary()
        assert "test" in s
        assert "1.50" in s
