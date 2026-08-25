"""
Tests for portfolio allocation and accounting.
"""

import numpy as np
import pandas as pd
import pytest

from trading_system.portfolio.allocator import (
    equal_weight, volatility_weight, sharpe_weight, drawdown_adjusted_weight,
)


@pytest.fixture
def equity_curves():
    """Generate sample equity curves for multiple strategies."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")

    curves = {}
    for i, name in enumerate(["strategy_a", "strategy_b", "strategy_c"]):
        returns = np.random.randn(n) * (0.001 + i * 0.0005)
        curves[name] = pd.Series(
            10000 * np.cumprod(1 + returns),
            index=dates,
        )

    return curves


class TestEqualWeight:
    def test_basic(self):
        weights = equal_weight(3)
        assert len(weights) == 3
        assert pytest.approx(sum(weights)) == 1.0
        assert all(w == pytest.approx(1/3) for w in weights)

    def test_single(self):
        weights = equal_weight(1)
        assert weights[0] == pytest.approx(1.0)


class TestVolatilityWeight:
    def test_basic(self, equity_curves):
        weights = volatility_weight(equity_curves)
        assert len(weights) == 3
        assert pytest.approx(sum(weights)) == 1.0
        assert all(w >= 0 for w in weights)

    def test_lower_vol_gets_higher_weight(self, equity_curves):
        weights = volatility_weight(equity_curves)
        # Strategy with lower vol should have higher weight
        # (strategy_a has lowest vol, should have highest weight)
        assert weights[0] > weights[2]


class TestSharpeWeight:
    def test_basic(self, equity_curves):
        weights = sharpe_weight(equity_curves)
        assert len(weights) == 3
        assert pytest.approx(sum(weights)) == 1.0
        assert all(w >= 0 for w in weights)


class TestDrawdownWeight:
    def test_basic(self, equity_curves):
        weights = drawdown_adjusted_weight(equity_curves)
        assert len(weights) == 3
        assert pytest.approx(sum(weights)) == 1.0
