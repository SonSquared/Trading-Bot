"""
Tests for strategy signal generation.

Verifies that all strategies produce valid signals with no look-ahead bias.
"""

import numpy as np
import pandas as pd
import pytest

from trading_system.strategies import ALL_STRATEGIES, get_strategy, STRATEGY_REGISTRY


@pytest.fixture
def sample_ohlcv():
    """Generate realistic sample OHLCV data."""
    np.random.seed(42)
    n = 1000
    dates = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")

    returns = np.random.randn(n) * 0.01
    close = 100 * np.cumprod(1 + returns)
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


class TestAllStrategies:
    """Test all registered strategies."""

    def test_all_strategies_registered(self):
        assert len(ALL_STRATEGIES) >= 17  # At least 17 strategies

    def test_all_have_meta(self):
        for strategy in ALL_STRATEGIES:
            meta = strategy.meta()
            assert meta.name
            assert meta.family in ("trend", "momentum", "mean_reversion", "volatility")

    def test_all_have_default_params(self):
        for strategy in ALL_STRATEGIES:
            params = strategy.default_params()
            assert isinstance(params, dict)
            assert len(params) > 0

    def test_all_have_param_grid(self):
        for strategy in ALL_STRATEGIES:
            grid = strategy.param_grid()
            assert isinstance(grid, dict)
            assert len(grid) > 0
            for key, values in grid.items():
                assert isinstance(values, list)
                assert len(values) > 0


@pytest.fixture
def regime_buffet():
    """Synthetic data containing every regime a strategy family needs:
    squeeze -> gap-up trend -> oscillation -> downtrend -> calm.
    Every registered strategy must produce at least one non-zero signal
    on this fixture — a strategy that never fires anywhere is dead code
    (this tripwire exists because Supertrend silently traded nothing for
    months due to an NaN-poisoning bug in its indicator)."""
    rng = np.random.default_rng(42)
    phases = []
    # 1: quiet/squeeze
    phases.append(np.full(150, 100.0) + np.cumsum(rng.normal(0, 0.0002, 150)))
    # 2: gap + strong uptrend
    up = [phases[-1][-1] * 1.02]
    up += [up[-1] * (1 + 0.003 + rng.normal(0, 0.001)) for _ in range(149)]
    phases.append(np.array(up))
    # 3: oscillation (mean-reversion feeder)
    base = phases[-1][-1]
    phases.append(base * (1 + 0.008 * np.sin(np.arange(300) / 20)
                          + rng.normal(0, 0.001, 300)))
    # 4: downtrend
    dn = [phases[-1][-1]]
    dn += [dn[-1] * (1 - 0.003 + rng.normal(0, 0.001)) for _ in range(149)]
    phases.append(np.array(dn))
    # 5: calm again
    phases.append(np.full(100, phases[-1][-1])
                  + np.cumsum(rng.normal(0, 0.0002, 100)))

    close = np.concatenate(phases)
    n = len(close)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    noise = rng.normal(0, 0.0008, n)
    high = np.maximum(open_price, close) * (1 + np.abs(noise) + 0.0005)
    low = np.minimum(open_price, close) * (1 - np.abs(noise) - 0.0005)
    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 5000.0),
    })


class TestSignalValidity:
    """Test that signals are valid across all strategies."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.meta().name)
    def test_valid_signals(self, strategy, sample_ohlcv):
        """All strategies should produce signals in {-2, -1, 0, 1, 2}."""
        params = strategy.default_params()
        signals = strategy.generate_signals(sample_ohlcv, params)

        assert len(signals) == len(sample_ohlcv)
        valid_values = {-2, -1, 0, 1, 2}
        invalid = set(signals.unique()) - valid_values
        assert not invalid, f"{strategy.meta().name} produced invalid signals: {invalid}"

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.meta().name)
    def test_signals_not_all_zero(self, strategy, regime_buffet):
        """Every strategy must trade somewhere in a data set that offers
        every regime. A strategy that never fires is dead code, not a
        selective one — it would silently contribute 0.00% forever."""
        params = strategy.default_params()
        signals = strategy.generate_signals(regime_buffet, params)
        non_zero = (signals != 0).sum()
        assert non_zero > 0, (
            f"{strategy.meta().name} produced ZERO non-zero signals on the "
            f"regime buffet — it is dead code (indicator bug or logic lock)")


class TestCausrality:
    """Verify no look-ahead bias in signal generation."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.meta().name)
    def test_no_lookahead_bias(self, strategy, sample_ohlcv):
        """Signals at time T should not change if we add future data."""
        params = strategy.default_params()

        # Generate signals with first 500 candles
        signals_500 = strategy.generate_signals(sample_ohlcv[:500], params)
        # Generate signals with all 1000 candles
        signals_1000 = strategy.generate_signals(sample_ohlcv, params)

        # First 500 signals should be identical
        common = min(len(signals_500), len(signals_1000))
        np.testing.assert_array_equal(
            signals_500.values[:common],
            signals_1000.values[:common],
            err_msg=f"{strategy.meta().name} has look-ahead bias!",
        )


class TestStrategyLookup:
    def test_get_strategy(self):
        for name in STRATEGY_REGISTRY:
            strategy = get_strategy(name)
            assert strategy is not None

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            get_strategy("nonexistent_strategy")
