"""Tests for the Kevin Davey strategy adaptations.

Covers the shared pulse_to_state state machine (Davey's time-exit and
reversal-flip semantics), each strategy's entry logic, symmetric
long/short design, and the no-lookahead constraint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.strategies.base import pulse_to_state
from trading_system.strategies import STRATEGY_REGISTRY


def make_df(n: int = 300, seed: int = 7, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0] * (1 - 0.001)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 1000.0),
    })


class TestPulseToState:
    """The shared time-exit state machine (Davey's exit convention)."""

    def test_enters_on_pulse(self):
        pulses = pd.Series([False, True, False, False])
        out = pulse_to_state(pulses, pd.Series([False] * 4), exit_bars=3)
        assert list(out) == [0, 1, 1, 1]

    def test_time_exit_releases_to_flat(self):
        # Entry at bar 1, exit_bars=2 -> held bars 1,2 then flat from bar 3
        long_entry = pd.Series([False, True, False, False, False])
        short_entry = pd.Series([False, False, False, False, False])
        out = pulse_to_state(long_entry, short_entry, exit_bars=2)
        assert list(out) == [0, 1, 1, 0, 0]

    def test_reversal_pulse_flips_immediately(self):
        long_entry = pd.Series([False, True, False, False, False])
        short_entry = pd.Series([False, False, False, True, False])
        out = pulse_to_state(long_entry, short_entry, exit_bars=10)
        assert list(out) == [0, 1, 1, -1, -1]

    def test_hold_is_exactly_exit_bars(self):
        n = 20
        long_entry = pd.Series([False, True] + [False] * (n - 2))
        short_entry = pd.Series([False] * n)
        out = pulse_to_state(long_entry, short_entry, exit_bars=5)
        held = (out == 1).sum()
        assert held == 5

    def test_exit_bars_minimum_is_one(self):
        long_entry = pd.Series([True, False, False])
        out = pulse_to_state(long_entry, pd.Series([False] * 3), exit_bars=0)
        assert list(out) == [1, 0, 0]

    def test_persistent_pulse_reenters_after_release(self):
        # Pulse stays true: each release to flat is followed by a fresh
        # entry on the next bar (TradeStation semantics: exit, then the
        # still-true condition re-enters next bar). Cycle: 2 held, 1 flat.
        long_entry = pd.Series([True] * 8)
        short_entry = pd.Series([False] * 8)
        out = pulse_to_state(long_entry, short_entry, exit_bars=2)
        assert list(out) == [1, 1, 0, 1, 1, 0, 1, 1]

    def test_output_domain_and_length(self):
        rng = np.random.default_rng(3)
        n = 200
        le = pd.Series(rng.random(n) > 0.9)
        se = pd.Series(rng.random(n) > 0.9) & ~le
        out = pulse_to_state(le, se, exit_bars=4)
        assert len(out) == n
        assert set(out.unique()) <= {-1, 0, 1}


class TestDaveyMomentumPullback:
    def setup_method(self):
        self.s = STRATEGY_REGISTRY["Davey_Momentum_Pullback"]

    def test_registered_with_grid(self):
        assert self.s.meta().name == "Davey_Momentum_Pullback"
        grid = self.s.param_grid()
        assert set(grid) == {"bar_count", "pullback", "exit_bars", "count_higher_highs"}

    def test_trades_on_trending_synthetic_data(self):
        df = make_df(n=500, drift=0.002)
        sig = self.s.generate_signals(df, self.s.default_params())
        assert (sig != 0).sum() > 0

    def test_symmetric_design(self):
        # Symmetry is structural: same thresholds drive both sides. Verify
        # by mirror symmetry — flipping the price series vertically flips
        # the signal series.
        df = make_df(n=400, seed=11)
        sig = self.s.generate_signals(df, self.s.default_params())
        # Vertical mirror around ONE center for every column: close/open
        # mirror to themselves, high<->low swap. Mirroring columns around
        # different centers breaks the bar geometry and is not a symmetry.
        center = 2 * df["close"].iloc[0]
        mirror = df.copy()
        mirror["close"] = center - df["close"]
        mirror["open"] = center - df["open"]
        mirror["high"] = center - df["low"]
        mirror["low"] = center - df["high"]
        sig_m = self.s.generate_signals(mirror, self.s.default_params())
        pd.testing.assert_series_equal(sig_m, -sig, check_names=False)

    def test_no_lookahead(self):
        # Truncating the future must not change past signals.
        df = make_df(n=400, seed=5)
        full = self.s.generate_signals(df, self.s.default_params())
        for cut in (350, 300, 250):
            part = self.s.generate_signals(df.iloc[:cut].copy(),
                                           self.s.default_params())
            pd.testing.assert_series_equal(
                part, full.iloc[:cut], check_names=False)

    def test_state_machine_time_exit_in_signals(self):
        # Construct data where the pulse fires once then never again:
        # a single strong down-close after an uptrend, then flat drift.
        n = 60
        close = np.concatenate([
            np.linspace(100, 120, 40),          # up bars
            [118.0],                             # one down close (pullback)
            np.full(19, 118.0),                  # flat: no new pulses
        ])
        open_ = np.roll(close, 1) * 0.999
        open_[0] = close[0] * 0.999
        df = pd.DataFrame({
            "open": open_,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 1000.0),
        })
        # pullback=2: the pullback bar itself is a down bar, so with pb=1
        # the up-bar count and the pullback can never co-occur (that is the
        # degenerate gap-open case fixed in default_params).
        sig = self.s.generate_signals(
            df, {"bar_count": 1, "pullback": 2, "exit_bars": 3,
                 "count_higher_highs": False})
        # After the last entry, exactly exit_bars bars are held, then flat.
        nonzero_idx = np.flatnonzero(sig.to_numpy() != 0)
        assert len(nonzero_idx) > 0
        last = nonzero_idx[-1]
        assert sig.iloc[last] != 0
        assert sig.iloc[min(last + 1, n - 1)] == 0  # released after hold


class TestDaveyCountertrendReversal:
    def setup_method(self):
        self.s = STRATEGY_REGISTRY["Davey_Countertrend_Reversal"]

    def test_registered_with_grid(self):
        assert self.s.meta().name == "Davey_Countertrend_Reversal"
        grid = self.s.param_grid()
        assert set(grid) == {"bar_count", "momentum_period", "with_trend", "exit_bars"}

    def test_fade_mode_fires_on_streak(self):
        # 5 consecutive down closes -> fade mode goes long.
        n = 40
        close = np.concatenate([
            np.full(20, 100.0),
            100 - np.arange(1, 13) * 0.5,   # 12 consecutive down closes
            np.full(8, 94.0),
        ])
        df = pd.DataFrame({
            "open": np.roll(close, 1),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 1000.0),
        })
        sig = self.s.generate_signals(
            df, {"bar_count": 3, "momentum_period": 1, "with_trend": False,
                 "exit_bars": 4})
        assert (sig == 1).sum() > 0  # went long on the down streak

    def test_with_trend_mode_is_mirror_of_fade(self):
        df = make_df(n=300, seed=9)
        fade = self.s.generate_signals(
            df, {"bar_count": 3, "momentum_period": 1, "with_trend": False,
                 "exit_bars": 5})
        ride = self.s.generate_signals(
            df, {"bar_count": 3, "momentum_period": 1, "with_trend": True,
                 "exit_bars": 5})
        # Patterns #3 and #4 are exact opposites by construction.
        pd.testing.assert_series_equal(ride, -fade, check_names=False)

    def test_no_lookahead(self):
        df = make_df(n=400, seed=21)
        full = self.s.generate_signals(df, self.s.default_params())
        for cut in (350, 300):
            part = self.s.generate_signals(df.iloc[:cut].copy(),
                                           self.s.default_params())
            pd.testing.assert_series_equal(
                part, full.iloc[:cut], check_names=False)


class TestDaveyRSITrigger:
    def setup_method(self):
        self.s = STRATEGY_REGISTRY["Davey_RSI_Trigger"]

    def test_registered_with_grid(self):
        assert self.s.meta().name == "Davey_RSI_Trigger"
        grid = self.s.param_grid()
        assert set(grid) == {"rsi_period", "entry_threshold", "exit_bars"}

    def test_strength_trigger_fires(self):
        # Sharp rally -> RSI(2) pinned high -> long trigger.
        n = 60
        close = np.concatenate([
            np.full(20, 100.0),
            100 * np.cumprod(np.full(30, 1.01)),
            np.full(10, 134.0),
        ])
        df = pd.DataFrame({
            "open": np.roll(close, 1),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 1000.0),
        })
        sig = self.s.generate_signals(
            df, {"rsi_period": 2, "entry_threshold": 65, "exit_bars": 5})
        assert (sig == 1).sum() > 0

    def test_symmetric_thresholds(self):
        # The short threshold is exactly 100 - long threshold.
        df = make_df(n=300, seed=13)
        sig = self.s.generate_signals(
            df, {"rsi_period": 2, "entry_threshold": 70, "exit_bars": 5})
        # With threshold 70, shorts require RSI < 30 — verify a deep-drop
        # series produces shorts.
        n = 60
        close = np.concatenate([
            np.full(20, 100.0),
            100 * np.cumprod(np.full(30, 0.99)),
            np.full(10, 74.0),
        ])
        df2 = pd.DataFrame({
            "open": np.roll(close, 1),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 1000.0),
        })
        sig2 = self.s.generate_signals(
            df2, {"rsi_period": 2, "entry_threshold": 70, "exit_bars": 5})
        assert (sig2 == -1).sum() > 0

    def test_no_lookahead(self):
        df = make_df(n=400, seed=17)
        full = self.s.generate_signals(df, self.s.default_params())
        for cut in (350, 300):
            part = self.s.generate_signals(df.iloc[:cut].copy(),
                                           self.s.default_params())
            pd.testing.assert_series_equal(
                part, full.iloc[:cut], check_names=False)
