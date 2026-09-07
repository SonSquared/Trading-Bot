"""
Davey Momentum Pullback Strategy.

Adapted from Kevin J. Davey's price-action patterns #1 and #2
("Directional Bars and Pullback Momentum" and "Higher Highs, Lower Lows
and Pullback Momentum", kjtradingsystems.com, 2020), with his own testing
conventions applied:

- Symmetric long/short entries (same parameters both sides) — Davey's
  curve-fitting safeguard against parameter multiplication.
- Time exit: exit after ``exit_bars`` candles unless a reversal signal
  fires first. Implemented as a signal state machine so the shared
  next-open backtester and the bot's hysteresis behave identically.
- Counted-conditions (bar counts) instead of optimizable indicator
  thresholds where possible — fewer degrees of freedom, fewer ways to
  curve-fit.

Signals are state-based: +1 while a long is "held" by this strategy,
-1 while a short is held, 0 flat. A reversal signal flips the state
immediately; the time exit releases it to 0.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.strategies.base import BaseStrategy, StrategyMeta, pulse_to_state


class DaveyMomentumPullbackStrategy(BaseStrategy):
    """Davey's pullback-momentum entries with a time exit."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Davey_Momentum_Pullback",
            family="momentum",
            description=(
                "Kevin Davey's directional-bars / higher-highs momentum "
                "plus pullback entry, symmetric long/short, time exit"
            ),
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "bar_count": 1,
            # pullback=1 is degenerate when open == previous close (gap-open
            # data): a long then requires an up candle closing below the
            # prior close — self-contradictory. Davey's pullback is a lookback
            # variable, and 2 is the robust choice on such data.
            "pullback": 2,
            "exit_bars": 5,
            "count_higher_highs": False,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "bar_count": [1, 2, 3],
            "pullback": [1, 2, 3],
            "exit_bars": [3, 5, 8],
            "count_higher_highs": [False, True],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        bcount = int(params.get("bar_count", 1))
        pullback = int(params.get("pullback", 1))
        exit_bars = max(1, int(params.get("exit_bars", 5)))
        use_hh = bool(params.get("count_higher_highs", False))

        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]

        # Momentum component: directional-bar count or higher-high/lower-low count
        if use_hh:
            up_events = (high > high.shift(1)).astype(int)
            down_events = (low < low.shift(1)).astype(int)
        else:
            up_events = (close - open_ > 0).astype(int)
            down_events = (close - open_ < 0).astype(int)

        up_count = up_events.rolling(bcount, min_periods=1).sum()
        down_count = down_events.rolling(bcount, min_periods=1).sum()

        # Pullback component: current close below/above close N bars ago
        ref_close = close.shift(pullback)
        pullback_long = close < ref_close
        pullback_short = close > ref_close

        long_pulse = (up_count > down_count) & pullback_long
        short_pulse = (down_count > up_count) & pullback_short

        # State machine: hold the position for exit_bars candles unless a
        # reversal pulse flips it first (Davey: "exit after X bars after
        # entry, unless a reversal signal occurs first").
        long_entry = long_pulse & ~(long_pulse.shift(1, fill_value=False))
        short_entry = short_pulse & ~(short_pulse.shift(1, fill_value=False))

        return pulse_to_state(long_entry, short_entry, exit_bars)
