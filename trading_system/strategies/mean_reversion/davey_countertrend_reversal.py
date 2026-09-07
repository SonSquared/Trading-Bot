"""
Davey Countertrend Reversal Strategy.

Adapted from Kevin J. Davey's price-action patterns #3 and #4
("Consecutive Up/Down With Momentum" and "Reverse Consecutive Up/Down
With Momentum", kjtradingsystems.com, 2020):

- Pattern #3 (counter-trend, ``with_trend=False``): N consecutive up
  closes plus positive momentum -> go SHORT (fade the stretch);
  symmetric for longs.
- Pattern #4 (with-trend, ``with_trend=True``): N consecutive up closes
  plus positive momentum -> go LONG; symmetric for shorts.

Davey's conventions preserved: symmetric long/short entries, counted
conditions (no optimizable threshold on the streak itself), and a
time exit after ``exit_bars`` candles unless a reversal signal fires
first. Signals are state-based (+1 / -1 / 0).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.strategies.base import BaseStrategy, StrategyMeta, pulse_to_state


class DaveyCountertrendReversalStrategy(BaseStrategy):
    """Davey's consecutive-close streak entries, fade or ride the streak."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Davey_Countertrend_Reversal",
            family="mean_reversion",
            description=(
                "Kevin Davey's consecutive up/down close entries with "
                "momentum confirm, counter-trend or with-trend mode, "
                "symmetric long/short, time exit"
            ),
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "bar_count": 3,
            "momentum_period": 1,
            "with_trend": False,
            "exit_bars": 5,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "bar_count": [2, 3, 4, 5],
            "momentum_period": [1, 3, 5],
            "with_trend": [False, True],
            "exit_bars": [3, 5, 8],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        bcount = int(params.get("bar_count", 3))
        pcount = int(params.get("momentum_period", 1))
        with_trend = bool(params.get("with_trend", False))
        exit_bars = max(1, int(params.get("exit_bars", 5)))

        close = df["close"]

        # Consecutive up/down close counters (Davey's UCounter / DCounter)
        up_step = (close > close.shift(1)).astype(int)
        down_step = (close < close.shift(1)).astype(int)

        up_counter = pd.Series(0, index=df.index, dtype=int)
        down_counter = pd.Series(0, index=df.index, dtype=int)
        up_arr = up_step.to_numpy()
        down_arr = down_step.to_numpy()
        u = 0
        d = 0
        for i in range(len(df)):
            if up_arr[i]:
                u += 1
            else:
                u = 0
            if down_arr[i]:
                d += 1
            else:
                d = 0
            up_counter.iloc[i] = u
            down_counter.iloc[i] = d

        # Momentum over the last PCount bars
        mom_up = close > close.shift(pcount)
        mom_down = close < close.shift(pcount)

        long_pulse = (down_counter >= bcount) & mom_down if not with_trend \
            else (up_counter >= bcount) & mom_up
        short_pulse = (up_counter >= bcount) & mom_up if not with_trend \
            else (down_counter >= bcount) & mom_down

        # State machine with time exit and reversal flip (shared helper).
        long_entry = long_pulse & ~(long_pulse.shift(1, fill_value=False))
        short_entry = short_pulse & ~(short_pulse.shift(1, fill_value=False))

        return pulse_to_state(long_entry, short_entry, exit_bars)
