"""
Davey RSI Trigger Strategy.

Adapted from Kevin J. Davey's published short-term RSI trigger ("Strategy
2 - 2 Period RSI: if RSI(2) > Threshold then buy", kjtradingsystems.com),
a short-horizon strength trigger. Davey's conventions preserved:

- Symmetric long/short entries: long when RSI(period) > entry_threshold
  (buying strength), short when RSI(period) < 100 - entry_threshold
  (selling weakness). One threshold parameter drives both sides, per
  his symmetry principle against curve-fitting.
- Time exit after ``exit_bars`` candles unless a reversal signal fires
  first.
- Signals are state-based (+1 / -1 / 0) so the shared next-open
  backtester and the live bot's hysteresis behave identically.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.indicators import rsi
from trading_system.strategies.base import BaseStrategy, StrategyMeta, pulse_to_state


class DaveyRSITriggerStrategy(BaseStrategy):
    """Davey's short-period RSI strength trigger with a time exit."""

    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name="Davey_RSI_Trigger",
            family="momentum",
            description=(
                "Kevin Davey's 2-period RSI strength trigger "
                "(RSI > threshold buys, symmetric short side), "
                "time exit"
            ),
        )

    def default_params(self) -> dict[str, Any]:
        return {
            "rsi_period": 2,
            "entry_threshold": 65,
            "exit_bars": 5,
        }

    def param_grid(self) -> dict[str, list]:
        return {
            "rsi_period": [2, 3, 4],
            "entry_threshold": [60, 65, 70, 75],
            "exit_bars": [3, 5, 8],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = int(params.get("rsi_period", 2))
        threshold = float(params.get("entry_threshold", 65))
        exit_bars = max(1, int(params.get("exit_bars", 5)))

        r = rsi(df["close"], period)
        long_pulse = r > threshold          # buying strength
        short_pulse = r < (100.0 - threshold)  # symmetric: selling weakness
        # NaN warm-up rows produce no pulse by construction (NaN compares False)

        long_entry = long_pulse & ~(long_pulse.shift(1, fill_value=False))
        short_entry = short_pulse & ~(short_pulse.shift(1, fill_value=False))

        return pulse_to_state(long_entry, short_entry, exit_bars)
