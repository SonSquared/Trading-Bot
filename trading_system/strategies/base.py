"""
Abstract strategy base class.

All strategies must implement generate_signals() which takes a DataFrame
and parameters dict, and returns a Series of integer signals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd



def pulse_to_state(
    long_entry: pd.Series,
    short_entry: pd.Series,
    exit_bars: int,
) -> pd.Series:
    """Convert entry pulses into a held-state signal series.

    Implements Kevin Davey's time-exit convention: enter on a pulse, hold
    the position for ``exit_bars`` candles, exit to flat unless an opposite
    pulse flips the position first. Returns +1 (long), -1 (short), 0 flat —
    the state encoding shared by the next-open backtester and the live bot.

    Args:
        long_entry: Boolean series of fresh long-entry pulses (true only on
            the first bar of a new long condition, not while it persists).
        short_entry: Boolean series of fresh short-entry pulses.
        exit_bars: Candles to hold before releasing to flat (>= 1).
    """
    exit_bars = max(1, int(exit_bars))
    long_arr = long_entry.to_numpy(dtype=bool)
    short_arr = short_entry.to_numpy(dtype=bool)

    state = 0
    bars_held = 0
    out = pd.Series(0, index=long_entry.index, dtype=int)
    for i in range(len(out)):
        if state == 1:
            if short_arr[i]:
                state = -1
                bars_held = 0
            elif bars_held >= exit_bars:
                state = 0
                bars_held = 0
        elif state == -1:
            if long_arr[i]:
                state = 1
                bars_held = 0
            elif bars_held >= exit_bars:
                state = 0
                bars_held = 0
        else:
            if long_arr[i]:
                state = 1
                bars_held = 0
            elif short_arr[i]:
                state = -1
                bars_held = 0

        if state != 0:
            bars_held += 1
        out.iloc[i] = state
    return out


@dataclass
class StrategyMeta:
    """Metadata about a strategy."""
    name: str
    family: str  # "trend", "momentum", "mean_reversion", "volatility"
    description: str = ""
    param_ranges: dict[str, list] = field(default_factory=dict)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def meta(self) -> StrategyMeta:
        """Return strategy metadata."""
        ...

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        params: dict[str, Any],
    ) -> pd.Series:
        """
        Generate trading signals from OHLCV data.

        IMPORTANT: This function MUST NOT use future data.
        Signals at index i must only use data up to and including index i.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            params: Strategy-specific parameters

        Returns:
            Series of integer signals: -2 (strong short), -1 (short),
            0 (flat), 1 (long), 2 (strong long)
        """
        ...

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Return default parameters for this strategy."""
        ...

    @abstractmethod
    def param_grid(self) -> dict[str, list]:
        """
        Return parameter grid for optimization.

        Each key maps to a list of values to test.
        """
        ...

    def validate_params(self, params: dict[str, Any]) -> bool:
        """Check if parameters are valid for this strategy."""
        return True

    def required_columns(self) -> list[str]:
        """Return the DataFrame columns required by this strategy."""
        return ["open", "high", "low", "close", "volume"]

    def __repr__(self) -> str:
        m = self.meta()
        return f"<{m.name} ({m.family})>"
