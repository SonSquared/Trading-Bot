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

from trading_system.strategies.signal import Signal


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
