"""Regime detection for exposure scaling (plan MD Task 5).

The multiplier is a *risk reducer*: it scales the per-trade risk budget
down (to zero) as volatility rises, and never scales risk up. Bounded
[0, 1] by construction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegimeDetector:
    """Piecewise volatility governor.

    vol <= calm      -> 1.0 (full risk budget)
    calm < vol < hot -> linear down to zero
    vol >= hot       -> 0.0 (stand aside)
    """

    calm: float = 0.02
    hot: float = 0.10

    def multiplier(self, volatility: float) -> float:
        if volatility <= self.calm:
            return 1.0
        if volatility >= self.hot:
            return 0.0
        return (self.hot - volatility) / (self.hot - self.calm)
