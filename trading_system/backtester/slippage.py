"""
Slippage models for realistic execution simulation.

Supports: fixed slippage, ATR-adaptive slippage, and none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.config import SlippageConfig
from trading_system.indicators import atr


class SlippageModel:
    """Base slippage model."""

    def __init__(self, config: SlippageConfig):
        self.config = config

    def calculate_slippage(
        self,
        price: float,
        position_size: float,
        is_long: bool,
        candle_data: pd.Series | None = None,
        atr_series: pd.Series | None = None,
    ) -> float:
        """
        Calculate slippage for a trade.

        Returns the slippage-adjusted execution price.
        Positive slippage means worse fill (costs money).
        """
        raise NotImplementedError


class FixedSlippage(SlippageModel):
    """Fixed percentage slippage."""

    def calculate_slippage(
        self,
        price: float,
        position_size: float,
        is_long: bool,
        candle_data: pd.Series | None = None,
        atr_series: pd.Series | None = None,
    ) -> float:
        slippage_pct = self.config.base_slippage
        direction = 1 if is_long else -1
        return price * (1 + direction * slippage_pct)


class ATRAdaptiveSlippage(SlippageModel):
    """ATR-based adaptive slippage — more slippage in volatile markets."""

    def calculate_slippage(
        self,
        price: float,
        position_size: float,
        is_long: bool,
        candle_data: pd.Series | None = None,
        atr_series: pd.Series | None = None,
    ) -> float:
        base = self.config.base_slippage

        if atr_series is not None and candle_data is not None and len(atr_series) > 0:
            idx = candle_data.name if candle_data.name is not None else atr_series.index[-1]
            if idx in atr_series.index:
                current_atr = atr_series.loc[idx]
                atr_pct = current_atr / price if price > 0 else 0
                slippage_pct = base + self.config.atr_multiplier * atr_pct
            else:
                slippage_pct = base
        else:
            slippage_pct = base

        direction = 1 if is_long else -1
        return price * (1 + direction * slippage_pct)


class NoSlippage(SlippageModel):
    """No slippage (ideal execution)."""

    def calculate_slippage(
        self,
        price: float,
        position_size: float,
        is_long: bool,
        candle_data: pd.Series | None = None,
        atr_series: pd.Series | None = None,
    ) -> float:
        return price


def create_slippage_model(config: SlippageConfig) -> SlippageModel:
    """Factory function to create the appropriate slippage model."""
    models = {
        "fixed": FixedSlippage,
        "atr_adaptive": ATRAdaptiveSlippage,
        "none": NoSlippage,
    }
    model_cls = models.get(config.model, ATRAdaptiveSlippage)
    return model_cls(config)


def calculate_slippage_cost(
    intended_price: float,
    actual_price: float,
    position_size: float,
) -> float:
    """Calculate the monetary cost of slippage."""
    return abs(actual_price - intended_price) * abs(position_size)
