"""Volatility strategies."""

from trading_system.strategies.volatility.atr_breakout import ATRBreakoutStrategy
from trading_system.strategies.volatility.vol_expansion import VolExpansionStrategy
from trading_system.strategies.volatility.keltner_breakout import KeltnerBreakoutStrategy
from trading_system.strategies.volatility.regime_volatility import RegimeVolatilityStrategy

VOLATILITY_STRATEGIES = [
    ATRBreakoutStrategy(),
    VolExpansionStrategy(),
    KeltnerBreakoutStrategy(),
    RegimeVolatilityStrategy(),
]

__all__ = [
    "ATRBreakoutStrategy", "VolExpansionStrategy",
    "KeltnerBreakoutStrategy", "RegimeVolatilityStrategy",
    "VOLATILITY_STRATEGIES",
]
