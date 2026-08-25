"""
Trading strategies framework.

18 strategies across 4 families:
- Trend Following (5)
- Momentum (4)
- Mean Reversion (4)
- Volatility (4)
"""

from trading_system.strategies.trend import TREND_STRATEGIES
from trading_system.strategies.momentum import MOMENTUM_STRATEGIES
from trading_system.strategies.mean_reversion import MEAN_REVERSION_STRATEGIES
from trading_system.strategies.volatility import VOLATILITY_STRATEGIES
from trading_system.strategies.base import BaseStrategy, StrategyMeta

# Master list of all strategies
ALL_STRATEGIES = (
    TREND_STRATEGIES
    + MOMENTUM_STRATEGIES
    + MEAN_REVERSION_STRATEGIES
    + VOLATILITY_STRATEGIES
)

# Lookup by name
STRATEGY_REGISTRY: dict[str, BaseStrategy] = {
    s.meta().name: s for s in ALL_STRATEGIES
}

# Lookup by family
STRATEGIES_BY_FAMILY: dict[str, list[BaseStrategy]] = {
    "trend": TREND_STRATEGIES,
    "momentum": MOMENTUM_STRATEGIES,
    "mean_reversion": MEAN_REVERSION_STRATEGIES,
    "volatility": VOLATILITY_STRATEGIES,
}


def get_strategy(name: str) -> BaseStrategy:
    """Get a strategy by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]


def get_strategies_by_family(family: str) -> list[BaseStrategy]:
    """Get all strategies in a family."""
    return STRATEGIES_BY_FAMILY.get(family, [])


__all__ = [
    "ALL_STRATEGIES", "STRATEGY_REGISTRY", "STRATEGIES_BY_FAMILY",
    "get_strategy", "get_strategies_by_family",
    "BaseStrategy", "StrategyMeta",
]
