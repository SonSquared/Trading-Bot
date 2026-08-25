"""Momentum strategies."""

from trading_system.strategies.momentum.rsi_momentum import RSIMomentumStrategy
from trading_system.strategies.momentum.roc_momentum import ROCMomentumStrategy
from trading_system.strategies.momentum.stochastic_momentum import StochasticMomentumStrategy
from trading_system.strategies.momentum.multi_momentum import MultiMomentumStrategy

MOMENTUM_STRATEGIES = [
    RSIMomentumStrategy(),
    ROCMomentumStrategy(),
    StochasticMomentumStrategy(),
    MultiMomentumStrategy(),
]

__all__ = [
    "RSIMomentumStrategy", "ROCMomentumStrategy",
    "StochasticMomentumStrategy", "MultiMomentumStrategy",
    "MOMENTUM_STRATEGIES",
]
