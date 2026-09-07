"""Momentum strategies."""

from trading_system.strategies.momentum.rsi_momentum import RSIMomentumStrategy
from trading_system.strategies.momentum.roc_momentum import ROCMomentumStrategy
from trading_system.strategies.momentum.stochastic_momentum import StochasticMomentumStrategy
from trading_system.strategies.momentum.multi_momentum import MultiMomentumStrategy
from trading_system.strategies.momentum.davey_momentum_pullback import DaveyMomentumPullbackStrategy
from trading_system.strategies.momentum.davey_rsi_trigger import DaveyRSITriggerStrategy

MOMENTUM_STRATEGIES = [
    RSIMomentumStrategy(),
    ROCMomentumStrategy(),
    StochasticMomentumStrategy(),
    MultiMomentumStrategy(),
    DaveyMomentumPullbackStrategy(),
    DaveyRSITriggerStrategy(),
]

__all__ = [
    "RSIMomentumStrategy", "ROCMomentumStrategy",
    "StochasticMomentumStrategy", "MultiMomentumStrategy",
    "DaveyMomentumPullbackStrategy", "DaveyRSITriggerStrategy",
    "MOMENTUM_STRATEGIES",
]
