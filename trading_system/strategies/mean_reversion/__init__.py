"""Mean reversion strategies."""

from trading_system.strategies.mean_reversion.rsi_reversion import RSIReversionStrategy
from trading_system.strategies.mean_reversion.bollinger_reversion import BollingerReversionStrategy
from trading_system.strategies.mean_reversion.zscore_reversion import ZScoreReversionStrategy
from trading_system.strategies.mean_reversion.bb_squeeze import BBSqueezeStrategy

MEAN_REVERSION_STRATEGIES = [
    RSIReversionStrategy(),
    BollingerReversionStrategy(),
    ZScoreReversionStrategy(),
    BBSqueezeStrategy(),
]

__all__ = [
    "RSIReversionStrategy", "BollingerReversionStrategy",
    "ZScoreReversionStrategy", "BBSqueezeStrategy",
    "MEAN_REVERSION_STRATEGIES",
]
