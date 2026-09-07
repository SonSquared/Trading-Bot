"""Mean reversion strategies."""

from trading_system.strategies.mean_reversion.rsi_reversion import RSIReversionStrategy
from trading_system.strategies.mean_reversion.bollinger_reversion import BollingerReversionStrategy
from trading_system.strategies.mean_reversion.zscore_reversion import ZScoreReversionStrategy
from trading_system.strategies.mean_reversion.bb_squeeze import BBSqueezeStrategy
from trading_system.strategies.mean_reversion.davey_countertrend_reversal import DaveyCountertrendReversalStrategy

MEAN_REVERSION_STRATEGIES = [
    RSIReversionStrategy(),
    BollingerReversionStrategy(),
    ZScoreReversionStrategy(),
    BBSqueezeStrategy(),
    DaveyCountertrendReversalStrategy(),
]

__all__ = [
    "RSIReversionStrategy", "BollingerReversionStrategy",
    "ZScoreReversionStrategy", "BBSqueezeStrategy",
    "DaveyCountertrendReversalStrategy",
    "MEAN_REVERSION_STRATEGIES",
]
