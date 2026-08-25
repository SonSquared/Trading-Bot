"""Trend following strategies."""

from trading_system.strategies.trend.ma_crossover import MACrossoverStrategy
from trading_system.strategies.trend.macd_strategy import MACDStrategy
from trading_system.strategies.trend.adx_trend import ADXTrendStrategy
from trading_system.strategies.trend.donchian_breakout import DonchianBreakoutStrategy
from trading_system.strategies.trend.supertrend import SupertrendStrategy

TREND_STRATEGIES = [
    MACrossoverStrategy(),
    MACDStrategy(),
    ADXTrendStrategy(),
    DonchianBreakoutStrategy(),
    SupertrendStrategy(),
]

__all__ = [
    "MACrossoverStrategy", "MACDStrategy", "ADXTrendStrategy",
    "DonchianBreakoutStrategy", "SupertrendStrategy",
    "TREND_STRATEGIES",
]
