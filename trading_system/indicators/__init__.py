"""
Technical indicators library.

All indicators are strictly causal — they only use data available up to the current candle.
"""

from trading_system.indicators.trend import (
    sma, ema, dema, wma, macd, adx, donchian_channel, supertrend, parabolic_sar,
)
from trading_system.indicators.momentum import (
    rsi, roc, stochastic, williams_r, cci, mfi, stoch_rsi, awesome_oscillator,
    ultimate_oscillator,
)
from trading_system.indicators.volatility import (
    atr, true_range, bollinger_bands, keltner_channel, historical_volatility,
    normalized_atr, chandelier_exit, volatility_regime,
)
from trading_system.indicators.volume import (
    obv, vwap, volume_sma, relative_volume, accumulation_distribution,
    chaikin_money_flow, volume_profile,
)
from trading_system.indicators.utils import (
    crossover, crossunder, above, below, zscore, percentile_rank,
    returns, log_returns, rolling_max, rolling_min, rolling_rank, heikin_ashi,
)

__all__ = [
    # Trend
    "sma", "ema", "dema", "wma", "macd", "adx", "donchian_channel",
    "supertrend", "parabolic_sar",
    # Momentum
    "rsi", "roc", "stochastic", "williams_r", "cci", "mfi", "stoch_rsi",
    "awesome_oscillator", "ultimate_oscillator",
    # Volatility
    "atr", "true_range", "bollinger_bands", "keltner_channel",
    "historical_volatility", "normalized_atr", "chandelier_exit", "volatility_regime",
    # Volume
    "obv", "vwap", "volume_sma", "relative_volume", "accumulation_distribution",
    "chaikin_money_flow", "volume_profile",
    # Utils
    "crossover", "crossunder", "above", "below", "zscore", "percentile_rank",
    "returns", "log_returns", "rolling_max", "rolling_min", "rolling_rank",
    "heikin_ashi",
]
