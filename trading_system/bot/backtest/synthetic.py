"""Deterministic synthetic candles, so the harness runs on a clean checkout.

**This is not market data.** It exists for two reasons only:

  * a committed command that reproduces the exact same numbers with no
    download and no network (`--synthetic`), and
  * hermetic fixtures for the tests.

Every report labels a synthetic run as synthetic, because a synthetic result is
evidence about the harness and nothing else. Real conclusions must come from the
stored OHLCV under ``data/raw``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_system.bot.backtest.historical import timeframe_interval

# Rough per-hour volatility and starting prices, only so the synthetic series
# moves on a plausible scale. Not calibrated to anything.
_DEFAULT_VOL = {"BTC/USDT:USDT": 0.011, "ETH/USDT:USDT": 0.014}
_DEFAULT_PRICE = {"BTC/USDT:USDT": 30000.0, "ETH/USDT:USDT": 2000.0}

# One slow cycle (up/down/chop) across the generated span, so different windows
# of a synthetic run differ in character the way real regimes do.
_CYCLE_HOURS = 24 * 120


def build_synthetic_frames(
    pairs: list[str],
    timeframe: str = "1h",
    start: str = "2022-01-01",
    days: int = 400,
    seed: int = 11,
    volatility: dict[str, float] | None = None,
    price: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame]:
    """One deterministic OHLCV frame per pair (same inputs -> same bytes)."""
    interval = timeframe_interval(timeframe)
    bars_per_day = int(pd.Timedelta(days=1) / interval)
    periods = max(bars_per_day * days, 500)
    index = pd.date_range(
        pd.Timestamp(start, tz="UTC"), periods=periods, freq=interval,
    )
    vol_cfg = {**_DEFAULT_VOL, **(volatility or {})}
    price_cfg = {**_DEFAULT_PRICE, **(price or {})}

    frames: dict[str, pd.DataFrame] = {}
    for offset, pair in enumerate(sorted(pairs)):
        rng = np.random.default_rng(seed + offset)
        sigma = float(vol_cfg.get(pair, 0.012))
        cycle = np.sin(np.linspace(0, 2 * np.pi * periods / _CYCLE_HOURS, periods))
        drift = cycle * sigma * 0.08
        log_returns = drift + rng.normal(0.0, sigma, periods)
        close = float(price_cfg.get(pair, 1000.0)) * np.exp(np.cumsum(log_returns))
        open_ = np.concatenate([[close[0]], close[:-1]])
        span = np.abs(rng.normal(0.0, sigma * 0.6, periods)) + 1e-9
        high = np.maximum(open_, close) * (1.0 + span)
        low = np.minimum(open_, close) * (1.0 - span)
        volume = np.abs(rng.normal(1000.0, 200.0, periods)) + 1.0
        frames[pair] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close,
             "volume": volume},
            index=index,
        )
    return frames
