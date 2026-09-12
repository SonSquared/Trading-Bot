"""
Market data collector for the AI trading bot.

Gathers prices, technical indicators, and funding rates from Binance
and formats them into human-readable context blocks for the AI engine.

Indicator API notes — this project's library is DataFrame/Series based:
    rsi(close_series, period)                      -> Series
    ema(close_series, period)                      -> Series
    macd(close_series)                             -> DataFrame(macd, signal, histogram)
    bollinger_bands(close_series, period, std)     -> DataFrame(upper, middle, lower, ...)
    atr(df, period)                                -> Series   (df needs high/low/close)
    adx(df, period)                                -> DataFrame(adx, plus_di, minus_di)
    stochastic(df)                                 -> DataFrame(k, d)
    volume_sma(df, period)                         -> Series   (uses df["volume"])

All indicator values are NaN during their warm-up window, so every value
pulled from a series goes through _last(), which returns None for
NaN/inf/missing values. Formatting renders those as "N/A (warming up)"
instead of NaN leaking into the AI prompt.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from trading_system.bot.exchange import ExchangeInterface
from trading_system.indicators import (
    adx,
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    stochastic,
    volume_sma,
)

logger = structlog.get_logger(__name__)

MIN_CANDLES = 50


def _fmt_price(price: float) -> str:
    if price >= 1000:
        return f"${price:,.2f}"
    if price >= 1:
        return f"${price:.4f}"
    return f"${price:.6f}"


def _fmt_pct(pct: float) -> str:
    return f"{pct:+.2f}%"


def _fmt_volume(vol: float) -> str:
    if vol >= 1e9:
        return f"${vol / 1e9:.2f}B"
    if vol >= 1e6:
        return f"${vol / 1e6:.2f}M"
    if vol >= 1e3:
        return f"${vol / 1e3:.1f}K"
    return f"${vol:.0f}"


def _fmt_opt(value: float | None, fmt) -> str:
    """Format a value that may be None (indicator still warming up)."""
    return fmt(value) if value is not None else "N/A (warming up)"


def _last(series: pd.Series | None) -> float | None:
    """Last finite value of a series, or None if empty/NaN/inf."""
    if series is None or len(series) == 0:
        return None
    try:
        val = float(series.iloc[-1])
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def compute_indicators(df: pd.DataFrame) -> dict[str, Any]:
    """Compute technical indicators from OHLCV data.

    Returns {"ok": True, ...latest values...} or {"ok": False, "error": ...}
    when there is not enough usable data. Never raises.
    """
    required = {"open", "high", "low", "close", "volume"}
    if df is None or df.empty or not required.issubset(set(df.columns)):
        return {"ok": False, "error": "missing or empty OHLCV data"}

    n = len(df)
    if n < MIN_CANDLES:
        return {"ok": False, "error": f"insufficient data ({n} candles, need >= {MIN_CANDLES})"}

    close = df["close"]
    high = df["high"]
    low = df["low"]

    ind: dict[str, Any] = {"ok": True}

    price = _last(close)
    if price is None or price <= 0:
        return {"ok": False, "error": "no finite close prices"}
    ind["price"] = price

    prev = _last(close.iloc[:-1])
    ind["price_change_1h"] = (price - prev) / prev * 100 if prev else 0.0
    prev24 = float(close.iloc[-25]) if n >= 25 else 0.0
    ind["price_change_24h"] = (price - prev24) / prev24 * 100 if prev24 else 0.0

    # --- Trend ---
    ind["ema_9"] = _last(ema(close, 9))
    ind["ema_21"] = _last(ema(close, 21))
    ind["ema_50"] = _last(ema(close, 50))
    e9, e21, e50 = ind["ema_9"], ind["ema_21"], ind["ema_50"]
    if None not in (e9, e21, e50):
        if e9 > e21 > e50:
            ind["ema_trend"] = "bullish (EMA9 > EMA21 > EMA50)"
        elif e9 < e21 < e50:
            ind["ema_trend"] = "bearish (EMA9 < EMA21 < EMA50)"
        else:
            ind["ema_trend"] = "mixed"
    else:
        ind["ema_trend"] = "warming up"

    ind["rsi_14"] = _last(rsi(close, 14))

    macd_df = macd(close)
    ind["macd"] = _last(macd_df["macd"])
    ind["macd_signal"] = _last(macd_df["signal"])
    ind["macd_histogram"] = _last(macd_df["histogram"])
    hist, hist_prev = ind["macd_histogram"], _last(macd_df["histogram"].iloc[:-1])
    if hist is not None and hist_prev is not None:
        ind["macd_cross"] = (
            "bullish" if hist > 0 and hist_prev <= 0
            else "bearish" if hist < 0 and hist_prev >= 0
            else "none"
        )
    else:
        ind["macd_cross"] = "warming up"

    # --- Volatility ---
    bb = bollinger_bands(close, 20, 2.0)
    ind["bb_upper"] = _last(bb["upper"])
    ind["bb_middle"] = _last(bb["middle"])
    ind["bb_lower"] = _last(bb["lower"])
    if None not in (ind["bb_upper"], ind["bb_lower"], ind["bb_middle"]) and ind["bb_middle"]:
        ind["bb_width_pct"] = (ind["bb_upper"] - ind["bb_lower"]) / ind["bb_middle"] * 100
    else:
        ind["bb_width_pct"] = None

    atr_14 = _last(atr(df, 14))
    ind["atr_14"] = atr_14
    ind["atr_pct"] = atr_14 / price * 100 if atr_14 else None

    # --- Momentum extras (project API: these take the DataFrame) ---
    st = stochastic(df)
    ind["stoch_k"] = _last(st["k"])
    ind["stoch_d"] = _last(st["d"])

    ind["adx"] = _last(adx(df, 14)["adx"])

    # --- Volume ---
    vol_now = _last(df["volume"])
    vol_avg = _last(volume_sma(df, 20))
    ind["volume"] = vol_now
    ind["volume_sma_20"] = vol_avg
    ind["volume_ratio"] = vol_now / vol_avg if (vol_now is not None and vol_avg) else None

    # --- Key levels ---
    ind["recent_high_24"] = float(high.tail(25).max())
    ind["recent_low_24"] = float(low.tail(25).min())
    if n >= 168:
        ind["recent_high_7d"] = float(high.tail(168).max())
        ind["recent_low_7d"] = float(low.tail(168).min())
    else:
        ind["recent_high_7d"] = None
        ind["recent_low_7d"] = None

    return ind


def format_indicators(pair: str, ind: dict[str, Any], funding_rate: float = 0.0) -> str:
    """Format a computed indicator dict into a readable block for the AI."""
    if not ind.get("ok"):
        return f"Pair: {pair}\nData unavailable: {ind.get('error', 'unknown error')}"

    rsi_val = ind["rsi_14"]
    adx_val = ind["adx"]
    vol_ratio = ind["volume_ratio"]

    lines = [
        f"Pair: {pair}",
        f"Current Price: {_fmt_price(ind['price'])}",
        f"1h Change: {_fmt_pct(ind['price_change_1h'])}",
        f"24h Change: {_fmt_pct(ind['price_change_24h'])}",
        "",
        "--- Trend ---",
        f"EMA 9: {_fmt_opt(ind['ema_9'], _fmt_price)}",
        f"EMA 21: {_fmt_opt(ind['ema_21'], _fmt_price)}",
        f"EMA 50: {_fmt_opt(ind['ema_50'], _fmt_price)}",
        f"EMA Trend: {ind['ema_trend']}",
        f"ADX: {_fmt_opt(adx_val, lambda v: f'{v:.1f}')}"
        + (
            f" {'(strong trend)' if adx_val > 25 else '(weak trend)' if adx_val < 20 else '(moderate)'}"
            if adx_val is not None else ""
        ),
        "",
        "--- Momentum ---",
        f"RSI(14): {_fmt_opt(rsi_val, lambda v: f'{v:.1f}')}"
        + (
            f" {'(oversold)' if rsi_val < 30 else '(overbought)' if rsi_val > 70 else '(neutral)'}"
            if rsi_val is not None else ""
        ),
        f"MACD: {_fmt_opt(ind['macd'], lambda v: f'{v:.4f}')}"
        f" | Signal: {_fmt_opt(ind['macd_signal'], lambda v: f'{v:.4f}')}",
        f"MACD Histogram: {_fmt_opt(ind['macd_histogram'], lambda v: f'{v:.4f}')}"
        f" ({ind['macd_cross']})",
        f"Stochastic K/D: {_fmt_opt(ind['stoch_k'], lambda v: f'{v:.1f}')}"
        f" / {_fmt_opt(ind['stoch_d'], lambda v: f'{v:.1f}')}",
        "",
        "--- Volatility ---",
        f"ATR(14): {_fmt_opt(ind['atr_14'], _fmt_price)}"
        + (f" ({ind['atr_pct']:.2f}% of price)" if ind["atr_pct"] else ""),
        f"Bollinger Width: {_fmt_opt(ind['bb_width_pct'], lambda v: f'{v:.2f}%')}",
        f"BB Upper: {_fmt_opt(ind['bb_upper'], _fmt_price)}",
        f"BB Middle: {_fmt_opt(ind['bb_middle'], _fmt_price)}",
        f"BB Lower: {_fmt_opt(ind['bb_lower'], _fmt_price)}",
        "",
        "--- Volume ---",
        f"Current: {_fmt_opt(ind['volume'], _fmt_volume)}",
        f"20-period Avg: {_fmt_opt(ind['volume_sma_20'], _fmt_volume)}",
        f"Volume Ratio: {_fmt_opt(vol_ratio, lambda v: f'{v:.2f}x')}"
        + (
            f" {'(elevated)' if vol_ratio > 1.5 else '(low)' if vol_ratio < 0.5 else '(normal)'}"
            if vol_ratio is not None else ""
        ),
        "",
        "--- Key Levels ---",
        f"24h High: {_fmt_price(ind['recent_high_24'])}",
        f"24h Low: {_fmt_price(ind['recent_low_24'])}",
    ]

    if ind.get("recent_high_7d") is not None:
        lines.append(f"7d High: {_fmt_price(ind['recent_high_7d'])}")
        lines.append(f"7d Low: {_fmt_price(ind['recent_low_7d'])}")

    lines.append("")
    lines.append(f"Funding Rate: {funding_rate * 100:.4f}%")

    return "\n".join(lines)


def format_market_context(pair: str, df: pd.DataFrame, funding_rate: float = 0.0) -> str:
    """Convenience wrapper: compute indicators from a DataFrame, then format."""
    return format_indicators(pair, compute_indicators(df), funding_rate)


def fetch_market_context(
    exchange: ExchangeInterface,
    pairs: list[str],
    timeframe: str = "1h",
) -> tuple[str, bool]:
    """Fetch and format market context for all configured pairs.

    Returns (context_text, ok) where ok is True only if at least one pair
    produced a complete, computable indicator set.
    """
    contexts: list[str] = []
    any_ok = False

    for pair in pairs:
        try:
            df = exchange.get_ohlcv(pair, timeframe, limit=200)
            funding = exchange.get_funding_rate(pair)
            ind = compute_indicators(df)
            if ind.get("ok"):
                any_ok = True
                contexts.append(format_indicators(pair, ind, funding))
            else:
                contexts.append(
                    f"Pair: {pair} - data unavailable: {ind.get('error', 'unknown')}"
                )
        except Exception as e:
            logger.error("market_data_fetch_failed", pair=pair, error=str(e))
            contexts.append(f"Pair: {pair} - fetch error: {e}")

    return "\n\n".join(contexts), any_ok
