"""
Dynamic Risk Manager with Regime-Aware Position Sizing.

Instead of switching strategies during choppy markets (which failed —
no mean reversion strategy works on this data), this module keeps the
existing strategies but scales position sizes based on the current
market regime:

  TRENDING:     Full position size (1.0x multiplier)
  CHOPPY:       Reduced position size (0.3x multiplier)
  TRANSITIONAL: Moderate position size (0.7x multiplier)

This is implemented as a pre-computed multiplier series that can be
applied to any backtest or live trading configuration.

Performance: Pre-computes all indicators once, then classifies
regime in a fast numpy loop (no repeated indicator calculation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from trading_system.indicators import adx, atr, bollinger_bands

logger = structlog.get_logger(__name__)


@dataclass
class RegimeRiskConfig:
    """Configuration for regime-aware risk scaling."""
    # Position size multipliers per regime
    trending_mult: float = 1.0     # Full size in trends
    transitional_mult: float = 0.7  # Slightly reduced during transitions
    choppy_mult: float = 0.3       # Aggressive reduction in choppy markets

    # Regime detector parameters
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    adx_choppy_threshold: float = 20.0
    bb_period: int = 20
    bb_std: float = 2.0
    bb_width_lookback: int = 50
    atr_period: int = 14
    atr_lookback: int = 20

    # Smoothing to prevent rapid regime flickering
    min_regime_duration: int = 3  # candles to stay in a regime before switching

    def get_multiplier(self, regime_val: str, confidence: float) -> float:
        """Get the risk multiplier for a regime string and confidence."""
        base = {
            "trending": self.trending_mult,
            "choppy": self.choppy_mult,
            "transitional": self.transitional_mult,
        }[regime_val]

        neutral = self.transitional_mult
        return neutral + (base - neutral) * confidence


class DynamicRiskManager:
    """
    Pre-computes a regime-based risk multiplier for every candle.

    Optimized: computes all indicators once, then classifies regime
    in a fast numpy loop (~0.5s for 10K candles instead of ~30min).
    """

    def __init__(self, config: RegimeRiskConfig | None = None):
        self.config = config or RegimeRiskConfig()

    def compute_multipliers(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a regime-based risk multiplier for each candle.

        Pre-computes ADX, BB Width, and ATR once, then classifies
        regime in a fast loop.
        """
        n = len(df)
        multipliers = np.full(n, self.config.transitional_mult, dtype=np.float64)

        warmup = max(self.config.adx_period, self.config.bb_period,
                     self.config.atr_period) + 30

        if n < warmup:
            return pd.Series(multipliers, index=df.index)

        # ── Pre-compute all indicators ONCE ───────────────────────
        adx_df = adx(df, self.config.adx_period)
        adx_values = adx_df["adx"].values.astype(np.float64)

        plus_di = adx_df["plus_di"].values.astype(np.float64)
        minus_di = adx_df["minus_di"].values.astype(np.float64)

        bb = bollinger_bands(df["close"], self.config.bb_period, self.config.bb_std)
        bb_width_raw = ((bb["upper"] - bb["lower"]) / bb["middle"]).values.astype(np.float64)

        atr_series = atr(df, self.config.atr_period)
        atr_values = atr_series.values.astype(np.float64)
        close_vals = df["close"].values.astype(np.float64)

        # ── Classify regime at each candle ────────────────────────
        prev_regime = "transitional"
        regime_count = 0

        for i in range(warmup, n):
            # ADX vote
            adx_val = adx_values[i]
            if np.isnan(adx_val):
                adx_vote = "transitional"
                adx_conf = 0.5
            elif adx_val > self.config.adx_trend_threshold:
                adx_vote = "trending"
                adx_conf = min((adx_val - self.config.adx_trend_threshold) / 20.0, 1.0)
            elif adx_val < self.config.adx_choppy_threshold:
                adx_vote = "choppy"
                adx_conf = min((self.config.adx_choppy_threshold - adx_val) / 15.0, 1.0)
            else:
                adx_vote = "transitional"
                adx_conf = 0.5

            # Trend direction
            pdi = plus_di[i] if not np.isnan(plus_di[i]) else 0
            mdi = minus_di[i] if not np.isnan(minus_di[i]) else 0

            # BB Width vote
            bw = bb_width_raw[i] if not np.isnan(bb_width_raw[i]) else 1.0
            bw_avg = np.nanmean(bb_width_raw[max(0, i - self.config.bb_width_lookback):i + 1])
            if bw_avg <= 0:
                bw_avg = 1.0
            bw_ratio = bw / bw_avg

            if bw_ratio > 1.3:
                bb_vote = "trending"
            elif bw_ratio < 0.8:
                bb_vote = "choppy"
            else:
                bb_vote = "transitional"

            # ATR vote
            atr_val = atr_values[i] if not np.isnan(atr_values[i]) else 0
            atr_avg = np.nanmean(atr_values[max(0, i - self.config.atr_lookback):i + 1])
            if atr_avg <= 0:
                atr_avg = 1.0
            atr_ratio = atr_val / atr_avg

            if atr_ratio > 1.2:
                atr_vote = "trending"
            elif atr_ratio < 0.85:
                atr_vote = "choppy"
            else:
                atr_vote = "transitional"

            # Majority vote with ADX as primary
            votes = {"trending": 0, "choppy": 0, "transitional": 0}
            votes[adx_vote] += 2  # ADX double weight
            votes[bb_vote] += 1
            votes[atr_vote] += 1

            if votes["trending"] >= 3:
                regime = "trending"
                confidence = np.mean([adx_conf, bw_ratio / 2, atr_ratio / 2])
                confidence = min(confidence, 1.0)
            elif votes["choppy"] >= 3:
                regime = "choppy"
                confidence = np.mean([adx_conf, 1.0 / max(bw_ratio, 0.1), 1.0 / max(atr_ratio, 0.1)])
                confidence = min(confidence, 1.0)
            else:
                regime = "transitional"
                confidence = 0.5

            # Enforce minimum regime duration (prevent flickering)
            if regime != prev_regime:
                regime_count = 1
            else:
                regime_count += 1

            if regime_count < self.config.min_regime_duration:
                effective_regime = prev_regime
                effective_confidence = 0.5
            else:
                effective_regime = regime
                effective_confidence = confidence

            multipliers[i] = self.config.get_multiplier(effective_regime, effective_confidence)
            prev_regime = effective_regime

        return pd.Series(multipliers, index=df.index)

    def compute_regime_labels(self, df: pd.DataFrame) -> pd.Series:
        """Compute regime labels for each candle (for analysis)."""
        n = len(df)
        warmup = max(self.config.adx_period, self.config.bb_period,
                     self.config.atr_period) + 30

        labels = np.full(n, "transitional", dtype=object)

        if n < warmup:
            return pd.Series(labels, index=df.index)

        adx_df = adx(df, self.config.adx_period)
        adx_values = adx_df["adx"].values.astype(np.float64)
        bb = bollinger_bands(df["close"], self.config.bb_period, self.config.bb_std)
        bb_width_raw = ((bb["upper"] - bb["lower"]) / bb["middle"]).values.astype(np.float64)
        atr_series = atr(df, self.config.atr_period)
        atr_values = atr_series.values.astype(np.float64)

        prev_regime = "transitional"
        regime_count = 0

        for i in range(warmup, n):
            adx_val = adx_values[i]
            if np.isnan(adx_val):
                adx_vote = "transitional"
            elif adx_val > self.config.adx_trend_threshold:
                adx_vote = "trending"
            elif adx_val < self.config.adx_choppy_threshold:
                adx_vote = "choppy"
            else:
                adx_vote = "transitional"

            bw = bb_width_raw[i] if not np.isnan(bb_width_raw[i]) else 1.0
            bw_avg = np.nanmean(bb_width_raw[max(0, i - self.config.bb_width_lookback):i + 1])
            if bw_avg <= 0:
                bw_avg = 1.0
            bw_ratio = bw / bw_avg

            if bw_ratio > 1.3:
                bb_vote = "trending"
            elif bw_ratio < 0.8:
                bb_vote = "choppy"
            else:
                bb_vote = "transitional"

            atr_val = atr_values[i] if not np.isnan(atr_values[i]) else 0
            atr_avg = np.nanmean(atr_values[max(0, i - self.config.atr_lookback):i + 1])
            if atr_avg <= 0:
                atr_avg = 1.0
            atr_ratio = atr_val / atr_avg

            if atr_ratio > 1.2:
                atr_vote = "trending"
            elif atr_ratio < 0.85:
                atr_vote = "choppy"
            else:
                atr_vote = "transitional"

            votes = {"trending": 0, "choppy": 0, "transitional": 0}
            votes[adx_vote] += 2
            votes[bb_vote] += 1
            votes[atr_vote] += 1

            if votes["trending"] >= 3:
                regime = "trending"
            elif votes["choppy"] >= 3:
                regime = "choppy"
            else:
                regime = "transitional"

            if regime != prev_regime:
                regime_count = 1
            else:
                regime_count += 1

            if regime_count < self.config.min_regime_duration:
                labels[i] = prev_regime
            else:
                labels[i] = regime

            prev_regime = labels[i]

        return pd.Series(labels, index=df.index)

    def get_regime_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of regime distribution."""
        labels = self.compute_regime_labels(df)
        multipliers = self.compute_multipliers(df)

        summary = {}
        for regime in ["trending", "transitional", "choppy"]:
            mask = labels == regime
            count = int(mask.sum())
            if count > 0:
                avg_mult = float(multipliers[mask].mean())
            else:
                avg_mult = 0.0

            summary[regime] = {
                "count": count,
                "pct": count / len(labels) * 100,
                "avg_multiplier": avg_mult,
            }

        summary["overall"] = {
            "mean_multiplier": float(multipliers.mean()),
            "min_multiplier": float(multipliers.min()),
            "max_multiplier": float(multipliers.max()),
        }

        return summary
