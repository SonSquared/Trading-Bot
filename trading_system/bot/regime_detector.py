"""
Market Regime Detector.

Classifies the current market regime as:
  - TRENDING: Strong directional movement (ADX > 25, rising volatility)
  - CHOPPY: Range-bound, sideways market (ADX < 20, low volatility)
  - TRANSITIONAL: Shifting between regimes

Uses a combination of:
  1. ADX (Average Directional Index) — trend strength
  2. Bollinger Band Width — volatility compression/expansion
  3. ATR relative to price — normalized volatility
  4. Price vs SMA — directional bias

Each indicator votes, and the regime is determined by majority vote
with configurable thresholds for regime transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import structlog

from trading_system.indicators import adx, atr, bollinger_bands

logger = structlog.get_logger(__name__)


class MarketRegime(Enum):
    TRENDING = "trending"
    CHOPPY = "choppy"
    TRANSITIONAL = "transitional"


@dataclass
class RegimeState:
    """Current regime classification with confidence."""
    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    adx_value: float
    bb_width: float
    atr_pct: float
    trend_direction: int  # +1 up, -1 down, 0 flat
    indicators: dict  # individual indicator votes

    def is_trending(self) -> bool:
        return self.regime == MarketRegime.TRENDING

    def is_choppy(self) -> bool:
        return self.regime == MarketRegime.CHOPPY

    def trend_weight(self) -> float:
        """How much to weight trend strategies (0.0 to 1.0)."""
        if self.regime == MarketRegime.TRENDING:
            return min(0.7 + self.confidence * 0.3, 1.0)
        elif self.regime == MarketRegime.CHOPPY:
            return max(0.3 - self.confidence * 0.3, 0.0)
        else:
            return 0.5  # Equal split during transitions

    def mr_weight(self) -> float:
        """How much to weight mean-reversion strategies (0.0 to 1.0)."""
        return 1.0 - self.trend_weight()


class RegimeDetector:
    """
    Detects market regime using multiple indicators.

    ADX-based:
      - ADX > 25: trending
      - ADX < 20: choppy
      - ADX 20-25: transitional

    BB Width-based:
      - Width > 1.5x average: expanding (trending)
      - Width < 0.7x average: compressing (choppy/pre-breakout)

    ATR-based:
      - ATR rising: trending
      - ATR falling/flat: choppy
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_choppy_threshold: float = 20.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_width_lookback: int = 50,
        atr_period: int = 14,
        atr_lookback: int = 20,
        ema_smoothing: int = 3,
    ):
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_choppy_threshold = adx_choppy_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_width_lookback = bb_width_lookback
        self.atr_period = atr_period
        self.atr_lookback = atr_lookback
        self.ema_smoothing = ema_smoothing

        self._prev_regime = MarketRegime.TRANSITIONAL

    def detect(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect the current market regime.

        Args:
            df: OHLCV DataFrame with at least 50 candles

        Returns:
            RegimeState with regime classification
        """
        if len(df) < max(self.adx_period, self.bb_period, self.atr_period) + 20:
            return RegimeState(
                regime=MarketRegime.TRANSITIONAL,
                confidence=0.0,
                adx_value=0, bb_width=0, atr_pct=0,
                trend_direction=0, indicators={},
            )

        # ── 1. ADX indicator ──────────────────────────────────────
        adx_df = adx(df, self.adx_period)
        adx_val = float(adx_df["adx"].iloc[-1])
        plus_di = float(adx_df["plus_di"].iloc[-1])
        minus_di = float(adx_df["minus_di"].iloc[-1])

        if adx_val > self.adx_trend_threshold:
            adx_vote = "trending"
            adx_conf = min((adx_val - self.adx_trend_threshold) / 20.0, 1.0)
        elif adx_val < self.adx_choppy_threshold:
            adx_vote = "choppy"
            adx_conf = min((self.adx_choppy_threshold - adx_val) / 15.0, 1.0)
        else:
            adx_vote = "transitional"
            adx_conf = 0.5

        # Trend direction from +DI vs -DI
        if plus_di > minus_di + 5:
            trend_dir = 1
        elif minus_di > plus_di + 5:
            trend_dir = -1
        else:
            trend_dir = 0

        # ── 2. Bollinger Band Width ───────────────────────────────
        bb = bollinger_bands(df["close"], self.bb_period, self.bb_std)
        bb_width = (bb["upper"] - bb["lower"]) / bb["middle"]
        bb_width_val = float(bb_width.iloc[-1])
        bb_width_avg = float(bb_width.iloc[-self.bb_width_lookback:].mean())
        bb_width_ratio = bb_width_val / bb_width_avg if bb_width_avg > 0 else 1.0

        if bb_width_ratio > 1.3:
            bb_vote = "trending"  # Expanding bands = trending
            bb_conf = min((bb_width_ratio - 1.0) / 0.5, 1.0)
        elif bb_width_ratio < 0.8:
            bb_vote = "choppy"  # Compressing bands = range-bound
            bb_conf = min((1.0 - bb_width_ratio) / 0.3, 1.0)
        else:
            bb_vote = "transitional"
            bb_conf = 0.5

        # ── 3. ATR volatility ─────────────────────────────────────
        atr_series = atr(df, self.atr_period)
        atr_val = float(atr_series.iloc[-1])
        atr_pct = atr_val / float(df["close"].iloc[-1]) * 100
        atr_avg = float(atr_series.iloc[-self.atr_lookback:].mean())
        atr_ratio = atr_val / atr_avg if atr_avg > 0 else 1.0

        if atr_ratio > 1.2:
            atr_vote = "trending"  # Rising volatility
            atr_conf = min((atr_ratio - 1.0) / 0.3, 1.0)
        elif atr_ratio < 0.85:
            atr_vote = "choppy"  # Falling volatility
            atr_conf = min((1.0 - atr_ratio) / 0.2, 1.0)
        else:
            atr_vote = "transitional"
            atr_conf = 0.5

        # ── Majority vote with ADX as primary ─────────────────────
        votes = {"adx": adx_vote, "bb": bb_vote, "atr": atr_vote}
        vote_counts = {"trending": 0, "choppy": 0, "transitional": 0}
        for v in votes.values():
            vote_counts[v] += 1

        # ADX gets double weight
        vote_counts[adx_vote] += 1

        if vote_counts["trending"] >= 3:
            regime = MarketRegime.TRENDING
            confidence = np.mean([adx_conf, bb_conf, atr_conf])
        elif vote_counts["choppy"] >= 3:
            regime = MarketRegime.CHOPPY
            confidence = np.mean([adx_conf, bb_conf, atr_conf])
        else:
            regime = MarketRegime.TRANSITIONAL
            confidence = 0.5

        # Smooth regime transitions (avoid flickering)
        if regime != self._prev_regime:
            if regime == MarketRegime.TRENDING and self._prev_regime == MarketRegime.CHOPPY:
                # Big jump — require 2 consecutive confirmations
                regime = MarketRegime.TRANSITIONAL
                confidence = 0.3
            elif regime == MarketRegime.CHOPPY and self._prev_regime == MarketRegime.TRENDING:
                regime = MarketRegime.TRANSITIONAL
                confidence = 0.3

        self._prev_regime = regime

        state = RegimeState(
            regime=regime,
            confidence=confidence,
            adx_value=adx_val,
            bb_width=bb_width_val,
            atr_pct=atr_pct,
            trend_direction=trend_dir,
            indicators=votes,
        )

        logger.info("regime_detected",
                     regime=regime.value,
                     confidence=f"{confidence:.2f}",
                     adx=f"{adx_val:.1f}",
                     bb_ratio=f"{bb_width_ratio:.2f}",
                     atr_ratio=f"{atr_ratio:.2f}")

        return state

    def get_regime_history(self, df: pd.DataFrame, lookback: int = 100) -> list[dict]:
        """Get regime classification history for backtesting."""
        history = []
        for i in range(max(self.adx_period + 20, len(df) - lookback), len(df)):
            sub_df = df.iloc[:i + 1]
            state = self.detect(sub_df)
            history.append({
                "timestamp": str(df.index[i]),
                "regime": state.regime.value,
                "confidence": state.confidence,
                "adx": state.adx_value,
                "trend_weight": state.trend_weight(),
                "mr_weight": state.mr_weight(),
            })
        return history
