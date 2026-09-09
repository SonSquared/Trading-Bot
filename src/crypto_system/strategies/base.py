"""Intent-only strategy protocol and transparent baselines (plan MD Task 6).

Strategies produce **immutable OrderIntent lists** — nothing else. They
cannot place, sign, or route orders; they never see credentials; they
cannot express exchange-specific operations. The risk governor is the only
authorizer downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import pandas as pd

from crypto_system.models import OrderIntent


class BaseStrategy(ABC):
    """Contract: generate() -> list[OrderIntent]. Nothing else is allowed."""

    name: str = "base"
    symbol: str = "BTCUSDT"

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def signal(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        """Return a position series in {-1, 0, +1} indexed like df."""

    def generate(
        self,
        df: pd.DataFrame,
        params: Mapping[str, Any] | None = None,
        *,
        notional: float = 1000.0,
    ) -> list[OrderIntent]:
        """Emit an intent for every position change (next-open semantics are
        applied by the simulator, not here)."""
        params = params or self.default_params()
        pos = self.signal(df, params)
        changes = pos.diff().fillna(pos.iloc[0])
        intents: list[OrderIntent] = []
        for ts, delta in changes.items():
            if delta == 0:
                continue
            side = "long" if delta > 0 else "short"
            intents.append(
                OrderIntent(
                    symbol=self.symbol,
                    side=side,  # type: ignore[arg-type]
                    notional=notional,
                    strategy=self.name,
                )
            )
        return intents


class TrendSleeve(BaseStrategy):
    """Donchian-style trend follower: long new N-bar highs, short new lows."""

    name = "trend"

    def default_params(self) -> dict[str, Any]:
        return {"lookback": 40, "exit": 10}

    def signal(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        n = int(params.get("lookback", 40))
        x = int(params.get("exit", 10))
        upper = df["high"].rolling(n).max().shift(1)
        lower = df["low"].rolling(n).min().shift(1)
        exit_upper = df["high"].rolling(x).max().shift(1)
        exit_lower = df["low"].rolling(x).min().shift(1)

        pos = pd.Series(0.0, index=df.index)
        state = 0.0
        for i in range(len(df)):
            if state == 0:
                if df["close"].iloc[i] > upper.iloc[i]:
                    state = 1.0
                elif df["close"].iloc[i] < lower.iloc[i]:
                    state = -1.0
            elif state == 1 and df["close"].iloc[i] < exit_lower.iloc[i]:
                state = 0.0
            elif state == -1 and df["close"].iloc[i] > exit_upper.iloc[i]:
                state = 0.0
            pos.iloc[i] = state
        return pos


class MeanReversionSleeve(BaseStrategy):
    """Bollinger reversion: fade extreme z-scores, exit at the mean."""

    name = "reversion"

    def default_params(self) -> dict[str, Any]:
        return {"window": 14, "entry": 30, "exit": 55}

    def signal(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.Series:
        window = int(params.get("window", 14))
        entry_pct = float(params.get("entry", 30))
        exit_pct = float(params.get("exit", 55))
        mean = df["close"].rolling(window).mean()
        std = df["close"].rolling(window).std()
        z = (df["close"] - mean) / std
        entry = abs(z) > abs(_z_from_pct(entry_pct))
        exit_ = abs(z) < abs(_z_from_pct(exit_pct))

        pos = pd.Series(0.0, index=df.index)
        state = 0.0
        for i in range(len(df)):
            if state == 0:
                if z.iloc[i] < 0 and entry.iloc[i]:
                    state = 1.0  # fade extreme lows
                elif z.iloc[i] > 0 and entry.iloc[i]:
                    state = -1.0
            elif state == 1 and (exit_.iloc[i] or z.iloc[i] >= 0):
                state = 0.0
            elif state == -1 and (exit_.iloc[i] or z.iloc[i] <= 0):
                state = 0.0
            pos.iloc[i] = state
        return pos


def _z_from_pct(pct: float) -> float:
    """Map a percentile to a z threshold (two-sided normal approx)."""

    # 30 -> ~1.04, 55 -> ~0.76 (two-sided coverage pct)
    from statistics import NormalDist

    return NormalDist().inv_cdf(1 - (100 - pct) / 200)
