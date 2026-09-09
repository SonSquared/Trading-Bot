"""Purged walk-forward evaluation and net-of-cost simulation (plan MD Task 6).

Leakage resistance:
- Chronological folds with an embargo gap: train always ends at least
  ``embargo_bars`` before test begins, so no label/feature overlap survives.
- Parameter search runs ONLY on training folds (``select_params``);
  out-of-sample scoring happens afterwards, once, on the untouched test fold.

Simulation semantics (matching the platform's execution model):
- next-open fills: a signal known at bar t is first held from bar t+1,
- net returns deduct taker fees and slippage on every position change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from crypto_system.strategies.base import BaseStrategy


@dataclass(frozen=True)
class Fold:
    train: pd.DataFrame
    test: pd.DataFrame

    def __iter__(self):
        """Allow ``for train_df, test_df in folds`` unpacking."""
        return iter((self.train, self.test))


class PurgedWalkForward:
    def __init__(self, *, n_folds: int = 4, embargo_bars: int = 6) -> None:
        self.n_folds = n_folds
        self.embargo_bars = embargo_bars

    def split(self, df: pd.DataFrame) -> list[Fold]:
        n = len(df)
        if n < self.n_folds * 10:
            raise ValueError(f"too few rows ({n}) for {self.n_folds} folds")
        fold_len = n // (self.n_folds + 1)
        folds: list[Fold] = []
        for k in range(self.n_folds):
            train_end = fold_len * (k + 1)
            test_start = train_end + self.embargo_bars
            test_end = min(test_start + fold_len, n)
            if test_start >= n:
                break
            folds.append(
                Fold(
                    train=df.iloc[:train_end],
                    test=df.iloc[test_start:test_end],
                )
            )
        return folds

    @staticmethod
    def select_params(
        strategy: BaseStrategy,
        train_df: pd.DataFrame,
        grid: Iterable[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Pick the best params by mean net return ON TRAIN ONLY."""
        best: tuple[float, Mapping[str, Any]] | None = None
        for params in grid:
            sig = strategy.signal(train_df, params)
            score = float(
                simulate_net(
                    sig, train_df["close"], fee_rate=0.0005, slippage_bps=2.0
                ).sum()
            )
            if best is None or score > best[0]:
                best = (score, dict(params))
        if best is None:
            raise ValueError("empty parameter grid")
        return best[1]


def simulate_net(
    signal: pd.Series,
    close: pd.Series,
    *,
    fee_rate: float,
    slippage_bps: float,
) -> pd.Series:
    """Next-open net returns of a {-1,0,+1} position series.

    The return earned over bar t uses the position decided at t-1 (next-open
    semantics), so no same-bar lookahead is possible. Costs are charged on
    every position change.
    """
    aligned_pos = signal.shift(1).fillna(0.0)  # next-open: lag the position
    gross = aligned_pos * close.pct_change().fillna(0.0)
    changes = aligned_pos.diff().abs().fillna(aligned_pos.abs().iloc[0])
    cost = changes * (fee_rate + slippage_bps / 10_000.0)
    return gross - cost
