"""Evaluation statistics (plan MD Task 6): block bootstrap CIs, multiple-
testing adjustment, and risk metrics. Deterministic under an explicit seed."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def bootstrap_mean_ci(
    returns: Sequence[float] | pd.Series,
    *,
    n_boot: int = 1000,
    block: int = 10,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of returns.

    Block resampling preserves the short-range autocorrelation that makes
    iid bootstrap CIs too narrow on trading returns.
    """
    arr = np.asarray(list(returns), dtype=float)
    n = len(arr)
    if n < block * 2:
        block = max(1, n // 2)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([arr[s: s + block] for s in starts])[:n]
        means[b] = sample.mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    """Family-wise alpha under Bonferroni (never <= 0)."""
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    return alpha / n_tests


def sharpe(returns: pd.Series, *, periods_per_year: int) -> float:
    """Annualized Sharpe.

    Zero dispersion is degenerate: a riskless gain scores 0.0, a riskless
    loss scores -inf (it must never pass a Sharpe floor).
    """
    std = returns.std()
    if pd.isna(std) or std == 0:
        return 0.0 if float(returns.mean()) >= 0 else float("-inf")
    return float(returns.mean() / std * (periods_per_year ** 0.5))


def max_drawdown(equity: pd.Series | Sequence[float]) -> float:
    """Worst peak-to-trough drawdown as a negative fraction."""
    if isinstance(equity, pd.Series):
        eq = equity.astype(float)
    else:
        eq = pd.Series(list(equity), dtype=float)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())
