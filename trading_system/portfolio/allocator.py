"""
Portfolio allocation methods.

Supports: equal weight, volatility-weighted, risk parity, drawdown-based.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


def equal_weight(n_strategies: int) -> np.ndarray:
    """Equal allocation across all strategies."""
    return np.ones(n_strategies) / n_strategies


def volatility_weight(
    equity_curves: dict[str, pd.Series],
    lookback: int = 100,
) -> np.ndarray:
    """
    Weight inversely proportional to volatility.
    Lower volatility = higher weight.
    """
    vols = []
    names = list(equity_curves.keys())
    for name in names:
        curve = equity_curves[name]
        recent = curve.pct_change().tail(lookback).dropna()
        vol = recent.std() if len(recent) > 0 else 1.0
        vols.append(max(vol, 1e-8))

    inv_vols = 1.0 / np.array(vols)
    weights = inv_vols / inv_vols.sum()
    return weights


def risk_parity(
    equity_curves: dict[str, pd.Series],
    lookback: int = 200,
) -> np.ndarray:
    """
    Risk parity allocation: each strategy contributes equally to portfolio risk.
    """
    returns_dict = {}
    for name, curve in equity_curves.items():
        recent = curve.pct_change().tail(lookback).dropna()
        returns_dict[name] = recent

    returns_df = pd.DataFrame(returns_dict)
    vols = returns_df.std()

    inv_vols = 1.0 / vols.replace(0, np.inf)
    weights = inv_vols / inv_vols.sum()
    return weights.values


def sharpe_weight(
    equity_curves: dict[str, pd.Series],
    lookback: int = 200,
) -> np.ndarray:
    """Weight proportional to trailing Sharpe ratio."""
    sharpes = []
    names = list(equity_curves.keys())
    for name in names:
        curve = equity_curves[name]
        recent = curve.pct_change().tail(lookback).dropna()
        if len(recent) > 10 and recent.std() > 0:
            sharpe = recent.mean() / recent.std() * np.sqrt(365 * 24)
        else:
            sharpe = 0
        sharpes.append(max(sharpe, 0.01))

    sharpes = np.array(sharpes)
    weights = sharpes / sharpes.sum()
    return weights


def drawdown_adjusted_weight(
    equity_curves: dict[str, pd.Series],
    lookback: int = 200,
) -> np.ndarray:
    """
    Reduce weight for strategies in drawdown.
    Strategies in drawdown get less capital.
    """
    adjustments = []
    names = list(equity_curves.keys())
    for name in names:
        curve = equity_curves[name]
        recent = curve.tail(lookback)
        peak = recent.max()
        current = recent.iloc[-1]
        dd = (current - peak) / peak if peak > 0 else 0

        # Reduce weight based on drawdown
        if dd < -0.10:
            adj = 0.3
        elif dd < -0.05:
            adj = 0.6
        elif dd < -0.02:
            adj = 0.8
        else:
            adj = 1.0
        adjustments.append(adj)

    adjustments = np.array(adjustments)
    weights = adjustments / adjustments.sum()
    return weights


def create_allocator(method: str) -> callable:
    """Factory function for allocation methods."""
    methods = {
        "equal": lambda curves, **kw: equal_weight(len(curves)),
        "volatility": lambda curves, **kw: volatility_weight(curves),
        "risk_parity": lambda curves, **kw: risk_parity(curves),
        "sharpe": lambda curves, **kw: sharpe_weight(curves),
        "drawdown": lambda curves, **kw: drawdown_adjusted_weight(curves),
    }
    return methods.get(method, lambda curves, **kw: equal_weight(len(curves)))
