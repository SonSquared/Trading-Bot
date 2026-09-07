"""
Multi-objective scoring function for strategy ranking.

Prioritizes robustness and risk-adjusted returns over absolute profit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trading_system.backtester.results import BacktestResults


@dataclass
class ScoringWeights:
    """Weights for the composite scoring function."""
    sharpe: float = 0.25
    sortino: float = 0.15
    calmar: float = 0.15
    profit_factor: float = 0.10
    oos_ratio: float = 0.10
    param_stability: float = 0.10
    regime_stability: float = 0.05
    cost_robustness: float = 0.05
    timeframe_stability: float = 0.05
    overfitting_penalty: float = 0.05


DEFAULT_WEIGHTS = ScoringWeights()


def calculate_composite_score(
    results: BacktestResults,
    oos_sharpe: float = 0.0,
    is_sharpe: float = 0.0,
    param_stability: float = 0.0,
    regime_stability: float = 0.0,
    cost_robustness: float = 0.0,
    timeframe_stability: float = 0.0,
    weights: ScoringWeights | None = None,
) -> dict[str, float]:
    """
    Calculate composite score for a strategy.

    Returns dict with 'composite_score' and component scores.
    """
    w = weights or DEFAULT_WEIGHTS

    # Normalize metrics to [0, 1] range using sigmoid-like transformation
    sharpe_norm = _normalize_sharpe(results.sharpe)
    sortino_norm = _normalize_sortino(results.sortino)
    calmar_norm = _normalize_calmar(results.calmar)
    pf_norm = _normalize_profit_factor(results.profit_factor)

    # OOS ratio: how well does out-of-sample match in-sample
    if is_sharpe > 0 and oos_sharpe > 0:
        oos_ratio = min(oos_sharpe / is_sharpe, 1.5) / 1.5  # Cap at 1.5x
    elif is_sharpe > 0 and oos_sharpe <= 0:
        oos_ratio = 0.0  # OOS failed completely
    else:
        oos_ratio = 0.5  # Default if no data

    # Overfitting penalty
    overfitting_penalty = _calculate_overfitting_penalty(results, is_sharpe, oos_sharpe)

    # Minimum trade count requirement (penalize strategies with too few trades)
    trade_penalty = 0.0
    if results.total_trades < 30:
        trade_penalty = (30 - results.total_trades) / 30 * 0.5

    # Composite score
    score = (
        w.sharpe * sharpe_norm
        + w.sortino * sortino_norm
        + w.calmar * calmar_norm
        + w.profit_factor * pf_norm
        + w.oos_ratio * oos_ratio
        + w.param_stability * param_stability
        + w.regime_stability * regime_stability
        + w.cost_robustness * cost_robustness
        + w.timeframe_stability * timeframe_stability
        - w.overfitting_penalty * overfitting_penalty
        - trade_penalty
    )

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    return {
        "composite_score": score,
        "sharpe_norm": sharpe_norm,
        "sortino_norm": sortino_norm,
        "calmar_norm": calmar_norm,
        "pf_norm": pf_norm,
        "oos_ratio": oos_ratio,
        "param_stability": param_stability,
        "regime_stability": regime_stability,
        "cost_robustness": cost_robustness,
        "timeframe_stability": timeframe_stability,
        "overfitting_penalty": overfitting_penalty,
        "trade_penalty": trade_penalty,
    }


def _normalize_sharpe(sharpe: float) -> float:
    """Normalize Sharpe ratio to [0, 1] using sigmoid."""
    # Sharpe of 2 is excellent, 0 is neutral, -2 is terrible
    return 1 / (1 + np.exp(-sharpe))


def _normalize_sortino(sortino: float) -> float:
    """Normalize Sortino ratio."""
    return 1 / (1 + np.exp(-sortino / 2))


def _normalize_calmar(calmar: float) -> float:
    """Normalize Calmar ratio."""
    return 1 / (1 + np.exp(-calmar / 2))


def _normalize_profit_factor(pf: float) -> float:
    """Normalize profit factor. PF > 1.5 is excellent."""
    if pf == float("inf"):
        return 1.0
    if pf <= 0:
        return 0.0
    return 1 / (1 + np.exp(-(pf - 1.2) * 2))


def _calculate_overfitting_penalty(
    results: BacktestResults,
    is_sharpe: float,
    oos_sharpe: float,
) -> float:
    """
    Calculate overfitting penalty (0 = no penalty, 1 = severe penalty).

    Factors:
    - Large IS/OOS performance gap
    - Too few trades
    - Extremely high Sharpe (too good to be true)
    - High return concentration
    """
    penalty = 0.0

    # Factor 1: IS/OOS gap
    if is_sharpe > 0:
        ratio = oos_sharpe / is_sharpe if is_sharpe > 0 else 0
        if ratio < 0.3:
            penalty += 0.3
        elif ratio < 0.5:
            penalty += 0.15

    # Factor 2: Too few trades
    if results.total_trades < 20:
        penalty += 0.2
    elif results.total_trades < 50:
        penalty += 0.1

    # Factor 3: Extremely high Sharpe (may be overfit)
    if results.sharpe > 4:
        penalty += 0.2
    elif results.sharpe > 3:
        penalty += 0.1

    # Factor 4: Return concentration
    if results.total_trades > 0 and results.trades:
        pnls = [t["pnl"] for t in results.trades]
        if pnls:
            top_10_pct = max(1, len(pnls) // 10)
            top_trades = sorted(pnls, reverse=True)[:top_10_pct]
            total_pnl = sum(pnls)
            if total_pnl != 0 and abs(sum(top_trades) / total_pnl) > 0.5:
                penalty += 0.15

    # Factor 5: Unrealistic win rate with low trade count
    if results.win_rate > 0.8 and results.total_trades < 100:
        penalty += 0.1

    return min(1.0, penalty)
