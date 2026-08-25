"""
Strategy scoring and ranking system.

Ranks strategies based on composite score that considers:
- Risk-adjusted returns
- Robustness
- Out-of-sample performance
- Overfitting risk
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from trading_system.optimization.scoring import calculate_composite_score, ScoringWeights
from trading_system.backtester.results import BacktestResults

logger = structlog.get_logger(__name__)


class StrategyScorer:
    """Scores and ranks strategies using composite scoring."""

    def __init__(self, weights: ScoringWeights | None = None):
        self.weights = weights or ScoringWeights()

    def score_experiment(
        self,
        results: dict[str, Any],
        oos_sharpe: float = 0.0,
        is_sharpe: float = 0.0,
        param_stability: float = 0.0,
        regime_stability: float = 0.0,
        cost_robustness: float = 0.0,
        timeframe_stability: float = 0.0,
    ) -> dict[str, Any]:
        """Score a single experiment."""
        # Convert dict to BacktestResults-like object
        r = type("R", (), results)()

        score = calculate_composite_score(
            r,
            oos_sharpe=oos_sharpe,
            is_sharpe=is_sharpe,
            param_stability=param_stability,
            regime_stability=regime_stability,
            cost_robustness=cost_robustness,
            timeframe_stability=timeframe_stability,
            weights=self.weights,
        )

        return {**results, **score}

    def rank_experiments(
        self,
        experiments: list[dict[str, Any]],
        top_n: int = 50,
    ) -> list[dict[str, Any]]:
        """Rank all experiments by composite score."""
        scored = []
        for exp in experiments:
            results = exp.get("results", {})
            if not results or results.get("total_trades", 0) < 10:
                continue

            scored.append({
                **exp,
                "composite_score": results.get("composite_score", 0),
            })

        # Sort by composite score
        scored.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        return scored[:top_n]

    def get_top_strategies(
        self,
        experiments: list[dict[str, Any]],
        n: int = 10,
    ) -> list[dict[str, Any]]:
        """Get top N unique strategies (best params per strategy)."""
        # Group by strategy name
        by_strategy: dict[str, list] = {}
        for exp in experiments:
            name = exp.get("strategy_name", "")
            if name not in by_strategy:
                by_strategy[name] = []
            by_strategy[name].append(exp)

        # Get best experiment per strategy
        best_per_strategy = []
        for name, exps in by_strategy.items():
            best = max(exps, key=lambda x: x.get("results", {}).get("composite_score", 0))
            best_per_strategy.append(best)

        # Sort by composite score
        best_per_strategy.sort(
            key=lambda x: x.get("results", {}).get("composite_score", 0),
            reverse=True,
        )

        return best_per_strategy[:n]
