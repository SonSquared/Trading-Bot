"""
Strategy correlation analysis for portfolio diversification.

Measures correlation between strategy equity curves to ensure
selected strategies are complementary, not redundant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from typing import Any

logger = structlog.get_logger(__name__)


class StrategyCorrelationAnalyzer:
    """Analyzes correlation between strategy returns for diversification."""

    def calculate_returns_matrix(
        self,
        equity_curves: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Calculate returns correlation matrix from equity curves."""
        returns_dict = {}
        for name, curve in equity_curves.items():
            if len(curve) > 1:
                returns_dict[name] = curve.pct_change().dropna()

        if not returns_dict:
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_dict)
        return returns_df.corr()

    def find_diversified_set(
        self,
        strategies: list[dict],
        equity_curves: dict[str, pd.Series],
        n_select: int = 3,
        max_correlation: float = 0.7,
    ) -> list[str]:
        """
        Select N strategies that are maximally diversified.

        Uses a greedy algorithm to select strategies with low correlation.
        """
        if not strategies or not equity_curves:
            return []

        corr_matrix = self.calculate_returns_matrix(equity_curves)

        if corr_matrix.empty:
            return [s.get("strategy_name", "") for s in strategies[:n_select]]

        # Greedy selection: start with highest-scoring, then add least correlated
        selected = []
        remaining = [s.get("strategy_name", "") for s in strategies]

        # Always start with the best strategy
        if remaining:
            best = remaining[0]
            selected.append(best)
            remaining.remove(best)

        while len(selected) < n_select and remaining:
            best_candidate = None
            min_max_corr = float("inf")

            for candidate in remaining:
                # Max correlation with already selected
                max_corr_with_selected = 0
                for sel in selected:
                    if candidate in corr_matrix.index and sel in corr_matrix.columns:
                        corr = abs(corr_matrix.loc[candidate, sel])
                        max_corr_with_selected = max(max_corr_with_selected, corr)

                if max_corr_with_selected < min_max_corr:
                    min_max_corr = max_corr_with_selected
                    best_candidate = candidate

            if best_candidate and min_max_corr < max_correlation:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                # If all remaining are too correlated, take the next best anyway
                if remaining:
                    selected.append(remaining[0])
                    remaining.pop(0)

        return selected

    def get_correlation_summary(
        self,
        equity_curves: dict[str, pd.Series],
    ) -> dict[str, Any]:
        """Get summary of strategy correlations."""
        corr_matrix = self.calculate_returns_matrix(equity_curves)

        if corr_matrix.empty:
            return {"error": "No data for correlation analysis"}

        # Get unique pairs
        names = corr_matrix.columns.tolist()
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append({
                    "strategy_a": names[i],
                    "strategy_b": names[j],
                    "correlation": float(corr_matrix.iloc[i, j]),
                })

        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return {
            "avg_correlation": float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()) if len(names) > 1 else 0,
            "max_correlation": pairs[0]["correlation"] if pairs else 0,
            "min_correlation": pairs[-1]["correlation"] if pairs else 0,
            "pairs": pairs,
            "matrix": corr_matrix.to_dict(),
        }
