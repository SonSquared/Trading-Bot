"""
Monte Carlo simulation for strategy robustness testing.

Generates distribution of possible outcomes by:
- Shuffling trade order
- Bootstrap resampling of returns
- Randomizing slippage and execution
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class MonteCarloAnalyzer:
    """Monte Carlo analysis for strategy robustness."""

    def __init__(self, n_simulations: int = 10000, seed: int = 42):
        self.n_simulations = n_simulations
        self.seed = seed

    def run_trade_shuffling(
        self,
        trades: list[dict],
        initial_capital: float = 10000.0,
    ) -> dict[str, Any]:
        """
        Shuffle trade order to estimate outcome distribution.

        This tests whether the strategy's performance depends on
        the specific sequence of wins and losses.
        """
        if not trades or len(trades) < 10:
            return self._empty_result()

        pnls = np.array([t["pnl"] for t in trades])
        rng = np.random.RandomState(self.seed)

        final_equities = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(self.n_simulations):
            shuffled = rng.permutation(pnls)
            equity = np.cumsum(shuffled) + initial_capital
            equity = np.insert(equity, 0, initial_capital)

            # Max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max
            max_dd = abs(drawdowns.min())

            # Sharpe (approximate)
            returns = np.diff(equity) / equity[:-1]
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)) if np.std(returns) > 0 else 0

            final_equities.append(equity[-1])
            max_drawdowns.append(max_dd)
            sharpe_ratios.append(sharpe)

        return self._compute_statistics(final_equities, max_drawdowns, sharpe_ratios, initial_capital)

    def run_bootstrap_returns(
        self,
        returns: pd.Series,
        initial_capital: float = 10000.0,
        horizon: int = 1000,
    ) -> dict[str, Any]:
        """Bootstrap resampling of returns to project forward."""
        if returns.empty or len(returns) < 20:
            return self._empty_result()

        ret_arr = returns.values
        rng = np.random.RandomState(self.seed)

        final_equities = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(self.n_simulations):
            # Sample returns with replacement
            sampled = rng.choice(ret_arr, size=horizon, replace=True)
            equity_curve = initial_capital * np.cumprod(1 + sampled)
            equity_curve = np.insert(equity_curve, 0, initial_capital)

            # Max drawdown
            running_max = np.maximum.accumulate(equity_curve)
            dd = (equity_curve - running_max) / running_max
            max_dd = abs(dd.min())

            # Sharpe
            std = np.std(sampled)
            sharpe = (np.mean(sampled) / std * np.sqrt(365 * 24)) if std > 0 else 0

            final_equities.append(equity_curve[-1])
            max_drawdowns.append(max_dd)
            sharpe_ratios.append(sharpe)

        return self._compute_statistics(final_equities, max_drawdowns, sharpe_ratios, initial_capital)

    def run_slippage_randomization(
        self,
        trades: list[dict],
        base_slippage: float = 0.0001,
        slippage_range: float = 0.0005,
        initial_capital: float = 10000.0,
    ) -> dict[str, Any]:
        """Test strategy under randomized slippage conditions."""
        if not trades or len(trades) < 10:
            return self._empty_result()

        rng = np.random.RandomState(self.seed)

        final_equities = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            equity = initial_capital
            peak = equity
            max_dd = 0.0

            for trade in trades:
                pnl = trade["pnl"]
                # Add random slippage cost
                random_slippage = rng.uniform(
                    base_slippage, base_slippage + slippage_range
                )
                notional = abs(trade.get("position_size", 1.0)) * trade.get("entry_price", equity)
                slippage_cost = notional * random_slippage * 2  # Entry + exit
                equity += pnl - slippage_cost

                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_equities.append(equity)
            max_drawdowns.append(max_dd)

        returns = np.array(final_equities) / initial_capital - 1
        sharpe_ratios = []

        for eq in final_equities:
            r = (eq / initial_capital - 1)
            sharpe_ratios.append(r)

        return self._compute_statistics(final_equities, max_drawdowns, sharpe_ratios, initial_capital)

    def _compute_statistics(
        self,
        final_equities: list[float],
        max_drawdowns: list[float],
        sharpe_ratios: list[float],
        initial_capital: float,
    ) -> dict[str, Any]:
        """Compute Monte Carlo statistics from simulation results."""
        finals = np.array(final_equities)
        dds = np.array(max_drawdowns)
        sharpes = np.array(sharpe_ratios)

        returns = finals / initial_capital - 1

        # Probability of losing money
        prob_loss = (finals < initial_capital).mean()

        # Ruin probability (portfolio drops below 50%)
        prob_ruin = (finals < initial_capital * 0.5).mean()

        # Confidence intervals
        ci_95 = np.percentile(finals, [2.5, 50, 97.5])
        ci_99 = np.percentile(finals, [0.5, 50, 99.5])

        return {
            "n_simulations": len(final_equities),
            "prob_loss": float(prob_loss),
            "prob_ruin": float(prob_ruin),
            "expected_return": float(returns.mean()),
            "median_return": float(np.median(returns)),
            "worst_case_return": float(returns.min()),
            "best_case_return": float(returns.max()),
            "expected_max_drawdown": float(dds.mean()),
            "worst_max_drawdown": float(dds.max()),
            "median_max_drawdown": float(np.median(dds)),
            "max_drawdown_95": float(np.percentile(dds, 95)),
            "expected_sharpe": float(sharpes.mean()),
            "median_sharpe": float(np.median(sharpes)),
            "final_equity_95_ci": ci_95.tolist(),
            "final_equity_99_ci": ci_99.tolist(),
            "percentile_5_return": float(np.percentile(returns, 5)),
            "percentile_25_return": float(np.percentile(returns, 25)),
            "percentile_75_return": float(np.percentile(returns, 75)),
            "percentile_95_return": float(np.percentile(returns, 95)),
        }

    def _empty_result(self) -> dict[str, Any]:
        """Return empty result when insufficient data."""
        return {
            "n_simulations": 0,
            "error": "Insufficient trades/data for Monte Carlo analysis",
            "prob_loss": 0.5,
            "prob_ruin": 0.0,
            "expected_return": 0.0,
        }
