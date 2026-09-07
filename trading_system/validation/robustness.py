"""
Robustness testing for strategy validation.

Tests:
- Parameter perturbation sensitivity
- Transaction cost stress testing
- Execution assumption sensitivity
- Time-period stability
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from trading_system.backtester.engine import BacktestEngine
from trading_system.config import BacktestConfig, FeeConfig
from trading_system.optimization.param_space import perturb_params
from trading_system.strategies import get_strategy

logger = structlog.get_logger(__name__)


class RobustnessAnalyzer:
    """Comprehensive robustness testing for strategies."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.engine = BacktestEngine(config)

    def test_parameter_stability(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        best_params: dict[str, Any],
        pair: str = "",
        timeframe: str = "",
        n_steps: int = 5,
        perturbation_pct: float = 0.3,
        funding_rates: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Test how sensitive strategy performance is to parameter changes.

        A robust strategy should perform reasonably well with nearby parameters.
        """
        strategy = get_strategy(strategy_name)
        perturbed = perturb_params(best_params, n_steps=n_steps, perturbation_pct=perturbation_pct)

        results_list = []
        for params in perturbed:
            try:
                signals = strategy.generate_signals(df, params)
                result = self.engine.run(
                    df, signals, strategy_name, params, pair, timeframe, funding_rates
                )
                results_list.append({
                    "params": params,
                    "sharpe": result.sharpe,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "is_original": params == best_params,
                })
            except Exception:
                continue

        if not results_list:
            return {"stability_score": 0.0, "results": []}

        sharpes = [r["sharpe"] for r in results_list]
        original_sharpe = sharpes[0] if results_list else 0

        # Stability: what fraction of perturbations remain profitable
        profitable = sum(1 for s in sharpes if s > 0)
        stability_score = profitable / len(sharpes) if sharpes else 0

        # Sharpe degradation: how much does Sharpe drop on average
        positive_sharpes = [s for s in sharpes if s > 0]
        avg_sharpe = np.mean(positive_sharpes) if positive_sharpes else 0

        return {
            "stability_score": stability_score,
            "total_tested": len(perturbed),
            "profitable_variations": profitable,
            "avg_sharpe": avg_sharpe,
            "original_sharpe": original_sharpe,
            "min_sharpe": min(sharpes) if sharpes else 0,
            "max_sharpe": max(sharpes) if sharpes else 0,
            "sharpe_std": np.std(sharpes) if sharpes else 0,
            "results": results_list,
        }

    def test_cost_robustness(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        params: dict[str, Any],
        pair: str = "",
        timeframe: str = "",
        cost_multipliers: list[float] | None = None,
        funding_rates: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Test strategy performance under increased transaction costs.

        A robust strategy should remain profitable with higher costs.
        """
        if cost_multipliers is None:
            cost_multipliers = [1.0, 1.5, 2.0, 3.0]

        strategy = get_strategy(strategy_name)
        signals = strategy.generate_signals(df, params)

        cost_results = []
        for mult in cost_multipliers:
            # Create modified config with scaled costs
            modified_fees = FeeConfig(
                maker_fee=self.config.fees.maker_fee * mult,
                taker_fee=self.config.fees.taker_fee * mult,
                funding_rate_model=self.config.fees.funding_rate_model,
            )
            modified_config = BacktestConfig(
                fees=modified_fees,
                slippage=self.config.slippage,
                execution=self.config.execution,
            )
            engine = BacktestEngine(modified_config)
            result = engine.run(df, signals, strategy_name, params, pair, timeframe, funding_rates)

            cost_results.append({
                "cost_multiplier": mult,
                "sharpe": result.sharpe,
                "total_return": result.total_return,
                "net_profit": result.net_profit,
                "total_fees": result.total_fees,
                "profitable": result.total_return > 0,
            })

        # Cost robustness: how many cost levels remain profitable
        profitable_count = sum(1 for r in cost_results if r["profitable"])
        cost_robustness = profitable_count / len(cost_results) if cost_results else 0

        # Sharpe degradation rate
        if len(cost_results) >= 2:
            base_sharpe = cost_results[0]["sharpe"]
            worst_sharpe = cost_results[-1]["sharpe"]
            if base_sharpe > 0:
                sharpe_degradation = (base_sharpe - worst_sharpe) / base_sharpe
            else:
                sharpe_degradation = 1.0
        else:
            sharpe_degradation = 0.0

        return {
            "cost_robustness_score": cost_robustness,
            "profitable_at_all_levels": profitable_count == len(cost_results),
            "profitable_at_1_5x": any(r["profitable"] for r in cost_results if r["cost_multiplier"] == 1.5),
            "sharpe_degradation": sharpe_degradation,
            "results": cost_results,
        }

    def test_time_period_stability(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        params: dict[str, Any],
        pair: str = "",
        timeframe: str = "",
        n_periods: int = 4,
        funding_rates: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Test strategy performance across different time periods."""
        strategy = get_strategy(strategy_name)
        period_size = len(df) // n_periods

        period_results = []
        for i in range(n_periods):
            start = i * period_size
            end = min(start + period_size, len(df))
            period_df = df.iloc[start:end]

            try:
                signals = strategy.generate_signals(period_df, params)
                result = self.engine.run(
                    period_df, signals, strategy_name, params, pair, timeframe,
                    funding_rates.iloc[start:end] if funding_rates is not None else None,
                )
                period_results.append({
                    "period": i,
                    "start": str(period_df.index[0]),
                    "end": str(period_df.index[-1]),
                    "sharpe": result.sharpe,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "trades": result.total_trades,
                    "profitable": result.total_return > 0,
                })
            except Exception:
                continue

        profitable_periods = sum(1 for r in period_results if r["profitable"])

        return {
            "n_periods": n_periods,
            "profitable_periods": profitable_periods,
            "stability_score": profitable_periods / n_periods if n_periods > 0 else 0,
            "avg_sharpe": np.mean([r["sharpe"] for r in period_results]) if period_results else 0,
            "min_sharpe": min((r["sharpe"] for r in period_results), default=0),
            "results": period_results,
        }

    def run_full_robustness(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        best_params: dict[str, Any],
        pair: str = "",
        timeframe: str = "",
        funding_rates: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Run all robustness tests and return combined results."""
        logger.info("robustness_test_start", strategy=strategy_name, pair=pair, timeframe=timeframe)

        param_stability = self.test_parameter_stability(
            df, strategy_name, best_params, pair, timeframe,
            funding_rates=funding_rates,
        )

        cost_robustness = self.test_cost_robustness(
            df, strategy_name, best_params, pair, timeframe,
            funding_rates=funding_rates,
        )

        time_stability = self.test_time_period_stability(
            df, strategy_name, best_params, pair, timeframe,
            funding_rates=funding_rates,
        )

        # Combined robustness score
        combined_score = (
            param_stability["stability_score"] * 0.4
            + cost_robustness["cost_robustness_score"] * 0.3
            + time_stability["stability_score"] * 0.3
        )

        return {
            "parameter_stability": param_stability,
            "cost_robustness": cost_robustness,
            "time_period_stability": time_stability,
            "combined_robustness_score": combined_score,
        }
