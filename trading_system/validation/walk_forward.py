"""
Walk-forward analysis for realistic out-of-sample performance estimation.

Implements both expanding and rolling window walk-forward tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

from trading_system.backtester.engine import BacktestEngine
from trading_system.backtester.results import BacktestResults
from trading_system.config import BacktestConfig
from trading_system.optimization.param_space import generate_grid
from trading_system.strategies import get_strategy

logger = structlog.get_logger(__name__)


class WalkForwardAnalyzer:
    """Walk-forward analysis for strategy validation."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.engine = BacktestEngine(config)

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        param_grid: dict[str, list],
        n_windows: int = 10,
        train_pct: float = 0.7,
        pair: str = "",
        timeframe: str = "",
        funding_rates: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Run walk-forward analysis.

        For each window:
        1. Train on the first train_pct of the window
        2. Optimize parameters on training data
        3. Test on the remaining test_pct (out-of-sample)
        """
        strategy = get_strategy(strategy_name)
        n_candles = len(df)
        window_size = n_candles // n_windows
        train_size = int(window_size * train_pct)
        test_size = window_size - train_size

        logger.info(
            "walk_forward_start",
            strategy=strategy_name,
            windows=n_windows,
            window_size=window_size,
            train_size=train_size,
            test_size=test_size,
        )

        oos_results = []
        best_params_per_window = []

        for w in range(n_windows):
            start = w * window_size
            end = min(start + window_size, n_candles)

            if end - start < train_size + 10:
                continue

            train_end = start + train_size
            test_start = train_end
            test_end = end

            train_df = df.iloc[start:train_end]
            test_df = df.iloc[test_start:test_end]

            # Optimize on training data
            best_params = self._optimize_on_train(
                strategy, param_grid, train_df, pair, timeframe
            )

            best_params_per_window.append(best_params)

            # Test on out-of-sample data
            signals = strategy.generate_signals(test_df, best_params)
            result = self.engine.run(
                test_df, signals, strategy_name, best_params, pair, timeframe,
                funding_rates.iloc[test_start:test_end] if funding_rates is not None else None,
            )

            oos_results.append({
                "window": w,
                "train_start": str(train_df.index[0]),
                "train_end": str(train_df.index[-1]),
                "test_start": str(test_df.index[0]),
                "test_end": str(test_df.index[-1]),
                "best_params": best_params,
                "sharpe": result.sharpe,
                "total_return": result.total_return,
                "max_drawdown": result.max_drawdown,
                "trades": result.total_trades,
            })

        # Aggregate results
        agg = self._aggregate_oos_results(oos_results)

        return {
            "strategy": strategy_name,
            "n_windows": n_windows,
            "windows": oos_results,
            "aggregate": agg,
            "best_params_per_window": best_params_per_window,
        }

    def _optimize_on_train(
        self,
        strategy,
        param_grid: dict[str, list],
        train_df: pd.DataFrame,
        pair: str,
        timeframe: str,
    ) -> dict[str, Any]:
        """Find best parameters on training data."""
        all_params = generate_grid(param_grid)

        best_sharpe = -float("inf")
        best_params = all_params[0] if all_params else {}

        for params in all_params[:500]:  # Limit to prevent excessive computation
            try:
                signals = strategy.generate_signals(train_df, params)
                result = self.engine.run(
                    train_df, signals, strategy.name if hasattr(strategy, 'name') else "",
                    params, pair, timeframe,
                )

                # Use composite metric
                score = result.sharpe * 0.5 + result.sortino * 0.3 + result.calmar * 0.2
                if score > best_sharpe:
                    best_sharpe = score
                    best_params = params
            except Exception:
                continue

        return best_params

    def _aggregate_oos_results(self, oos_results: list[dict]) -> dict[str, Any]:
        """Aggregate out-of-sample results across all windows."""
        if not oos_results:
            return {}

        sharpes = [r["sharpe"] for r in oos_results]
        returns = [r["total_return"] for r in oos_results]
        drawdowns = [r["max_drawdown"] for r in oos_results]
        trades = [r["trades"] for r in oos_results]

        return {
            "avg_sharpe": np.mean(sharpes),
            "median_sharpe": np.median(sharpes),
            "std_sharpe": np.std(sharpes),
            "min_sharpe": np.min(sharpes),
            "max_sharpe": np.max(sharpes),
            "pct_profitable_windows": sum(1 for s in sharpes if s > 0) / len(sharpes),
            "avg_return": np.mean(returns),
            "avg_max_drawdown": np.mean(drawdowns),
            "worst_max_drawdown": max(drawdowns),
            "total_trades": sum(trades),
            "avg_trades_per_window": np.mean(trades),
        }
