"""
Parallel experiment runner for large-scale backtesting.

Supports multiprocessing for running thousands of backtests efficiently.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import structlog

from trading_system.backtester.engine import BacktestEngine
from trading_system.config import BacktestConfig, SystemConfig
from trading_system.strategies import get_strategy

logger = structlog.get_logger(__name__)


def _run_single_experiment(args: tuple) -> dict | None:
    """
    Worker function for parallel backtesting.

    Runs one strategy/parameter/data combination.
    This function must be importable at module level for multiprocessing.
    """
    strategy_name, params, df_dict, config_dict, pair, timeframe, funding_dict = args

    try:
        # Reconstruct objects (they need to be picklable)
        config = BacktestConfig(
            fees=__import__("trading_system.config", fromlist=["FeeConfig"]).FeeConfig(**config_dict.get("fees", {})),
            slippage=__import__("trading_system.config", fromlist=["SlippageConfig"]).SlippageConfig(**config_dict.get("slippage", {})),
            execution=__import__("trading_system.config", fromlist=["ExecutionConfig"]).ExecutionConfig(**config_dict.get("execution", {})),
        )

        engine = BacktestEngine(config)
        strategy = get_strategy(strategy_name)

        df = pd.DataFrame(df_dict)
        df.index = pd.to_datetime(df.index)
        signals = strategy.generate_signals(df, params)

        funding_rates = None
        if funding_dict:
            funding_rates = pd.Series(funding_dict)
            funding_rates.index = pd.to_datetime(funding_rates.index)

        results = engine.run(
            df=df,
            signals=signals,
            strategy_name=strategy_name,
            params=params,
            pair=pair,
            timeframe=timeframe,
            funding_rates=funding_rates,
        )

        return {
            "experiment_id": _make_experiment_id(strategy_name, params, pair, timeframe),
            "strategy_name": strategy_name,
            "parameters": params,
            "pair": pair,
            "timeframe": timeframe,
            "results": results.to_dict(),
            "success": True,
        }

    except Exception as e:
        return {
            "experiment_id": _make_experiment_id(strategy_name, params, pair, timeframe),
            "strategy_name": strategy_name,
            "parameters": params,
            "pair": pair,
            "timeframe": timeframe,
            "results": {},
            "success": False,
            "error": str(e),
        }


def _make_experiment_id(
    strategy_name: str,
    params: dict,
    pair: str,
    timeframe: str,
) -> str:
    """Create a unique, deterministic experiment ID."""
    param_str = str(sorted(params.items()))
    raw = f"{strategy_name}_{pair}_{timeframe}_{param_str}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


class ExperimentRunner:
    """Runs backtesting experiments in parallel."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.engine = BacktestEngine(config.backtest)

    def run_grid_search(
        self,
        strategy_name: str,
        param_grid: dict[str, list],
        df: pd.DataFrame,
        pair: str,
        timeframe: str,
        funding_rates: pd.Series | None = None,
        n_workers: int | None = None,
    ) -> list[dict]:
        """Run all parameter combinations for a strategy."""
        from trading_system.optimization.param_space import generate_grid

        all_params = generate_grid(param_grid)
        total = len(all_params)

        logger.info(
            "grid_search_start",
            strategy=strategy_name,
            pair=pair,
            timeframe=timeframe,
            total_combinations=total,
        )

        results = []
        successful = 0
        failed = 0

        # For small grids, run sequentially (avoids multiprocessing overhead)
        if total < 100:
            for i, params in enumerate(all_params):
                result = self._run_experiment(
                    strategy_name, params, df, pair, timeframe, funding_rates
                )
                results.append(result)
                if result["success"]:
                    successful += 1
                else:
                    failed += 1

                if (i + 1) % 50 == 0:
                    logger.info("progress", completed=i + 1, total=total)
        else:
            # Parallel execution for large grids
            workers = n_workers or self.config.optimization.n_workers
            config_dict = {
                "fees": {
                    "maker_fee": self.config.backtest.fees.maker_fee,
                    "taker_fee": self.config.backtest.fees.taker_fee,
                },
                "slippage": {
                    "model": self.config.backtest.slippage.model,
                    "base_slippage": self.config.backtest.slippage.base_slippage,
                    "atr_multiplier": self.config.backtest.slippage.atr_multiplier,
                },
                "execution": {
                    "model": self.config.backtest.execution.model,
                    "delay_candles": self.config.backtest.execution.delay_candles,
                    "initial_capital": self.config.backtest.execution.initial_capital,
                    "leverage": self.config.backtest.execution.leverage,
                    "risk_per_trade": self.config.backtest.execution.risk_per_trade,
                },
            }

            df_dict = df.reset_index().to_dict(orient="list")
            funding_dict = {}
            if funding_rates is not None and not funding_rates.empty:
                funding_dict = funding_rates.to_dict()

            args_list = [
                (strategy_name, params, df_dict, config_dict, pair, timeframe, funding_dict)
                for params in all_params
            ]

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run_single_experiment, args): i for i, args in enumerate(args_list)}

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if result and result["success"]:
                        successful += 1
                    else:
                        failed += 1

                    if len(results) % 100 == 0:
                        logger.info(
                            "progress",
                            completed=len(results),
                            total=total,
                            successful=successful,
                            failed=failed,
                        )

        logger.info(
            "grid_search_complete",
            strategy=strategy_name,
            total=total,
            successful=successful,
            failed=failed,
        )

        return results

    def run_random_search(
        self,
        strategy_name: str,
        param_grid: dict[str, list],
        df: pd.DataFrame,
        pair: str,
        timeframe: str,
        funding_rates: pd.Series | None = None,
        n_samples: int = 1000,
        seed: int = 42,
        n_workers: int | None = None,
    ) -> list[dict]:
        """Run random parameter sampling."""
        from trading_system.optimization.param_space import sample_random

        all_params = sample_random(param_grid, n_samples, seed)

        logger.info("random_search_start", strategy=strategy_name, n_samples=n_samples)

        results = []
        for i, params in enumerate(all_params):
            result = self._run_experiment(
                strategy_name, params, df, pair, timeframe, funding_rates
            )
            results.append(result)

            if (i + 1) % 100 == 0:
                logger.info("progress", completed=i + 1, total=n_samples)

        return results

    def _run_experiment(
        self,
        strategy_name: str,
        params: dict,
        df: pd.DataFrame,
        pair: str,
        timeframe: str,
        funding_rates: pd.Series | None = None,
    ) -> dict:
        """Run a single backtest experiment."""
        try:
            strategy = get_strategy(strategy_name)
            signals = strategy.generate_signals(df, params)

            results = self.engine.run(
                df=df,
                signals=signals,
                strategy_name=strategy_name,
                params=params,
                pair=pair,
                timeframe=timeframe,
                funding_rates=funding_rates,
            )

            return {
                "experiment_id": _make_experiment_id(strategy_name, params, pair, timeframe),
                "strategy_name": strategy_name,
                "parameters": params,
                "pair": pair,
                "timeframe": timeframe,
                "results": results.to_dict(),
                "success": True,
            }
        except Exception as e:
            logger.error("experiment_failed", strategy=strategy_name, error=str(e))
            return {
                "experiment_id": _make_experiment_id(strategy_name, params, pair, timeframe),
                "strategy_name": strategy_name,
                "parameters": params,
                "pair": pair,
                "timeframe": timeframe,
                "results": {},
                "success": False,
                "error": str(e),
            }
