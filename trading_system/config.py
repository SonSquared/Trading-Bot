"""
Configuration loader and validator for the trading system.

Loads YAML configs, merges defaults, and validates required fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).parent.parent / "configs"
DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class ExchangeConfig:
    name: str = "binance"
    market_type: str = "futures"  # futures or spot
    pairs: list[str] = field(default_factory=lambda: ["BTC/USDT:USDT", "ETH/USDT:USDT"])
    timeframes: list[str] = field(default_factory=lambda: ["15m", "30m", "1h", "4h"])
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = True


@dataclass
class DataConfig:
    start_date: str = "2022-01-01"
    end_date: str = ""  # empty = latest available
    data_dir: str = str(DATA_DIR / "raw")
    processed_dir: str = str(DATA_DIR / "processed")
    results_dir: str = str(DATA_DIR / "results")


@dataclass
class FeeConfig:
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%
    funding_rate_model: str = "actual"  # "actual" or "average"
    average_funding_rate: float = 0.0001  # used if model = "average"


@dataclass
class SlippageConfig:
    model: str = "atr_adaptive"  # "fixed", "atr_adaptive", "none"
    base_slippage: float = 0.0001  # 0.01%
    atr_multiplier: float = 0.1


@dataclass
class ExecutionConfig:
    model: str = "next_open"  # "next_open", "next_open_delay"
    delay_candles: int = 0
    initial_capital: float = 10000.0
    leverage: float = 1.0
    max_leverage: float = 5.0
    position_sizing: str = "fixed_fraction"  # "fixed_fraction", "kelly", "volatility"
    risk_per_trade: float = 0.02  # 2% risk per trade
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0


@dataclass
class BacktestConfig:
    fees: FeeConfig = field(default_factory=FeeConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


@dataclass
class DataSplitConfig:
    in_sample_end: str = "2023-06-30"
    validation_end: str = "2023-12-31"
    out_of_sample_end: str = "2024-06-30"
    # Everything after out_of_sample_end is the final holdout


@dataclass
class OptimizationConfig:
    method: str = "grid"  # "grid", "random", "bayesian"
    n_random_samples: int = 10000
    n_optuna_trials: int = 5000
    n_workers: int = 8
    batch_size: int = 100
    objective_weights: dict[str, float] = field(default_factory=lambda: {
        "sharpe": 0.25,
        "sortino": 0.15,
        "calmar": 0.15,
        "profit_factor": 0.10,
        "oos_ratio": 0.10,
        "param_stability": 0.10,
        "regime_stability": 0.05,
        "cost_robustness": 0.05,
        "timeframe_stability": 0.05,
        "overfitting_penalty": 0.05,
    })


@dataclass
class RiskConfig:
    max_position_pct: float = 0.25  # 25% of portfolio
    max_portfolio_exposure: float = 1.0  # 100%
    max_risk_per_trade: float = 0.02  # 2%
    max_daily_loss: float = 0.05  # 5%
    max_drawdown: float = 0.15  # 15%
    max_simultaneous_positions: int = 3
    max_order_size: float = 0.25  # 25% of portfolio
    emergency_stop: bool = False


@dataclass
class ValidationConfig:
    walk_forward_windows: int = 10
    walk_forward_train_pct: float = 0.7
    monte_carlo_simulations: int = 10000
    monte_carlo_confidence: float = 0.95
    parameter_perturbation_range: float = 0.3  # ±30%
    parameter_perturbation_steps: int = 5
    cost_stress_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0])


@dataclass
class StrategyConfig:
    families: list[str] = field(default_factory=lambda: [
        "trend", "momentum", "mean_reversion", "volatility"
    ])
    enabled_strategies: list[str] = field(default_factory=list)  # empty = all


@dataclass
class BotConfig:
    mode: str = "paper"  # "backtest", "paper", "dry_run", "live"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    check_interval_seconds: int = 60
    state_file: str = str(DATA_DIR / "bot_state.json")
    log_file: str = str(DATA_DIR / "bot.log")
    notification_enabled: bool = False
    notification_webhook: str = ""


@dataclass
class SystemConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    bot: BotConfig = field(default_factory=BotConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SystemConfig:
        """Load configuration from a YAML file, merging with defaults."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        config = cls()
        _apply_dict(config, raw)
        return config

    @classmethod
    def default(cls) -> SystemConfig:
        """Load the default configuration."""
        default_path = CONFIG_DIR / "default.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls()

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, sort_keys=False)

    def get_pair_safe_name(self, pair: str) -> str:
        """Convert pair name to filesystem-safe string."""
        return pair.replace("/", "_").replace(":", "_")

    def get_data_path(self, pair: str, timeframe: str) -> Path:
        """Get the path for storing/loading data for a specific pair and timeframe."""
        pair_name = self.get_pair_safe_name(pair)
        return Path(self.data.processed_dir) / pair_name / f"{timeframe}.parquet"

    def get_results_path(self, experiment_name: str = "results") -> Path:
        """Get the path for the results database."""
        return Path(self.data.results_dir) / f"{experiment_name}.db"


def _apply_dict(obj: Any, data: dict) -> None:
    """Recursively apply a dictionary to a dataclass."""
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if hasattr(obj, key):
            attr = getattr(obj, key)
            if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                _apply_dict(attr, value)
            else:
                setattr(obj, key, value)


def _to_dict(obj: Any) -> dict:
    """Convert a dataclass to a dictionary."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj
