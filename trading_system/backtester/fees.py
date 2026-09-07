"""
Fee calculation for perpetual futures.

Handles maker fees, taker fees, spread costs, and funding rate accrual.
"""

from __future__ import annotations

import pandas as pd
import structlog

from trading_system.config import FeeConfig

logger = structlog.get_logger(__name__)


class FeeCalculator:
    """Calculates all trading fees including funding rates."""

    def __init__(self, config: FeeConfig):
        self.config = config

    def calculate_entry_fee(
        self,
        entry_price: float,
        position_size: float,
        is_maker: bool = False,
    ) -> float:
        """Calculate fee for entering a position."""
        fee_rate = self.config.maker_fee if is_maker else self.config.taker_fee
        return abs(position_size) * entry_price * fee_rate

    def calculate_exit_fee(
        self,
        exit_price: float,
        position_size: float,
        is_maker: bool = False,
    ) -> float:
        """Calculate fee for exiting a position."""
        fee_rate = self.config.maker_fee if is_maker else self.config.taker_fee
        return abs(position_size) * exit_price * fee_rate

    def calculate_spread_cost(
        self,
        price: float,
        position_size: float,
        spread_pct: float = 0.0,
    ) -> float:
        """Calculate spread cost."""
        spread = spread_pct if spread_pct > 0 else 0.0002  # 0.02% default
        return abs(position_size) * price * spread

    def calculate_funding_cost(
        self,
        position_size: float,
        entry_price: float,
        funding_rates: pd.Series,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
    ) -> float:
        """
        Calculate total funding cost for a position.

        Funding is charged every 8 hours (00:00, 08:00, 16:00 UTC).
        The rate is applied to the notional value of the position.
        """
        if funding_rates is None or funding_rates.empty:
            return 0.0

        # Find funding rates between entry and exit
        mask = (funding_rates.index >= entry_time) & (funding_rates.index <= exit_time)
        applicable_rates = funding_rates[mask]

        if applicable_rates.empty:
            return 0.0

        notional = abs(position_size) * entry_price
        total_funding = 0.0

        for ts, rate in applicable_rates.items():
            # For long positions, positive rate costs money; negative rate earns money
            # For short positions, it's reversed
            if position_size > 0:
                total_funding += notional * rate
            else:
                total_funding -= notional * rate

        return total_funding

    def calculate_total_cost(
        self,
        entry_price: float,
        exit_price: float,
        position_size: float,
        entry_fee: float,
        exit_fee: float,
        spread_cost: float,
        funding_cost: float,
    ) -> dict[str, float]:
        """Calculate total transaction cost breakdown."""
        return {
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "spread_cost": spread_cost,
            "funding_cost": funding_cost,
            "total_cost": entry_fee + exit_fee + spread_cost + funding_cost,
        }

    def scale_costs(self, costs: dict[str, float], multiplier: float) -> dict[str, float]:
        """Scale all costs by a multiplier for stress testing."""
        return {k: v * multiplier for k, v in costs.items()}


def batch_calculate_funding_costs(
    trades: list[dict],
    funding_rates: pd.Series,
) -> list[float]:
    """
    Batch calculate funding costs for multiple trades.

    Each trade dict must have: entry_price, position_size, entry_time, exit_time
    """
    if funding_rates is None or funding_rates.empty:
        return [0.0] * len(trades)

    costs = []
    for trade in trades:
        mask = (
            (funding_rates.index >= trade["entry_time"]) &
            (funding_rates.index <= trade["exit_time"])
        )
        rates = funding_rates[mask]

        notional = abs(trade["position_size"]) * trade["entry_price"]
        funding = 0.0
        for _, rate in rates.items():
            if trade["position_size"] > 0:
                funding += notional * rate
            else:
                funding -= notional * rate

        costs.append(funding)

    return costs
