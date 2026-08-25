"""
Backtest result container with comprehensive metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestResults:
    """Complete results from a backtest run."""

    # Metadata
    strategy_name: str = ""
    parameters: dict = field(default_factory=dict)
    pair: str = ""
    timeframe: str = ""
    data_start: str = ""
    data_end: str = ""

    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)

    # Returns
    total_return: float = 0.0
    cagr: float = 0.0
    annualized_return: float = 0.0
    monthly_returns: pd.Series = field(default_factory=pd.Series)
    yearly_returns: pd.Series = field(default_factory=pd.Series)

    # Risk
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # candles
    volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Risk-adjusted
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    return_to_dd: float = 0.0

    # Trading statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    avg_holding_period: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0

    # Execution statistics
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding_costs: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    turnover: float = 0.0

    # Trade log
    trades: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to a dictionary (for storage)."""
        return {
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "pair": self.pair,
            "timeframe": self.timeframe,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "total_return": self.total_return,
            "cagr": self.cagr,
            "annualized_return": self.annualized_return,
            "max_drawdown": self.max_drawdown,
            "avg_drawdown": self.avg_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "volatility": self.volatility,
            "downside_deviation": self.downside_deviation,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "return_to_dd": self.return_to_dd,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expectancy": self.expectancy,
            "avg_holding_period": self.avg_holding_period,
            "max_consecutive_losses": self.max_consecutive_losses,
            "max_consecutive_wins": self.max_consecutive_wins,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "total_funding_costs": self.total_funding_costs,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "net_profit": self.net_profit,
            "turnover": self.turnover,
            "n_trades": self.total_trades,  # alias for convenience
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Strategy: {self.strategy_name} | {self.pair} {self.timeframe}\n"
            f"Period: {self.data_start} to {self.data_end}\n"
            f"Total Return: {self.total_return:.2%} | CAGR: {self.cagr:.2%}\n"
            f"Sharpe: {self.sharpe:.2f} | Sortino: {self.sortino:.2f} | Calmar: {self.calmar:.2f}\n"
            f"Max Drawdown: {self.max_drawdown:.2%}\n"
            f"Trades: {self.total_trades} | Win Rate: {self.win_rate:.1%}\n"
            f"Profit Factor: {self.profit_factor:.2f}\n"
            f"Fees: ${self.total_fees:.2f} | Slippage: ${self.total_slippage:.2f} | "
            f"Funding: ${self.total_funding_costs:.2f}"
        )
