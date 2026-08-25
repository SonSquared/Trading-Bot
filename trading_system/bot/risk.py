"""
Live risk management module.

Enforces strict limits on position size, drawdown, daily loss, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from trading_system.config import RiskConfig

logger = structlog.get_logger(__name__)


@dataclass
class RiskState:
    """Current risk state of the bot."""
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    max_drawdown_reached: float = 0.0
    total_positions: int = 0
    daily_trades: int = 0
    consecutive_losses: int = 0
    emergency_stop: bool = False
    stop_reason: str = ""
    last_reset_date: str = ""


class RiskManager:
    """Enforces risk limits for live/paper trading."""

    def __init__(self, config: RiskConfig, initial_capital: float = 10000.0):
        self.config = config
        self.state = RiskState(
            peak_equity=initial_capital,
            current_equity=initial_capital,
        )

    def can_open_position(
        self,
        proposed_size: float,
        current_exposure: float,
        account_equity: float,
    ) -> tuple[bool, str]:
        """Check if a new position can be opened."""
        # Emergency stop
        if self.state.emergency_stop:
            return False, f"Emergency stop active: {self.state.stop_reason}"

        # Daily loss limit
        if abs(self.state.daily_pnl) > self.config.max_daily_loss * self.state.peak_equity:
            return False, f"Daily loss limit reached: {self.state.daily_pnl:.2f}"

        # Max drawdown
        if self.state.peak_equity > 0:
            current_dd = (self.state.peak_equity - account_equity) / self.state.peak_equity
            if current_dd > self.config.max_drawdown:
                return False, f"Max drawdown exceeded: {current_dd:.2%}"

        # Max simultaneous positions
        if self.state.total_positions >= self.config.max_simultaneous_positions:
            return False, f"Max positions reached: {self.state.total_positions}"

        # Max position size
        position_pct = abs(proposed_size) / account_equity if account_equity > 0 else 1.0
        if position_pct > self.config.max_position_pct:
            return False, f"Position too large: {position_pct:.2%} > {self.config.max_position_pct:.2%}"

        # Max portfolio exposure
        new_exposure = current_exposure + abs(proposed_size)
        exposure_pct = new_exposure / account_equity if account_equity > 0 else 1.0
        if exposure_pct > self.config.max_portfolio_exposure:
            return False, f"Portfolio exposure too high: {exposure_pct:.2%}"

        # Max risk per trade
        risk_pct = abs(proposed_size) * 0.02 / account_equity if account_equity > 0 else 1.0
        if risk_pct > self.config.max_risk_per_trade:
            return False, f"Risk per trade too high: {risk_pct:.2%}"

        return True, "OK"

    def update(self, equity: float, trade_pnl: float = 0.0) -> None:
        """Update risk state after a new trade or equity change."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Reset daily counter if new day
        if today != self.state.last_reset_date:
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.last_reset_date = today

        self.state.current_equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self.state.daily_pnl += trade_pnl

        if trade_pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

    def trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop."""
        self.state.emergency_stop = True
        self.state.stop_reason = reason
        logger.critical("emergency_stop_triggered", reason=reason)

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop (requires manual confirmation in production)."""
        self.state.emergency_stop = False
        self.state.stop_reason = ""
        logger.info("emergency_stop_reset")

    def get_status(self) -> dict[str, Any]:
        """Get current risk status."""
        return {
            "emergency_stop": self.state.emergency_stop,
            "stop_reason": self.state.stop_reason,
            "daily_pnl": self.state.daily_pnl,
            "daily_pnl_limit": self.config.max_daily_loss * self.state.peak_equity,
            "current_drawdown": (
                (self.state.peak_equity - self.state.current_equity) / self.state.peak_equity
                if self.state.peak_equity > 0 else 0
            ),
            "max_drawdown_limit": self.config.max_drawdown,
            "consecutive_losses": self.state.consecutive_losses,
            "total_positions": self.state.total_positions,
            "daily_trades": self.state.daily_trades,
        }

    def check_abnormal_conditions(
        self,
        price_change_pct: float,
        api_error: bool = False,
        data_stale: bool = False,
    ) -> list[str]:
        """Check for abnormal conditions that should trigger protective action."""
        warnings = []

        if api_error:
            warnings.append("API error detected")
            self.trigger_emergency_stop("API error")

        if data_stale:
            warnings.append("Data feed is stale")

        if abs(price_change_pct) > 0.10:  # >10% move
            warnings.append(f"Abnormal price movement: {price_change_pct:.1%}")
            self.trigger_emergency_stop(f"Abnormal price movement: {price_change_pct:.1%}")

        if self.state.consecutive_losses >= 5:
            warnings.append(f"Consecutive losses: {self.state.consecutive_losses}")

        return warnings
