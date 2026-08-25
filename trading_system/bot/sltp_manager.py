"""
Stop-Loss Manager.

Simple ATR-based stop-loss only. No take-profit, no trailing stop,
no breakeven — the strategy's own signal handles normal exits.

The stop-loss acts purely as disaster protection against:
- Exchange/API outages
- Flash crashes
- Black swan events
- Bot disconnection

Configuration: 3x ATR (wide enough to avoid noise, tight enough to
limit catastrophic loss to ~4-5% per trade).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import structlog

from trading_system.indicators import atr

logger = structlog.get_logger(__name__)


@dataclass
class PositionLevel:
    """Tracks SL level for a single position."""
    pair: str
    side: str  # "long" or "short"
    entry_price: float
    entry_time: str
    size: float
    strategy: str
    stop_loss: float = 0.0

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.side == "long":
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop-loss is hit."""
        if self.side == "long":
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss


class SLTPManager:
    """
    Simple stop-loss manager.

    Only monitors stop-loss at 3x ATR. The strategy signal
    handles normal exits (entry/exit based on indicators).
    This SL is purely for disaster protection.
    """

    def __init__(
        self,
        sl_atr_mult: float = 3.0,
        atr_period: int = 14,
    ):
        self.sl_atr_mult = sl_atr_mult
        self.atr_period = atr_period
        self.positions: dict[str, PositionLevel] = {}
        self._closed_positions: list[dict] = []

    def open_position(
        self,
        pair: str,
        side: str,
        entry_price: float,
        size: float,
        strategy: str,
        df: pd.DataFrame,
    ) -> PositionLevel:
        """Register a new position and compute SL level."""
        atr_series = atr(df, self.atr_period)
        current_atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else entry_price * 0.02

        if side == "long":
            stop_loss = entry_price - current_atr * self.sl_atr_mult
        else:
            stop_loss = entry_price + current_atr * self.sl_atr_mult

        key = f"{pair}_{strategy}"
        level = PositionLevel(
            pair=pair,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            size=size,
            strategy=strategy,
            stop_loss=stop_loss,
        )
        self.positions[key] = level

        logger.info("sl_position_opened",
                     pair=pair, side=side, entry=entry_price,
                     sl=stop_loss, atr=current_atr,
                     sl_pct=f"{abs(entry_price - stop_loss) / entry_price * 100:.1f}%")

        return level

    def check_price(self, pair: str, current_price: float) -> list[dict]:
        """
        Check all positions for SL hits.

        Returns list of actions: [{"action": "close", "pair": ..., "reason": "stop_loss"}]
        """
        actions = []
        keys_to_remove = []

        for key, pos in self.positions.items():
            if pos.pair != pair:
                continue

            if pos.should_stop_loss(current_price):
                pnl = pos.unrealized_pnl(current_price)
                actions.append({
                    "action": "close",
                    "pair": pos.pair,
                    "side": "sell" if pos.side == "long" else "buy",
                    "reason": "stop_loss",
                    "pnl_pct": pnl,
                    "entry": pos.entry_price,
                    "exit": current_price,
                    "strategy": pos.strategy,
                })
                self._closed_positions.append({
                    **actions[-1],
                    "entry_time": pos.entry_time,
                    "exit_time": datetime.now(timezone.utc).isoformat(),
                })
                keys_to_remove.append(key)
                logger.info("sl_triggered", pair=pair,
                            entry=pos.entry_price, exit=current_price,
                            pnl=f"{pnl*100:.2f}%")

        for key in keys_to_remove:
            del self.positions[key]

        return actions

    def get_status(self) -> dict[str, Any]:
        """Get current SL status for all positions."""
        return {
            "open_positions": len(self.positions),
            "closed_positions": len(self._closed_positions),
            "positions": {
                key: {
                    "pair": pos.pair,
                    "side": pos.side,
                    "entry": pos.entry_price,
                    "sl": pos.stop_loss,
                    "sl_pct": f"{abs(pos.entry_price - pos.stop_loss) / pos.entry_price * 100:.1f}%",
                }
                for key, pos in self.positions.items()
            },
            "recent_closes": self._closed_positions[-5:] if self._closed_positions else [],
        }

    def get_closed_trades(self) -> list[dict]:
        """Get all closed trade records."""
        return self._closed_positions.copy()

    def get_win_rate(self) -> float:
        """Calculate win rate from closed trades."""
        if not self._closed_positions:
            return 0.0
        wins = sum(1 for t in self._closed_positions if t.get("pnl_pct", 0) > 0)
        return wins / len(self._closed_positions)
