"""
Bot state persistence using JSON files.

Ensures the bot can recover from restarts without duplicate orders.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BotState:
    """Persistent bot state management."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("state_load_failed", path=str(self.state_file))
        return self._default_state()

    def _default_state(self) -> dict[str, Any]:
        return {
            "positions": {},
            "pending_orders": {},
            "last_signals": {},
            "trades_today": 0,
            "daily_pnl": 0.0,
            "last_check_time": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_order_ids": [],
        }

    def save(self) -> None:
        """Save state to file."""
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.state_file, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except IOError as e:
            logger.error("state_save_failed", error=str(e))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a state value."""
        self._state[key] = value
        self.save()

    def record_trade(self, trade: dict) -> None:
        """Record a trade in state."""
        trades = self._state.setdefault("trade_history", [])
        trades.append({
            **trade,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.save()

    def has_pending_order(self, pair: str) -> bool:
        """Check if there's a pending order for a pair."""
        return pair in self._state.get("pending_orders", {})

    def add_pending_order(self, pair: str, order_id: str) -> None:
        """Record a pending order."""
        self._state.setdefault("pending_orders", {})[pair] = order_id
        self.save()

    def remove_pending_order(self, pair: str) -> None:
        """Remove a pending order."""
        self._state.get("pending_orders", {}).pop(pair, None)
        self.save()

    def was_order_filled(self, order_id: str) -> bool:
        """Check if an order was already processed (duplicate prevention)."""
        return order_id in self._state.get("last_order_ids", [])

    def record_order(self, order_id: str) -> None:
        """Record an order ID for duplicate prevention."""
        ids = self._state.setdefault("last_order_ids", [])
        ids.append(order_id)
        # Keep only last 1000 order IDs
        if len(ids) > 1000:
            self._state["last_order_ids"] = ids[-1000:]
        self.save()

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self._state["trades_today"] = 0
        self._state["daily_pnl"] = 0.0
        self.save()
