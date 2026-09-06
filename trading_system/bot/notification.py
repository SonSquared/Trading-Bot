"""
Notification system for bot alerts.

Supports webhook notifications (Telegram, Discord, Slack).
"""

from __future__ import annotations


import requests
import structlog

logger = structlog.get_logger(__name__)


class NotificationManager:
    """Send notifications for bot events."""

    def __init__(self, enabled: bool = False, webhook_url: str = ""):
        self.enabled = enabled
        self.webhook_url = webhook_url

    def send(self, message: str, level: str = "info") -> bool:
        """Send a notification."""
        if not self.enabled or not self.webhook_url:
            return False

        try:
            payload = {
                "text": f"[{level.upper()}] {message}",
                "level": level,
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info("notification_sent", level=level)
                return True
            else:
                logger.warning("notification_failed", status=response.status_code)
                return False

        except Exception as e:
            logger.error("notification_error", error=str(e))
            return False

    def notify_trade(self, trade: dict) -> None:
        """Notify about a new trade."""
        msg = (
            f"Trade: {trade.get('pair')} {trade.get('side')} "
            f"@ {trade.get('price', 0):.2f} | "
            f"P&L: ${trade.get('pnl', 0):.2f}"
        )
        self.send(msg)

    def notify_error(self, error: str) -> None:
        """Notify about an error."""
        self.send(f"ERROR: {error}", level="error")

    def notify_emergency_stop(self, reason: str) -> None:
        """Notify about emergency stop."""
        self.send(f"EMERGENCY STOP: {reason}", level="critical")

    def notify_daily_summary(self, summary: dict) -> None:
        """Send daily summary notification."""
        msg = (
            f"Daily Summary:\n"
            f"Equity: ${summary.get('equity', 0):.2f}\n"
            f"Daily P&L: ${summary.get('daily_pnl', 0):.2f}\n"
            f"Trades: {summary.get('trades_today', 0)}\n"
            f"Open Positions: {summary.get('positions', 0)}"
        )
        self.send(msg, level="info")
