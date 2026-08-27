"""
Telegram Notification System for Trading Bot.

Sends clean, professional trade alerts and status updates.
No cluttered separators or excessive emojis — just the data you need.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import requests
import structlog

logger = structlog.get_logger(__name__)


class TelegramNotifier:
    """Send clean notifications via Telegram Bot API."""

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, bot_token: str = "", chat_id: str = "", enabled: bool = False):
        self.enabled = enabled
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = self.BASE_URL.format(token=bot_token) if bot_token else ""

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram with retry."""
        if not self.enabled or not self.base_url or not self.chat_id:
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        # Try curl first (more reliable on Windows/SSL issues)
        try:
            if self._send_via_curl(payload):
                return True
        except Exception:
            pass

        # Fallback to requests
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json=payload,
                timeout=20,
            )
            if resp.status_code == 200:
                logger.info("telegram_sent_requests")
                return True
            else:
                logger.warning("telegram_send_failed", status=resp.status_code, text=resp.text)
                return False
        except Exception as e:
            logger.error("telegram_error", error=str(e))
            return False

    def _send_via_curl(self, payload: dict) -> bool:
        """Send via curl subprocess with retry (works around SSL timeout on some Windows setups)."""
        import subprocess
        import json as _json
        import time as _time

        url = f"{self.base_url}/sendMessage"
        for attempt in range(3):
            try:
                result = subprocess.run(
                    [
                        "curl", "-s", "-m", "20",
                        "-X", "POST", url,
                        "-H", "Content-Type: application/json",
                        "-d", _json.dumps(payload),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=25,
                )

                if result.stdout:
                    data = _json.loads(result.stdout)
                    if data.get("ok"):
                        logger.info("telegram_sent_curl")
                        return True
                    else:
                        desc = data.get("description", "")
                        # Don't retry on client errors (4xx)
                        if "Bad Request" in desc or "Forbidden" in desc:
                            logger.warning("telegram_curl_bad_request", desc=desc)
                            return False
                        logger.warning("telegram_curl_failed", desc=desc)
                # If no output or failed, retry after delay
                if attempt < 2:
                    _time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("telegram_curl_error", error=str(e), attempt=attempt)
                if attempt < 2:
                    _time.sleep(2 ** attempt)

        return False

    # ── Trade Alerts ────────────────────────────────────────────

    def notify_trade_open(
        self,
        pair: str,
        side: str,
        price: float,
        amount: float,
        confidence: float,
        strategy: str = "portfolio",
        mode: str = "paper",
    ) -> bool:
        """Send trade open notification."""
        direction = "LONG" if side == "buy" else "SHORT"
        emoji = "🟢" if side == "buy" else "🔴"
        mode_tag = "[PAPER]" if mode == "paper" else "[LIVE]"

        msg = (
            f"{emoji} <b>OPENED {direction}</b> {mode_tag}\n"
            f"{pair} @ ${price:,.2f}\n"
            f"Size: ${amount:,.2f} | Strategy: {strategy}\n"
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self._send_message(msg)

    def notify_trade_close(
        self,
        pair: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        pnl_usd: float,
        reason: str,
        mode: str = "paper",
    ) -> bool:
        """Send trade close notification with P&L."""
        is_profit = pnl_pct >= 0
        emoji = "✅" if is_profit else "❌"
        mode_tag = "[PAPER]" if mode == "paper" else "[LIVE]"

        msg = (
            f"{emoji} <b>CLOSED</b> {mode_tag}\n"
            f"{pair} | Entry ${entry_price:,.2f} → Exit ${exit_price:,.2f}\n"
            f"P&L: <b>{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%</b> "
            f"(${pnl_usd:+.2f})\n"
            f"Reason: {reason}\n"
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self._send_message(msg)

    def notify_sltp_trigger(
        self,
        pair: str,
        side: str,
        entry_price: float,
        trigger_price: float,
        pnl_pct: float,
        trigger_type: str,
    ) -> bool:
        """Send stop-loss or take-profit trigger notification."""
        label = "STOP LOSS" if trigger_type == "stop_loss" else "TAKE PROFIT"
        emoji = "🛑" if trigger_type == "stop_loss" else "🎯"

        msg = (
            f"{emoji} <b>{label}</b>\n"
            f"{pair} | Entry ${entry_price:,.2f} → Trigger ${trigger_price:,.2f}\n"
            f"P&L: {pnl_pct:+.2f}%\n"
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self._send_message(msg)

    # ── Daily Summary ───────────────────────────────────────────

    def notify_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        trades_today: int,
        open_positions: list[dict],
        win_rate: float = 0.0,
        total_trades: int = 0,
    ) -> bool:
        """Send clean portfolio summary."""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"

        msg = f"📊 <b>PORTFOLIO</b>\n"
        msg += f"Equity: ${equity:,.2f} ({daily_pnl_pct:+.1f}%)\n"
        msg += f"{pnl_emoji} P&L: ${daily_pnl:+.2f}\n"

        if total_trades > 0:
            msg += f"Trades: {total_trades} ({win_rate:.0f}% win)\n"

        if open_positions:
            msg += f"\n<b>Open ({len(open_positions)}):</b>\n"
            for pos in open_positions:
                pair = pos.get("pair", "?")
                side = "LONG" if pos.get("side") == "long" else "SHORT"
                emoji = "🟢" if side == "LONG" else "🔴"
                entry = pos.get("entry_price", 0)
                current = pos.get("current_price", 0)
                pnl = pos.get("unrealized_pnl_pct", 0)
                pnl_usd = pos.get("unrealized_pnl_usd", 0)

                if entry > 0 and current > 0:
                    msg += f"{emoji} {pair} {side} ${entry:,.0f}→${current:,.0f} "
                    msg += f"({pnl:+.1f}%, ${pnl_usd:+.2f})\n"
                else:
                    msg += f"{emoji} {pair} {side}\n"
        else:
            msg += "\nNo open positions\n"

        msg += f"{datetime.now(timezone.utc).strftime('%b %d, %H:%M UTC')}"
        return self._send_message(msg)

    # ── Portfolio Status ────────────────────────────────────────

    def notify_portfolio_status(self, status: dict) -> bool:
        """Send portfolio status snapshot."""
        msg = "📋 <b>STATUS</b>\n"

        # Equity and return
        equity = status.get("equity", 0)
        initial = status.get("initial_capital", 97)
        ret = (equity - initial) / initial * 100
        msg += f"Equity: ${equity:,.2f} ({ret:+.1f}%)\n"

        # Strategies
        strategies = status.get("strategies", [])
        if strategies:
            msg += f"\n<b>Signals:</b>\n"
            for s in strategies:
                sig = s.get("last_signal", "N/A")
                sig_emoji = "🟢" if sig == "LONG" else "🔴" if sig == "SHORT" else "⚪"
                msg += f"{sig_emoji} {s['label']} ({s['weight']:.0%})\n"

        # Risk
        risk = status.get("risk", {})
        if risk:
            msg += f"\n<b>Risk:</b>\n"
            msg += f"DD: {risk.get('current_drawdown', 0):.1f}%\n"
            msg += f"Positions: {risk.get('open_positions', 0)}/{risk.get('max_positions', 3)}\n"

        msg += f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        return self._send_message(msg)

    # ── Errors & Alerts ─────────────────────────────────────────

    def notify_error(self, error: str, context: str = "") -> bool:
        """Send error notification."""
        msg = f"⚠️ <b>ERROR</b>\n{error}\n"
        if context:
            msg += f"Context: {context}\n"
        msg += f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        return self._send_message(msg)

    def notify_emergency_stop(self, reason: str) -> bool:
        """Send emergency stop notification."""
        msg = (
            f"🚨 <b>EMERGENCY STOP</b>\n"
            f"Reason: {reason}\n"
            f"All trading halted.\n"
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self._send_message(msg)

    def notify_bot_start(self, mode: str, n_strategies: int) -> bool:
        """Send bot startup notification."""
        msg = (
            f"🤖 <b>BOT STARTED</b>\n"
            f"Mode: {mode.upper()} | Strategies: {n_strategies}\n"
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self._send_message(msg)

    def notify_bot_stop(self, reason: str = "Manual stop") -> bool:
        """Send bot shutdown notification."""
        msg = (
            f"🛑 <b>BOT STOPPED</b>\n"
            f"Reason: {reason}\n"
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        return self._send_message(msg)

    # ── Health Check ────────────────────────────────────────────

    def test_connection(self) -> bool:
        """Test Telegram bot connection. Returns True if API responds."""
        if not self.enabled or not self.base_url:
            logger.warning("telegram_not_configured")
            return False

        # Try curl first (works around SSL issues on some Windows setups)
        try:
            import subprocess
            result = subprocess.run(
                ["curl", "-s", "-m", "10", f"{self.base_url}/getMe"],
                capture_output=True, text=True, timeout=15,
            )
            if result.stdout:
                import json as _json
                data = _json.loads(result.stdout)
                if data.get("ok"):
                    bot_name = data["result"].get("username", "unknown")
                    logger.info("telegram_connected_curl", bot=bot_name)
                    return True
        except Exception:
            pass

        # Fallback to requests
        try:
            resp = requests.get(f"{self.base_url}/getMe", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                bot_name = data.get("result", {}).get("username", "unknown")
                logger.info("telegram_connected_requests", bot=bot_name)
                return True
            else:
                logger.warning("telegram_connect_failed", status=resp.status_code)
                return False
        except Exception as e:
            logger.error("telegram_connect_error", error=str(e))
            return False
