"""
Telegram Notification System for Trading Bot.

Sends formatted trade alerts, daily summaries, error notifications,
and portfolio status updates via Telegram Bot API.

Setup:
  1. Message @BotFather on Telegram → /newbot → get token
  2. Message your bot → /start → get your chat_id
  3. Add to bot_live.yaml:
       telegram:
         enabled: true
         bot_token: "YOUR_BOT_TOKEN"
         chat_id: "YOUR_CHAT_ID"
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import requests
import structlog

logger = structlog.get_logger(__name__)


class TelegramNotifier:
    """Send rich notifications via Telegram Bot API."""

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, bot_token: str = "", chat_id: str = "", enabled: bool = False):
        self.enabled = enabled
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = self.BASE_URL.format(token=bot_token) if bot_token else ""

    def _send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram. Uses curl as primary, requests as fallback."""
        if not self.enabled or not self.base_url or not self.chat_id:
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            return self._send_via_curl(payload)
        except Exception:
            pass

        # Fallback to requests
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json=payload,
                timeout=15,
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
        """Send via curl subprocess (works around SSL timeout on some Windows setups)."""
        import subprocess
        import json as _json

        url = f"{self.base_url}/sendMessage"
        result = subprocess.run(
            [
                "curl", "-s", "-m", "15",
                "-X", "POST", url,
                "-H", "Content-Type: application/json",
                "-d", _json.dumps(payload),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

        if result.stdout:
            data = _json.loads(result.stdout)
            if data.get("ok"):
                logger.info("telegram_sent_curl")
                return True
            else:
                logger.warning("telegram_curl_failed", desc=data.get("description", ""))
                return False
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
        emoji = "🟢" if side == "buy" else "🔴"
        direction = "LONG" if side == "buy" else "SHORT"
        mode_emoji = "📝" if mode == "paper" else "💰" if mode == "live" else "👀"

        msg = (
            f"{emoji} <b>NEW {direction} POSITION</b> {mode_emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Pair: <b>{pair}</b>\n"
            f"💵 Entry: <b>${price:,.2f}</b>\n"
            f"📦 Size: {amount:.6f}\n"
            f"🎯 Confidence: {confidence:.1%}\n"
            f"🧠 Strategy: {strategy}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
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
        pnl_emoji = "📈" if is_profit else "📉"
        mode_emoji = "📝" if mode == "paper" else "💰" if mode == "live" else "👀"

        msg = (
            f"{emoji} <b>POSITION CLOSED</b> {mode_emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Pair: <b>{pair}</b>\n"
            f"💵 Entry: ${entry_price:,.2f} → Exit: ${exit_price:,.2f}\n"
            f"{pnl_emoji} P&L: <b>{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%</b> (${'+' if pnl_usd >= 0 else ''}{pnl_usd:.2f})\n"
            f"📋 Reason: {reason}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
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
        emoji = "🛑" if trigger_type == "stop_loss" else "🎯"
        label = "STOP LOSS" if trigger_type == "stop_loss" else "TAKE PROFIT"

        msg = (
            f"{emoji} <b>{label} TRIGGERED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Pair: <b>{pair}</b>\n"
            f"💵 Entry: ${entry_price:,.2f}\n"
            f"📍 Trigger: ${trigger_price:,.2f}\n"
            f"📉 P&L: <b>{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%</b>\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
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
        """Send daily summary at end of day."""
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"

        positions_text = ""
        for pos in open_positions:
            pair = pos.get("pair", "?")
            side = "🟢LONG" if pos.get("side") == "long" else "🔴SHORT"
            pnl = pos.get("unrealized_pnl_pct", 0)
            positions_text += f"  {side} {pair} ({'+' if pnl >= 0 else ''}{pnl:.2f}%)\n"

        if not positions_text:
            positions_text = "  No open positions\n"

        msg = (
            f"📊 <b>DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Equity: <b>${equity:,.2f}</b>\n"
            f"{pnl_emoji} Daily P&L: <b>{'+' if daily_pnl >= 0 else ''}{daily_pnl_pct:.2f}%</b> (${'+' if daily_pnl >= 0 else ''}{daily_pnl:.2f})\n"
            f"📋 Trades today: {trades_today}\n"
            f"🏆 Win rate: {win_rate:.1f}% ({total_trades} total)\n"
            f"\n📍 <b>Open Positions:</b>\n{positions_text}"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return self._send_message(msg)

    # ── Portfolio Status ────────────────────────────────────────

    def notify_portfolio_status(self, status: dict) -> bool:
        """Send portfolio status snapshot."""
        strategies_text = ""
        for s in status.get("strategies", []):
            sig = s.get("last_signal", "N/A")
            sig_emoji = "🟢" if sig == 1 else "🔴" if sig == -1 else "⚪"
            strategies_text += f"  {sig_emoji} {s['label']} (w={s['weight']:.0%}) → {sig}\n"

        risk = status.get("risk", {})
        sltp = status.get("sltp", {})

        msg = (
            f"📋 <b>PORTFOLIO STATUS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 Mode: {status.get('mode', 'unknown').upper()}\n"
            f"🔄 Running: {'✅' if status.get('running') else '❌'}\n\n"
            f"🧠 <b>Strategies:</b>\n{strategies_text}\n"
            f"🛡️ <b>Risk:</b>\n"
            f"  Positions: {risk.get('open_positions', 0)}/{risk.get('max_positions', 3)}\n"
            f"  Daily P&L: {risk.get('daily_pnl', 0):.2%}\n"
            f"  Drawdown: {risk.get('current_drawdown', 0):.2%}\n\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return self._send_message(msg)

    # ── Errors & Alerts ─────────────────────────────────────────

    def notify_error(self, error: str, context: str = "") -> bool:
        """Send error notification."""
        msg = (
            f"🚨 <b>ERROR</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"❌ {error}\n"
        )
        if context:
            msg += f"📍 Context: {context}\n"
        msg += f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        return self._send_message(msg)

    def notify_emergency_stop(self, reason: str) -> bool:
        """Send emergency stop notification."""
        msg = (
            f"🚨🚨🚨 <b>EMERGENCY STOP</b> 🚨🚨🚨\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"⛔ Reason: {reason}\n"
            f"🛑 All trading has been halted.\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return self._send_message(msg)

    def notify_bot_start(self, mode: str, n_strategies: int) -> bool:
        """Send bot startup notification."""
        msg = (
            f"🚀 <b>BOT STARTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🤖 Mode: <b>{mode.upper()}</b>\n"
            f"🧠 Strategies: {n_strategies}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return self._send_message(msg)

    def notify_bot_stop(self, reason: str = "Manual stop") -> bool:
        """Send bot shutdown notification."""
        msg = (
            f"🛑 <b>BOT STOPPED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 Reason: {reason}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        return self._send_message(msg)

    # ── Health Check ────────────────────────────────────────────

    def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        if not self.enabled or not self.base_url:
            logger.warning("telegram_not_configured")
            return False

        # Try curl first (works around SSL issues on some Windows setups)
        try:
            import subprocess
            result = subprocess.run(
                ["curl", "-s", "-m", "15", f"{self.base_url}/getMe"],
                capture_output=True, text=True, timeout=20,
            )
            if result.stdout:
                import json as _json
                data = _json.loads(result.stdout)
                if data.get("ok"):
                    bot_name = data["result"].get("username", "unknown")
                    logger.info("telegram_connected_curl", bot=bot_name)
                    return self._send_message(f"Trading Bot Connected!\nBot: @{bot_name}")
        except Exception:
            pass

        # Fallback to requests
        try:
            resp = requests.get(f"{self.base_url}/getMe", timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                bot_name = data.get("result", {}).get("username", "unknown")
                logger.info("telegram_connected_requests", bot=bot_name)
                return self._send_message(f"Trading Bot Connected!\nBot: @{bot_name}")
            else:
                logger.warning("telegram_connect_failed", status=resp.status_code)
                return False
        except Exception as e:
            logger.error("telegram_connect_error", error=str(e))
            return False
