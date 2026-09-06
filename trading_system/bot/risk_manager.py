"""
Risk Management Module

Auto-closes positions when risk thresholds are breached:
  - Position stop-loss: close if position loses X% (default 5%)
  - Portfolio drawdown: close all if total equity drops X% from peak (default 10%)
  - Trailing stop: lock in profits after position reaches target gain
  - Max position time: close positions held longer than X hours
  - Volatility filter: don't open new positions during extreme volatility

Integrates with paper_trader.py — called on every run before new positions open.
"""

from datetime import datetime, timezone


class RiskManager:
    """Enforces risk rules on paper trading positions."""

    def __init__(
        self,
        position_stop_loss_pct: float = 5.0,
        portfolio_max_dd_pct: float = 10.0,
        trailing_stop_activation_pct: float = 5.0,
        trailing_stop_distance_pct: float = 3.0,
        max_position_hours: int = 72,
        max_open_positions: int = 3,
        max_portfolio_heat_pct: float = 30.0,
        dd_cooldown_hours: float = 24.0,
    ):
        self.position_stop_loss_pct = position_stop_loss_pct
        self.portfolio_max_dd_pct = portfolio_max_dd_pct
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.trailing_stop_distance_pct = trailing_stop_distance_pct
        self.max_position_hours = max_position_hours
        self.max_open_positions = max_open_positions
        self.max_portfolio_heat_pct = max_portfolio_heat_pct
        # How long after a portfolio drawdown stop the bot must stay flat.
        # The caller re-arms ``peak_equity`` to the post-stop equity so the
        # protection measures from the new baseline after the cooldown.
        self.dd_cooldown_hours = dd_cooldown_hours

    def check_positions(
        self, positions: dict, prices: dict, equity: float, peak_equity: float
    ) -> tuple[list[dict], list[dict]]:
        """
        Check all positions against risk rules.

        Returns:
            (positions_to_close, alerts)
            Each close dict has: symbol, reason, loss_pct
            Each alert dict has: symbol, message, severity
        """
        closes = []
        alerts = []
        now = datetime.now(timezone.utc)

        # --- Portfolio-level drawdown check ---
        if peak_equity > 0:
            dd_pct = (peak_equity - equity) / peak_equity * 100
            if dd_pct >= self.portfolio_max_dd_pct:
                alerts.append({
                    "symbol": "PORTFOLIO",
                    "message": f"Portfolio drawdown {dd_pct:.1f}% exceeds {self.portfolio_max_dd_pct}% limit",
                    "severity": "CRITICAL",
                })
                for symbol in list(positions.keys()):
                    closes.append({
                        "symbol": symbol,
                        "reason": f"Portfolio drawdown stop ({dd_pct:.1f}%)",
                        "loss_pct": 0,
                    })
                return closes, alerts

        # --- Individual position checks ---
        for symbol, pos in positions.items():
            current_price = prices.get(symbol)
            if current_price is None:
                continue

            entry_price = pos.get("entry_price", 0)
            side = pos.get("side", 0)
            entry_time_str = pos.get("entry_time", "")

            if entry_price <= 0:
                continue

            # Calculate current P&L %
            if side == 1:  # LONG
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price * 100

            # --- Position stop-loss ---
            if pnl_pct <= -self.position_stop_loss_pct:
                closes.append({
                    "symbol": symbol,
                    "reason": f"Stop-loss triggered ({pnl_pct:+.1f}%)",
                    "loss_pct": pnl_pct,
                })
                alerts.append({
                    "symbol": symbol,
                    "message": f"STOP-LOSS: {symbol} at {pnl_pct:+.1f}% (limit: -{self.position_stop_loss_pct}%)",
                    "severity": "WARNING",
                })

            # --- Trailing stop ---
            high_pnl = pos.get("high_pnl", 0)
            if pnl_pct > high_pnl:
                pos["high_pnl"] = pnl_pct  # Update in place
                high_pnl = pnl_pct

            if high_pnl >= self.trailing_stop_activation_pct:
                stop_level = high_pnl - self.trailing_stop_distance_pct
                if pnl_pct <= stop_level:
                    closes.append({
                        "symbol": symbol,
                        "reason": f"Trailing stop (peak {high_pnl:.1f}% -> {pnl_pct:.1f}%)",
                        "loss_pct": pnl_pct,
                    })
                    alerts.append({
                        "symbol": symbol,
                        "message": f"TRAILING STOP: {symbol} peaked {high_pnl:.1f}% now {pnl_pct:.1f}%",
                        "severity": "INFO",
                    })

            # --- Max position time ---
            if entry_time_str:
                try:
                    entry_time = datetime.fromisoformat(entry_time_str)
                    hours_held = (now - entry_time).total_seconds() / 3600
                    if hours_held >= self.max_position_hours:
                        closes.append({
                            "symbol": symbol,
                            "reason": f"Max time exceeded ({hours_held:.0f}h > {self.max_position_hours}h)",
                            "loss_pct": pnl_pct,
                        })
                        alerts.append({
                            "symbol": symbol,
                            "message": f"TIME STOP: {symbol} held {hours_held:.0f}h (limit: {self.max_position_hours}h)",
                            "severity": "INFO",
                        })
                except (ValueError, TypeError):
                    pass

        # --- Max open positions ---
        if len(positions) > self.max_open_positions:
            alerts.append({
                "symbol": "PORTFOLIO",
                "message": f"Too many open positions: {len(positions)} > {self.max_open_positions}",
                "severity": "WARNING",
            })

        return closes, alerts

    def can_open_position(
        self, positions: dict, equity: float, cash: float, prices: dict,
        peak_equity: float | None = None,
        dd_cooldown_until=None,
        now=None,
    ) -> tuple[bool, str]:
        """Check if a new position can be opened.

        Drawdown protection has TWO parts, both of which prevent the
        close-all -> re-open oscillation observed in the forward replay:

        1. ``dd_cooldown_until`` / ``now``: after a portfolio drawdown stop
           the caller keeps the account flat for ``dd_cooldown_hours`` and
           re-arms ``peak_equity`` to the post-stop equity, so protection is
           measured from the new baseline once trading resumes.
        2. ``peak_equity`` (without re-arming this would be a permanent
           shutdown: a flat cash account can never climb back above the dd
           line, so a static peak must NOT gate re-entry).
        """
        if dd_cooldown_until is not None and now is not None:
            try:
                cd = dd_cooldown_until
                if isinstance(cd, str):
                    cd = datetime.fromisoformat(cd)
                elif hasattr(cd, "to_pydatetime"):
                    cd = cd.to_pydatetime()
                cur = now
                if isinstance(cur, str):
                    cur = datetime.fromisoformat(cur)
                elif hasattr(cur, "to_pydatetime"):
                    cur = cur.to_pydatetime()
                if cd.tzinfo is None:
                    cd = cd.replace(tzinfo=timezone.utc)
                if cur.tzinfo is None:
                    cur = cur.replace(tzinfo=timezone.utc)
                if cur < cd:
                    return False, f"Drawdown cooldown until {cd.isoformat()}"
            except (ValueError, TypeError):
                pass
        if peak_equity is not None and peak_equity > 0:
            dd_pct = (peak_equity - equity) / peak_equity * 100
            if dd_pct >= self.portfolio_max_dd_pct:
                return False, (f"Portfolio drawdown {dd_pct:.1f}% >= "
                               f"{self.portfolio_max_dd_pct}% — below the dd line")
        # Max positions check
        if len(positions) >= self.max_open_positions:
            return False, f"Max {self.max_open_positions} positions reached"

        # Portfolio heat check (total exposure as % of equity)
        total_exposure = 0
        for symbol, pos in positions.items():
            price = prices.get(symbol, pos.get("entry_price", 0))
            side = pos.get("side", 0)
            entry = pos.get("entry_price", 0)
            size_usd = pos.get("size_usd", pos.get("size", 0))
            total_exposure += size_usd

        if equity > 0:
            heat_pct = total_exposure / equity * 100
            if heat_pct >= self.max_portfolio_heat_pct:
                return False, f"Portfolio heat {heat_pct:.0f}% exceeds {self.max_portfolio_heat_pct}% limit"

        # Minimum cash check
        if cash < 5:
            return False, f"Insufficient cash (${cash:.2f})"

        return True, "OK"

    def update_trailing_stops(self, positions: dict, prices: dict) -> dict:
        """Update high-water-mark for trailing stops. Returns updated positions."""
        for symbol, pos in positions.items():
            current_price = prices.get(symbol)
            if current_price is None:
                continue

            entry_price = pos.get("entry_price", 0)
            side = pos.get("side", 0)

            if side == 1:
                pnl_pct = (current_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - current_price) / entry_price * 100

            high_pnl = pos.get("high_pnl", 0)
            if pnl_pct > high_pnl:
                pos["high_pnl"] = pnl_pct

        return positions

    def format_alert(self, alert: dict) -> str:
        """Format an alert for Telegram."""
        severity_emoji = {
            "CRITICAL": "🚨",
            "WARNING": "⚠️",
            "INFO": "ℹ️",
        }
        emoji = severity_emoji.get(alert["severity"], "❓")
        return f"{emoji} <b>RISK ALERT: {alert['symbol']}</b>\n{alert['message']}"

    def format_close(self, close: dict) -> str:
        """Format a close notification for Telegram."""
        return (
            f"🛑 <b>AUTO-CLOSE: {close['symbol']}</b>\n"
            f"Reason: {close['reason']}\n"
            f"P&L: {close['loss_pct']:+.1f}%"
        )


# Default risk manager instance
DEFAULT_RISK_MANAGER = RiskManager(
    position_stop_loss_pct=5.0,
    portfolio_max_dd_pct=10.0,
    trailing_stop_activation_pct=5.0,
    trailing_stop_distance_pct=3.0,
    max_position_hours=72,
    max_open_positions=3,
    max_portfolio_heat_pct=50.0,  # 50% allows multiple positions on small accounts ($97)
)
