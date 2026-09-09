"""Risk governor (plan MD Task 5).

The ONLY authorizer of order intents. Every sizing decision is
stop-distance based:

    qty = equity * risk_per_trade * regime_multiplier / (price * stop_distance)

then clamped by every cap (symbol, cluster, gross, net, max positions,
margin reserve). Nothing downstream may enlarge a decision: a rejected
order stays rejected, an accepted size stays final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from crypto_system.models import OrderIntent, RiskDecision, RiskLimits
from crypto_system.risk.killswitch import KillSwitch
from crypto_system.risk.regime import RegimeDetector


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float
    cash: float
    gross_notional: float
    net_notional: float
    positions: Mapping[str, Mapping[str, float | str]]
    daily_pnl: float
    peak_equity: float


@dataclass(frozen=True)
class MarketState:
    price: float
    cluster_of: Mapping[str, str] = field(default_factory=dict)


class RiskGovernor:
    def __init__(
        self,
        limits: RiskLimits,
        *,
        killswitch: KillSwitch | None = None,
        regime: RegimeDetector | None = None,
    ) -> None:
        self.limits = limits
        self.killswitch = killswitch or KillSwitch()
        self.regime = regime or RegimeDetector()

    def evaluate(
        self,
        intent: OrderIntent,
        account: AccountSnapshot,
        market: MarketState,
        *,
        stop_distance: float,
        leverage: float = 1.0,
        volatility: float = 0.02,
        data_ok: bool = True,
    ) -> RiskDecision:
        limits = self.limits
        checks: dict[str, str] = {}

        # Non-cap gates first: halts and data health.
        if not data_ok:
            return RiskDecision(
                intent_digest=intent.digest(), accepted=False,
                reason="data fault: refusing to size on suspect data",
                checks=checks,
            )
        if self.killswitch.latched:
            return RiskDecision(
                intent_digest=intent.digest(), accepted=False,
                reason=f"halt latched: {'; '.join(self.killswitch.reasons)}",
                checks=checks,
            )

        # Halt conditions (latch if tripped).
        if account.daily_pnl < -limits.max_daily_loss * account.equity:
            self.killswitch.trip(
                "daily_loss",
                f"daily pnl {account.daily_pnl:.2f} beyond "
                f"{limits.max_daily_loss:.1%} of equity",
            )
        dd = 1.0 - (account.equity / account.peak_equity if account.peak_equity else 1.0)
        if dd > limits.max_drawdown:
            self.killswitch.trip(
                "drawdown",
                f"drawdown {dd:.1%} beyond limit {limits.max_drawdown:.1%}",
            )
        if self.killswitch.latched:
            return RiskDecision(
                intent_digest=intent.digest(), accepted=False,
                reason=f"halt tripped: {'; '.join(self.killswitch.reasons)}",
                checks=checks,
            )

        # No averaging down / pyramiding into an open position.
        if intent.symbol in account.positions:
            return RiskDecision(
                intent_digest=intent.digest(), accepted=False,
                reason=f"open position for {intent.symbol} — adding is prohibited "
                f"(no martingale, no averaging down)",
                checks=checks,
            )
        if len(account.positions) >= limits.max_positions:
            return RiskDecision(
                intent_digest=intent.digest(), accepted=False,
                reason=f"max positions ({limits.max_positions}) reached",
                checks=checks,
            )

        # Stop-distance sizing with regime scaling.
        stop = max(stop_distance, limits.min_stop_distance)
        vol_multiplier = self.regime.multiplier(volatility)
        risk_budget = account.equity * limits.max_risk_per_trade * vol_multiplier
        raw_qty = risk_budget / (market.price * stop)

        # Cap clamps — applied in order of tightness.
        cluster = market.cluster_of.get(intent.symbol, "default")
        cluster_used = sum(
            float(p["qty"]) * float(p["price"])
            for sym, p in account.positions.items()
            if market.cluster_of.get(sym, "default") == cluster
        )
        symbol_room = max(0.0, limits.max_symbol_notional - 0.0)
        gross_room = max(0.0, limits.max_gross_notional - account.gross_notional)
        cluster_room = max(0.0, limits.max_cluster_notional - cluster_used)
        net_dir = 1.0 if intent.side == "long" else -1.0
        net_room = max(
            0.0,
            limits.max_net_notional - net_dir * account.net_notional,
        )
        qty_cap = min(
            raw_qty,
            symbol_room / market.price,
            gross_room / market.price,
            cluster_room / market.price,
            net_room / market.price,
        )

        # Margin reserve: initial margin must fit within free cash reserve.
        leverage = min(leverage, limits.max_leverage)
        max_margin = account.cash * (1.0 - limits.margin_reserve)
        qty_cap = min(qty_cap, max_margin * leverage / market.price)

        if qty_cap <= 0:
            return RiskDecision(
                intent_digest=intent.digest(), accepted=False,
                reason="no room under position/symbol/cluster/gross/net caps",
                checks=checks,
            )

        checks["sizing"] = f"stop={stop:.4f} regime_x={vol_multiplier:.2f}"
        checks["caps"] = (
            f"symbol<= {limits.max_symbol_notional:.0f} "
            f"gross<= {limits.max_gross_notional:.0f} "
            f"cluster<= {limits.max_cluster_notional:.0f}"
        )
        return RiskDecision(
            intent_digest=intent.digest(),
            accepted=True,
            reason="ok",
            quantity=qty_cap,
            stop_distance=stop,
            leverage=leverage,
            checks=checks,
        )
