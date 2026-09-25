"""What the exchange will actually accept, and what that implies at a given
account size.

Paper trading has no venue: a $9.70 order fills happily in the ledger and the
"strategy" looks fine, while the real exchange would reject it outright.
That is not a simulation detail — it decides whether the account can trade at
all. This module is the single owner of those rules so the agent, the reports
and the feasibility check all answer the same way.

Real Binance USDⓈ-M values used by ``BUILTIN_LIMITS`` (fetched 2026-09-25 from
``ccxt.binanceusdm().load_markets()``; the exchange itself is authoritative and
``ExchangeInterface.get_market_limits`` prefers live values when reachable):

    BTC/USDT:USDT  MIN_NOTIONAL 50 USDT | LOT_SIZE step 0.001 | taker 0.05%
    ETH/USDT:USDT  MIN_NOTIONAL 20 USDT | LOT_SIZE step 0.001 | taker 0.05%

The builtin copy exists because the cloud runner is geo-blocked from Binance
(HTTP 451) and therefore cannot read the filters at all; a check that silently
disappears when the venue is unreachable is worse than no check, so the values
carry a source tag and every caller can report which one it used.

The arithmetic that matters at small size:

    risk_usd     = equity * risk_pct / 100          (the hard risk cap)
    notional_max = risk_usd / (stop_pct / 100)      (notional the cap allows)
    notional_min = max(venue MIN_NOTIONAL, paper floor)

``notional_min > notional_max`` means no legal order exists at that equity
under those risk rules, and ``min_equity_for_pair`` says how much equity would
be needed to change that.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# Binance USDⓈ-M taker fee, matching PAPER_TAKER_FEE_PCT in ai_agent.py.
# Maker is 0.02%, but the agent sends market orders, so taker is the honest
# number for a round trip.
DEFAULT_TAKER_FEE_PCT = 0.05

# Date the builtin table below was read off the live exchange.
BUILTIN_CAPTURED_AT = "2026-09-25"

# Tolerance for "exactly at the minimum" comparisons on float notionals.
_EPS = 1e-9


@dataclass(frozen=True)
class MarketLimits:
    """Venue-accepted order bounds for one pair."""

    pair: str
    min_notional: float      # MIN_NOTIONAL: minimum order value in quote ccy
    amount_step: float       # LOT_SIZE stepSize
    min_amount: float        # LOT_SIZE minQty
    taker_fee_pct: float
    source: str = "unknown"  # "exchange" | "builtin" | "unknown"

    def as_dict(self) -> dict:
        return asdict(self)


BUILTIN_LIMITS: dict[str, MarketLimits] = {
    "BTC/USDT:USDT": MarketLimits(
        pair="BTC/USDT:USDT", min_notional=50.0, amount_step=0.001,
        min_amount=0.001, taker_fee_pct=DEFAULT_TAKER_FEE_PCT, source="builtin",
    ),
    "ETH/USDT:USDT": MarketLimits(
        pair="ETH/USDT:USDT", min_notional=20.0, amount_step=0.001,
        min_amount=0.001, taker_fee_pct=DEFAULT_TAKER_FEE_PCT, source="builtin",
    ),
}


def builtin_limits(pair: str) -> MarketLimits | None:
    """Builtin limits for ``pair``, or None when the table has no entry."""
    return BUILTIN_LIMITS.get(pair)


def resolve_limits(
    pair: str, fetched: MarketLimits | None = None
) -> MarketLimits | None:
    """Live limits when available, else the builtin table, else None."""
    if fetched is not None:
        return fetched
    return builtin_limits(pair)


# ---------------------------------------------------------------------------
# Order legality
# ---------------------------------------------------------------------------

def floor_to_step(qty: float, step: float) -> float:
    """Largest step-multiple at or below ``qty`` (never rounds up)."""
    if step <= 0:
        return qty
    steps = math.floor(qty / step + _EPS)
    return round(steps * step, 12)


def min_legal_notional(limits: MarketLimits, floor_usd: float = 0.0) -> float:
    """Smallest order value the venue (and the paper floor) will accept."""
    return max(float(limits.min_notional), float(floor_usd))


def max_notional_for_risk(equity: float, risk_pct: float, stop_pct: float) -> float:
    """Largest notional whose stop-loss loss still equals ``risk_pct`` of equity.

    This is the honest ceiling on position size: the risk cap, not an
    arbitrary notional cap, is what protects the account.
    """
    if equity <= 0 or risk_pct <= 0 or stop_pct <= 0:
        return 0.0
    risk_usd = equity * risk_pct / 100.0
    return risk_usd / (stop_pct / 100.0)


def round_trip_fee(notional: float, taker_fee_pct: float) -> float:
    """Taker fee paid on entry AND exit for ``notional``."""
    return abs(notional) * taker_fee_pct / 100.0 * 2.0


def min_equity_required(
    floor_usd: float,
    risk_pct: float,
    stop_pct: float,
    max_position_size_pct: float,
) -> float:
    """Equity at which ``floor_usd`` becomes placeable under BOTH caps.

    The venue floor has to clear the risk cap AND the position-size cap, so
    the answer is the LARGER of the two demands — reporting only the risk one
    understates the requirement whenever the size cap is the tighter rule.
    """
    candidates: list[float] = []
    if risk_pct > 0 and stop_pct > 0:
        candidates.append(floor_usd * stop_pct / risk_pct)
    if max_position_size_pct > 0:
        candidates.append(floor_usd * 100.0 / max_position_size_pct)
    return max(candidates) if candidates else math.inf


def order_problems(limits: MarketLimits, notional: float, qty: float) -> list[str]:
    """Reasons the venue would reject this order ([] means it is legal)."""
    problems: list[str] = []
    if qty < limits.min_amount - _EPS:
        problems.append(
            f"quantity {qty:g} below minimum {limits.min_amount:g} "
            f"(step {limits.amount_step:g})"
        )
    if notional < limits.min_notional - _EPS:
        problems.append(
            f"notional ${notional:,.2f} below venue minimum "
            f"${limits.min_notional:,.2f}"
        )
    return problems


# ---------------------------------------------------------------------------
# Sizing at a given account size
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SizingBounds:
    """What a pair can legally be traded at, for one equity and stop distance."""

    pair: str
    equity: float
    stop_pct: float
    floor_usd: float       # smallest legal order
    ceiling_usd: float     # largest order the risk rules allow
    min_equity: float      # equity at which floor <= ceiling
    feasible: bool
    limits_source: str

    def as_dict(self) -> dict:
        return asdict(self)


def sizing_bounds(
    pair: str,
    limits: MarketLimits,
    *,
    equity: float,
    risk_pct: float,
    stop_pct: float,
    max_position_size_pct: float,
    floor_usd: float = 0.0,
) -> SizingBounds:
    """Legal order range for ``pair`` at ``equity`` with a ``stop_pct`` stop.

    ``ceiling_usd`` is the tighter of the risk cap and the position-size cap,
    so the returned range is what BOTH rules allow.
    """
    floor = min_legal_notional(limits, floor_usd)
    risk_ceiling = max_notional_for_risk(equity, risk_pct, stop_pct)
    size_ceiling = equity * max_position_size_pct / 100.0
    ceiling = min(risk_ceiling, size_ceiling)
    min_equity = min_equity_required(
        floor, risk_pct, stop_pct, max_position_size_pct
    )
    return SizingBounds(
        pair=pair,
        equity=round(equity, 4),
        stop_pct=stop_pct,
        floor_usd=round(floor, 4),
        ceiling_usd=round(ceiling, 4),
        min_equity=round(min_equity, 4) if math.isfinite(min_equity) else math.inf,
        feasible=floor <= ceiling + _EPS,
        limits_source=limits.source,
    )


def leverage_for_margin(
    notional: float, equity: float, max_margin_pct: float = 100.0
) -> float:
    """Leverage needed to hold ``notional`` while tying up <= max_margin_pct.

    Leverage is NOT a risk control: risk is ``notional * stop_pct`` and is
    untouched by it. It only decides how much margin is committed (and how far
    liquidation sits), so it is derived from margin appetite, never chosen to
    make an oversized position "fit". A result <= 1 means 1x already suffices.
    """
    if notional <= 0 or equity <= 0 or max_margin_pct <= 0:
        return 0.0
    margin_budget = equity * max_margin_pct / 100.0
    return notional / margin_budget


def pair_feasibility(
    pair: str,
    limits: MarketLimits,
    *,
    equity: float,
    risk_pct: float,
    max_stop_pct: float,
    max_position_size_pct: float,
    floor_usd: float = 0.0,
) -> dict:
    """Per-pair verdict at this equity, sized at the widest legal stop.

    Uses ``max_stop_pct`` because a wider stop permits a larger notional for
    the same risk — so this is the most favourable reading of the rules. If it
    is infeasible here, it is infeasible everywhere.
    """
    b = sizing_bounds(
        pair, limits, equity=equity, risk_pct=risk_pct, stop_pct=max_stop_pct,
        max_position_size_pct=max_position_size_pct, floor_usd=floor_usd,
    )
    out = b.as_dict()
    out["venue_min_notional"] = limits.min_notional
    out["amount_step"] = limits.amount_step
    if b.feasible:
        out["verdict"] = (
            f"tradable: orders ${b.floor_usd:,.2f}-${b.ceiling_usd:,.2f}"
        )
    else:
        out["verdict"] = (
            f"NOT tradable at ${equity:,.2f}: venue minimum ${b.floor_usd:,.2f} "
            f"exceeds the ${b.ceiling_usd:,.2f} the risk rules allow "
            f"(needs ~${b.min_equity:,.2f} equity)"
        )
    return out


def project_scale(
    *,
    equity: float,
    notional: float,
    risk_pct: float,
    rr: float,
    taker_fee_pct: float = DEFAULT_TAKER_FEE_PCT,
    trades_per_month: float,
    win_rate: float = 0.5,
) -> dict:
    """A typical trade and a typical month, net of fees, at this size.

    A projection with stated assumptions, not a forecast: ``win_rate`` is an
    input because the realized sample is far too small to estimate it, and
    ``trades_per_month`` is supplied by the caller from observed history.
    """
    risk_usd = equity * risk_pct / 100.0
    fees = round_trip_fee(notional, taker_fee_pct)
    win_net = risk_usd * rr - fees
    loss_net = risk_usd + fees
    expectancy = win_rate * win_net - (1 - win_rate) * loss_net
    month = expectancy * trades_per_month
    return {
        "equity": round(equity, 2),
        "notional": round(notional, 2),
        "stop_pct": round(risk_pct / 100.0 / (notional / equity) * 100, 3)
        if equity > 0 and notional > 0 else None,
        "risk_usd": round(risk_usd, 4),
        "fees_round_trip": round(fees, 4),
        "fees_pct_of_risk": round(fees / risk_usd * 100, 2) if risk_usd else None,
        "win_net": round(win_net, 4),
        "loss_net": round(loss_net, 4),
        "expectancy_per_trade": round(expectancy, 4),
        "trades_per_month": trades_per_month,
        "month_usd": round(month, 4),
        "month_pct": round(month / equity * 100, 3) if equity else None,
        "assumptions": {
            "win_rate": win_rate,
            "rr": rr,
            "taker_fee_pct": taker_fee_pct,
        },
    }
