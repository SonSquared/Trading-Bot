"""
Shared cost model for the paper trader, the walk-forward backtests, and the
forward-run evaluator.

One set of constants so "P&L" means the same thing everywhere:

- ``FEE_RATE``       0.05% taker fee, charged on the notional of each side
                      (entry AND exit).
- ``SLIPPAGE_RATE``  0.02% per side, embedded in the fill price.
- ``FUNDING_RATE_8H`` 0.0001 (~0.01%) charged at every 8-hour UTC boundary
                      (00:00, 08:00, 16:00) while a position is held — the
                      standard perpetual-futures funding schedule.

This replaces the old ``FUNDING_PER_CANDLE = 0.00005`` per-4h-candle
approximation used by the walk-forward scripts: 0.00005 per 4h candle is
exactly half of ``FUNDING_RATE_8H``, so per-position totals are identical
for positions held a whole number of 4h candles.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

FEE_RATE = 0.0005        # 0.05% taker fee per side
SLIPPAGE_RATE = 0.0002   # 0.02% per side
FUNDING_RATE_8H = 0.0001  # ~0.01% every 8h (00:00 / 08:00 / 16:00 UTC)

# Legacy alias kept so old code that referenced FUNDING_PER_CANDLE still
# imports; it equals FUNDING_RATE_8H per 4h candle (2 candles = 8h).
FUNDING_PER_CANDLE = FUNDING_RATE_8H / 2


def as_utc(dt: Any) -> pd.Timestamp:
    """Coerce a timestamp (aware, naive, or pd.Timestamp) to tz-aware UTC."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def next_funding_boundary(dt: Any) -> pd.Timestamp:
    """The next 8h funding boundary (00:00/08:00/16:00 UTC) strictly after dt."""
    ts = as_utc(dt)
    hour = ts.hour
    for target in (0, 8, 16):
        if target > hour:
            return ts.normalize() + pd.Timedelta(hours=target)
    return ts.normalize() + pd.Timedelta(days=1)


def funding_boundaries_in(start_dt: Any, end_dt: Any) -> int:
    """Number of 8h funding boundaries in ``[start_dt, end_dt)``.

    A boundary exactly at ``start`` is charged (conservative: a position
    held at a settlement pays it), a boundary exactly at ``end`` is not.
    """
    start = as_utc(start_dt)
    end = as_utc(end_dt)
    if end <= start:
        return 0
    # First boundary >= start: start itself only if it is exactly on a
    # boundary (e.g. entry at 08:00:00 pays the 08:00 settlement); otherwise
    # the next boundary strictly after start.
    if start.hour in (0, 8, 16) and start.minute == 0 and start.second == 0 and start.microsecond == 0:
        b = start
    else:
        b = next_funding_boundary(start)
    count = 0
    while b < end:
        count += 1
        b += pd.Timedelta(hours=8)
    return count


def funding_cost(notional: float, entry_dt: Any, exit_dt: Any) -> float:
    """Funding accrued on ``notional`` between entry and exit (8h schedule)."""
    if notional <= 0:
        return 0.0
    return notional * FUNDING_RATE_8H * funding_boundaries_in(entry_dt, exit_dt)


def charge_funding(state: dict, now: datetime | None = None) -> tuple[float, dict]:
    """Charge 8h funding on all open positions in a paper state dict.

    Mutates ``state`` in place: ``cash`` and ``total_pnl`` are debited and
    each position tracks ``last_funding_time`` and cumulative
    ``funding_paid``. Returns ``(total_charged, {pair: amount})``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    now_ts = as_utc(now)

    total = 0.0
    per_pair: dict[str, float] = {}
    for pair, pos in state.get("positions", {}).items():
        entry_price = pos.get("entry_price", 0)
        size_usd = pos.get("size_usd", 0)
        if entry_price <= 0 or size_usd <= 0:
            continue
        qty = size_usd / entry_price

        last = pos.get("last_funding_time")
        if last:
            start = last
        else:
            start = pos.get("entry_time") or datetime.now(timezone.utc).isoformat()

        n = funding_boundaries_in(start, now_ts)
        if n <= 0:
            continue
        amt = qty * entry_price * FUNDING_RATE_8H * n
        state["cash"] -= amt
        # Funding is a real cash cost — booking it against total_pnl keeps
        # the displayed P&L consistent with equity (cash-based P&L).
        state["total_pnl"] = state.get("total_pnl", 0.0) - amt
        pos["last_funding_time"] = now.isoformat()
        pos["funding_paid"] = pos.get("funding_paid", 0.0) + amt
        total += amt
        per_pair[pair] = amt
    return total, per_pair