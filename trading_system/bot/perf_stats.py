#!/usr/bin/env python3
"""Rolling performance statistics for the AI trading bot.

The weekly report used to answer only one question — "what happened in the
last 7 days?" — which is useless until enough trades close: a 50% win rate on
2 trades looks identical to a 50% win rate on 200. This module answers the
questions that actually inform a paper->live decision:

  * rolling win rate / expectancy / profit factor over 7d, 30d and all-time
  * realized max drawdown on the closed-trade equity curve (and where we are
    relative to the peak right now)
  * realized R:R (avg win / avg loss) next to the *planned* R:R the AI asked
    for, so "the AI's setups are drifting" is visible before it costs money
  * per-slot performance (which wakeup slot earns and which bleeds), which is
    the only way to tell whether e.g. the 08:00 session deserves to trade

Pure computation: no I/O, no clock reads unless the caller omits `now`, no
LLM calls, no orders. Everything is derived from the continuity files the
trader already writes, so the numbers can always be traced back to the ledger
and journal.

Definitions (documented because these get quoted as "the strategy's results"):

  realized equity curve
      ``start_equity + cumulative net_pnl``, ordered by close time. It is the
      *closed-trade* curve: an open position contributes nothing until it
      closes, so the drawdown here is realized drawdown, not mark-to-market.
  max drawdown
      the largest peak-to-trough fall on that curve, in $ and as % of the
      peak that preceded it.
  profit factor
      gross wins / |gross losses|. ``None`` until a loss exists — printing
      "inf" would read as a result when it is really an absence of data.
  expectancy
      mean net_pnl per closed trade.
  realized R:R
      ``avg_win / avg_loss`` (in $). ``None`` until both sides exist. The
      ``planned`` block reports what the entry records asked for
      (``take_profit_pct / stop_loss_pct``) before the fact.
  slot
      the wakeup slot that OPENED the trade. Attribution uses the journal:
      the newest wakeup at/before the entry time (within
      ``OPEN_MATCH_MINUTES``) is the wakeup that placed the order, and on a
      slot-based scheduler every wakeup serves the most recent scheduled slot
      at/before it (a catch-up wake can lag its slot by hours). So a trade
      opened by the 11:11 catch-up for the dropped 08:00 slot is credited to
      08:00, not to "14:00-ish".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

SLOT_HOURS: tuple[tuple[int, int], ...] = ((0, 0), (6, 0), (8, 0), (14, 0), (20, 0), (23, 0))

# How close a wakeup must be to an entry for us to call it "the wakeup that
# opened this trade". The open happens seconds after the wakeup starts; the
# slack absorbs clock skew and slow fills.
OPEN_MATCH_MINUTES = 30


def parse_ts(value: str | None) -> datetime | None:
    """Parse an ISO timestamp into an aware UTC datetime (None if unusable)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def slot_label(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def _slot_dt(day: datetime, hour: int, minute: int) -> datetime:
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def slot_at_or_before(
    ts: datetime, slots: tuple[tuple[int, int], ...] = SLOT_HOURS
) -> tuple[datetime, str]:
    """The most recent scheduled slot at or before ``ts`` (UTC)."""
    for day_offset in (0, 1):
        base = ts - timedelta(days=day_offset)
        candidates = [
            (_slot_dt(base, hour, minute), slot_label(hour, minute))
            for hour, minute in slots
            if _slot_dt(base, hour, minute) <= ts
        ]
        if candidates:
            return max(candidates, key=lambda c: c[0])
    # Unreachable for a non-empty slot list; fail visibly rather than crash.
    return ts, "off-slot"


def attribute_slot(
    entry_time: datetime | None,
    wakeups: list[datetime] | None = None,
    slots: tuple[tuple[int, int], ...] = SLOT_HOURS,
) -> str:
    """Slot whose wakeup opened the position (see module docstring)."""
    if entry_time is None:
        return "unknown"
    if wakeups:
        matched = [
            w
            for w in wakeups
            if w <= entry_time + timedelta(seconds=60)
            and entry_time - w <= timedelta(minutes=OPEN_MATCH_MINUTES)
        ]
        if matched:
            return slot_at_or_before(max(matched), slots)[1]
    return slot_at_or_before(entry_time, slots)[1]


def _num(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _pnl(trade: dict) -> float:
    """Net (round-trip, both fees) P&L for a closed trade."""
    for key in ("net_pnl", "cash_delta", "gross_pnl"):
        if trade.get(key) is not None:
            return _num(trade[key])
    return 0.0


def _pnl_pct(trade: dict) -> float:
    for key in ("pnl_pct_net", "pnl_pct"):
        if trade.get(key) is not None:
            return _num(trade[key])
    return 0.0


def _streaks(pnls: list[float]) -> tuple[int, int, int]:
    """(max_win_streak, max_loss_streak, current_streak) in trade order."""
    max_w = max_l = cur = 0
    for pnl in pnls:
        if pnl > 0:
            cur = cur + 1 if cur > 0 else 1
            max_w = max(max_w, cur)
        elif pnl < 0:
            cur = cur - 1 if cur < 0 else -1
            max_l = max(max_l, abs(cur))
        else:
            cur = 0
    return max_w, max_l, cur


def compute_metrics(trades: list[dict]) -> dict:
    """Rolling stats for a list of closed trades (order-insensitive)."""
    pnls = [_pnl(t) for t in trades]
    pcts = [_pnl_pct(t) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    win_pcts = [p for p in pcts if p > 0]
    loss_pcts = [p for p in pcts if p < 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    n = len(pnls)
    max_w, max_l, current = _streaks(pnls)

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": n - len(wins) - len(losses),
        "win_rate": (len(wins) / n * 100) if n else 0.0,
        "net_pnl": sum(pnls),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        # None, not infinity: no losses yet means the ratio is undefined.
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy": (sum(pnls) / n) if n else 0.0,
        "avg_win": (gross_profit / len(wins)) if wins else 0.0,
        "avg_loss": (-sum(losses) / len(losses)) if losses else 0.0,
        "avg_win_pct": (sum(win_pcts) / len(win_pcts)) if win_pcts else 0.0,
        "avg_loss_pct": (-sum(loss_pcts) / len(loss_pcts)) if loss_pcts else 0.0,
        "realized_rr": (gross_profit / len(wins)) / (-sum(losses) / len(losses))
        if wins and losses
        else None,
        "largest_win": max(pnls) if pnls else 0.0,
        "largest_loss": min(pnls) if pnls else 0.0,
        "max_win_streak": max_w,
        "max_loss_streak": max_l,
        "current_streak": current,
    }


def drawdown(points: list[tuple[datetime, float]]) -> dict:
    """Max + current drawdown on an equity curve (ordered oldest -> newest)."""
    if not points:
        return {
            "peak": 0.0, "max_dd": 0.0, "max_dd_pct": 0.0,
            "current_dd": 0.0, "current_dd_pct": 0.0,
        }
    peak = points[0][1]
    max_dd = 0.0
    max_dd_pct = 0.0
    for _, equity in points:
        peak = max(peak, equity)
        drop = peak - equity
        if drop > max_dd:
            max_dd = drop
            max_dd_pct = (drop / peak * 100) if peak else 0.0
    current = points[-1][1]
    current_dd = peak - current
    return {
        "peak": peak,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "current_dd": current_dd,
        "current_dd_pct": (current_dd / peak * 100) if peak else 0.0,
    }


def equity_curve(
    trades: list[dict], start_equity: float = 0.0
) -> list[tuple[datetime, float]]:
    """Realized equity curve: starting equity, then + net_pnl per close.

    The starting equity is the curve's FIRST point on purpose. Without it the
    peak could never be the pre-trade balance, so the account's first loss
    would not register as drawdown at all — a drawdown report that understates
    itself is worse than none.
    """
    now = datetime.now(timezone.utc)
    dated: list[tuple[datetime, dict]] = []
    undated: list[dict] = []
    for t in trades:
        ts = parse_ts(t.get("close_time"))
        if ts is None:
            undated.append(t)
        else:
            dated.append((ts, t))
    ordered: list[tuple[datetime | None, dict]] = sorted(
        dated, key=lambda item: item[0]
    ) + [(None, t) for t in undated]

    first_ts = dated[0][0] if dated else now
    curve: list[tuple[datetime, float]] = [(first_ts, float(start_equity))]
    equity = float(start_equity)
    last_ts = first_ts
    for ts, t in ordered:
        equity += _pnl(t)
        last_ts = ts or last_ts
        curve.append((last_ts, equity))
    return curve


def _window(trades: list[dict], now: datetime, days: int) -> list[dict]:
    cutoff = now - timedelta(days=days)
    out = []
    for t in trades:
        ts = parse_ts(t.get("close_time"))
        if ts is None:  # legacy record without a close time: all-time only
            continue
        if ts >= cutoff:
            out.append(t)
    return out


def _previous_window(trades: list[dict], now: datetime, days: int) -> list[dict]:
    end = now - timedelta(days=days)
    start = end - timedelta(days=days)
    out = []
    for t in trades:
        ts = parse_ts(t.get("close_time"))
        if ts is not None and start <= ts < end:
            out.append(t)
    return out


def slot_breakdown(
    trades: list[dict],
    journal: list[dict] | None = None,
    slots: tuple[tuple[int, int], ...] = SLOT_HOURS,
) -> list[dict]:
    """Per-slot trade performance + how often that slot decided."""
    wakeups = [
        ts
        for ts in (parse_ts(e.get("timestamp")) for e in (journal or []))
        if ts is not None
    ]
    wakeup_counts: dict[str, int] = {}
    for w in wakeups:
        label = slot_at_or_before(w, slots)[1]
        wakeup_counts[label] = wakeup_counts.get(label, 0) + 1

    buckets: dict[str, list[dict]] = {
        slot_label(h, m): [] for h, m in slots
    }
    for t in trades:
        key = attribute_slot(parse_ts(t.get("entry_time")), wakeups, slots)
        buckets.setdefault(key, []).append(t)

    rows = []
    for hour, minute in slots:
        label = slot_label(hour, minute)
        metrics = compute_metrics(buckets.get(label, []))
        rows.append(
            {
                "slot": label,
                "wakeups": wakeup_counts.get(label, 0),
                "trades": metrics["trades"],
                "wins": metrics["wins"],
                "win_rate": metrics["win_rate"],
                "net_pnl": metrics["net_pnl"],
                "avg_pnl": metrics["expectancy"],
            }
        )
    return rows


def ledger_audit(closed: list[dict], start_equity: float, ledger_equity: float) -> dict:
    """Reconcile reported P&L against the account's own equity.

    These must agree to the cent once every position is closed. When they do
    not, the review says so rather than quoting a P&L the account never earned:
    the 2026-09-14 paper record was written before net-at-both-fees accounting,
    so it is $0.50 optimistic — a real drift this check surfaced from cloud
    data on 2026-09-22. Do NOT silently rewrite history to make it zero.

    Only meaningful with no open positions (cash is realized equity then).
    """
    realized = float(start_equity) + sum(_pnl(t) for t in closed)
    return {
        "realized_equity": realized,
        "ledger_equity": float(ledger_equity),
        "drift": realized - float(ledger_equity),
    }


def planned_stats(opens: list[dict] | None) -> dict:
    """What the AI *asked* for at entry: planned R:R and confidence."""
    rrs = []
    confs = []
    for o in opens or []:
        sl = _num(o.get("stop_loss_pct"), 0.0)
        tp = _num(o.get("take_profit_pct"), 0.0)
        if sl > 0 and tp > 0:
            rrs.append(tp / sl)
        conf = o.get("confidence")
        if conf is not None:
            confs.append(_num(conf))
    return {
        "entries": len(opens or []),
        "avg_planned_rr": (sum(rrs) / len(rrs)) if rrs else None,
        "avg_confidence": (sum(confs) / len(confs)) if confs else None,
    }


def build_perf(
    closed_trades: list[dict],
    journal: list[dict] | None = None,
    opens: list[dict] | None = None,
    start_equity: float = 0.0,
    now: datetime | None = None,
    primary_days: int = 7,
    windows: tuple[int, ...] = (7, 30),
    slots: tuple[tuple[int, int], ...] = SLOT_HOURS,
    ledger_equity: float | None = None,
) -> dict:
    """Everything the weekly report needs, in one dict.

    ``windows`` are lookback lengths in days; ``primary_days`` also gets a
    trend block comparing it with the immediately preceding window of the
    same length ("this week vs last week"). ``ledger_equity`` (cash, with no
    positions open) enables the reconciliation check.
    """
    now = now or datetime.now(timezone.utc)

    window_metrics = {
        f"{days}d": compute_metrics(_window(closed_trades, now, days)) for days in windows
    }
    window_metrics["all"] = compute_metrics(list(closed_trades))

    prev = compute_metrics(_previous_window(closed_trades, now, primary_days))
    cur = compute_metrics(_window(closed_trades, now, primary_days))
    trend = {
        "window_days": primary_days,
        "prev_trades": prev["trades"],
        "prev_win_rate": prev["win_rate"],
        "prev_net_pnl": prev["net_pnl"],
        "prev_expectancy": prev["expectancy"],
        "win_rate_delta": cur["win_rate"] - prev["win_rate"],
        "net_pnl_delta": cur["net_pnl"] - prev["net_pnl"],
        "expectancy_delta": cur["expectancy"] - prev["expectancy"],
    }

    return {
        "windows": window_metrics,
        "trend": trend,
        "drawdown": drawdown(equity_curve(closed_trades, start_equity)),
        "slots": slot_breakdown(closed_trades, journal, slots),
        "planned": planned_stats(opens),
        "audit": ledger_audit(closed_trades, start_equity, ledger_equity)
        if ledger_equity is not None
        else None,
    }


def _pf(metrics: dict) -> str:
    pf = metrics.get("profit_factor")
    return f"PF {pf:.2f}" if pf is not None else "PF n/a"


def _rr(metrics: dict) -> str:
    rr = metrics.get("realized_rr")
    return f"R:R {rr:.2f}" if rr is not None else "R:R n/a"


def summarize(perf: dict) -> list[str]:
    """Plain-text lines shared by the stdout report and the Telegram message."""
    lines: list[str] = []
    for key in (k for k in perf.get("windows", {}) if k != "all"):
        m = perf["windows"][key]
        lines.append(
            f"{key:>3}: {m['trades']} trades | {m['win_rate']:.0f}% win | "
            f"{_pf(m)} | exp {m['expectancy']:+,.2f} | {_rr(m)} | "
            f"net {m['net_pnl']:+,.2f}"
        )

    m_all = perf.get("windows", {}).get("all")
    if m_all:
        streak = m_all.get("current_streak", 0)
        streak_txt = f"{streak}W" if streak > 0 else (f"{abs(streak)}L" if streak < 0 else "flat")
        lines.append(
            f"All: {m_all['trades']} trades | {m_all['win_rate']:.0f}% win | "
            f"{_pf(m_all)} | exp {m_all['expectancy']:+,.2f} | {_rr(m_all)}"
        )
        lines.append(
            f"     avg win {m_all['avg_win']:+,.2f} / avg loss {m_all['avg_loss']:,.2f} | "
            f"streak {streak_txt} "
            f"(best {m_all['max_win_streak']}W / worst {m_all['max_loss_streak']}L)"
        )

    dd = perf.get("drawdown") or {}
    if dd:
        lines.append(
            f"Drawdown: max {dd['max_dd_pct']:.2f}% (${dd['max_dd']:,.2f}) | "
            f"now {dd['current_dd_pct']:.2f}% off peak ${dd['peak']:,.2f}"
        )

    slots = perf.get("slots") or []
    traded = [s for s in slots if s["trades"]]
    if traded:
        chunks = []
        for s in slots:
            if s["trades"]:
                chunks.append(
                    f"{s['slot']} {s['trades']}t {s['win_rate']:.0f}% {s['net_pnl']:+,.2f}"
                )
            else:
                chunks.append(f"{s['slot']} no trades")
        for i in range(0, len(chunks), 3):
            prefix = "Slots: " if i == 0 else "       "
            lines.append(prefix + " | ".join(chunks[i:i + 3]))
    else:
        lines.append("Slots: no closed trades yet")

    planned = perf.get("planned") or {}
    if planned.get("avg_planned_rr") is not None:
        conf = planned.get("avg_confidence")
        conf_txt = f" | avg conf {conf:.0f}" if conf is not None else ""
        lines.append(
            f"Planned: R:R {planned['avg_planned_rr']:.2f} avg over "
            f"{planned['entries']} entries{conf_txt}"
        )

    audit = perf.get("audit")
    # Only shout when it actually differs: a cent of float noise is not news,
    # a dollar of unexplained P&L is.
    if audit and abs(audit["drift"]) > 0.01:
        lines.append(
            f"⚠ Ledger check: reported P&L {audit['drift']:+,.2f} vs realized cash "
            f"${audit['ledger_equity']:,.2f} (older record predates fee-inclusive P&L)"
        )

    trend = perf.get("trend") or {}
    if trend:
        lines.append(
            f"Trend ({trend['window_days']}d vs prev): "
            f"win {trend['win_rate_delta']:+.0f}pp | "
            f"exp {trend['expectancy_delta']:+,.2f} | "
            f"net {trend['net_pnl_delta']:+,.2f}"
        )
    return lines
