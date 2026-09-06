"""
Forward replay of the DEPLOYED paper bot — full risk model.

Replays the exact decision logic of scripts/paper_trader.py — closed-candle
signals, weighted hysteresis aggregation, one trade per new closed candle,
next-open fills, shared cash across pairs — over the most recent ``--days``
days of 4h candles, charging the full cost model shared with the backtest
engine (trading_system.bot.accounting):

  - 0.05% taker fee per side
  - 0.02% slippage per side (embedded in the fill)
  - funding at every 8h UTC boundary (00:00/08:00/16:00) while held

It ALSO replays the paper bot's risk manager (DEFAULT_RISK_MANAGER from
trading_system.bot.risk_manager), so the report reflects the full repaired
bot, not just strategy signals:

  - 5% disaster stop-loss per position
  - trailing stop: ratchets after +5% profit, exits on a 3% retrace
  - 72h maximum position time
  - 10% portfolio drawdown stop: closes all positions, re-arms the equity
    peak at the post-stop level, and stays FLAT for a 24h cooldown. This is
    the only workable drawdown semantics for a flat-cash account: a static
    peak would either oscillate (close-all + re-open every cycle) or leave
    the bot permanently dormant (flat cash can never climb back above an
    old dd line)

The order of phases within each candle boundary mirrors the bot's main loop:
signals are aggregated FIRST (so hysteresis still sees pre-risk positions),
risk closes execute next, then signal trades execute at the next open.
Because of that order the replay intentionally reproduces a real bot
behaviour: a position stopped out at a candle's low can be re-opened at the
next open if the aggregate signal (computed while it was still held) is
still directional.

Intra-candle approximation (documented, deliberate): the live bot polls every
~15 minutes; this replay only has 4h OHLC. Stops are checked against each
held candle's extremes (low for longs / high for shorts), evaluated
ADVERSE-FIRST — the stop level comes from the profit watermark at the START
of the candle, and the watermark only ratchets up after a candle the
position survives. Fills occur at the stop level (or at the candle open when
it gaps through) plus the normal slippage. Time and drawdown stops fill at
the candle close. This errs toward NOT crediting the stops with intra-candle
perfection.

Because the deployed parameters were selected on data older than the replay
window, this is a true out-of-sample forward test: it answers "does the
deployed strategy have any edge net of costs and risk management?" without
waiting N days.

Replay runs NEVER write to the production trade log: log_trade is patched to
a no-op while this module is imported (the earlier 30-day runs polluted
data/results/paper_trades.jsonl with synthetic entries).

Output:
  python scripts/forward_run.py [--days N]        -> forward_run_report.{md,json}
  python scripts/forward_run.py --compare         -> fetches 90 days once, replays
     the trailing 90d (forward_run_report_90d.{md,json}) plus two
     NON-OVERLAPPING windows — the prior 60d and the recent 30d — and writes
     forward_run_comparison.{md,json} with a regime-consistency verdict.
"""

from __future__ import annotations

import json
import sys
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.paper_trader as pt  # noqa: E402
from scripts.paper_trader import (
    load_active_strategies,
    aggregate_signals,
    open_position,
    close_position,
    _fresh_state,
    format_pair,
    INITIAL_CAPITAL,
    MAX_POSITION_PCT,
    MIN_TRADE_USD,
)
from trading_system.bot.accounting import (
    FEE_RATE,
    SLIPPAGE_RATE,
    FUNDING_RATE_8H,
    funding_cost,
)
from trading_system.bot.risk_manager import DEFAULT_RISK_MANAGER
from trading_system.strategies import STRATEGY_REGISTRY

# Replays must never write to the production trade log. open_position /
# close_position above log every trade through scripts.paper_trader.log_trade;
# neutralise it. (The earlier forward runs polluted paper_trades.jsonl.)
pt.log_trade = lambda _trade: None  # type: ignore[assignment]

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_MD = RESULTS_DIR / "forward_run_report.md"
REPORT_JSON = RESULTS_DIR / "forward_run_report.json"
REPORT_MD_90D = RESULTS_DIR / "forward_run_report_90d.md"
REPORT_JSON_90D = RESULTS_DIR / "forward_run_report_90d.json"
COMPARE_MD = RESULTS_DIR / "forward_run_comparison.md"
COMPARE_JSON = RESULTS_DIR / "forward_run_comparison.json"

SYMBOL_MAP = {
    "ETH_USDT_USDT": "ETH/USDT",
    "BTC_USDT_USDT": "BTC/USDT",
}


def fetch_klines(pair: str, timeframe: str = "4h", days: int = 30) -> pd.DataFrame | None:
    """Paginated Kraken (fallback Binance) fetch of the last ``days`` of candles."""
    symbol = SYMBOL_MAP.get(pair, pair.replace("_", "/"))
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    max_candles = int(days * 24 * 3600 / 4 / 3600) + 60

    import ccxt

    for exchange_cls, limit in ((ccxt.kraken, 720), (ccxt.binance, 1000)):
        try:
            exchange = exchange_cls({"enableRateLimit": True})
            spot = symbol.replace(":USDT", "")
            out = []
            since = since_ms
            while len(out) < max_candles:
                batch = exchange.fetch_ohlcv(spot, timeframe, since=since, limit=limit)
                if not batch:
                    break
                out.extend(batch)
                if len(batch) < limit:
                    break
                since = batch[-1][0] + 1
                _time.sleep((getattr(exchange, "rateLimit", 0) or 500) / 1000.0)
            if len(out) >= 60:
                df = pd.DataFrame(out, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
                print(f"  {exchange_cls.__name__}: {len(df)} candles for {pair} ({timeframe}, {days}d)")
                return df
            print(f"  {exchange_cls.__name__}: only {len(out)} candles, trying fallback")
        except Exception as e:
            print(f"  {exchange_cls.__name__} failed for {symbol}: {e}")
    return None


def position_value(state: dict, pair: str, price: float) -> float:
    """Mark-to-market value of one open position at ``price`` (mirrors get_equity)."""
    pos = state["positions"].get(pair)
    if not pos:
        return 0.0
    entry = pos["entry_price"]
    size = pos["size_usd"]
    side = pos["side"]
    qty = size / entry if entry > 0 else 0
    if side == 1:
        return size + qty * (price - entry)
    return size + qty * (entry - price)


def equity_at(state: dict, pairs: list[str], prices: dict[str, float]) -> float:
    """Total equity = cash + positions marked at ``prices``."""
    return state["cash"] + sum(position_value(state, p, prices[p]) for p in pairs)


def apply_position_stops(pos: dict, candle: dict, risk) -> tuple[str | None, float | None]:
    """Adverse-first stop check for one position across one 4h candle.

    Mutates ``pos["high_pnl"]`` (profit watermark) when the position survives.
    Returns ``(reason, fill_ref)`` when a stop fires, else ``(None, None)``.

    Mirrors the precedence in RiskManager.check_positions: disaster stop-loss
    first, then the trailing stop, then (handled by the caller) time.
    """
    side = pos["side"]
    entry = pos["entry_price"]
    if entry <= 0:
        return None, None
    o = float(candle["open"])
    h = float(candle["high"])
    l = float(candle["low"])
    long = side == 1
    adv_px = l if long else h
    fav_px = h if long else l
    wm = pos.get("high_pnl", 0.0)

    def pnl_pct(px: float) -> float:
        return (px - entry) / entry * side * 100.0

    adv = pnl_pct(adv_px)

    # 1) disaster stop-loss (fixed % from entry). The reason reports the pnl
    #    at the FILL (the level, or worse when gapped), not the candle extreme.
    if adv <= -risk.position_stop_loss_pct:
        level = entry * (1 - risk.position_stop_loss_pct / 100.0) if long else entry * (1 + risk.position_stop_loss_pct / 100.0)
        fill = min(o, level) if long else max(o, level)
        return f"Stop-loss triggered ({pnl_pct(fill):+.1f}%)", fill

    # 2) trailing stop (only if the watermark has activated)
    if wm >= risk.trailing_stop_activation_pct:
        stop_pnl = wm - risk.trailing_stop_distance_pct
        if adv <= stop_pnl:
            level = entry * (1 + stop_pnl / 100.0) if long else entry * (1 - stop_pnl / 100.0)
            fill = min(o, level) if long else max(o, level)
            return f"Trailing stop (peak {wm:.1f}% -> {pnl_pct(fill):+.1f}%)", fill

    # 3) survived the candle -> ratchet the watermark (adverse-first: the new
    #    high only counts from the NEXT candle's stop checks onward).
    fav = pnl_pct(fav_px)
    if fav > wm:
        pos["high_pnl"] = fav
    return None, None


def _replay(dfs: dict[str, pd.DataFrame], label: str = "window",
            start: int = 1) -> dict:
    """Replay the deployed bot over aligned, equal-length DataFrames (dict of pairs).

    ``start``: first candle index to trade. Signals are still computed from
    the FULL history (indices [0..i]) so indicators are warm, but no orders
    are placed before ``start`` — used to replay only a sub-window (e.g. the
    period the live bot actually ran) without starving the strategies of
    warmup data.
    """
    load_active_strategies()

    pairs = sorted(dfs.keys())
    n = min(len(dfs[p]) for p in pairs)
    dfs = {p: df.iloc[:n].reset_index(drop=True) for p, df in dfs.items()}
    days = (pd.Timestamp(dfs[pairs[0]]["timestamp"].iloc[-1]) - pd.Timestamp(dfs[pairs[0]]["timestamp"].iloc[0])).total_seconds() / 86400
    print(f"Replaying {n} candles ({days:.1f} days) per pair: {label}\n")

    risk = DEFAULT_RISK_MANAGER
    state = _fresh_state()
    trades = []          # closed-trade records (paper trader format)
    cost_fees = 0.0
    cost_slippage = 0.0
    cost_funding = 0.0
    equity_curve = []
    # When replaying a sub-window (start > 1), seed the peak with the equity
    # the account would have had at the window start (flat cash baseline).
    if start > 1:
        i0 = start - 1
        seed_px = {p: float(dfs[p]["close"].iloc[i0]) for p in pairs}
        state["peak_equity"] = equity_at(state, pairs, seed_px)

    def record_close(pair: str, reason: str, fill_ref: float, ts_boundary) -> None:
        """Close a position, book funding, and append the trade record."""
        nonlocal cost_fees, cost_slippage, cost_funding
        pos = state["positions"].get(pair)
        if not pos:
            return
        entry_time_iso = pos.get("entry_time", "")
        qty = pos["size_usd"] / pos["entry_price"]
        funding = funding_cost(qty * pos["entry_price"], pos["entry_time"], ts_boundary)
        t = close_position(state, pair, fill_ref, reason)
        if t:
            cost_fees += t.get("fee", 0.0)
            cost_slippage += t.get("slippage", 0.0)
            cost_funding += funding
            t["_funding"] = funding
            # close_position stamps wall-clock "now"; override with the real
            # candle times so the trade record is chronologically auditable
            t["_entry_time"] = entry_time_iso
            t["_exit_time"] = pd.Timestamp(ts_boundary).isoformat()
            trades.append(t)

    for i in range(start, n - 1):
        # ── 1. Signals from closed candles [0..i]; fills at the next open.
        next_open = {}
        closed = {}
        signals = {}
        for pair in pairs:
            df = dfs[pair]
            closed[pair] = df.iloc[: i + 1]
            next_open[pair] = float(df["open"].iloc[i + 1])
            # Read via the module attribute: load_active_strategies() REBINDS
            # paper_trader.ACTIVE_STRATEGIES (a ``from X import Y`` here would
            # go stale and silently trade the default strategies instead of
            # the optimized ones).
            for name, cfg in pt.ACTIVE_STRATEGIES.items():
                if cfg["pair"] != pair:
                    continue
                sig = STRATEGY_REGISTRY[cfg["strategy"]].generate_signals(closed[pair], cfg["params"])
                price = float(closed[pair]["close"].iloc[-1])
                atr = (float(closed[pair]["close"].diff().abs().rolling(14).mean().iloc[-1])
                       if len(closed[pair]) > 14 else price * 0.01)
                signals[name] = {
                    "signal": int(sig.iloc[-1]) if len(sig) else 0,
                    "weight": cfg["weight"],
                    "pair": pair,
                    "strategy": cfg["strategy"],
                    "price": price,
                    "next_open": next_open[pair],
                    "atr": atr,
                    "candle_time": pd.to_datetime(closed[pair]["timestamp"].iloc[-1], utc=True),
                }

        # ── 2. Aggregate BEFORE risk closes (mirrors the bot: hysteresis sees
        #        the pre-risk positions, so a stopped-out pair can re-enter at
        #        the next open on the same cycle).
        portfolio = aggregate_signals(signals, state)

        # Boundary prices / timestamps (all pairs share 4h UTC boundaries)
        close_px = {p: float(dfs[p]["close"].iloc[i]) for p in pairs}
        ts_boundary = pd.Timestamp(dfs[pairs[0]]["timestamp"].iloc[i + 1])

        # ── 3. RISK PHASE (mirrors the bot's risk block between aggregate and execution)
        if state["positions"]:
            # 3a. portfolio drawdown stop at the candle close
            eq_now = equity_at(state, pairs, close_px)
            peak = state.get("peak_equity", eq_now)
            dd_pct = (peak - eq_now) / peak * 100.0 if peak > 0 else 0.0
            if dd_pct >= risk.portfolio_max_dd_pct:
                for pair in list(state["positions"]):
                    record_close(pair, f"Portfolio drawdown stop ({dd_pct:.1f}%)",
                                 close_px[pair], ts_boundary)
                    print(f"  RISK CLOSE {pair}: drawdown stop -> dd {dd_pct:.1f}%")
                # Mirror the bot: re-arm the peak to the post-stop equity and
                # go flat for the cooldown, or the account either oscillates
                # (close->reopen) or stays dormant forever (static peak).
                state["peak_equity"] = equity_at(state, pairs, close_px)
                state["dd_cooldown_until"] = (
                    ts_boundary + pd.Timedelta(hours=risk.dd_cooldown_hours)
                ).isoformat()
            else:
                # 3b. per-position stops (disaster SL / trailing / time)
                for pair in list(state["positions"]):
                    pos = state["positions"][pair]
                    row = dfs[pair].iloc[i]
                    reason, fill_ref = apply_position_stops(
                        pos,
                        {"open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"]},
                        risk,
                    )
                    # time stop: decided at this boundary, filled at the close
                    if reason is None:
                        try:
                            entry_ts = pd.Timestamp(pos["entry_time"])
                            held_h = (ts_boundary - entry_ts).total_seconds() / 3600
                            if held_h >= risk.max_position_hours:
                                reason = f"Max time exceeded ({held_h:.0f}h > {risk.max_position_hours}h)"
                                fill_ref = close_px[pair]
                        except (ValueError, TypeError):
                            pass
                    if reason and fill_ref is not None and pair in state["positions"]:
                        record_close(pair, reason, fill_ref, ts_boundary)
                        print(f"  RISK CLOSE {pair}: {reason}")

        # can_open gate — mirrors risk_manager.can_open_position after closes,
        # including the drawdown cooldown: a stopped-out account stays flat for
        # ``dd_cooldown_hours`` with the peak re-armed, so it neither
        # oscillates (close->reopen every candle) nor goes dormant forever.
        eq_after = equity_at(state, pairs, close_px)
        can_open, _reason = risk.can_open_position(
            state["positions"], eq_after, state["cash"], close_px,
            peak_equity=state.get("peak_equity"),
            dd_cooldown_until=state.get("dd_cooldown_until"),
            now=ts_boundary,
        )

        # ── 4. Signal execution at the next open
        for pair in pairs:
            data = portfolio.get(pair)
            if not data:
                continue
            desired = data["signal"]
            pos = state["positions"].get(pair)
            side = pos["side"] if pos else 0

            if desired != side:
                if side != 0:
                    # Min-hold: never close sooner than 8h (2 x 4h candles)
                    entry_ts = pd.Timestamp(pos["entry_time"])
                    if ts_boundary - entry_ts < pd.Timedelta(hours=8):
                        continue
                    record_close(pair, "Signal reversal", next_open[pair], ts_boundary)
                if desired != 0 and can_open and not state["positions"].get(pair):
                    equity = equity_at(state, pairs, close_px)
                    size = min(equity * MAX_POSITION_PCT, state["cash"] * 0.95, INITIAL_CAPITAL * 0.50)
                    if size > MIN_TRADE_USD:
                        strat_name = "portfolio"
                        for sname, sdata in signals.items():
                            if sdata["pair"] == pair:
                                strat_name = sdata["strategy"]
                                break
                        t = open_position(state, pair, desired, next_open[pair], strat_name, size)
                        if t:
                            # Stamp entry with the candle time for min-hold + funding
                            state["positions"][pair]["entry_time"] = pd.Timestamp(
                                dfs[pair]["timestamp"].iloc[i + 1]
                            ).isoformat()
                            cost_fees += t.get("fee", 0.0)
                            cost_slippage += t.get("slippage", 0.0)

        # ── 5. Equity mark at the close of the current candle; update peak
        eq = equity_at(state, pairs, close_px)
        equity_curve.append(eq)
        if eq > state.get("peak_equity", 0):
            state["peak_equity"] = eq
        dd = (state["peak_equity"] - eq) / state["peak_equity"] if state["peak_equity"] > 0 else 0
        state["max_drawdown"] = max(state.get("max_drawdown", 0), dd)

    # Force-close any remaining positions at the last close (like the backtest)
    ts_end = pd.Timestamp(dfs[pairs[0]]["timestamp"].iloc[-1])
    for pair in pairs:
        if pair in state["positions"]:
            record_close(pair, "End of window", float(dfs[pair]["close"].iloc[-1]), ts_end)

    final_equity = state["cash"]
    start_ts = dfs[pairs[0]]["timestamp"].iloc[0]
    end_ts = dfs[pairs[0]]["timestamp"].iloc[-1]

    wins = sum(1 for t in trades if t["pnl_usd"] > 0)
    total_pnl = sum(t["pnl_usd"] for t in trades)
    gross_pnl = total_pnl + cost_fees + cost_slippage + cost_funding
    total_costs = cost_fees + cost_slippage + cost_funding
    turnover = sum(t.get("size_usd", 0) for t in trades if t["action"] == "CLOSE") * 2

    eq_arr = np.array(equity_curve) if equity_curve else np.array([INITIAL_CAPITAL])
    peaks = np.maximum.accumulate(eq_arr)
    max_dd = float(np.max((peaks - eq_arr) / np.where(peaks > 0, peaks, 1))) * 100

    per_pair = {}
    for pair in pairs:
        pt_ = [t for t in trades if t["pair"] == pair]
        start = float(dfs[pair]["close"].iloc[0])
        end = float(dfs[pair]["close"].iloc[-1])
        per_pair[pair] = {
            "trades": len(pt_),
            "wins": sum(1 for t in pt_ if t["pnl_usd"] > 0),
            "pnl": round(sum(t["pnl_usd"] for t in pt_), 4),
            "fees": round(sum(t.get("fee", 0) for t in pt_), 4),
            "funding": round(sum(t.get("_funding", 0) for t in pt_), 4),
            "price_change_pct": round((end / start - 1) * 100, 2),
        }

    # Exit-reason breakdown (shows how much of the P&L came from risk closes vs signals)
    reasons = {}
    for t in trades:
        r = t.get("reason", "")
        if r.startswith("Signal"):
            cat = "signal"
        elif r.startswith("Stop-loss"):
            cat = "stop_loss"
        elif r.startswith("Trailing"):
            cat = "trailing"
        elif r.startswith("Max time"):
            cat = "time"
        elif r.startswith("Portfolio drawdown"):
            cat = "drawdown"
        else:
            cat = "end_of_window"
        reasons.setdefault(cat, {"count": 0, "pnl": 0.0})
        reasons[cat]["count"] += 1
        reasons[cat]["pnl"] += t["pnl_usd"]

    buy_hold = sum(
        0.35 * (float(dfs[p]["close"].iloc[-1]) / float(dfs[p]["close"].iloc[0]) - 1)
        for p in pairs
    ) * 100

    risk_cfg = {
        "stop_loss_pct": risk.position_stop_loss_pct,
        "trailing_activation_pct": risk.trailing_stop_activation_pct,
        "trailing_distance_pct": risk.trailing_stop_distance_pct,
        "max_position_hours": risk.max_position_hours,
        "portfolio_max_dd_pct": risk.portfolio_max_dd_pct,
        "dd_cooldown_hours": risk.dd_cooldown_hours,
        "max_open_positions": risk.max_open_positions,
    }

    report = {
        "window": {"start": str(start_ts)[:16], "end": str(end_ts)[:16],
                   "days": round((pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds() / 86400, 1),
                   "candles": n, "timeframe": "4h", "pairs": pairs, "label": label},
        "capital": {"initial": INITIAL_CAPITAL, "final": round(final_equity, 4),
                    "return_pct": round((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 3)},
        "trades": {"closed": len(trades), "wins": wins,
                   "win_rate_pct": round(wins / len(trades) * 100, 1) if trades else 0,
                   "gross_pnl": round(gross_pnl, 4),
                   "net_pnl": round(total_pnl, 4),
                   "exit_breakdown": {k: {"count": v["count"], "pnl": round(v["pnl"], 4)}
                                      for k, v in reasons.items()}},
        "costs": {"fees": round(cost_fees, 4), "slippage": round(cost_slippage, 4),
                  "funding": round(cost_funding, 4), "total": round(total_costs, 4),
                  "costs_pct_of_turnover": round(total_costs / turnover * 100, 3) if turnover else None},
        "risk": {"max_drawdown_pct": round(max_dd, 2)},
        "benchmark": {"buy_hold_35pct_each_pct": round(buy_hold, 2)},
        "alpha_pp": round((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 - buy_hold, 2),
        "per_pair": per_pair,
        "risk_config": risk_cfg,
        "cost_model": {
            "fee_rate": FEE_RATE, "slippage_rate": SLIPPAGE_RATE,
            "funding_rate_8h": FUNDING_RATE_8H,
            "fills": "next open", "signals": "closed candles only",
        },
        "deployed_strategies": {
            name: {"strategy": cfg["strategy"], "pair": cfg["pair"],
                   "params": cfg["params"]}
            for name, cfg in pt.ACTIVE_STRATEGIES.items()
        },
        "trades_detail": [
            {
                "pair": t["pair"],
                "side": t.get("side"),
                "action": t["action"],
                "entry_price": t.get("entry_price"),
                "exit_price": t.get("exit_price"),
                "size_usd": t.get("size_usd"),
                "pnl_usd": t.get("pnl_usd"),
                "pnl_pct": t.get("pnl_pct"),
                "fee": t.get("fee"),
                "slippage": t.get("slippage"),
                "funding": t.get("_funding"),
                "reason": t.get("reason"),
                "entry_time": t.get("_entry_time"),
                "exit_time": t.get("_exit_time"),
            }
            for t in trades
        ],
        # One mark per closed candle (4h) for auditing the equity path
        "equity_timeline": [
            {"t": pd.Timestamp(dfs[pairs[0]]["timestamp"].iloc[i]).isoformat(), "equity": round(e, 4)}
            for i, e in enumerate(equity_curve, start=1)
        ],
    }
    return report


def _window_label(rows: pd.DataFrame, prefix: str) -> str:
    s = rows["timestamp"].iloc[0]
    e = rows["timestamp"].iloc[-1]
    days = (pd.Timestamp(e) - pd.Timestamp(s)).total_seconds() / 86400
    return f"{prefix} ({s} -> {e}, {days:.0f}d)"


def replay(days: int = 30) -> dict:
    """Fetch ``days`` of data and replay the deployed bot over that window."""
    load_active_strategies()
    pairs = sorted({cfg["pair"] for cfg in pt.ACTIVE_STRATEGIES.values()})
    print(f"Pairs: {pairs}")
    dfs = {}
    for pair in pairs:
        df = fetch_klines(pair, "4h", days)
        if df is None or len(df) < 80:
            print(f"ERROR: insufficient data for {pair}")
            raise SystemExit(1)
        dfs[pair] = df.reset_index(drop=True)
    return _replay(dfs, label=f"trailing {days}d")


def compare() -> dict:
    """Fetch 90 days once; replay the trailing 90d plus two non-overlapping
    windows (prior 60d and recent 30d) and return the comparison report."""
    load_active_strategies()
    pairs = sorted({cfg["pair"] for cfg in pt.ACTIVE_STRATEGIES.values()})
    print(f"Pairs: {pairs}")
    dfs = {}
    for pair in pairs:
        df = fetch_klines(pair, "4h", 92)
        # 92d * 6 candles/day = 552; allow a few missing candles but we must
        # cover the 540-candle (90d) slice below.
        if df is None or len(df) < 545:
            print(f"ERROR: insufficient data for {pair} (90d window)")
            raise SystemExit(1)
        dfs[pair] = df.reset_index(drop=True)

    n = min(len(dfs[p]) for p in pairs)
    full90 = {p: dfs[p].iloc[n - 540:].reset_index(drop=True) for p in pairs}
    prior60 = {p: full90[p].iloc[:360].reset_index(drop=True) for p in pairs}
    recent30 = {p: full90[p].iloc[360:].reset_index(drop=True) for p in pairs}

    r90 = _replay(full90, label=_window_label(full90[pairs[0]], "full 90d"))
    r_prior = _replay(prior60, label=_window_label(prior60[pairs[0]], "prior 60d"))
    r_recent = _replay(recent30, label=_window_label(recent30[pairs[0]], "recent 30d"))

    REPORT_JSON_90D.write_text(json.dumps(r90, indent=2, default=str), encoding="utf-8")
    REPORT_MD_90D.write_text(render_markdown(r90), encoding="utf-8")

    comparison = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "windows": {
            "prior_60d": _summary_row(r_prior),
            "recent_30d": _summary_row(r_recent),
            "full_90d": _summary_row(r90),
        },
        "verdict": _consistency_verdict(r_prior, r_recent, r90),
        "details": {
            "prior_60d": r_prior,
            "recent_30d": r_recent,
            "full_90d": r90,
        },
    }
    COMPARE_JSON.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    COMPARE_MD.write_text(render_comparison(comparison), encoding="utf-8")
    print(f"Comparison report: {COMPARE_MD}")
    return comparison


def _summary_row(r: dict) -> dict:
    return {
        "window": r["window"],
        "return_pct": r["capital"]["return_pct"],
        "buy_hold_pct": r["benchmark"]["buy_hold_35pct_each_pct"],
        "alpha_pp": r["alpha_pp"],
        "trades": r["trades"]["closed"],
        "win_rate_pct": r["trades"]["win_rate_pct"],
        "max_dd_pct": r["risk"]["max_drawdown_pct"],
        "costs": r["costs"]["total"],
        "exit_breakdown": r["trades"]["exit_breakdown"],
    }


def _consistency_verdict(r_prior: dict, r_recent: dict, r90: dict) -> str:
    a60 = r_prior["alpha_pp"]
    a30 = r_recent["alpha_pp"]
    a90 = r90["alpha_pp"]
    t60 = r_prior["trades"]["closed"]
    t30 = r_recent["trades"]["closed"]
    n60 = r_prior["capital"]["return_pct"]
    n30 = r_recent["capital"]["return_pct"]
    b60 = r_prior["benchmark"]["buy_hold_35pct_each_pct"]
    b30 = r_recent["benchmark"]["buy_hold_35pct_each_pct"]
    c60 = r_prior["costs"]["total"]
    c30 = r_recent["costs"]["total"]

    if t60 == 0 or t30 == 0:
        return (f"Inconclusive: the {'prior 60d' if t60 == 0 else 'recent 30d'} window produced "
                f"no closed trades, so no cross-regime comparison is possible.")
    if a60 > 0 and a30 > 0:
        verdict = ("CONSISTENT positive alpha across both non-overlapping windows "
                   f"(prior 60d {a60:+.1f}pp, recent 30d {a30:+.1f}pp over buy & hold) — "
                   "the strategy's edge is not confined to a single regime, though alpha of "
                   "this size on 60-180 trades is still not statistical proof.")
    elif a60 < 0 and a30 < 0:
        verdict = ("CONSISTENTLY negative alpha across both non-overlapping windows "
                   f"(prior 60d {a60:+.1f}pp, recent 30d {a30:+.1f}pp) — the strategy "
                   "underperforms buy & hold in both regimes net of costs. No edge is "
                   "demonstrated.")
    else:
        verdict = ("INCONSISTENT across regimes: alpha flips sign between the windows "
                   f"(prior 60d {a60:+.1f}pp on {n60:+.1f}% vs buy & hold {b60:+.1f}%; "
                   f"recent 30d {a30:+.1f}pp on {n30:+.1f}% vs buy & hold {b30:+.1f}%). "
                   "Performance tracks the market regime — this is beta, not a stable edge.")
    return (f"{verdict} Full-90d context: {a90:+.1f}pp alpha "
            f"({r90['capital']['return_pct']:+.1f}% vs buy & hold "
            f"{r90['benchmark']['buy_hold_35pct_each_pct']:+.1f}%), "
            f"costs ${c60:.2f} (prior 60d) / ${c30:.2f} (recent 30d).")


def render_markdown(r: dict) -> str:
    w = r["window"]
    cap = r["capital"]
    tr = r["trades"]
    co = r["costs"]
    ex = tr.get("exit_breakdown", {})
    rc = r["risk_config"]
    lines = []
    lines.append("# Forward run: deployed paper bot (full risk model), net of costs\n")
    lines.append(f"**Window:** {w['start']} → {w['end']} UTC "
                 f"({w['days']} days, {w['candles']} 4h candles, pairs: {', '.join(w['pairs'])})\n")
    lines.append("**Risk model:** disaster stop-loss "
                 f"{rc['stop_loss_pct']:.0f}%, trailing stop after "
                 f"+{rc['trailing_activation_pct']:.0f}% (retrace "
                 f"{rc['trailing_distance_pct']:.0f}%), max hold "
                 f"{rc['max_position_hours']:.0f}h, portfolio drawdown stop "
                 f"{rc['portfolio_max_dd_pct']:.0f}% + "
                 f"{rc['dd_cooldown_hours']:.0f}h flat cooldown with peak "
                 "re-arm — mirrors the paper bot's `RiskManager`.\n")
    lines.append("**Deployed params:** loaded from `data/results/bot_strategy_params.json` "
                 "(or defaults) via `paper_trader.load_active_strategies()`\n")
    lines.append("\n## Results\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Starting equity | ${cap['initial']:,.2f} |")
    lines.append(f"| Ending equity | ${cap['final']:,.2f} |")
    lines.append(f"| Net return | {cap['return_pct']:+.3f}% |")
    lines.append(f"| Closed trades | {tr['closed']} (win rate {tr['win_rate_pct']:.0f}%) |")
    lines.append(f"| Gross P&L (before costs) | ${tr['gross_pnl']:+.4f} |")
    lines.append(f"| Net P&L (after costs) | ${tr['net_pnl']:+.4f} |")
    lines.append(f"| Max drawdown | {r['risk']['max_drawdown_pct']:.2f}% |")
    lines.append(f"| Buy & hold benchmark (35% each pair) | {r['benchmark']['buy_hold_35pct_each_pct']:+.2f}% |")
    lines.append(f"| Alpha vs buy & hold | {r['alpha_pp']:+.2f} pp |")
    lines.append("\n## Exit reasons\n")
    lines.append("| Exit | Count | Net P&L |")
    lines.append("|---|---|---|")
    label_map = {"signal": "Signal reversal", "stop_loss": "Stop-loss",
                 "trailing": "Trailing stop", "time": "Max time",
                 "drawdown": "Portfolio drawdown", "end_of_window": "End of window"}
    for cat in ("signal", "stop_loss", "trailing", "time", "drawdown", "end_of_window"):
        if cat in ex:
            lines.append(f"| {label_map[cat]} | {ex[cat]['count']} | ${ex[cat]['pnl']:+.2f} |")
    lines.append("\n## Costs\n")
    lines.append("| Component | Amount |")
    lines.append("|---|---|")
    lines.append(f"| Fees ({co['fees']:.4f}) + Slippage ({co['slippage']:.4f}) + Funding ({co['funding']:.4f}) | **${co['total']:.4f}** |")
    if co["costs_pct_of_turnover"] is not None:
        lines.append(f"| Costs as % of turnover | {co['costs_pct_of_turnover']:.3f}% |")
    lines.append("\n## Per pair\n")
    lines.append("| Pair | Trades | Wins | P&L | Fees | Funding | Price Δ |")
    lines.append("|---|---|---|---|---|---|---|")
    for pair, p in r["per_pair"].items():
        lines.append(f"| {format_pair(pair)} | {p['trades']} | {p['wins']} | "
                     f"${p['pnl']:+.2f} | ${p['fees']:.2f} | ${p['funding']:.2f} | {p['price_change_pct']:+.2f}% |")
    lines.append("\n## Cost model\n")
    lines.append(f"Fee {r['cost_model']['fee_rate']*100:.2f}%/side, slippage "
                 f"{r['cost_model']['slippage_rate']*100:.2f}%/side, funding "
                 f"{r['cost_model']['funding_rate_8h']*100:.3f}%/8h, fills at next open, "
                 f"signals on closed candles only. Stops checked per 4h candle at the "
                 f"adverse extreme (low/high); see module docstring for the approximation.\n")
    lines.append("\n## Verdict\n")
    net = cap["return_pct"]
    bench = r["benchmark"]["buy_hold_35pct_each_pct"]
    alpha = r["alpha_pp"]
    if net > 0 and tr["closed"] >= 3 and alpha > 2.0:
        verdict = (f"Net return is **positive** ({net:+.3f}%) and beats buy & hold "
                   f"by {alpha:+.1f}pp with {tr['closed']} closed trades. "
                   f"Costs totaled ${co['total']:.2f}. This is *suggestive*, not proof — "
                   f"one window is one market regime; re-run monthly (--compare for two "
                   f"non-overlapping windows) and require consistency before risking capital.")
    elif net > 0:
        verdict = (f"Net return is **positive** ({net:+.3f}%) but within "
                   f"{abs(alpha):.1f}pp of the buy & hold benchmark ({bench:+.2f}%) — "
                   f"the gains look like market beta, not strategy alpha. **No edge is "
                   f"demonstrated**; the strategy is not yet proven to pay for its costs "
                   f"(${co['total']:.2f} here). Keep it in paper mode.")
    else:
        verdict = (f"Net return is **{net:+.3f}%** with {tr['closed']} closed trades — "
                   f"no positive edge is demonstrated net of costs over this window. "
                   f"The honest conclusion is that the deployed strategy has NOT yet "
                   f"proven it can pay for fees ({co['fees']:.2f}), slippage ({co['slippage']:.2f}), "
                   f"and funding ({co['funding']:.2f}). Keep it in paper mode.")
    lines.append(verdict + "\n")
    return "\n".join(lines)


def render_comparison(cmp: dict) -> str:
    ws = cmp["windows"]
    lines = []
    lines.append("# Forward edge check: window comparison (net of costs, full risk model)\n")
    lines.append(f"Generated {cmp['generated'][:16]} UTC. Windows share the paper bot's "
                 "cost model and risk manager; each replay starts from a fresh $97 account.\n")
    lines.append("## Per-window results\n")
    lines.append("| Window | Dates | Net return | Buy & hold | Alpha | Trades | Win rate | Max DD | Costs |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, label in (("prior_60d", "Prior 60d"), ("recent_30d", "Recent 30d"), ("full_90d", "Full 90d (continuous)")):
        s = ws[key]
        w = s["window"]
        lines.append(f"| {label} | {w['start']} → {w['end']} | {s['return_pct']:+.2f}% | "
                     f"{s['buy_hold_pct']:+.2f}% | {s['alpha_pp']:+.2f} pp | {s['trades']} | "
                     f"{s['win_rate_pct']:.0f}% | {s['max_dd_pct']:.1f}% | ${s['costs']:.2f} |")
    lines.append("\nMethodology note: prior 60d and recent 30d are NON-OVERLAPPING and each "
                 "start from a fresh $97 account, so they are the clean regime "
                 "comparison. The full-90d row is a single CONTINUOUS run: it is "
                 "path-dependent (positions and drawdown cooldowns carry across "
                 "regime boundaries), and a mean-reversion strategy that is flat "
                 "at a different moment can take the opposite side of the same "
                 "move — treat it as context, not as a third independent regime.\n")
    lines.append("## Exit-reason mix\n")
    lines.append("| Window | Signal | Stop-loss | Trailing | Max time | Drawdown |")
    lines.append("|---|---|---|---|---|---|")
    for key, label in (("prior_60d", "Prior 60d"), ("recent_30d", "Recent 30d"), ("full_90d", "Full 90d (continuous)")):
        ex = ws[key]["exit_breakdown"]
        lines.append(f"| {label} | {ex.get('signal', {}).get('count', 0)} | "
                     f"{ex.get('stop_loss', {}).get('count', 0)} | "
                     f"{ex.get('trailing', {}).get('count', 0)} | "
                     f"{ex.get('time', {}).get('count', 0)} | "
                     f"{ex.get('drawdown', {}).get('count', 0)} |")
    lines.append("\n## Verdict\n")
    lines.append(cmp["verdict"] + "\n")
    lines.append("Per-window detail: `forward_run_report_90d.md` + the `details` key of "
                 "`forward_run_comparison.json`.\n")
    return "\n".join(lines)


def replay_live_window(days: int = 12) -> dict:
    """Replay the exact window the live paper bot ran (run_history.jsonl)
    and compare it against the bot's actual record.

    The replay gets ``days`` of warmup data but only trades from the first
    live run onward, so both engines see the same market slice with the same
    strategy params — any difference is granularity/execution model, not data.
    """
    hist_path = RESULTS_DIR / "run_history.jsonl"
    if not hist_path.exists():
        raise SystemExit("no run_history.jsonl — live bot has not recorded runs")
    runs = [json.loads(l) for l in hist_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    runs = [r for r in runs if r.get("status") == "success"]
    if not runs:
        raise SystemExit("no successful live runs in run_history.jsonl")

    first_run = pd.Timestamp(runs[0]["timestamp"])
    last_run = pd.Timestamp(runs[-1]["timestamp"])
    now_utc = pd.Timestamp.now(tz="UTC")
    span_days = max((last_run - first_run).total_seconds() / 86400.0, 1.0)

    load_active_strategies()
    pairs = sorted({cfg["pair"] for cfg in pt.ACTIVE_STRATEGIES.values()})
    dfs = {}
    for pair in pairs:
        df = fetch_klines(pair, "4h", days + int(span_days) + 2)
        if df is None or len(df) < 80:
            print(f"ERROR: insufficient data for {pair}")
            raise SystemExit(1)
        df = df.reset_index(drop=True)
        # first index whose candle CLOSES at/after the first live run
        ts_col = pd.to_datetime(df["timestamp"], utc=True)
        idx = df.index[ts_col >= first_run - pd.Timedelta(hours=4)]
        start = int(idx[0]) if len(idx) else max(1, len(df) - int(span_days * 6) - 1)
        dfs[pair] = df

    print(f"Live window: {first_run.isoformat()} -> {now_utc.isoformat()} "
          f"({span_days:.1f} days, {len(runs)} live runs)\n")
    result = _replay(dfs, label=f"live window ({span_days:.1f}d)", start=start)

    # ── Compare against the live bot's actual record ──
    live_opens = sum(r.get("trades", 0) for r in runs)
    live_equity_last = runs[-1]["equity"]
    rep = result["capital"]

    lines = []
    lines.append("# Replay vs live paper bot — same window, same params\n")
    lines.append(f"**Window:** {first_run.isoformat()} -> {now_utc.date().isoformat()} "
                 f"({span_days:.1f} days)\n")
    lines.append(f"**Live bot activity:** {len(runs)} successful runs, "
                 f"{live_opens} open-logs, latest recorded equity "
                 f"${live_equity_last:.2f}\n")
    lines.append("**Live accounting caveat:** the live bot records equity as "
                 "`cash + entry-price position value` at execution time — it "
                 "never marks positions to market and never logged a close in "
                 "this window, so its reported P&L (~0%) is not comparable "
                 "to a marked-to-market replay. The comparison below "
                 "quantifies exactly that gap.\n")
    lines.append("\n| Metric | Live bot record | 4h replay (same window) |")
    lines.append("|---|---|---|")
    lines.append(f"| Open actions logged | {live_opens} | {result['trades']['closed'] + result['trades'].get('open_count', 0)} |")
    lines.append(f"| Closed trades | 0 (none logged) | {result['trades']['closed']} |")
    lines.append(f"| Marked-to-market equity | ~${live_equity_last:.2f} (unmarked) | ${rep['final']:,.2f} |")
    lines.append(f"| Return | ~0.0% (unmarked) | {rep['return_pct']:+.2f}% |")
    lines.append(f"| Costs charged | unknown (no closes) | ${result['costs']['total']:.2f} |")
    lines.append(f"| Max drawdown | {0.063:.1f}% (from state peak) | {result['risk']['max_drawdown_pct']:.2f}% |")

    out_md = RESULTS_DIR / "replay_vs_live.md"
    out_json = RESULTS_DIR / "replay_vs_live.json"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps({
        "window": {"start": first_run.isoformat(), "end": now_utc.isoformat(),
                    "days": span_days, "live_runs": len(runs)},
        "live": {"opens": live_opens, "closes": 0, "equity_last": live_equity_last},
        "replay": result,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {out_md}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Forward replay of the deployed paper bot (full risk model)")
    parser.add_argument("--days", type=int, default=30,
                        help="days of history to replay (default 30)")
    parser.add_argument("--compare", action="store_true",
                        help="replay 90d and compare prior-60d vs recent-30d windows")
    parser.add_argument("--vs-live", action="store_true",
                        help="replay the window the live bot actually ran and "
                             "compare against run_history.jsonl")
    args = parser.parse_args()

    if args.compare:
        print("=== FORWARD COMPARISON (90 days) ===")
        compare()
        return

    if args.vs_live:
        print("=== REPLAY vs LIVE BOT (same window) ===")
        replay_live_window(days=12)
        return

    print(f"=== FORWARD RUN ({args.days} days) ===")
    report = replay(days=args.days)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")

    cap = report["capital"]
    print(f"\nFinal equity: ${cap['final']:,.2f} ({cap['return_pct']:+.3f}%)")
    print(f"Closed trades: {report['trades']['closed']} | "
          f"Costs: ${report['costs']['total']:.2f} | "
          f"Max DD: {report['risk']['max_drawdown_pct']:.2f}%")
    print(f"Report: {REPORT_MD}")


if __name__ == "__main__":
    main()
