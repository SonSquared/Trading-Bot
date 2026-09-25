#!/usr/bin/env python3
"""AI Trading Bot — venue feasibility check.

Answers the question the paper ledger cannot: what can THIS account size
legally trade on the configured venue?

    python scripts/ai_venue_check.py            # human-readable
    python scripts/ai_venue_check.py --json     # machine-readable

The account size and the risk rules come from configs/ai_bot.yaml; the order
rules (MIN_NOTIONAL, LOT_SIZE, taker fee) come from the exchange itself via
ccxt, falling back to the dated table in trading_system/bot/venue_limits.py
when the venue is unreachable (geo-blocked runners). Which source was used is
always printed, so the answer is never silently stale.

Zero LLM calls, no orders — read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from trading_system.bot.account import (
    CONFIG_KEY,
    legacy_scale_notice,
    resolve_starting_equity,
)
from trading_system.bot.venue_limits import (
    BUILTIN_CAPTURED_AT,
    DEFAULT_TAKER_FEE_PCT,
    MarketLimits,
    leverage_for_margin,
    pair_feasibility,
    project_scale,
    resolve_limits,
    round_trip_fee,
)

# Margin must stay inside this share of equity for the position to be
# holdable; used only to report the leverage needed.
MARGIN_BUDGET_PCT = 50.0

# Fallback trade frequency when no history exists to measure it.
ASSUMED_TRADES_PER_MONTH = 20.0

# A paper order below this is refused regardless of venue (see ai_agent.py).
MIN_TRADE_USD = 5.0


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise click.ClickException(f"No config at {path}")
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _fetch_limits(pairs: list[str]) -> tuple[dict[str, MarketLimits], str]:
    """Venue limits per pair plus a note saying where they came from."""
    from trading_system.bot.exchange import ExchangeInterface
    from trading_system.config import ExchangeConfig

    exchange = ExchangeInterface(ExchangeConfig())
    reachable = exchange.connect()
    limits: dict[str, MarketLimits] = {}
    for pair in pairs:
        fetched = exchange.get_market_limits(pair) if reachable else None
        resolved = resolve_limits(pair, fetched)
        if resolved is not None:
            limits[pair] = resolved
    live = sorted(p for p, m in limits.items() if m.source == "exchange")
    if live:
        note = f"live from the exchange for {', '.join(live)}"
    else:
        note = (
            f"builtin table captured {BUILTIN_CAPTURED_AT} "
            f"(the exchange was unreachable from this machine)"
        )
    return limits, note


def observed_trades_per_month(data_dir: Path, days: int = 30) -> float | None:
    """Trades per month measured from trades.jsonl, or None if unmeasurable."""
    path = data_dir / "trades.jsonl"
    if not path.exists():
        return None
    from datetime import datetime

    closes: list[datetime] = []
    for line in path.read_text().strip().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("side") != "close":
            continue
        ts = row.get("timestamp") or row.get("close_time")
        if not ts:
            continue
        try:
            closes.append(datetime.fromisoformat(str(ts).replace("Z", "+00:00")))
        except (ValueError, TypeError):
            continue
    if len(closes) < 2:
        return None
    span_days = (max(closes) - min(closes)).total_seconds() / 86400.0
    if span_days <= 0:
        return None
    return len(closes) / span_days * 30.0


def build_feasibility(
    cfg: dict, limits: dict[str, MarketLimits], limits_note: str,
    data_dir: Path | None = None, trades_per_month: float | None = None,
) -> dict:
    """Everything the check reports, as a plain dict."""
    equity = resolve_starting_equity(cfg)[0]
    bot_cfg = cfg.get("bot", {}) or {}
    risk = cfg.get("risk", {}) or {}
    risk_pct = float(risk.get("max_risk_per_trade_pct", 2.0))
    max_stop = float(risk.get("stop_loss_max_pct", 5.0))
    max_size_pct = float(risk.get("max_position_size_pct", 10.0))
    max_heat_pct = float(risk.get("max_portfolio_heat_pct", 30.0))
    # A single position cannot be larger than everything allowed, so the
    # per-trade ceiling is the SMALLER of the size cap and the total-exposure
    # cap. This is the same rule the agent enforces (AIAgent._venue_rejection),
    # and quoting only the risk-derived figure would overstate what is placeable.
    cap_pct = min(max_size_pct, max_heat_pct)

    observed = trades_per_month
    if observed is None and data_dir is not None:
        observed = observed_trades_per_month(data_dir)
    rate = observed if observed is not None else ASSUMED_TRADES_PER_MONTH
    rate_basis = (
        "observed from trades.jsonl"
        if observed is not None
        else f"assumed {ASSUMED_TRADES_PER_MONTH:g}/month, no history to measure"
    )

    pairs = list(bot_cfg.get("pairs") or [])
    per_pair = []
    tradable = []
    for pair in pairs:
        lim = limits.get(pair)
        if lim is None:
            per_pair.append({"pair": pair, "verdict": "unknown order rules"})
            continue
        row = pair_feasibility(
            pair, lim, equity=equity, risk_pct=risk_pct,
            max_stop_pct=max_stop, max_position_size_pct=cap_pct,
            floor_usd=MIN_TRADE_USD,
        )
        row["taker_fee_pct"] = lim.taker_fee_pct
        per_pair.append(row)
        if row["feasible"]:
            tradable.append(row)

    best = tradable[0] if tradable else None
    projection = None
    fees = None
    leverage = None
    if best is not None:
        notional = best["ceiling_usd"]
        # The risk a trade of THIS size actually takes at the widest legal stop —
        # which is less than the 2% cap whenever another cap bounds the size
        # first. Using the cap here would overstate both the loss and the
        # expectancy, i.e. exactly the optimism this check exists to remove.
        implied_risk_pct = notional / equity * 100.0 * (max_stop / 100.0) if equity else 0.0
        risk_usd = equity * implied_risk_pct / 100.0
        fees = {
            "notional": round(notional, 2),
            "round_trip_fee": round(
                round_trip_fee(notional, limits[best["pair"]].taker_fee_pct), 4
            ),
            "taker_fee_pct": limits[best["pair"]].taker_fee_pct,
            "implied_risk_pct_of_equity": round(implied_risk_pct, 3),
            "implied_risk_usd": round(risk_usd, 4),
        }
        fees["pct_of_risked_amount"] = (
            round(fees["round_trip_fee"] / risk_usd * 100, 2) if risk_usd else None
        )
        leverage = {
            "notional": round(notional, 2),
            "margin_budget_pct": MARGIN_BUDGET_PCT,
            "leverage_needed": round(
                leverage_for_margin(notional, equity, MARGIN_BUDGET_PCT), 3
            ),
            "note": (
                "Leverage does not change the risk cap — risk is "
                "notional x stop distance. It only decides how much margin is "
                "tied up, so 1x is enough whenever the figure above is <= 1."
            ),
        }
        projection = project_scale(
            equity=equity,
            notional=notional,
            risk_pct=implied_risk_pct,
            rr=float(risk.get("min_risk_reward_ratio", 1.5)),
            taker_fee_pct=limits[best["pair"]].taker_fee_pct,
            trades_per_month=rate,
            win_rate=0.5,
        )

    return {
        "equity": equity,
        "config_key": CONFIG_KEY,
        "pairs": pairs,
        "limits_source": limits_note,
        "risk": {
            "risk_pct": risk_pct,
            "max_stop_pct": max_stop,
            "max_position_size_pct": max_size_pct,
            "max_portfolio_heat_pct": max_heat_pct,
            "effective_position_cap_pct": cap_pct,
            "cap_that_binds": (
                "total exposure" if max_heat_pct < max_size_pct
                else "per-trade size" if max_size_pct < max_heat_pct
                else "both (they coincide)"
            ),
            "max_notional_at_max_stop": round(
                equity * risk_pct / 100.0 / (max_stop / 100.0), 2
            ),
            "max_notional_effective": round(
                min(
                    equity * cap_pct / 100.0,
                    equity * risk_pct / 100.0 / (max_stop / 100.0),
                ),
                2,
            ),
        },
        "per_pair": per_pair,
        "tradable_pairs": [r["pair"] for r in tradable],
        "fee_drag": fees,
        "leverage": leverage,
        "typical_trade_and_month": projection,
        "trades_per_month": rate,
        "trades_per_month_basis": rate_basis,
        "legacy_scale_notice": (
            legacy_scale_notice(
                _ledger_origin(data_dir), equity
            ) if data_dir is not None else None
        ),
    }


def _ledger_origin(data_dir: Path | None) -> float | None:
    if data_dir is None:
        return None
    path = data_dir / "paper_ledger.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("start_equity")
    except (json.JSONDecodeError, OSError):
        return None


def render(r: dict) -> str:
    """The human-readable answer, including what it does NOT promise."""
    lines = [
        "AI BOT VENUE FEASIBILITY",
        "=" * 40,
        f"Account (from {r['config_key']}): ${r['equity']:,.2f}",
        f"Order rules: {r['limits_source']}",
        "",
        "Risk rules in force:",
        f"  Risk per trade:      {r['risk']['risk_pct']:g}% of equity "
        f"= ${r['equity'] * r['risk']['risk_pct'] / 100:,.2f}",
        f"  Max stop distance:   {r['risk']['max_stop_pct']:g}%",
        f"  Max position size:   {r['risk']['max_position_size_pct']:g}% "
        f"of equity",
        f"  Max total exposure:  {r['risk']['max_portfolio_heat_pct']:g}% "
        f"of equity",
        f"  => risk cap alone allows: "
        f"${r['risk']['max_notional_at_max_stop']:,.2f}",
        f"  => a position can reach:  "
        f"${r['risk']['max_notional_effective']:,.2f} "
        f"(cap that binds: {r['risk']['cap_that_binds']}, "
        f"{r['risk']['effective_position_cap_pct']:g}%)",
        "",
        "Per pair:",
    ]
    for row in r["per_pair"]:
        if "floor_usd" not in row:
            lines.append(f"  {row['pair']}: {row['verdict']}")
            continue
        lines.append(
            f"  {row['pair']} (rules: {row.get('limits_source', '?')})"
        )
        lines.append(
            f"    venue minimum order: ${row['venue_min_notional']:,.2f} "
            f"| quantity step {row['amount_step']:g} "
            f"| taker {row.get('taker_fee_pct', DEFAULT_TAKER_FEE_PCT):g}%"
        )
        if row["feasible"]:
            lines.append(
                f"    legal order range:   ${row['floor_usd']:,.2f} - "
                f"${row['ceiling_usd']:,.2f}"
            )
        else:
            lines.append(
                f"    venue minimum:       ${row['venue_min_notional']:,.2f} "
                f"(largest a position may be: ${row['ceiling_usd']:,.2f})"
            )
        lines.append(f"    -> {row['verdict']}")

    if r["fee_drag"]:
        f = r["fee_drag"]
        lines += [
            "",
            "Fee drag (market orders in and out):",
            f"  on a ${f['notional']:,.2f} order: ${f['round_trip_fee']:,.4f} "
            f"round trip at {f['taker_fee_pct']:g}% per side",
            f"  that order risks ${f['implied_risk_usd']:,.4f} at a "
            f"{r['risk']['max_stop_pct']:g}% stop "
            f"({f['implied_risk_pct_of_equity']:g}% of equity, under the "
            f"{r['risk']['risk_pct']:g}% cap), so fees are "
            f"{f['pct_of_risked_amount']:g}% of the amount risked",
        ]
        lev = r["leverage"]
        lines += [
            "",
            "Leverage:",
            f"  to hold ${lev['notional']:,.2f} with margin under "
            f"{lev['margin_budget_pct']:g}% of equity: "
            f"{lev['leverage_needed']:g}x needed",
            f"  {lev['note']}",
        ]

    p = r["typical_trade_and_month"]
    lines += [
        "",
        "Typical trade and month (projection, not a promise). Both assume every",
        "trade is sized to the cap above — a smaller position scales it down:",
    ]
    if p is None:
        lines.append(
            "  No configured pair can be traded at this size, so there is no "
            "typical trade to project. See the per-pair verdicts above."
        )
    else:
        lines += [
            f"  Assumes: {p['assumptions']['win_rate']:.0%} win rate, "
            f"R:R {p['assumptions']['rr']:g}, "
            f"{r['trades_per_month']:.1f} trades/month "
            f"({r['trades_per_month_basis']})",
            "  NOT a promise: the realized sample is far too small to estimate "
            "a win rate, so 50% is a stated assumption.",
            f"  A typical trade risks ${p['risk_usd']:,.2f}; a win nets "
            f"${p['win_net']:,.2f}, a loss costs ${p['loss_net']:,.2f} "
            f"(fees included)",
            f"  Expectancy: ${p['expectancy_per_trade']:+,.4f} per trade",
            f"  A typical month: {p['month_usd']:+,.2f} USD "
            f"({p['month_pct']:+.2f}% of equity)",
        ]
    if r["legacy_scale_notice"]:
        lines += ["", f"WARNING: {r['legacy_scale_notice']}"]
    lines += ["", "This is arithmetic on the rules, not a forecast of returns."]
    return "\n".join(lines)


@click.command()
@click.option("--config", default="configs/ai_bot.yaml", help="Config file path")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of text")
@click.option(
    "--trades-per-month", default=None, type=float,
    help="Override the observed trade frequency used in the projection",
)
def main(config: str, as_json: bool, trades_per_month: float | None) -> None:
    """Report what this account size can legally trade on the venue."""
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

    cfg = _load_config(Path(config))
    bot_cfg = cfg.get("bot", {}) or {}
    data_dir = Path(bot_cfg.get("data_dir", "data/ai_bot"))

    pairs = list(bot_cfg.get("pairs") or [])
    if not pairs:
        raise click.ClickException(
            f"No bot.pairs configured in {config} — nothing to check."
        )

    limits, note = _fetch_limits(pairs)
    r = build_feasibility(
        cfg, limits, note, data_dir=data_dir, trades_per_month=trades_per_month
    )
    click.echo(json.dumps(r, indent=2, default=str) if as_json else render(r))


if __name__ == "__main__":
    main()
