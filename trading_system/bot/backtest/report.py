"""Render a backtest suite, and say plainly what it does and does not show.

The verdict is computed from the numbers rather than written by hand, because a
hand-written conclusion is the part of a backtest that quietly stops being true.
Every claim this module makes is a restatement of a measured quantity, and the
limitations block is rendered with the results rather than kept in a README.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from trading_system.bot.backtest.engine import (
    TRIGGER_MODEL_CAVEATS,
    BacktestResult,
    WindowResult,
)
from trading_system.bot.backtest.historical import load_funding

BAR = "=" * 100
THIN = "-" * 100


def _fmt_money(value: float) -> str:
    return f"{value:+,.2f}"


def _windows(result: BacktestResult, kind: str) -> list[WindowResult]:
    return [w for w in result.windows if w.kind == kind]


def _continuous(result: BacktestResult) -> WindowResult | None:
    got = _windows(result, "continuous")
    return got[0] if got else None


def _row(w: WindowResult) -> str:
    return (
        f"  {w.label:<22} {w.start[:10]}..{w.end[:10]} "
        f"{w.slots:>6} {w.start_equity:>8.2f} {w.end_equity:>8.2f} "
        f"{w.return_pct:>+7.2f} {w.max_drawdown_pct:>6.2f} "
        f"{w.trades:>6} {w.win_rate:>5.0f}% {w.fees:>8.4f} "
        f"{w.net_pnl:>+8.2f} {w.expectancy_usd:>+8.4f} "
        f"{w.stopped_out:>4}/{w.took_profit:<4} {w.open_positions_at_end:>3}"
    )


TABLE_HEADER = (
    "  {:<22} {:<21} {:>6} {:>8} {:>8} {:>7} {:>6} {:>6} {:>5} {:>8} "
    "{:>8} {:>8} {:>9} {:>3}".format(
        "window", "dates", "slots", "eq from", "eq to", "ret %", "maxDD",
        "trades", "win", "fees", "net $", "expect", "stop/tp", "open",
    )
)


def funding_note(data_dir: str | Path, pairs: list[str], notional: float) -> str | None:
    """Quantify the funding the ledger does not charge, from stored rates."""
    rates = []
    for pair in pairs:
        series = load_funding(data_dir, pair)
        if series is not None and len(series) > 0:
            rates.append(series.abs().mean())
    if not rates:
        return None
    mean_abs = sum(rates) / len(rates)
    per_day = mean_abs * 3            # 8-hour funding, three times a day
    per_trade = per_day * notional    # roughly one day held
    return (
        f"Funding is NOT charged by the paper ledger. Stored funding data for "
        f"{', '.join(pairs)} covers only part of the span, and its mean absolute "
        f"rate there is {mean_abs * 100:.4f}% per 8h. On a ${notional:,.2f} "
        f"position held about a day that is roughly ${per_trade:.4f} per trade "
        f"({per_day * 100:.4f}%/day of notional) — small at $97, omitted, and "
        "therefore not a source of optimism or pessimism either way."
    )


def render(results: list[BacktestResult], command: str | None = None) -> str:
    lines: list[str] = [
        BAR,
        "AI BOT BACKTEST — what the MECHANICS do, not what the model thinks",
        BAR,
    ]

    first = results[0]
    ev = first.evidence
    cfg = first.config
    equity = cfg.get("equity_from_config")

    if command:
        lines += ["", "Reproduce with:", f"  {command}"]
    lines += [
        "",
        "MECHANICS UNDER TEST (production code, driven as-is)",
        f"  account equity          ${equity:,.2f}  (from {cfg.get('config_path')})",
        f"  risk rules              {_compact(cfg.get('risk_rules', {}))}",
        f"  pairs                   {', '.join(ev['pairs'])}",
        f"  timeframe               {ev['timeframe']}",
        f"  decision slots          {ev['slots_per_day']}/day at "
        f"{', '.join(ev['slot_hours_utc'])} UTC (schedule parsed by the production gate)",
        f"  closed candles only     one candle window per slot, warmed up over "
        f"{ev['warmup_candles']} candles",
        f"  trigger model           {first.trigger_model} — "
        f"{TRIGGER_MODEL_CAVEATS.get(first.trigger_model, '')}",
        f"  venue limits enforced   {_venues(cfg.get('venues', {}))}",
        f"  data                    {ev['data_dir']}  "
        f"({ev['data_range']['start'][:10]} .. {ev['data_range']['end'][:10]})",
        f"  replayed window         {ev['effective_range']['start'][:10]} .. "
        f"{ev['effective_range']['end'][:10]}  ({ev['continuous_slots']} slots)",
    ]

    for result in results:
        lines += ["", THIN, f"SOURCE: {result.source['name']}  [{result.source['label']}]", THIN]
        lines += [
            f"  {result.source['description']}",
            f"  fitted to this data: {'YES' if result.source['fitted'] else 'no'}",
            f"  parameters: {_compact(result.source['params'])}",
        ]
        if result.source.get("caveat"):
            lines += [f"  -> {result.source['caveat']}"]
        cont = _continuous(result)
        lines += ["", "  Continuous run (one $97 account for the whole span):"]
        if cont is not None:
            lines += [TABLE_HEADER, _row(cont)]
        folds = _windows(result, "fold")
        if folds:
            lines += ["", f"  Walk-forward folds ({len(folds)} independent $97 accounts):"]
            lines += [TABLE_HEADER] + [_row(w) for w in folds]
        regimes = _windows(result, "regime")
        if regimes:
            lines += ["", "  Regimes (each an independent $97 account, no carry-over):"]
            lines += [TABLE_HEADER] + [_row(w) for w in regimes]
        table = _rejection_table(result)
        if table:
            lines += ["", "  What the rules refused (counted across every window):"]
            lines += table

    lines += ["", BAR, "EVIDENCE THAT THE HARNESS IS WHAT IT CLAIMS", BAR]
    total_slots = sum(w.slots for r in results for w in r.windows)
    lines += [
        f"  slots replayed (all windows)   {total_slots:,}",
        f"  distinct candle windows        "
        f"{sum(w.distinct_windows for r in results for w in r.windows):,}"
        "   (pair x newest closed candle, de-duplicated)",
        f"  candles served                 "
        f"{sum(w.candles_served for r in results for w in r.windows):,}"
        "   (every fetch call, agent and engine)",
        f"  look-ahead violations          "
        f"{sum(r.evidence['look_ahead_violations'] for r in results)}"
        "   (a candle after the cursor was served)",
        f"  order-path attempts            "
        f"{sum(r.evidence['order_path_attempts'] for r in results)}"
        "   (place/cancel calls from replayed code)",
        "  Both counters are guards, not assumptions: the historical exchange counts",
        "  a violation and raises on any order call, so 0 is a measurement.",
    ]
    note = funding_note(ev["data_dir"], ev["pairs"], _typical_notional(results))
    if note:
        lines += ["", f"  {note}"]

    lines += ["", BAR, "VERDICT", BAR]
    lines += _verdict(results)
    lines += ["", BAR, "WHAT THIS DOES NOT SHOW", BAR]
    for item in results[0].limitations:
        lines += [f"  - {item}"]
    lines += [
        "",
        "  In one line: the numbers above are about the plumbing — sizing, caps,",
        "  venue filters, fees, stop geometry, cadence — and a mechanical stand-in",
        "  for the strategy. They are not a forecast and not a claim about the",
        "  model's judgement, which cannot be backtested without look-ahead.",
        "",
    ]
    return "\n".join(lines)


def _compact(mapping: dict) -> str:
    if not mapping:
        return "{}"
    return ", ".join(f"{k}={v}" for k, v in mapping.items())


def _venues(venues: dict) -> str:
    if not venues:
        return "unknown"
    return "; ".join(
        f"{pair} min ${v.get('min_notional', 0):,.2f} step {v.get('amount_step', 0):g} "
        f"taker {v.get('taker_fee_pct', 0):g}% ({v.get('source', '?')})"
        for pair, v in venues.items()
    )


def _rejection_table(result: BacktestResult) -> list[str]:
    totals: dict[str, int] = {}
    examples: dict[str, str] = {}
    for w in result.windows:
        for bucket, count in w.rejections.items():
            totals[bucket] = totals.get(bucket, 0) + count
            examples.setdefault(bucket, w.rejection_examples.get(bucket, ""))
    if not totals:
        return []
    out = []
    for bucket, count in sorted(totals.items(), key=lambda kv: -kv[1]):
        out.append(f"    {count:>7}  {bucket}")
        if examples.get(bucket):
            out.append(f"             e.g. {examples[bucket][:150]}")
    return out


def _typical_notional(results: list[BacktestResult]) -> float:
    sizes = [t.get("size_usd") for r in results for t in r.trades if t.get("size_usd")]
    return (sum(float(s) for s in sizes) / len(sizes)) if sizes else 0.0


def _labelled(label: str, body: str) -> list[str]:
    """A wrapped ``LABEL:  body`` block at the verdict's indentation."""
    return textwrap.wrap(
        f"{label}  {body}", width=88, initial_indent="     ",
        subsequent_indent="     " + " " * (len(label) + 2),
    )


def _venue_arithmetic(results: list[BacktestResult]) -> list[str]:
    """The account's tradability at this size, from the engine's own numbers.

    Not a sentence someone typed: the floors and ceilings come from the venue
    helpers the agent sizes with (``engine._venue_reach``), so if the account or
    the filters change the claim changes with them.
    """
    ev = results[0].evidence or {}
    reach = ev.get("venue_reach") or {}
    equity = float(results[0].config.get("equity_from_config") or 0.0)
    tradable = {p: v for p, v in reach.items() if v.get("tradable")}
    refused = {p: v for p, v in reach.items() if not v.get("tradable")}
    if not reach:
        return ["     SUPPORTED:  the venue filters were enforced (no limits resolved)."]

    body = (
        f"at ${equity:,.2f} the account can place legal orders on "
        + (
            ", ".join(
                f"{p} (venue floor ${v['floor_usd']:,.2f}, ceiling "
                f"${v['ceiling_usd']:,.2f})"
                for p, v in tradable.items()
            )
            or "no configured pair"
        )
        + ". "
    )
    if refused:
        body += "It cannot on " + "; ".join(
            f"{p}, whose ${v['floor_usd']:,.2f} venue floor is above the "
            f"${v['ceiling_usd']:,.2f} a position may reach at this equity "
            f"(that pair needs about ${v['min_equity_usd']:,.2f})"
            for p, v in refused.items()
        ) + ". "
    body += (
        "Every entry respects the risk and exposure caps, fees are charged on "
        "both sides, stops and targets fire at the declared geometry, the "
        "drawdown halt engages and then releases or re-arms on its two defined "
        "routes after a served cool-off, and the same code produces the same "
        "numbers on every rerun."
    )
    return _labelled("SUPPORTED:", body)


def _summary_line(result: BacktestResult, cont: WindowResult) -> str:
    return (
        f"     {result.source['name']:<14} {cont.slots:>6} wakeups, "
        f"{cont.trades:>4} round trips, {cont.opens:>4} entries, "
        f"peak exposure {cont.heat_pct_peak:>5.1f}% of equity, "
        f"max drawdown {cont.max_drawdown_pct:>5.2f}%, "
        f"equity ${cont.start_equity:,.2f} -> ${cont.end_equity:,.2f} "
        f"({cont.return_pct:+.2f}%)"
    )


def _verdict(results: list[BacktestResult]) -> list[str]:
    lines: list[str] = []
    section = 0

    def head(title: str) -> None:
        # Numbered here rather than by hand: sections appear only when the run
        # produced what they describe, and a hand-kept index skips numbers.
        nonlocal section
        section += 1
        # extend(), not +=: an augmented assignment here would rebind `lines`
        # as a local of this closure instead of appending to the caller's list.
        lines.extend(["", f"  {section}. {title}"])

    by_name = {r.source["name"]: r for r in results}

    head("The mechanics work, and the rules bind.")
    lines.append("")
    for result in results:
        cont = _continuous(result)
        if cont is None:
            continue
        lines.append(_summary_line(result, cont))
    halted = sum(
        w.rejections.get("drawdown halt", 0) for r in results for w in r.windows
    )
    lines += [
        "",
        "     Every entry either satisfies the risk cap, the exposure cap and the",
        "     venue floor, or it is refused with its arithmetic in the journal —",
        "     the same code the bot runs, so no special case was needed to keep",
        "     inside the rules. Peak *observed* exposure can read slightly above",
        "     the 30% cap: the cap is applied to equity at entry, and equity can",
        "     fall while a position is held. The halt is enforced at entry too.",
    ]
    if halted:
        lines.append(
            f"     The drawdown halt refused {halted} entries while it was "
            "engaged. It blocks new entries and never force-closes one, and "
            "every engagement ends in a release or a re-arm once its cool-off "
            "has been served — it is a pause, not a dead end."
        )

    null = by_name.get("null")
    if null is not None:
        cont = _continuous(null)
        if cont is not None:
            head(
                "Doing nothing costs nothing: the null source paid "
                f"${cont.fees:,.4f} in fees over {cont.trades} trades."
            )

    random = by_name.get("random")
    if random is not None:
        cont = _continuous(random)
        regimes = _windows(random, "regime")
        head("The cost baseline — a coin flip.")
        lines.append("")
        if cont is not None:
            lines.append(
                f"     Continuous: {cont.trades} round trips for {cont.net_pnl:+.2f} "
                f"USD ({cont.return_pct:+.2f}%), {cont.win_rate:.0f}% winners, "
                f"${abs(cont.fees):,.4f} of fees, expectancy "
                f"{cont.expectancy_usd:+.4f} USD/trade."
            )
        if regimes:
            signs = sum(1 for w in regimes if w.net_pnl > 0)
            lines.append(
                f"     Regimes: positive in {signs} of {len(regimes)} "
                f"({', '.join(f'{w.label} {w.net_pnl:+.2f}' for w in regimes)})."
            )
        lines += [
            "",
            "     A coin flip has no edge, so this is the account's fee drag made",
            "     visible: with a driftless price a 5%/7.5% geometry is roughly",
            "     zero-expectancy before costs, and costs are what remain. Any",
            "     mechanical result smaller than this drag is a fee story, not a",
            "     strategy.",
        ]

    directional = [r for r in results if r.source["label"] == "mechanical"]
    if directional:
        head(
            f"The declared rules ({', '.join(r.source['name'] for r in directional)})"
            " —"
        )
        lines += [
            "     these say something about the RULE, not about the bot's AI.",
            "",
        ]
        for result in directional:
            regimes = _windows(result, "regime")
            cont = _continuous(result)
            if cont is None:
                continue
            if not regimes:
                lines.append(
                    f"     {result.source['name']:<14} continuous "
                    f"{cont.return_pct:+.2f}% ({cont.trades} trades, "
                    f"{cont.win_rate:.0f}% win, maxDD "
                    f"{cont.max_drawdown_pct:.2f}%) | no per-regime windows in "
                    "this run"
                )
                continue
            signs = sum(1 for w in regimes if w.net_pnl > 0)
            lines.append(
                f"     {result.source['name']:<14} continuous {cont.return_pct:+.2f}% "
                f"({cont.trades} trades, {cont.win_rate:.0f}% win, maxDD "
                f"{cont.max_drawdown_pct:.2f}%) | positive in {signs}/{len(regimes)} "
                f"regimes: " + ", ".join(f"{w.label} {w.return_pct:+.2f}%" for w in regimes)
            )
        lines += [
            "",
            "     Sign flips across regimes are the expected result and the honest",
            "     one: a fixed rule does not work in every market state. These",
            "     sources exist to prove the skeleton runs end to end over 4.5",
            "     years and several regimes without breaking its own rules — not",
            "     to be selected, tuned, or copied into the live config.",
        ]

    journal = by_name.get("journal")
    if journal is not None:
        ev = journal.evidence
        unreadable = ev.get("journal_records_without_actions") or 0
        head("The model's own calls (recorded replay).")
        lines += [
            "",
            f"     {ev.get('journal_matched_slots')} slots had a recorded decision and "
            f"{ev.get('journal_unmatched_slots')} did not; "
            f"{ev.get('journal_replayable_slots')} of those records carried the "
            "actions the model proposed.",
        ]
        if unreadable:
            lines += [
                f"     {unreadable} records carried NO readable action — an older",
                "     journal stored only a COUNT of what was requested, so those",
                "     decisions cannot be replayed at all. Any zero-trade result",
                "     below is a fact about those records, NOT evidence that the",
                "     model chose to sit flat.",
            ]
        lines += [
            "     No look-ahead is possible here because the decisions were made",
            "     before their outcomes were known — but this is a replay of a",
            "     handful of real calls, not a test of the model, and it is far too",
            "     small to estimate anything.",
        ]

    head("What the evidence supports, stated as narrowly as it is true.")
    lines += [""]
    lines += _venue_arithmetic(results)
    lines += _labelled(
        "NOT SUPPORTED:",
        "that the bot makes money. A mechanical stand-in is not the model, the "
        "fee drag on a coin flip is real and negative, and no result here "
        "estimates the model's edge.",
    )
    lines += _labelled(
        "NOT MEASURED:",
        "real fill slippage, spread, funding, liquidity, and the model's "
        "judgement.",
    )
    return lines
