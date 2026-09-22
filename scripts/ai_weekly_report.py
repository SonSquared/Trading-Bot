#!/usr/bin/env python3
"""AI Trading Bot — weekly report (Telegram + stdout).

Reads the shared continuity files (paper_ledger.json, journal.jsonl,
progress.json) and sends a Sunday-style summary: week P&L, win rate,
wakeups run/failed, open positions, and the AI's own progress notes.

Usage:
    python scripts/ai_weekly_report.py               # send via Telegram
    python scripts/ai_weekly_report.py --dry-run     # print, don't send
    python scripts/ai_weekly_report.py --days 7      # lookback window

Zero LLM calls, no orders — report only.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from trading_system.bot.perf_stats import build_perf, summarize
from trading_system.bot.telegram_notifier import TelegramNotifier, clip


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL continuity file, skipping corrupt lines (never blocks)."""
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        for line in path.read_text().strip().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return rows


def _load_ledger(data_dir: Path) -> dict:
    path = data_dir / "paper_ledger.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()) or {}
    except (json.JSONDecodeError, OSError):
        return {}


def _load_journal(data_dir: Path) -> list[dict]:
    path = data_dir / "journal.jsonl"
    if not path.exists():
        return []
    entries = []
    try:
        for line in path.read_text().strip().splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip corrupt lines, never block the report
    except OSError:
        pass
    return entries


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def build_report(data_dir: Path, days: int = 7) -> dict:
    """Compute the weekly summary dict from the shared files."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    ledger = _load_ledger(data_dir)
    journal = _load_journal(data_dir)

    closed = ledger.get("closed_trades", [])
    # Note: ledger has no close timestamp on very old records; the
    # close_time field was present from the first version of the ledger.
    week_trades = []
    for t in closed:
        ts = _parse_ts(t.get("close_time"))
        if ts is None or ts >= cutoff:
            week_trades.append(t)

    week_pnl = sum(float(t.get("net_pnl", 0)) for t in week_trades)
    week_wins = sum(1 for t in week_trades if float(t.get("net_pnl", 0)) > 0)

    def _fmt(t: dict) -> str:
        # Prefer the round-trip net % so the percentage and the dollar figure
        # describe the same thing (the $ is always net of both fees).
        pct = t.get("pnl_pct_net", t.get("pnl_pct", 0)) or 0
        return (
            f"{t.get('pair', '?')} {float(pct):+.2f}% "
            f"(${float(t.get('net_pnl', 0)):+.2f})"
        )

    sorted_trades = sorted(week_trades, key=lambda t: float(t.get("net_pnl", 0)))
    best = _fmt(sorted_trades[-1]) if sorted_trades else "n/a"
    worst = _fmt(sorted_trades[0]) if sorted_trades else "n/a"

    week_journal = []
    for e in journal:
        ts = _parse_ts(e.get("timestamp"))
        if ts is None or ts >= cutoff:
            week_journal.append(e)
    wakeups = len(week_journal)
    failed = sum(1 for e in week_journal if e.get("status") != "success")

    # The AI's own words: newest notes first, deduped.
    notes: list[str] = []
    for e in reversed(week_journal):
        note = (e.get("ai_reasoning") or "").strip()
        if note and note not in notes:
            notes.append(note)

    total_trades = len(closed)
    total_wins = sum(1 for t in closed if float(t.get("net_pnl", 0)) > 0)
    total_wr = (total_wins / total_trades * 100) if total_trades else 0.0

    equity = float(ledger.get("cash", 0))
    start_equity = float(ledger.get("start_equity", 0)) or equity

    positions = ledger.get("positions", {})
    open_positions = [
        {"pair": pair, "side": pos.get("side", "?"), "unrealized_pnl": 0.0}
        for pair, pos in positions.items()
    ]

    # Rolling performance review (win rate / drawdown / R:R / per-slot). Built
    # from the same continuity files, so every figure traces back to the ledger
    # and journal. `trades.jsonl` supplies the entry records, which is where the
    # PLANNED R:R (tp/sl the AI asked for) and its confidence live.
    opens = [
        t for t in _load_jsonl(data_dir / "trades.jsonl")
        if t.get("side") in ("long", "short")
    ]
    perf = build_perf(
        closed,
        journal=journal,
        opens=opens,
        start_equity=start_equity,
        now=now,
        primary_days=days,
        # The reported-P&L-vs-cash reconciliation only makes sense with no
        # positions open (otherwise cash is not realized equity yet).
        ledger_equity=float(ledger.get("cash", 0)) if not positions else None,
    )

    return {
        "perf": perf,
        "equity": equity,
        "start_equity": start_equity,
        "week_pnl": week_pnl,
        "week_pnl_pct": (week_pnl / start_equity * 100) if start_equity else 0.0,
        "week_trades": len(week_trades),
        "week_wins": week_wins,
        "best_trade": best,
        "worst_trade": worst,
        "total_trades": total_trades,
        "total_win_rate": total_wr,
        "wakeups": wakeups,
        "failed_wakeups": failed,
        "ai_notes": notes,
        "open_positions": open_positions,
    }


@click.command()
@click.option("--config", default="configs/ai_bot.yaml", help="Config file path")
@click.option("--dry-run", is_flag=True, help="Print the report without sending")
@click.option("--days", default=7, type=int, show_default=True, help="Lookback window")
def main(config: str, dry_run: bool, days: int) -> None:
    """Send the weekly AI bot report (P&L, win rate, AI notes)."""
    # Windows consoles default to cp1252; emoji/box chars would crash echo.
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    import os

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    import yaml

    cfg_path = Path(config)
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}

    data_dir = Path(cfg.get("bot", {}).get("data_dir", "data/ai_bot"))
    if not data_dir.exists():
        raise click.ClickException(f"No data dir at {data_dir} — run a wakeup first.")

    r = build_report(data_dir, days=days)

    # Always show the report on stdout so --dry-run (and CI logs) are useful.
    wr = (r["week_wins"] / r["week_trades"] * 100) if r["week_trades"] else 0.0
    click.echo(f"AI BOT WEEKLY REPORT ({days}d lookback)")
    click.echo(f"  Equity:        ${r['equity']:,.2f}")
    click.echo(f"  Week P&L:      ${r['week_pnl']:+,.2f} ({r['week_pnl_pct']:+.2f}%)")
    click.echo(f"  Closed trades: {r['week_trades']} | Wins: {r['week_wins']} ({wr:.0f}%)")
    click.echo(f"  Best/Worst:    {r['best_trade']} / {r['worst_trade']}")
    click.echo(f"  All-time:      {r['total_trades']} trades, {r['total_win_rate']:.0f}% win rate")
    click.echo(f"  Wakeups:       {r['wakeups']} run, {r['failed_wakeups']} failed")
    if r["open_positions"]:
        names = ", ".join(p["pair"] for p in r["open_positions"])
        click.echo(f"  Open:          {names}")
    click.echo("  PERFORMANCE (rolling)")
    for line in summarize(r["perf"]):
        click.echo(f"    {line}")
    for note in r["ai_notes"][:3]:
        click.echo(f"  AI note:       {clip(note, 150)}")

    if dry_run:
        click.echo("(dry-run: Telegram send skipped)")
        return

    tg_cfg = cfg.get("telegram", {})
    notifier = TelegramNotifier(
        # AI_-prefixed names take precedence (cloud secrets) so the AI bot
        # and the main bot can use different Telegram bots without clashing.
        bot_token=(
            os.environ.get("AI_TELEGRAM_BOT_TOKEN")
            or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        ),
        chat_id=(
            os.environ.get("AI_TELEGRAM_CHAT_ID")
            or os.environ.get("TELEGRAM_CHAT_ID", "")
        ),
        enabled=bool(tg_cfg.get("enabled", False)),
    )

    sent = notifier.notify_weekly_summary(
        equity=r["equity"],
        start_equity=r["start_equity"],
        week_pnl=r["week_pnl"],
        week_pnl_pct=r["week_pnl_pct"],
        week_trades=r["week_trades"],
        week_wins=r["week_wins"],
        best_trade=r["best_trade"],
        worst_trade=r["worst_trade"],
        total_trades=r["total_trades"],
        total_win_rate=r["total_win_rate"],
        wakeups=r["wakeups"],
        failed_wakeups=r["failed_wakeups"],
        ai_notes=r["ai_notes"],
        open_positions=r["open_positions"],
        perf=r["perf"],
    )

    if not sent:
        click.echo(
            "Telegram not configured or send failed — report NOT delivered. "
            "Set AI_TELEGRAM_BOT_TOKEN / AI_TELEGRAM_CHAT_ID (or the "
            "unprefixed TELEGRAM_* names) and telegram.enabled: true.",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
