#!/usr/bin/env python3
"""AI Trading Bot — daily digest (Telegram + stdout).

One heartbeat message per day (23:50 UTC, after the last wakeup): how many
wakeups ran and whether they succeeded, today's closed trades and P&L,
current equity, open positions, and the AI's last reasoning in its own
words. Because the digest runs on its own cron, a day with NO wakeups
still produces a loud "no wakeups ran today" alert — the heartbeat also
acts as a second watchdog.

Usage:
    python scripts/ai_daily_digest.py               # send via Telegram
    python scripts/ai_daily_digest.py --dry-run     # print, don't send

Zero LLM calls, no orders — report only.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from trading_system.bot.telegram_notifier import TelegramNotifier

# Records the last date a digest was actually delivered. A SUCCESSFUL
# send writes today's date; the digest skips a re-send for the same day.
# Missing/corrupt marker = "not sent yet" — a loud duplicate digest beats
# a silent gap, by design.
MARKER = "digest_sent.txt"


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
                continue  # skip corrupt lines, never block the digest
    except OSError:
        pass
    return entries


def summarize_day(data_dir: Path, now: datetime | None = None) -> dict:
    """Compute today's summary dict from the shared continuity files."""
    now = now or datetime.now(timezone.utc)
    day = now.strftime("%Y-%m-%d")

    journal = _load_journal(data_dir)
    todays = [e for e in journal if (e.get("timestamp") or "").startswith(day)]

    wakeups = len(todays)
    failed = [e for e in todays if e.get("status") != "success"]
    errors = [err for e in failed for err in (e.get("errors") or []) if err]

    closed_today: list[dict] = []
    for e in todays:
        for c in e.get("closed_triggers") or []:
            if isinstance(c, dict):
                closed_today.append(c)

    pnl_today = sum(float(c.get("net_pnl", 0)) for c in closed_today)
    ledger = _load_ledger(data_dir)
    last = todays[-1] if todays else None
    if last is not None and last.get("equity") is not None:
        equity = float(last["equity"])
    else:
        equity = float(ledger.get("cash", 0) or 0)

    positions = ledger.get("positions", {}) or {}

    # Idempotence guard: did today's digest already get DELIVERED? The
    # marker file lives in the same data dir and persists on the state
    # branch, so the cloud digest only ever fires once per day. A missing
    # or corrupt marker is treated as not-sent (resend loudly).
    already_sent = False
    marker_path = data_dir / MARKER
    if marker_path.exists():
        try:
            already_sent = marker_path.read_text().strip() == day
        except OSError:
            already_sent = False

    return {
        "day": day,
        "wakeups": wakeups,
        "expected_wakeups": 6,
        "failed": len(failed),
        "errors": errors,
        "closed": closed_today,
        "closed_count": len(closed_today),
        "pnl_today": pnl_today,
        "equity": equity,
        "open_positions": positions,
        "outlook": (last or {}).get("market_outlook"),
        "reasoning": (last or {}).get("ai_reasoning"),
        "already_sent": already_sent,
    }


def format_digest(s: dict) -> str:
    """Render the summary dict as the Telegram message text."""
    if s.get("already_sent"):
        return (
            f"AI BOT DAILY DIGEST — {s['day']}\n"
            f"{'=' * 30}\n"
            "✅ Already delivered earlier today — no duplicate send.\n"
            "(Idempotence guard: one digest per day, even if this job is\n"
            "re-run manually.)"
        )

    if s["wakeups"] == 0:
        return (
            f"AI BOT DAILY DIGEST — {s['day']}\n"
            f"{'=' * 30}\n"
            "⚠️ NO WAKEUPS RAN TODAY — the schedule did not fire.\n"
            "Check: GitHub → Actions → AI Trading Bot for red runs.\n"
            "(The watchdog patrols hourly; this digest is the second net.)"
        )

    ok = s["wakeups"] - s["failed"]
    if s["failed"]:
        head = f"⚠️ Wakeups: {ok}/{s['wakeups']} ok (of {s['expected_wakeups']} scheduled)"
    else:
        head = f"✅ Wakeups: {ok}/{s['wakeups']} ok (of {s['expected_wakeups']} scheduled)"

    lines = [
        f"AI BOT DAILY DIGEST — {s['day']}",
        "=" * 30,
        head,
    ]
    if s["errors"]:
        lines.append(f"  Last error: {s['errors'][-1][:140]}")

    if s["closed_count"]:
        wins = 0
        detail = []
        for c in s["closed"]:
            if float(c.get("net_pnl", 0)) > 0:
                wins += 1
            detail.append(
                f"  {c.get('pair', '?')} {float(c.get('pnl_pct', 0)):+.2f}% "
                f"(${float(c.get('net_pnl', 0)):+.2f})"
            )
        lines.append(
            f"P&L today: ${s['pnl_today']:+.2f} "
            f"({s['closed_count']} closed, {wins} wins)"
        )
        lines.extend(detail[:5])
    else:
        lines.append("P&L today: $0.00 (no trades closed)")

    lines.append(f"Equity: ${s['equity']:,.2f}")

    if s["open_positions"]:
        lines.append(f"OPEN ({len(s['open_positions'])}):")
        for pair, pos in list(s["open_positions"].items())[:5]:
            side = "LONG" if pos.get("side") in ("long", "buy", "LONG") else "SHORT"
            lines.append(f"  {pair} {side} @ ${pos.get('entry_price', 0):,.2f}")
    else:
        lines.append("OPEN: none")

    if s["outlook"]:
        lines.append(f"AI outlook: {s['outlook']}")
    if s["reasoning"]:
        text = " ".join(s["reasoning"].split())[:200]
        lines.append(f'AI said: "{text}"')

    stamp = datetime.now(timezone.utc).strftime("%b %d, %H:%M UTC")
    lines.append(f"\n{stamp} | Daily digest")
    return "\n".join(lines)


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252; emoji would crash click.echo."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@click.command()
@click.option("--config", default="configs/ai_bot.yaml", help="Config file path")
@click.option("--dry-run", is_flag=True, help="Print the digest without sending")
def main(config: str, dry_run: bool) -> None:
    """Send the daily AI bot digest (wakeups, P&L, equity, AI note)."""
    _force_utf8_stdout()
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

    s = summarize_day(data_dir)
    msg = format_digest(s)

    click.echo(msg)
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
    sent = notifier.send_message(msg)
    if not sent:
        click.echo(
            "Telegram not configured or send failed — digest NOT delivered. "
            "Set AI_TELEGRAM_BOT_TOKEN / AI_TELEGRAM_CHAT_ID (or the "
            "unprefixed TELEGRAM_* names) and telegram.enabled: true.",
            err=True,
        )
        sys.exit(1)

    # Mark today as delivered ONLY after a confirmed send, so a failed
    # send retries on the next trigger instead of going silent for the day.
    marker_path = data_dir / MARKER
    try:
        marker_path.write_text(s["day"])
        click.echo(f"Digest sent; marked delivered in {marker_path}")
    except OSError as e:
        click.echo(f"WARNING: digest sent but marker write failed: {e}", err=True)


if __name__ == "__main__":
    main()
