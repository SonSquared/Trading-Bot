#!/usr/bin/env python3
"""AI Trading Bot — daily digest (Telegram + stdout).

One heartbeat message per day: how many of the day's SCHEDULED wakeups
actually ran and whether they succeeded, that day's closed trades and P&L,
equity, open positions, and the AI's last reasoning in its own words.

Two hard-won rules are baked in (2026-09-16 audit):

1. THE REPORT IS ABOUT A SLOT-DEFINED DAY, NOT ABOUT WALL-CLOCK "TODAY".
   The digest is scheduled 23:50 UTC, but GitHub's scheduler held it ~2h late
   three days running (created 01:45, 01:55, 01:45), so every digest landed
   after midnight and reported the NEW day's first two hours while labelling
   itself with the new date. No digest ever described a complete day.
   ``target_day`` anchors the report to "the day whose last slot (23:00) has
   passed", which is stable whether the run happens at 23:50 or 02:10.

2. A MISSING WAKEUP MUST NOT LOOK LIKE A HEALTHY ONE.
   The old headline read "✅ Wakeups: 2/2 ok (of 6 scheduled)" on a day when
   4 of 6 slots never fired — a green tick in front of a real outage. The
   headline now counts SCHEDULE SLOTS served (a late catch-up wakeup counts
   for the slot it belongs to) and shouts about failures and silent gaps.

Because late runs are now harmless and the marker is authoritative, the
digest is also idempotent: ``digest_sent.txt`` records the delivered day, and
a re-trigger exits quietly instead of sending "already delivered" noise.

Usage:
    python scripts/ai_daily_digest.py               # send via Telegram
    python scripts/ai_daily_digest.py --dry-run     # print, don't send
    python scripts/ai_daily_digest.py --check-due   # exit 0 if due, 3 if not

Zero LLM calls, no orders — report only.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from trading_system.bot.telegram_notifier import TelegramNotifier, clip

# Records the last day whose digest was actually delivered. A SUCCESSFUL
# send writes the reported day; a re-trigger for the same day exits quietly.
# Missing/corrupt marker = "not sent yet" — a loud duplicate digest beats
# a silent gap, by design.
MARKER = "digest_sent.txt"

# The schedule's last wakeup hour (configs/ai_bot.yaml: daily_close at 23:00).
# A digest run at/after this hour reports ITS OWN day; earlier than this it
# reports the previous day (i.e. the run is late, not early).
LAST_SLOT_HOUR = 23

# Mirrors configs/ai_bot.yaml's schedule, used only if the config is unreadable.
DEFAULT_SLOTS: tuple[tuple[int, int], ...] = (
    (0, 0), (6, 0), (8, 0), (14, 0), (20, 0), (23, 0),
)

# A slot is considered served by any journal entry inside
# [slot - 2 min, slot + GRACE]. The grace covers a catch-up wakeup that ran
# late; the small backward tolerance covers a kick that ran just before the
# slot. Windows never overlap (the closest slots are 1h apart).
SLOT_GRACE_MINUTES = 180
SLOT_EARLY_TOLERANCE_MINUTES = 2

# A wakeup that missed the grace window can still have SERVED the slot as a
# late catch-up (the gate runs one catch-up per unserved slot, however late
# GitHub delivers it). Anything before the last slot's grace end counts.

CHECK_DUE_OK = 0
CHECK_DUE_ALREADY_SENT = 3


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def target_day(now: datetime, last_slot_hour: int = LAST_SLOT_HOUR) -> str:
    """The UTC day this digest reports on (YYYY-MM-DD).

    At/after the last slot hour -> today (the day that just finished).
    Before it -> yesterday (this run is late — GitHub held the 23:50 cron).
    """
    if now.hour >= last_slot_hour:
        return now.date().isoformat()
    return (now.date() - timedelta(days=1)).isoformat()


def _parse_ts(value: object) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def slot_report(
    entries: list[dict],
    day: str,
    slots: list[tuple[int, int]],
) -> tuple[list[str], list[str], list[str]]:
    """Which of ``day``'s schedule slots produced a wakeup?

    Returns three label lists (in schedule order):
        ran         — a wakeup ran within the grace window of the slot
        served_late — nothing in the window, but a later wakeup (before the
                      day's last slot + grace) served the slot as catch-up
        never_fired — nothing at all

    Rationale (2026-09-21 audit): GitHub delivered the 08:02 cron ~3h late;
    the gate correctly ran the 08:00 wakeup at 11:11 and skipped further
    kicks. But 11:11 sits outside the 180-min grace, so the digest called
    the slot "never fired" and hid the recovery. Late is worth knowing;
    missed is an outage — the digest now says which happened.

    Matching is 1:1 (greedy, nearest unfilled slot) so one wakeup can never
    cover two slots — otherwise a 08:30 wakeup would mark both 06:00 and
    08:00 as served and hide a genuine miss.
    """
    labels = [f"{h:02d}:{m:02d}" for h, m in slots]
    base = _parse_ts(f"{day}T00:00:00+00:00")
    if base is None:
        return [], [], labels
    slot_dts = [base + timedelta(hours=h, minutes=m) for h, m in slots]
    stamps = sorted(
        t for t in (_parse_ts(e.get("timestamp")) for e in entries) if t
    )
    early = timedelta(minutes=SLOT_EARLY_TOLERANCE_MINUTES)
    grace = timedelta(minutes=SLOT_GRACE_MINUTES)

    # Pass 1 — in-window matches: each wakeup takes its NEAREST slot whose
    # window contains it. Stamp-major on purpose: with a 180-min grace and
    # 1h slot spacing the windows overlap, and slot-major iteration would
    # let 06:00 steal the 08:30 wakeup that belongs to 08:00.
    in_window: dict[int, int] = {}   # slot idx -> stamp idx
    stamp_slot: dict[int, int] = {}  # stamp idx -> slot idx
    for ti, t in enumerate(stamps):
        best: int | None = None
        best_gap = None
        for si, slot in enumerate(slot_dts):
            if si in in_window or not (slot - early <= t <= slot + grace):
                continue
            gap = abs((t - slot).total_seconds())
            if best_gap is None or gap < best_gap:
                best, best_gap = si, gap
        if best is not None:
            in_window[best] = ti
            stamp_slot[ti] = best

    # Pass 2 — late catch-up: a leftover stamp before the last slot's grace
    # end serves its nearest unfilled earlier-or-equal slot (1:1 still).
    last_slot_end = slot_dts[-1] + grace if slot_dts else base
    served_late: set[int] = set()
    for ti, t in enumerate(stamps):
        if ti in stamp_slot or t > last_slot_end:
            continue
        best: int | None = None
        best_gap = None
        for si, slot in enumerate(slot_dts):
            if si in in_window or si in served_late or t < slot - early:
                continue
            gap = (t - slot).total_seconds()
            if best_gap is None or gap < best_gap:
                best, best_gap = si, gap
        if best is not None:
            served_late.add(best)
            stamp_slot[ti] = best

    taken = set(in_window) | served_late
    return (
        [labels[i] for i in sorted(in_window)],
        [labels[i] for i in sorted(served_late)],
        [labels[i] for i in range(len(labels)) if i not in taken],
    )


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


def summarize_day(
    data_dir: Path,
    now: datetime | None = None,
    slots: list[tuple[int, int]] | None = None,
    last_slot_hour: int = LAST_SLOT_HOUR,
) -> dict:
    """Compute the reported day's summary dict from the shared continuity files."""
    now = now or _utc_now()
    slots = list(slots) if slots else list(DEFAULT_SLOTS)
    day = target_day(now, last_slot_hour)

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

    ran_slots, late_slots, missing_slots = slot_report(journal, day, slots)

    # Idempotence guard: did this day's digest already get DELIVERED? The
    # marker file lives in the same data dir and persists on the state
    # branch, so the cloud digest only ever fires once per day.
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
        "expected_wakeups": len(slots),
        "slots_ran": ran_slots,
        "slots_served_late": late_slots,
        "slots_missing": missing_slots,
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


def _trade_pct(c: dict) -> float:
    """Round-trip net % when the ledger provides it, else the price move.

    Older records predate ``pnl_pct_net``; the dollar figure shown next to it
    is always net, so preferring the net percentage keeps the two consistent.
    """
    for key in ("pnl_pct_net", "pnl_pct"):
        if c.get(key) is not None:
            return float(c[key])
    return 0.0


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

    expected = int(s.get("expected_wakeups", len(DEFAULT_SLOTS)))
    ran = len(s.get("slots_ran") or [])
    late = list(s.get("slots_served_late") or [])
    missing = list(s.get("slots_missing") or [])
    failed = int(s.get("failed", 0))

    if s["wakeups"] == 0 and ran == 0 and not late:
        return (
            f"AI BOT DAILY DIGEST — {s['day']}\n"
            f"{'=' * 30}\n"
            f"⚠️ NO WAKEUPS RAN — none of the {expected} scheduled slots "
            "fired.\n"
            "Check: GitHub → Actions → AI Trading Bot for red runs.\n"
            "(The watchdog patrols hourly; this digest is the second net.)"
        )

    problems = []
    if failed:
        problems.append(f"{failed} FAILED")
    if late:
        shown = ", ".join(f"{m} UTC" for m in late[:4])
        more = f" (+{len(late) - 4} more)" if len(late) > 4 else ""
        problems.append(f"{len(late)} ran late: {shown}{more}")
    if missing:
        shown = ", ".join(f"{m} UTC" for m in missing[:4])
        more = f" (+{len(missing) - 4} more)" if len(missing) > 4 else ""
        problems.append(f"{len(missing)} never fired: {shown}{more}")

    if problems:
        icon = "✅" if not failed and not missing else "⚠️"
        head = (
            f"{icon} Wakeups: {ran + len(late)}/{expected} slots ran — "
            + " | ".join(problems)
        )
    else:
        head = f"✅ Wakeups: {ran}/{expected} slots ran, all ok"

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
                f"  {c.get('pair', '?')} {_trade_pct(c):+.2f}% "
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
        text = clip(s["reasoning"], 200)
        lines.append(f'AI said: "{text}"')

    stamp = _utc_now().strftime("%b %d, %H:%M UTC")
    lines.append(f"\n{stamp} | Daily digest")
    return "\n".join(lines)


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252; emoji would crash click.echo."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _slots_from_config(cfg: dict) -> list[tuple[int, int]]:
    """Schedule slots in UTC, honouring the config's tz_offset_hours."""
    tz_off = int((cfg.get("bot", {}) or {}).get("tz_offset_hours", 0) or 0)
    slots: list[tuple[int, int]] = []
    for entry in cfg.get("schedule") or []:
        if not isinstance(entry, dict) or "hour" not in entry:
            continue
        hour = int(entry["hour"])
        minute = int(entry.get("minute", 0) or 0)
        slots.append(((hour - tz_off) % 24, minute))
    return sorted(slots) or list(DEFAULT_SLOTS)


@click.command()
@click.option("--config", default="configs/ai_bot.yaml", help="Config file path")
@click.option("--dry-run", is_flag=True, help="Print the digest without sending")
@click.option(
    "--check-due/--no-check-due",
    default=False,
    help=f"Exit {CHECK_DUE_OK} if this day's digest is still due, "
    f"{CHECK_DUE_ALREADY_SENT} if it was already delivered. No send.",
)
def main(config: str, dry_run: bool, check_due: bool) -> None:
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
    slots = _slots_from_config(cfg)
    last_slot_hour = max(h for h, _ in slots)

    if check_due:
        # Used by the workflow's cheap pre-check: no state dir yet simply
        # means "not delivered", and the real run then fails loudly.
        if not data_dir.exists():
            click.echo("Due: no data dir yet (the scheduled run will report it).")
            return
        s = summarize_day(data_dir, slots=slots, last_slot_hour=last_slot_hour)
        if s["already_sent"]:
            click.echo(f"Not due: digest for {s['day']} already delivered.")
            sys.exit(CHECK_DUE_ALREADY_SENT)
        click.echo(f"Due: digest for {s['day']} not delivered yet.")
        return

    if not data_dir.exists():
        raise click.ClickException(f"No data dir at {data_dir} — run a wakeup first.")

    s = summarize_day(data_dir, slots=slots, last_slot_hour=last_slot_hour)
    msg = format_digest(s)

    click.echo(msg)

    if s["already_sent"]:
        # Quiet exit: sending "already delivered" would be noise, and the
        # marker is authoritative (persisted on the state branch).
        click.echo(f"(digest for {s['day']} already delivered — nothing sent)")
        return

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

    # Mark the REPORTED day as delivered ONLY after a confirmed send, so a
    # failed send retries on the next trigger instead of going silent.
    marker_path = data_dir / MARKER
    try:
        marker_path.write_text(s["day"])
        click.echo(f"Digest for {s['day']} sent; marked delivered in {marker_path}")
    except OSError as e:
        click.echo(f"WARNING: digest sent but marker write failed: {e}", err=True)


if __name__ == "__main__":
    main()
