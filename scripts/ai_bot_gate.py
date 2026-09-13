#!/usr/bin/env python3
"""Schedule gate + relay for the AI trading bot (mirrors scripts/schedule_gate.py).

GitHub's cron silently drops slots (proven 2026-09-13: four consecutive
wakeup slots never fired). This gate makes the workflow SELF-SCHEDULING
so trading no longer depends on cron:

    journal stale (no wakeup in GATE_MINUTES) -> exit 0: run NOW
        (the normal on-time wakeup, or catch-up after dropped slots)
    journal fresh (a wakeup just completed)   -> SLEEP until the next
        scheduled slot from configs/ai_bot.yaml, re-evaluate, repeat.
        This run becomes the relay that fires the next wakeup on time
        even if every cron slot is dropped.

Exit codes:
    0  -> run the wakeup (now, or after the in-process relay sleep)
    other -> real gate failure (red run) — never used for skips anymore

Design notes:
  - The workflow's concurrency group (cancel-in-progress: false) queues
    concurrent firings, so a cron slot landing while the relay sleeps
    waits its turn and its own gate check then sees fresh state —
    mutual exclusion without locks, no duplicate wakeups.
  - The relay runs BEFORE pip setup (see ai_bot.yml), so sleeping burns
    runner minutes but never the ~60s dependency install; on the public
    repo minutes are free.
  - Sleep happens in <=30-minute chunks so the job stays far inside its
    timeout budget and re-checks the journal after every chunk (a cron
    firing that ran a real wakeup while this one slept is honored
    immediately).
  - tz_offset_hours from the config shifts schedule times into UTC
    (local = UTC + offset).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

JOURNAL = Path(os.environ.get("AI_BOT_DATA_DIR", "data/ai_bot")) / "journal.jsonl"
CONFIG = Path(os.environ.get("AI_BOT_CONFIG", "configs/ai_bot.yaml"))
# Must exceed the redundancy gap (30 min) but stay well under the smallest
# real schedule gap (1h between 23:00 and 00:00).
GATE_MINUTES = float(os.environ.get("AI_BOT_GATE_MINUTES", "45"))
# Max single sleep chunk: keeps the job inside its timeout and re-checks
# the journal frequently (another run may have completed a wakeup).
MAX_SLEEP_SECONDS = 30 * 60
# Fallback if the config is missing/unparseable — mirrors configs/ai_bot.yaml.
DEFAULT_SLOTS: tuple[tuple[int, int], ...] = (
    (0, 0), (6, 0), (8, 0), (14, 0), (20, 0), (23, 0),
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _load_slots(config_path: Path) -> list[tuple[int, int]]:
    """Parse the wakeup schedule (hour, minute) from the config, in UTC.

    Regex, not yaml: the gate must run BEFORE pip setup on the runner, so
    it cannot rely on PyYAML being installed.
    """
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError:
        print(f"Gate: config {config_path} unreadable — using default schedule.")
        return list(DEFAULT_SLOTS)

    tz_match = re.search(r"tz_offset_hours:\s*(-?\d+)", text)
    tz_off = int(tz_match.group(1)) if tz_match else 0

    slots: list[tuple[int, int]] = []
    # Each schedule entry starts with "- name:"; hour precedes minute inside it.
    blocks = re.split(r"(?=^\s*-\s*name:)", text, flags=re.MULTILINE)
    for block in blocks:
        if "hour:" not in block:
            continue
        h = re.search(r"hour:\s*(\d+)", block)
        if not h:
            continue
        m = re.search(r"minute:\s*(\d+)", block)
        hour = int(h.group(1))
        minute = int(m.group(1)) if m else 0
        # local = UTC + offset  =>  utc_hour = (local_hour - offset) % 24
        slots.append(((hour - tz_off) % 24, minute))

    if not slots:
        print("Gate: no schedule entries found in config — using default schedule.")
        return list(DEFAULT_SLOTS)
    return sorted(slots)


def _next_slot_after(now: datetime, slots: list[tuple[int, int]]) -> datetime:
    """The first slot strictly after `now` (slots are UTC hour/minute)."""
    for day_offset in (0, 1):
        base = (now + timedelta(days=day_offset)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        for hour, minute in slots:
            candidate = base.replace(hour=hour, minute=minute)
            if candidate > now:
                return candidate
    # Unreachable with a non-empty slot list, but fail safe:
    return now + timedelta(hours=6)


def _last_journal_dt(journal_path: Path) -> datetime | None:
    """Timestamp of the most recent journal entry, or None if unreadable/absent."""
    if not journal_path.exists():
        return None
    try:
        lines = journal_path.read_text().strip().splitlines()
        if not lines:
            return None
        last = json.loads(lines[-1])
        ts = last.get("timestamp", "")
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        # A corrupt journal must not silently block trading — treat as stale.
        return None


def decide(
    now: datetime,
    last_dt: datetime | None,
    slots: list[tuple[int, int]],
) -> float:
    """Seconds to sleep before the wakeup should run (0.0 = run now).

    - No journal / stale journal -> 0 (run now).
    - Fresh journal -> sleep until the next scheduled slot, but never less
      than the remaining gate window (a manual wakeup 5 minutes before a
      slot must not produce a duplicate right after it).
    """
    if last_dt is None:
        return 0.0
    age = (now - last_dt).total_seconds()
    if age >= GATE_MINUTES * 60:
        return 0.0
    wait = (_next_slot_after(now, slots) - now).total_seconds()
    gate_remaining = GATE_MINUTES * 60 - age
    return max(wait, gate_remaining)


def main() -> int:
    slots = _load_slots(CONFIG)
    while True:
        now = _now_utc()
        last_dt = _last_journal_dt(JOURNAL)
        wait = decide(now, last_dt, slots)
        if wait <= 0:
            if last_dt is None:
                print("Gate: no journal yet — first run, proceed.")
            else:
                age_min = (now - last_dt).total_seconds() / 60
                print(f"Gate: last wakeup {age_min:.0f} min ago — proceed.")
            return 0
        chunk = min(wait, MAX_SLEEP_SECONDS)
        human = (
            f"{wait / 3600:.1f}h" if wait >= 3600 else f"{wait / 60:.0f}m"
        )
        print(
            f"Gate: last wakeup completed recently — relaying: sleeping "
            f"{chunk / 60:.0f}m of {human} until the next scheduled slot."
        )
        _sleep(chunk)


if __name__ == "__main__":
    sys.exit(main())
