#!/usr/bin/env python3
"""Schedule gate for the AI trading bot (strictly SLOT-BASED).

Context (2026-09-16 audit of the cloud journal + Actions history):

  GitHub's cron for this repo is unreliable. The 6 wakeup slots fire ~25% of
  the time, the "hourly" health patrol actually ran 6x/day (05:14, 10:23,
  15:14, 19:10, 22:24, 00:44), and the 23:50 digest landed at ~01:45 every
  day for three days straight. So this workflow is also kicked by the other
  AI-suite workflows completing (workflow_run mesh) — it runs far more often
  than 6x/day. That part is deliberate and load-bearing.

  The previous gate treated "journal older than 45 min" as "run a wakeup
  now". Under a mesh that fires every ~20 min, that fired a wakeup every
  time the journal aged past 45 min: the journal shows 22 wakeups on
  2026-09-14 and 25 on 2026-09-15 instead of the designed 6 — 4x the
  intended decision rate, and a digest that could never make sense of
  "of 6 scheduled".

This gate is therefore tied to the SCHEDULE, not to journal age:

    most recent slot has no journal entry at/after it
        -> RUN now. On time if we are at the slot; catch-up if GitHub
           dropped the firing that should have run it.
    that slot is served, next slot further away than RELAY_WINDOW
        -> SKIP. This firing is redundant (a mesh kick between slots);
           exit immediately, no wakeup, no LLM call.
    that slot is served, next slot within RELAY_WINDOW
        -> RELAY: sleep to the slot, then re-evaluate (-> RUN). This keeps
           the wakeup punctual when a kick happens to land just before one.

Result: exactly one wakeup per schedule slot per day, no matter how many
times the workflow is triggered. A slot nobody triggered is NOT papered
over — it is reported as "never fired" by the daily digest, which is the
honest signal the user asked for.

Exit codes / outputs:
    Always exits 0 (a skip is normal, not a failure) and writes
    ``run=true|false`` to $GITHUB_OUTPUT so the workflow can gate the
    wakeup step on it.

Design notes:
  - Runs BEFORE pip setup, so it may only use the stdlib (no PyYAML).
  - Sleeps in <=30-minute chunks so the job stays inside its timeout budget
    and re-checks the journal after every chunk.
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

# A slot counts as SERVED when a wakeup journaled within this many seconds
# before it: a kick landing a hair early must not trigger a duplicate.
SERVE_TOLERANCE_SECONDS = float(os.environ.get("AI_BOT_GATE_TOLERANCE", "120"))

# Only sleep to the next slot when it is this close. Further out, a skipped
# firing is cheaper and equally correct (the mesh will kick us again nearer
# the slot). Keeps every run short instead of holding a runner for hours.
RELAY_WINDOW_SECONDS = float(os.environ.get("AI_BOT_GATE_RELAY_WINDOW", "600"))

# Max single sleep chunk: keeps the job inside its timeout and re-checks
# the journal frequently.
MAX_SLEEP_SECONDS = 30 * 60

# decide() outcomes.
RUN = "run"
SKIP = "skip"
RELAY = "relay"

# Fallback if the config is missing/unparseable — mirrors configs/ai_bot.yaml.
DEFAULT_SLOTS: tuple[tuple[int, int], ...] = (
    (0, 0), (6, 0), (8, 0), (14, 0), (20, 0), (23, 0),
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _set_output(name: str, value: str) -> None:
    """Append ``name=value`` to $GITHUB_OUTPUT (no-op when run locally)."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")
    except OSError as e:  # pragma: no cover - never fail the gate on this
        print(f"Gate: WARNING could not write {name} to $GITHUB_OUTPUT: {e}")


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


def _slot_dt(now: datetime, hour: int, minute: int) -> datetime:
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _last_due_slot(now: datetime, slots: list[tuple[int, int]]) -> datetime:
    """The most recent schedule slot at or before ``now`` (UTC)."""
    for day_offset in (0, 1):
        base = now - timedelta(days=day_offset)
        candidates = [
            _slot_dt(base, hour, minute)
            for hour, minute in slots
            if _slot_dt(base, hour, minute) <= now
        ]
        if candidates:
            return max(candidates)
    # Unreachable with a non-empty slot list, but fail safe: treat the
    # journal as unserved so a wakeup still happens.
    return now - timedelta(days=2)


def _next_slot_after(now: datetime, slots: list[tuple[int, int]]) -> datetime:
    """The first slot strictly after ``now`` (slots are UTC hour/minute)."""
    for day_offset in (0, 1):
        base = now + timedelta(days=day_offset)
        for hour, minute in slots:
            candidate = _slot_dt(base, hour, minute)
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
        ts = str(last.get("timestamp", ""))
        if not ts:
            return None
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        # A naive timestamp must never blow up the comparison below.
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        # A corrupt journal must not silently block trading — treat as stale.
        return None


def decide(
    now: datetime,
    last_dt: datetime | None,
    slots: list[tuple[int, int]],
) -> tuple[str, float]:
    """What this firing should do: (RUN|SKIP|RELAY, seconds_to_sleep).

    - No journal yet            -> (RUN, 0)   first run ever.
    - Last due slot unserved    -> (RUN, 0)   on time, or catch-up.
    - Slot served, next is far  -> (SKIP, 0)  redundant mesh kick.
    - Slot served, next is near -> (RELAY, wait)
    """
    if last_dt is None:
        return RUN, 0.0

    due = _last_due_slot(now, slots)
    if last_dt < due - timedelta(seconds=SERVE_TOLERANCE_SECONDS):
        return RUN, 0.0

    wait = (_next_slot_after(now, slots) - now).total_seconds()
    if wait <= RELAY_WINDOW_SECONDS:
        return RELAY, wait
    return SKIP, 0.0


def main() -> int:
    slots = _load_slots(CONFIG)
    while True:
        now = _now_utc()
        last_dt = _last_journal_dt(JOURNAL)
        action, wait = decide(now, last_dt, slots)

        if action == RUN:
            _set_output("run", "true")
            if last_dt is None:
                print("Gate: no journal yet — first run, proceed.")
            else:
                due = _last_due_slot(now, slots)
                late_min = (now - due).total_seconds() / 60
                age_min = (now - last_dt).total_seconds() / 60
                if late_min < 1:
                    print(
                        f"Gate: slot {due:%H:%M} UTC is due — proceed. "
                        f"(last wakeup {age_min:.0f} min ago)"
                    )
                else:
                    print(
                        f"Gate: slot {due:%H:%M} UTC unserved for "
                        f"{late_min:.0f} min — catch-up, proceed. "
                        f"(last wakeup {age_min:.0f} min ago)"
                    )
            return 0

        if action == SKIP:
            _set_output("run", "false")
            due = _last_due_slot(now, slots)
            nxt = _next_slot_after(now, slots)
            last_s = last_dt.strftime("%H:%M") if last_dt else "?"
            print(
                f"Gate: slot {due:%H:%M} UTC already served "
                f"(last wakeup {last_s} UTC); next slot {nxt:%H:%M} UTC is "
                f"{(nxt - now).total_seconds() / 60:.0f} min away — SKIPPING "
                "this firing (redundant trigger, no wakeup)."
            )
            return 0

        # RELAY: a kick landed just before the next slot; sleep to it.
        chunk = min(wait, MAX_SLEEP_SECONDS)
        nxt = _next_slot_after(now, slots)
        print(
            f"Gate: relaying {chunk / 60:.0f} min to the {nxt:%H:%M} UTC slot."
        )
        _sleep(chunk)


if __name__ == "__main__":
    sys.exit(main())
