"""Wakeup Scheduler for the AI Trading Bot.

Adapts the Nate Herk 6-check stock schedule to 24/7 crypto:

  1. Asia Session Open   (00:00 UTC)
  2. Asia/London Overlap (06:00 UTC)
  3. London Open         (08:00 UTC)
  4. US Open             (14:00 UTC)
  5. US Midday           (20:00 UTC)
  6. Daily Close         (23:00 UTC)

Supports continuous mode, single named wakeup, and next-due wakeup.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)

# How long the loop sleeps between clock checks. Keeps the loop responsive
# to Ctrl+C and resilient to clock drift.
SLEEP_CHUNK_SECONDS = 60

# Default crypto-adapted schedule (UTC). Times are interpreted in the
# timezone given by tz_offset_hours relative to UTC.
DEFAULT_SCHEDULE: list[dict] = [
    {"name": "asia_session_open", "hour": 0, "minute": 0,
     "description": "Pre-market research, check news"},
    {"name": "asia_london_overlap", "hour": 6, "minute": 0,
     "description": "First trade check, scan for setups"},
    {"name": "london_open", "hour": 8, "minute": 0,
     "description": "Mid-morning review, manage positions"},
    {"name": "us_open", "hour": 14, "minute": 0,
     "description": "US session start, new opportunities"},
    {"name": "us_midday", "hour": 20, "minute": 0,
     "description": "Position management, adjust stops"},
    {"name": "daily_close", "hour": 23, "minute": 0,
     "description": "End-of-day review, record results"},
]


def _normalize_schedule(entries: list[dict] | None) -> list[dict]:
    """Validate and copy schedule entries. Raises ValueError on bad input."""
    if not entries:
        return [dict(e) for e in DEFAULT_SCHEDULE]
    if not isinstance(entries, list):
        raise ValueError("schedule must be a list of {name, hour, minute} dicts")
    out: list[dict] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict) or not {"name", "hour", "minute"} <= set(e):
            raise ValueError(
                f"schedule entry #{i}: must be a dict with 'name', 'hour', 'minute'"
            )
        if not isinstance(e["name"], str) or not e["name"]:
            raise ValueError(f"schedule entry #{i}: 'name' must be a non-empty string")
        if not (0 <= int(e["hour"]) <= 23 and 0 <= int(e["minute"]) <= 59):
            raise ValueError(f"schedule entry #{i}: hour/minute out of range")
        out.append(dict(e))
    return out


class Scheduler:
    """Runs the AI agent on a fixed schedule.

    Usage:
        scheduler = Scheduler(agent_fn=agent.run_wakeup)
        scheduler.run_forever()          # continuous loop
        scheduler.run_once("us_open")    # single wakeup by name
        scheduler.run_next()             # run the next scheduled wakeup
    """

    def __init__(
        self,
        agent_fn: Callable[[], dict],
        schedule: list[dict] | None = None,
        tz_offset_hours: float = 0,  # local offset from UTC, e.g. -5 for EST
    ):
        self.agent_fn = agent_fn
        self.schedule = _normalize_schedule(schedule)
        self.tz_offset_hours = float(tz_offset_hours)

    # ----- time helpers -----

    def _now(self) -> datetime:
        """True UTC now (all returned slots are expressed in true UTC)."""
        return datetime.now(timezone.utc)

    def _local_now(self) -> datetime:
        """Local wall-clock now (schedule hours are interpreted in this frame)."""
        return self._now() + timedelta(hours=self.tz_offset_hours)

    def _slot_today(self, entry: dict, local_now: datetime) -> datetime:
        return local_now.replace(
            hour=int(entry["hour"]), minute=int(entry["minute"]),
            second=0, microsecond=0,
        )

    def next_slot(self, now: datetime | None = None) -> dict:
        """The next wakeup slot strictly after `now`, across midnight.

        Schedule hours are interpreted in local time (UTC + tz_offset_hours);
        the returned "at" is converted back to TRUE UTC so sleeps and logs
        are correct regardless of offset. Handles the day rollover correctly:
        after 23:00 the next slot is tomorrow's 00:00 (first schedule entry),
        fixing the bug where the 00:00 wakeup was skipped every day.
        """
        now_utc = now if now is not None else self._now()
        local_now = now_utc + timedelta(hours=self.tz_offset_hours)

        candidates: list[tuple[datetime, dict]] = []
        for day_offset in (0, 1):
            base = (local_now + timedelta(days=day_offset)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            for s in self.schedule:
                local_dt = base + timedelta(
                    hours=int(s["hour"]), minutes=int(s["minute"])
                )
                if local_dt > local_now:
                    candidates.append((local_dt, s))

        if not candidates:
            raise RuntimeError("schedule produced no upcoming slots")

        local_dt, entry = min(candidates, key=lambda t: t[0])
        utc_dt = local_dt - timedelta(hours=self.tz_offset_hours)
        return {"name": entry["name"], "at": utc_dt, "info": entry}

    def _sleep_until(self, wake_at: datetime) -> None:
        """Sleep in chunks so Ctrl+C and clock drift are handled."""
        while True:
            now = self._now()
            remaining = (wake_at - now).total_seconds()
            if remaining <= 0:
                return
            time.sleep(min(remaining, SLEEP_CHUNK_SECONDS))

    # ----- modes -----

    def run_forever(self) -> None:
        """Run the scheduler loop. Blocks until interrupted."""
        logger.info("scheduler_started", wakeups=len(self.schedule),
                    tz_offset=self.tz_offset_hours)
        print("=" * 60)
        print("AI TRADING BOT - SCHEDULED WAKEUPS")
        print(f"Schedule: {len(self.schedule)} wakeups per day")
        for s in self.schedule:
            print(f"  {int(s['hour']):02d}:{int(s['minute']):02d} — {s.get('description', '')}")
        print("=" * 60)
        print()

        # Wakeups already past are NOT backfilled (a stale 4-hour-old signal
        # is worthless); we simply wait for the next one.
        try:
            while True:
                nxt = self.next_slot()
                wait_hours = (nxt["at"] - self._now()).total_seconds() / 3600
                logger.info("scheduler_sleeping", next_wakeup=nxt["name"],
                            wait_hours=round(wait_hours, 2))
                print(f"\n[Next] {nxt['name']} at "
                      f"{nxt['at'].strftime('%Y-%m-%d %H:%M UTC')} - "
                      f"sleeping {wait_hours:.1f} hours")
                self._sleep_until(nxt["at"])
                self._run_wakeup(nxt)
        except KeyboardInterrupt:
            logger.info("scheduler_stopped_by_user")
            print("\nScheduler stopped.")

    def run_next(self) -> dict | None:
        """Wait for and run the next scheduled wakeup. Returns its result."""
        nxt = self.next_slot()
        self._sleep_until(nxt["at"])
        return self._run_wakeup(nxt)

    def run_once(self, name: str | None = None) -> dict:
        """Run a single wakeup immediately.

        If name is given, run that specific entry. Otherwise run the most
        recent past slot of the day (or the first entry, before it starts).
        """
        now = self._local_now()
        if name:
            for s in self.schedule:
                if s["name"] == name:
                    return self._run_wakeup({"name": s["name"], "at": now, "info": s})
            raise ValueError(
                f"Unknown wakeup: {name!r}. Available: {[s['name'] for s in self.schedule]}"
            )

        most_recent = None
        local_now = self._local_now()
        for s in self.schedule:
            t = self._slot_today(s, local_now)
            if t <= local_now:
                most_recent = {"name": s["name"], "at": t, "info": s}
        if most_recent is None:
            first = self.schedule[0]
            most_recent = {"name": first["name"], "at": now, "info": first}
        return self._run_wakeup(most_recent)

    def run_all_now(self) -> list[dict]:
        """Run all wakeups immediately (manual override / testing)."""
        results = []
        for s in self.schedule:
            results.append(self._run_wakeup(
                {"name": s["name"], "at": self._now(), "info": s}))
        return results

    # ----- execution -----

    def _run_wakeup(self, wakeup: dict) -> dict:
        name = wakeup["name"]
        info = wakeup.get("info", {})
        print(f"\n{'=' * 60}")
        print(f">> WAKEUP: {name}")
        print(f"   {info.get('description', '')}")
        print(f"   Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'=' * 60}")

        start = time.time()
        try:
            result = self.agent_fn()
        except Exception as e:
            result = {"status": "error", "errors": [str(e)]}
            logger.error("wakeup_exception", name=name, error=str(e))

        elapsed = time.time() - start
        status = result.get("status", "unknown")
        trades = len(result.get("actions_taken", []) or [])
        errors = result.get("errors", []) or []

        print(f"\n{'─' * 60}")
        print(f"  Status: {status.upper()}")
        print(f"  Trades: {trades}")
        if errors:
            print(f"  Errors: {len(errors)}")
            for e in errors[:3]:
                print(f"    - {e}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"{'─' * 60}\n")
        return result
