"""Schedule gate: cheap pre-flight check before an expensive bot run.

GitHub Actions cron is not a clock — on free-tier private repos it fires
late and drops slots outright (observed 2026-09-07/08: a 2-hourly schedule
produced 5-7h gaps; slots at 02/04/08/10 UTC never ran at all). Two mitigations:

1. Redundant crons (bot.yml fires twice per even hour). The extra firings
   must NOT double-trade — that is this gate's job: exit 0 to run if the
   last *successful* bot run is older than MIN_RUN_INTERVAL_MINUTES,
   exit 3 to skip if a recent success exists.

2. Downstream duplicate-open protection can then safely exempt runs that
   the gate acknowledged: two runs trading within the interval can no
   longer happen, so a back-to-back OPEN of the same pair+side is a real
   concurrency bug (the Sep-3 signature), not an artifact of redundant
   schedules. paper_trader marks gate-acknowledged ledger entries with
   note "schedule-gate-ack" for that purpose.

Exit codes: 0 = run, 3 = skip (treated as success by the workflow).
All state is read from run_history.jsonl in the working tree (the git
bot-state branch is pulled into data/results/ before this gate runs).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path("data/results")
RUN_LOG = RESULTS / "run_history.jsonl"

MIN_RUN_INTERVAL_MINUTES = float(
    os.getenv("GATE_MIN_RUN_INTERVAL_MINUTES", "100")
)

# Exit code 3 = "skip, and that is fine". Distinct from 0/1 so a workflow
# step can keep the run green while doing nothing.
SKIP_EXIT = 3


def _iso(s) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, AttributeError):
        return None


def last_success(run_log: Path = RUN_LOG) -> datetime | None:
    """Timestamp of the newest successful/partial run-history entry."""
    if not run_log.exists():
        return None
    try:
        lines = run_log.read_text(encoding="utf-8", errors="replace") \
            .strip().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(entry.get("status", "")) in ("success", "partial"):
            ts = _iso(entry.get("timestamp", ""))
            if ts:
                return ts
    return None


def should_run(now: datetime | None = None,
               run_log: Path = RUN_LOG,
               min_interval_min: float = MIN_RUN_INTERVAL_MINUTES) -> tuple[bool, str]:
    """(run?, reason). True when the last success is older than the interval."""
    if now is None:
        now = datetime.now(timezone.utc)
    last = last_success(run_log)
    if last is None:
        return True, "no successful run on record — run"
    age_min = (now - last).total_seconds() / 60.0
    if age_min >= min_interval_min:
        return True, f"last success {age_min:.0f}m ago >= {min_interval_min:.0f}m — run"
    return False, (f"last success {age_min:.0f}m ago < {min_interval_min:.0f}m "
                   f"({last.isoformat()}) — redundant firing, skip")


def main(run_log: Path | None = None, now: datetime | None = None) -> int:
    run, reason = should_run(run_log=run_log or RUN_LOG, now=now)
    print(f"Schedule gate: {reason}")
    return 0 if run else SKIP_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
