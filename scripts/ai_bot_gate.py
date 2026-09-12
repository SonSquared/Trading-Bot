#!/usr/bin/env python3
"""Schedule gate for the AI trading bot (mirrors scripts/schedule_gate.py).

The ai_bot workflow fires each wakeup time TWICE (:00 and :30) because
GitHub's free-tier cron drops slots wholesale. This gate makes the second,
redundant firing a cheap no-op when the first one already completed:

    exit 0  -> run the wakeup
    exit 3  -> skip (a wakeup journaled within the last GATE_MINUTES)
    other   -> real gate failure (red run)

It runs BEFORE pip setup so skipped runs burn almost no Actions quota.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

JOURNAL = Path(os.environ.get("AI_BOT_DATA_DIR", "data/ai_bot")) / "journal.jsonl"
# Must exceed the redundancy gap (30 min) but stay well under the smallest
# real schedule gap (1h between 23:00 and 00:00).
GATE_MINUTES = float(os.environ.get("AI_BOT_GATE_MINUTES", "45"))


def main() -> int:
    if not JOURNAL.exists():
        print("Gate: no journal yet — first run, proceed.")
        return 0

    try:
        lines = JOURNAL.read_text().strip().splitlines()
        if not lines:
            print("Gate: journal is empty — proceed.")
            return 0
        last = json.loads(lines[-1])
    except (json.JSONDecodeError, OSError) as e:
        # A corrupt journal must not silently block trading — proceed and
        # let the wakeup itself surface the problem.
        print(f"Gate: journal unreadable ({e}) — proceed cautiously.")
        return 0

    ts = last.get("timestamp", "")
    try:
        from datetime import datetime
        last_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        age_minutes = (time.time() - last_dt.timestamp()) / 60
    except (ValueError, TypeError, OSError):
        print(f"Gate: last journal timestamp unparseable ({ts!r}) — proceed.")
        return 0

    if age_minutes < GATE_MINUTES:
        print(
            f"Gate: last wakeup completed {age_minutes:.0f} min ago "
            f"(status={last.get('status', '?')}) — skipping redundant firing."
        )
        return 3

    print(f"Gate: last wakeup {age_minutes:.0f} min ago — proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
