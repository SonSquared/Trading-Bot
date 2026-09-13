#!/usr/bin/env python3
"""Always-on Telegram command responder for the AI trading bot.

The scheduled wakeup responder (scripts/ai_telegram_commands.py) answers
commands only at wakeups — worst case ~6h later. This poller closes that
gap: a GitHub Actions job launches it every 15 minutes (backup crons,
plus every trading/digest/health-check completion via workflow_run); it
long-polls Telegram (server-held getUpdates, near-zero traffic) and
answers /status and friends within seconds. The NEXT generation cancels
this one (concurrency cancel-in-progress), so coverage is continuous and
billing is bounded to ~24×60 = 1,440 runner-minutes/day — free on public
repos (unlimited minutes), a decision point on private ones (2,000/month
cap).

Coordination with the wakeup responder, without double answers:
  - The update offset lives in data/ai_bot/telegram_offset.txt, re-read
    from disk before EVERY poll — so offsets the wakeup responder
    persists mid-flight are honored immediately.
  - update_ids already processed by THIS process are skipped in memory.
  - If another poller holds the long-poll (HTTP 409 Conflict), this
    instance exits 0 immediately: the redundant generation yields.

Exit codes: 0 = fine (deadline reached, conflict yielded, nothing
configured); 1 = configuration/infrastructure error that a rerun won't
fix. Never sends trading orders — report-only, same routing table as
the wakeup responder.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuses the EXACT routing/reply code the wakeup responder uses, so a
# command gets the same answer either way. Package import (tests, tools)
# and direct-script import (python scripts/ai_poller.py) both work, and
# resolve to the SAME module instance so shared globals stay in sync.
try:
    from scripts import ai_telegram_commands as cmds
except ImportError:  # direct execution: scripts/ itself is sys.path[0]
    import ai_telegram_commands as cmds  # noqa: F401


def _get_updates(token: str, offset: int | None, poll_seconds: int) -> tuple[dict | None, int]:
    """Long-poll getUpdates. Returns (data, http_status). Never raises.

    (data, 200)          -> updates (possibly empty)
    (None, 409)          -> another getUpdates consumer is active
    (None, other/0)      -> blip; caller retries
    """
    import requests

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/getUpdates",
            json={
                "offset": offset,
                "timeout": poll_seconds,
                "allowed_updates": ["message"],
            },
            # requests timeout must exceed the server hold, with headroom.
            timeout=poll_seconds + 15,
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            return data, 200
        if resp.status_code == 409:
            return None, 409
        print(f"WARNING: getUpdates failed: HTTP {resp.status_code} {data}",
              file=sys.stderr)
        return None, resp.status_code
    except Exception as e:  # noqa: BLE001 — responder must never crash the job
        print(f"WARNING: getUpdates error: {e}", file=sys.stderr)
        return None, 0


def poll_once(
    token: str,
    chat_id: str,
    *,
    poll_seconds: int = 50,
    get_updates: Callable[[str, int | None, int], tuple[dict | None, int]] = _get_updates,
    send: Callable[..., dict | None] = None,  # type: ignore[assignment]
    read_offset: Callable[[], int] = None,  # type: ignore[assignment]
    write_offset: Callable[[int], None] = None,  # type: ignore[assignment]
    load_ledger: Callable[[], dict] = None,  # type: ignore[assignment]
    load_journal: Callable[[], dict | None] = None,  # type: ignore[assignment]
    prices_fn: Callable[[list[str]], dict[str, float]] = None,  # type: ignore[assignment]
) -> dict:
    """One long-poll + answer cycle. Returns what happened.

    All dependencies are injectable for tests; defaults reuse the wakeup
    responder's module functions so behavior can never drift.
    """
    send = send or cmds._tg
    read_offset = read_offset or cmds._read_offset
    write_offset = write_offset or cmds._write_offset
    load_ledger = load_ledger or cmds._load_ledger
    load_journal = load_journal or cmds._last_journal
    prices_fn = prices_fn or cmds._current_prices

    data, status = get_updates(token, read_offset() or None, poll_seconds)
    if status == 409:
        return {"answered": 0, "max_update_id": 0, "conflict": True}
    if data is None:
        return {"answered": 0, "max_update_id": 0, "conflict": False}

    updates = data.get("result", [])
    if not updates:
        return {"answered": 0, "max_update_id": 0, "conflict": False}

    ledger = load_ledger()
    journal = load_journal()
    max_update_id = read_offset()  # never go backwards vs disk
    answered = 0

    for update in updates:
        uid = int(update.get("update_id", 0))
        if uid < max_update_id:
            continue  # already handled (by us or the wakeup responder)
        max_update_id = max(max_update_id, uid + 1)
        message = update.get("message") or {}
        text = message.get("text") or ""
        from_id = str((message.get("chat") or {}).get("id", ""))
        if from_id != chat_id:
            print(f"Ignored message from unauthorized chat {from_id}")
            continue
        prices = prices_fn(
            list(ledger.get("positions", {}).keys()) or
            ["BTC/USDT:USDT", "ETH/USDT:USDT"]
        )
        reply = cmds.route_command(text, ledger, journal, prices)
        if reply and send(token, "sendMessage", chat_id=chat_id, text=reply,
                          disable_web_page_preview=True):
            answered += 1

    write_offset(max_update_id)
    # Confirm the batch to Telegram SERVER-side before this process dies.
    # Each generation starts with a stale (or absent) offset file, so the
    # only durable record of "these updates were answered" is Telegram's
    # own cursor — advanced by one getUpdates call carrying the offset.
    # Without this, a fresh generation would be re-sent the updates
    # answered above and answer them again (double replies).
    if max_update_id:
        get_updates(token, max_update_id, 0)
    return {"answered": answered, "max_update_id": max_update_id,
            "conflict": False}


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-minutes", type=float, default=20.0,
                        help="Exit gracefully after this many minutes")
    parser.add_argument("--poll-seconds", type=int, default=50,
                        help="Telegram long-poll hold per request")
    args = parser.parse_args()

    token, chat_id = cmds._creds()
    if not token or not chat_id:
        print(
            "Telegram commands: not configured "
            "(set AI_TELEGRAM_BOT_TOKEN / AI_TELEGRAM_CHAT_ID) — exiting.",
            file=sys.stderr,
        )
        return 0

    deadline = time.monotonic() + args.max_minutes * 60
    total = 0
    while time.monotonic() < deadline - 5:
        r = poll_once(token, chat_id, poll_seconds=args.poll_seconds)
        if r["conflict"]:
            print("Another responder is long-polling (HTTP 409) — yielding.")
            return 0
        total += r["answered"]
    print(f"Responder deadline reached — answered {total} command(s) this "
          "generation; next generation takes over.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
