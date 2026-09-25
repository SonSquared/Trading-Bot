#!/usr/bin/env python3
"""Heartbeat guard — keeps the AI bot's pulse alive without a human.

WHY THIS EXISTS
---------------
The pulse is the Telegram poller: each ~21-minute generation long-polls
Telegram and, as its LAST step, dispatches the next generation (which cancels
it via ``concurrency: cancel-in-progress``). That design has one structural
weakness, and it is worth stating precisely because an earlier review of it
was wrong:

  * A generation that ends WITHOUT reaching its last step breaks the chain.
    That happens when a run is cancelled deliberately (the documented "stop
    the chain" switch — a cancelled run skips the relaunch, by design), when a
    runner is lost mid-job, or when the dispatch call itself fails.
  * The "four backup crons" in ai_poller.yml are NOT a safety net: over the
    100 poller runs ending 2026-09-25T06:22Z, only 13 were ``schedule``
    triggered (~25% delivery). GitHub thins this repo's crons severely.
  * Nothing repaired it. ``ai_health_check.yml`` watched only the *journal*
    (an 8h threshold on the last successful wakeup) and never looked at the
    poller at all, so a dead OR live-but-hung generation was invisible to it
    until it had already cost a wakeup or two — and even then it only alerted.

So recovery used to depend on the very thing that had just failed. This guard
inverts that: it is triggered independently of the chain and its only job is
to ensure a live generation exists.

WHAT IT DOES
------------
Reads the poller's run history and decides:

  healthy   an active generation younger than STALE_MINUTES -> do nothing
  dispatch  an active generation OLDER than STALE_MINUTES (a hung job),
            or no active generation and the last one ended more than
            GRACE_MINUTES ago (a dead chain) -> dispatch a fresh generation,
            which also cancels the hung one and frees the concurrency group
  alert     an active generation older than STALE_MINUTES but still ``queued``
            -> a runner shortage, not a hung job: dispatching would just stack
            more queued runs, so alert and wait for capacity

Thresholds are deliberately outside the normal envelope: STALE_MINUTES (40)
exceeds the poller job's own 30-minute timeout, so a legitimately long
generation is never mistaken for a hung one, and GRACE_MINUTES (10) is far
wider than the seconds it takes the relaunch dispatch to appear.

NO HOT LOOPS. The poller does not kick this guard's workflow (only the reverse
via dispatch), a dispatch makes the next check see a young active run, and the
per-generation anti-tight-loop pad in ai_poller.yml is untouched.

STDLIB ONLY, on purpose: a recovery path that needs ``pip install`` is a
recovery path that can fail for the wrong reason. Like scripts/ai_bot_gate.py,
this runs before any dependency setup.

Exit codes: 0 = nothing to do, or a repair was dispatched. 1 = the guard
itself could not do its job (API unreachable, dispatch failed) — that is the
one case worth a red run, because it means nobody is watching the pulse.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

REPO = os.environ.get("GITHUB_REPOSITORY", "")
TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
POLLER_WORKFLOW = os.environ.get("AI_HEARTBEAT_WORKFLOW", "ai_poller.yml")
REF = os.environ.get("AI_HEARTBEAT_REF", "main")

def _env_float(name: str, default: float) -> float:
    """Read a float threshold, defaulting LOUDLY rather than silently.

    GitHub passes an empty string for an unset workflow input, so "" must mean
    "use the default". Anything unparseable is reported: a wrong threshold is
    what decides whether a healthy generation gets cancelled.
    """
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(
            f"WARNING: {name}={raw!r} is not a number — using {default:g}",
            file=sys.stderr,
        )
        return default


# An active generation older than this is not "slow", it is stuck: the job's
# own timeout-minutes is 30, so anything past 40 is anomalous.
STALE_MINUTES = _env_float("AI_HEARTBEAT_STALE_MINUTES", 40.0)
# How long to allow between one generation ending and the next appearing.
GRACE_MINUTES = _env_float("AI_HEARTBEAT_GRACE_MINUTES", 10.0)
LOOKBACK = int(os.environ.get("AI_HEARTBEAT_LOOKBACK", "20"))

# GitHub run statuses that mean "a generation is or will be running".
ACTIVE_STATUSES = {"in_progress", "queued", "requested", "waiting", "pending"}
# Statuses meaning "it is running, not merely waiting for a runner".
RUNNING_STATUSES = {"in_progress"}

API = "https://api.github.com"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def decide_heartbeat(
    now: datetime,
    runs: list[dict],
    stale_minutes: float = STALE_MINUTES,
    grace_minutes: float = GRACE_MINUTES,
) -> dict:
    """What to do about the pulse: a pure function so it can be tested.

    ``runs`` is GitHub's run list (newest first is not assumed). Returns
    ``{"action", "reason", "detail", "run_number", "age_minutes"}`` where
    action is one of ``healthy`` / ``wait`` / ``dispatch`` / ``alert``.
    """
    ordered = sorted(
        runs, key=lambda r: _parse_ts(r.get("created_at")) or now, reverse=True
    )
    active = [r for r in ordered if r.get("status") in ACTIVE_STATUSES]

    if active:
        newest = active[0]
        started = _parse_ts(newest.get("created_at"))
        age = (now - started).total_seconds() / 60.0 if started else 0.0
        number = newest.get("run_number")
        if age < stale_minutes:
            return {
                "action": "healthy", "reason": "active",
                "run_number": number, "age_minutes": round(age, 1),
                "detail": (
                    f"generation #{number} is {newest.get('status')} "
                    f"({age:.0f} min old)"
                ),
            }
        if newest.get("status") in RUNNING_STATUSES:
            return {
                "action": "dispatch", "reason": "hung",
                "run_number": number, "age_minutes": round(age, 1),
                "detail": (
                    f"generation #{number} has been in_progress for {age:.0f} min "
                    f"(>= {stale_minutes:g}); a job past its 30-min timeout is "
                    f"orphaned, not slow"
                ),
            }
        return {
            "action": "alert", "reason": "queued",
            "run_number": number, "age_minutes": round(age, 1),
            "detail": (
                f"generation #{number} has been {newest.get('status')} for "
                f"{age:.0f} min — no runner is picking it up. Dispatching again "
                f"would only stack queued runs"
            ),
        }

    if not ordered:
        return {
            "action": "dispatch", "reason": "missing", "run_number": None,
            "age_minutes": None,
            "detail": "no generation has ever run",
        }

    newest = ordered[0]
    ended = _parse_ts(newest.get("updated_at")) or _parse_ts(newest.get("created_at"))
    age = (now - ended).total_seconds() / 60.0 if ended else 0.0
    number = newest.get("run_number")
    if age >= grace_minutes:
        return {
            "action": "dispatch", "reason": "dead",
            "run_number": number, "age_minutes": round(age, 1),
            "detail": (
                f"no active generation and #{number} ended {age:.0f} min ago "
                f"({newest.get('conclusion')}) with nothing to replace it — "
                f"the self-relaunch chain is broken"
            ),
        }
    return {
        "action": "wait", "reason": "grace", "run_number": number,
        "age_minutes": round(age, 1),
        "detail": (
            f"#{number} just ended ({age:.1f} min ago); the relaunch has "
            f"GRACE_MINUTES={grace_minutes:g} to appear"
        ),
    }


# ---------------------------------------------------------------------------
# GitHub / Telegram plumbing (stdlib urllib only)
# ---------------------------------------------------------------------------

def _api(method: str, path: str, token: str, body: dict | None = None) -> tuple[int, dict]:
    """One GitHub API call. Returns (status, parsed-json-or-{})."""
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:  # noqa: BLE001 — network failure is a result, not a crash
        return 0, {"error": str(e)[:200]}
    try:
        return status, json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return status, {}


def list_poller_runs(token: str, per_page: int = LOOKBACK) -> list[dict]:
    status, data = _api(
        "GET",
        f"/repos/{REPO}/actions/workflows/{POLLER_WORKFLOW}/runs?per_page={per_page}",
        token,
    )
    if status != 200:
        raise RuntimeError(f"poller run list failed: HTTP {status} {data}")
    return data.get("workflow_runs", [])


def dispatch_poller(token: str) -> bool:
    """Ask GitHub for a fresh generation (this also cancels the hung one)."""
    status, _ = _api(
        "POST",
        f"/repos/{REPO}/actions/workflows/{POLLER_WORKFLOW}/dispatches",
        token,
        {"ref": REF},
    )
    return status == 204


def alert(text: str) -> bool:
    """Telegram alert via urllib — never raises, never blocks a repair."""
    token = os.environ.get("AI_TELEGRAM_BOT_TOKEN") or os.environ.get(
        "TELEGRAM_BOT_TOKEN", ""
    )
    chat_id = os.environ.get("AI_TELEGRAM_CHAT_ID") or os.environ.get(
        "TELEGRAM_CHAT_ID", ""
    )
    if not token or not chat_id:
        print("No Telegram secrets — logging the alert instead.", file=sys.stderr)
        return False
    payload = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}
    ).encode()
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=payload
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status == 200
    except Exception as e:  # noqa: BLE001
        print(f"Telegram alert failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Decide and report, but dispatch nothing",
    )
    parser.add_argument(
        "--dispatch", action="store_true",
        help="Force a dispatch regardless of the decision (rehearsals only)",
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Emit the decision as JSON",
    )
    args = parser.parse_args()

    if not REPO or not TOKEN:
        print(
            "GITHUB_REPOSITORY / GH_TOKEN not set — nothing to guard. "
            "(On GitHub Actions both are always present.)",
            file=sys.stderr,
        )
        return 0

    try:
        runs = list_poller_runs(TOKEN)
    except RuntimeError as e:
        # Cannot see the pulse -> cannot honestly call it healthy.
        print(f"HEARTBEAT GUARD: {e}", file=sys.stderr)
        alert(f"⚠️ HEARTBEAT GUARD BLIND\n\n{e}\n\nNobody is watching the pulse.")
        return 1

    decision = decide_heartbeat(datetime.now(timezone.utc), runs)
    action, reason = decision["action"], decision["reason"]

    if args.as_json:
        print(json.dumps(decision, indent=2, default=str))
    else:
        print(f"HEARTBEAT GUARD: {action} ({reason}) — {decision['detail']}")

    if args.dispatch:
        # Rehearsal: prove the repair path end to end against the live pulse
        # without having to break anything first. It dispatches exactly what a
        # real recovery would, so "a new generation actually starts" is
        # verified rather than asserted.
        print("(--dispatch: forcing a repair to exercise the real recovery path)")
        action = "dispatch"

    if action in ("healthy", "wait"):
        return 0

    if action == "alert":
        alert(
            "⚠️ AI BOT HEARTBEAT DEGRADED\n\n"
            f"{decision['detail']}\n\n"
            "No action taken: dispatching again would stack queued runs. "
            "This is usually a GitHub runner shortage and clears itself."
        )
        return 0

    # action == "dispatch"
    if args.dry_run:
        print("(dry-run: no dispatch)")
        return 0

    if dispatch_poller(TOKEN):
        print("Dispatched a fresh poller generation — the chain is recovering.")
        alert(
            "🔧 AI BOT HEARTBEAT RECOVERED\n\n"
            f"{decision['detail']}\n\n"
            "A fresh generation was dispatched automatically; the previous one "
            "(if still alive) is cancelled by the shared concurrency group. "
            "No human action needed."
        )
        return 0

    print("Dispatch FAILED — the pulse is still down.", file=sys.stderr)
    alert(
        "🚨 AI BOT HEARTBEAT DOWN\n\n"
        f"{decision['detail']}\n\n"
        "The automatic repair could not dispatch a new generation "
        "(check the token's actions:write permission). "
        "Manual fix: Actions -> AI Bot Telegram Poller -> Run workflow."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
