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
  dispatch  an active generation that is hung — either in_progress past
            STALE_MINUTES (older than the poller job's own 30-minute timeout),
            or in_progress with no step transition for FROZEN_MINUTES (> the
            longest legitimate step, the 20-minute Telegram long poll) — or no
            active generation and the last one ended more than GRACE_MINUTES
            ago (a dead chain) -> dispatch a fresh generation, which also
            cancels the hung one and frees the concurrency group
  alert     an active generation older than STALE_MINUTES but still ``queued``
            -> a runner shortage, not a hung job: dispatching would just stack
            more queued runs, so alert and wait for capacity

Thresholds are deliberately outside the normal envelope: STALE_MINUTES (40)
exceeds the poller job's own 30-minute timeout, so a legitimately long
generation is never mistaken for a hung one, and GRACE_MINUTES (3) is an order
of magnitude wider than the seconds the relaunch dispatch takes to appear.

WHY A FROZEN ``updated_at`` IS ALSO A HANG SIGNAL
-------------------------------------------------
``updated_at`` freezing is NORMAL: a step that blocks — the Telegram long poll
— reports no transitions for up to 20 minutes. But a freeze LONGER than any
step can legitimately last is engine-independent evidence of a stuck
``in_progress`` run, and it does not depend on ``created_at`` at all. It exists
because ``created_at`` alone can only fire at STALE_MINUTES, and because the
Actions API can report a run as ``in_progress`` long after its job is gone:
on 2026-09-25 a generation whose job had been CANCELLED sat ``in_progress`` with
a frozen ``updated_at`` for 4.3 min while the runner finished an
uninterruptible anti-tight-loop sleep before honouring the cancel. That
generation was never coming back, and nothing about its age said so.

WHY "wait" RE-CHECKS IN THE SAME RUN
------------------------------------
A generation that ends WITHOUT reaching its relaunch step — cancelled, lost
runner, failed dispatch — is what breaks the chain, and the guard is kicked by
exactly that event (poller ``workflow_run`` completed fires for cancellations
too). At that instant the replacement legitimately does not exist yet, so the
first look can only ever say "wait". Deferring to the NEXT guard trigger was the
remaining hole: its own crons deliver only ~25% of their firings on this repo
(13 of the last 100 poller runs were ``schedule``), which is the very
unreliability the guard exists to remove. So the run sleeps out the remaining
grace and looks again: one completion event is enough, and recovery is bounded
by GRACE_MINUTES instead of by whether a cron lands.

Bounded, and never a hot loop: a dispatch makes the next check see a young
active generation, the poller does not kick this workflow back, and the poller
keeps its own per-generation anti-tight-loop pad.

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
import time
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
# A run that is in_progress but has reported no step transition for longer than
# the longest step can legitimately run is stuck, whatever created_at says. The
# longest legitimate step is the poller's Telegram long poll (--max-minutes 20),
# so this stays comfortably above it: freezing IS normal during a long poll, and
# only a freeze longer than a poll can last is evidence. A test reads that
# window from the real YAML; do not raise this above STALE_MINUTES, or the two
# rules collapse into one.
FROZEN_MINUTES = _env_float("AI_HEARTBEAT_FROZEN_MINUTES", 25.0)
# How long to allow between one generation ending and the next appearing. The
# relaunch dispatch shows up in seconds, so 3 min is already generous; keeping
# it small is what lets ONE run sleep it out and re-check.
GRACE_MINUTES = _env_float("AI_HEARTBEAT_GRACE_MINUTES", 3.0)
LOOKBACK = int(os.environ.get("AI_HEARTBEAT_LOOKBACK", "20"))
# A re-check sleeps at most this long, which must exceed GRACE_MINUTES * 60
# (+ rounding) and stay well inside the guard job's own timeout-minutes.
RECHECK_CAP_SECONDS = _env_float("AI_HEARTBEAT_RECHECK_CAP_SECONDS", 240.0)
# A generation stuck ``queued`` past STALE_MINUTES alerts once. After this many
# further minutes the guard goes quiet (the log still records it every run), so
# a runner shortage cannot become a Telegram message every ten minutes — that
# cry-wolf pattern is what trains you to ignore the real alerts.
ALERT_QUIET_MINUTES = _env_float("AI_HEARTBEAT_ALERT_QUIET_MINUTES", 20.0)

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
    frozen_minutes: float = FROZEN_MINUTES,
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
        number = newest.get("run_number")
        status = newest.get("status")
        started = _parse_ts(newest.get("created_at"))
        touched = _parse_ts(newest.get("updated_at")) or started
        age = (now - started).total_seconds() / 60.0 if started else 0.0
        frozen = (now - touched).total_seconds() / 60.0 if touched else 0.0

        if status in RUNNING_STATUSES:
            if frozen >= frozen_minutes and age < stale_minutes:
                return {
                    "action": "dispatch", "reason": "hung",
                    "run_number": number, "age_minutes": round(age, 1),
                    "frozen_minutes": round(frozen, 1),
                    "detail": (
                        f"generation #{number} is in_progress but has reported "
                        f"no step transition for {frozen:.0f} min "
                        f"(>= {frozen_minutes:g}) — longer than the Telegram "
                        f"long poll can hold, so it is stuck, not polling"
                    ),
                }
            if age >= stale_minutes:
                return {
                    "action": "dispatch", "reason": "hung",
                    "run_number": number, "age_minutes": round(age, 1),
                    "frozen_minutes": round(frozen, 1),
                    "detail": (
                        f"generation #{number} has been in_progress for "
                        f"{age:.0f} min (>= {stale_minutes:g}); a job past its "
                        f"30-min timeout is orphaned, not slow"
                    ),
                }
            return {
                "action": "healthy", "reason": "active",
                "run_number": number, "age_minutes": round(age, 1),
                "frozen_minutes": round(frozen, 1),
                "detail": (
                    f"generation #{number} is {status} "
                    f"({age:.0f} min old, last step transition "
                    f"{frozen:.0f} min ago)"
                ),
            }
        if age < stale_minutes:
            return {
                "action": "healthy", "reason": "active",
                "run_number": number, "age_minutes": round(age, 1),
                "frozen_minutes": round(frozen, 1),
                "detail": (
                    f"generation #{number} is {status} "
                    f"({age:.0f} min old)"
                ),
            }
        return {
            "action": "alert", "reason": "queued",
            "run_number": number, "age_minutes": round(age, 1),
            "frozen_minutes": round(frozen, 1),
            "detail": (
                f"generation #{number} has been {status} for "
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


def _sleep(seconds: float) -> None:
    """Sleep, isolated in one place so tests never actually wait."""
    if seconds > 0:
        time.sleep(seconds)


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
    parser.add_argument(
        "--no-recheck", action="store_true",
        help="Do not sleep out the grace and re-check (rehearsals)",
    )
    args = parser.parse_args()

    if not REPO or not TOKEN:
        print(
            "GITHUB_REPOSITORY / GH_TOKEN not set — nothing to guard. "
            "(On GitHub Actions both are always present.)",
            file=sys.stderr,
        )
        return 0

    def refresh() -> dict:
        return decide_heartbeat(datetime.now(timezone.utc), list_poller_runs(TOKEN))

    try:
        decision = refresh()
    except RuntimeError as e:
        # Cannot see the pulse -> cannot honestly call it healthy.
        print(f"HEARTBEAT GUARD: {e}", file=sys.stderr)
        alert(f"⚠️ HEARTBEAT GUARD BLIND\n\n{e}\n\nNobody is watching the pulse.")
        return 1

    if args.as_json:
        print(json.dumps(decision, indent=2, default=str))
    else:
        print(
            f"HEARTBEAT GUARD: {decision['action']} ({decision['reason']}) — "
            f"{decision['detail']}"
        )

    # Sleep out the grace and look again, so the completion event that woke us
    # is enough on its own (see the module docstring). Skipped in a dry run:
    # a rehearsal reports, it does not spend minutes.
    if decision["action"] == "wait" and not args.no_recheck and not args.dry_run:
        remaining = max(0.0, GRACE_MINUTES - float(decision.get("age_minutes") or 0.0))
        nap = min(remaining * 60.0 + 5.0, RECHECK_CAP_SECONDS)
        print(
            f"Re-checking in {nap:.0f}s — the relaunch gets "
            f"GRACE_MINUTES={GRACE_MINUTES:g} to appear"
        )
        _sleep(nap)
        try:
            decision = refresh()
        except RuntimeError as e:
            print(f"HEARTBEAT GUARD: {e}", file=sys.stderr)
            alert(f"⚠️ HEARTBEAT GUARD BLIND\n\n{e}\n\nNobody is watching the pulse.")
            return 1
        if decision["action"] == "wait":
            # The grace has now been slept out and nothing is active, so this is
            # not a slow handoff. Act on the fact rather than on the age
            # arithmetic: ``age_minutes`` is rounded, so a re-check can
            # otherwise land a second inside the window and exit cleanly —
            # re-opening exactly the hole this run exists to close.
            decision = {
                "action": "dispatch",
                "reason": "dead",
                "run_number": decision.get("run_number"),
                "age_minutes": decision.get("age_minutes"),
                "detail": (
                    f"{decision['detail']} — and still nothing active after "
                    f"sleeping the grace out"
                ),
            }
        print(
            f"HEARTBEAT GUARD after the grace: {decision['action']} "
            f"({decision['reason']}) — {decision['detail']}"
        )

    action = decision["action"]

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
        age = float(decision.get("age_minutes") or 0.0)
        if age - STALE_MINUTES > ALERT_QUIET_MINUTES:
            print(
                f"(already alerted for this generation — {age:.0f} min old, "
                f"{age - STALE_MINUTES:.0f} min past STALE_MINUTES — staying "
                "quiet so a runner shortage is not a message every ten minutes)"
            )
            return 0
        alert(
            "⚠️ AI BOT HEARTBEAT DEGRADED\n\n"
            f"{decision['detail']}\n\n"
            "No action taken: dispatching again would stack queued runs. "
            "This is usually a GitHub runner shortage and clears itself. "
            "If it persists past "
            f"{STALE_MINUTES + ALERT_QUIET_MINUTES:g} min the guard stops "
            "re-alerting (the Actions tab still shows it)."
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
