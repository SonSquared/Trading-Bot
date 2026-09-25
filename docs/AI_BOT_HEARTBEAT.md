# AI Bot — The Heartbeat: diagnosis and guard

**Read this before changing `ai_poller.yml`, `ai_heartbeat_guard.yml` or the
watchdog.** It records what the pulse actually does, what was *wrongly*
diagnosed about it on 2026-09-25, and why the recovery path is shaped the way
it is.

---

## 1. What was claimed, and what the evidence shows

A 2026-09-25 review reported the pulse **stalled for 7+ hours**: poller run
#1440 "`in_progress` since 06:01Z with a frozen `updated_at`". That was wrong.

**#1440 did not hang.** From the run's own job data:

| Fact | Value |
|---|---|
| Job started / completed | 06:01:27Z → 06:22:24Z (20m57s) |
| Steps | **all 10 succeeded**, including step 10 "Relaunch the next generation" |
| Conclusion | `cancelled` — by run #1441, created 06:22:21Z |
| Cycle | 06:01:21 → 06:22:21 = exactly 21 min |

21 minutes is the design: `--max-minutes 20` for the Telegram long-poll plus the
anti-tight-loop pad. Cancelling the previous generation is what the shared
concurrency group (`cancel-in-progress: true`) is *for*. And `updated_at` frozen
at the job's start is simply GitHub not refreshing the run record mid-job.

The seven hours came from comparing a **structlog local timestamp (UTC+7)**
against **UTC** run timestamps. Nothing was stalled; the 08:00 slot had not
happened yet.

**Lesson:** `scripts/*.py` logs are local time, the Actions API is UTC. Before
declaring a stall, read the clock from the source you are comparing against
(`curl -sI https://api.github.com | grep -i ^date`).

## 2. The fragility that is real

The pulse is a **serial chain**: each poller generation dispatches the next as
its LAST step. Therefore:

* a generation that ends **without** reaching step 10 breaks the chain — a
  deliberate cancel (the documented way to stop the chain, since the relaunch is
  gated on `!cancelled()`), a lost runner, or a failed dispatch call;
* the "four backup crons" are **not** a safety net. Over the 100 poller runs
  ending 2026-09-25T06:22Z: `workflow_dispatch` 32, `workflow_run` 55,
  **`schedule` 13** — ~25% delivery, matching how badly GitHub thins this
  repo's crons;
* nothing repaired it. `ai_health_check.yml` read **only the journal**, with an
  8-hour threshold on the last successful wakeup — it never looked at the
  poller, so a dead *or live-but-hung* generation was invisible to it, and even
  on alarm it only sent a Telegram message.

Recovery depended on the thing that had just failed.

## 3. The guard (`scripts/ai_heartbeat_guard.py`)

Triggered **independently of the chain** by `.github/workflows/ai_heartbeat_guard.yml`
(6 cron lines, plus `workflow_run` on every poller completion, plus a push
self-check on the guard itself). It reads the poller's run history and decides:

| Situation | Action |
|---|---|
| an active generation younger than `STALE_MINUTES` | **healthy** — do nothing |
| active, `in_progress`, older than `STALE_MINUTES` (a **hung** job) | **dispatch** a fresh generation, which also cancels the hung one and frees the concurrency group |
| active but still `queued`/`pending` past `STALE_MINUTES` | **alert only** — a runner shortage, not a hung job; dispatching would just stack queued runs |
| nothing active, last run ended more than `GRACE_MINUTES` ago (**dead chain**) | **dispatch** |
| nothing active, last run ended seconds ago | **wait** — the relaunch has the grace window to appear |

Thresholds: `STALE_MINUTES = 40` and `GRACE_MINUTES = 10`, both env-overridable
(`AI_HEARTBEAT_STALE_MINUTES`, `AI_HEARTBEAT_GRACE_MINUTES`).

* **40 > the poller job's own `timeout-minutes: 30`.** A job past its timeout is
  orphaned, not slow, so a legitimately long generation can never be mistaken for
  a hung one. (A test asserts this invariant, reading the real YAML.)
* **10 > the ≤5 min** an anti-tight-loop-padded generation can take, so a normal
  handoff is never read as death.

**No hot loops.** The poller does **not** kick the guard (one direction only:
the guard's repair is an explicit dispatch, not an event), a dispatch makes the
next check see a young active generation, and `ai_poller.yml`'s per-generation
anti-tight-loop pad is untouched. The guard uses its own concurrency group so it
cannot collide with the pulse it watches.

**Stdlib only, on purpose.** Like `scripts/ai_bot_gate.py`, it runs before any
dependency setup: a recovery path that needs `pip install` can fail for the
wrong reason.

## 4. What the watchdog can see now

`ai_health_check.yml` still owns the *trading* signal (no successful wakeup in
8h) and now, additionally:

* reports the poller's state (`#run status/conclusion, started N min ago`) in its
  alert, so the message says **which layer** is down;
* kicks the guard when it alarms — belt and braces, not a second policy engine.
  The thresholds stay in one file: the guard.

## 5. Rehearsing recovery (do this, do not assume it)

Recovery is only real if it has been exercised. Both paths can be rehearsed
without waiting for a real stall, using the env overrides:

```bash
# 1. DEAD CHAIN: stop the chain deliberately, then let the guard restart it.
#    (Cancelling a generation skips its relaunch by design.)
#    Actions -> AI Bot Telegram Poller -> Cancel run   # newest generation
#    wait > GRACE_MINUTES, then:
#    Actions -> AI Bot Heartbeat Guard -> Run workflow  -> expect "dispatch (dead)"

# 2. HUNG JOB: force the guard to classify the *live* generation as hung.
#    Run the guard with the staleness override set low enough that an
#    in-progress generation (normally 21 min) exceeds it:
AI_HEARTBEAT_STALE_MINUTES=1 python3 scripts/ai_heartbeat_guard.py
#    -> "dispatch (hung)": a replacement generation starts and the previous one
#    is cancelled. The pulse is never left down.
```

`--dry-run` decides and reports without dispatching, for a safe first look.

## 6. Residual risks (accepted, and stated)

* **Actions disabled for inactivity** (60 days without repo activity) stops
  everything, guard included. Only a human can re-enable it. The watchdog's
  alert names this cause.
* **A repo-wide runner shortage** leaves generations `queued`; the guard alerts
  rather than stacking more work, which is the honest response.
* The guard's own crons are subject to the same ~25% delivery. Its
  `workflow_run` trigger covers the failure it is most likely to meet (a
  generation that completed without relaunching); the dead-chain case depends on
  crons, so expect recovery in **minutes to a couple of hours**, not seconds.
