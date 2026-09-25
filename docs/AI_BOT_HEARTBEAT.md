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

### 2.1 The hole, caught in the job data

Run **#1446** is the proof, and it is not a hang:

| Step | Ran | Outcome |
|---|---|---|
| 7 Poll Telegram | 06:38:57 → 06:40:11 | `cancelled` (74s into a 20-min poll) |
| 8 Anti-tight-loop pad | 06:40:11 → **06:43:23** | `success` — padded the generation to its 5-min minimum |
| 10 **Relaunch the next generation** | 06:43:24 | **`skipped`** ← the chain lost a link |

A burst of kicks (bot/digest completions, crons, an operator dispatch) makes each
new generation cancel the previous one through the shared concurrency group.
Cancellations cascade, and the temporarily-cancelled run cannot relaunch —
that is fine **while a successor exists**, and a dead chain the moment one does
not. So the guard must not merely *detect* that; it must repair it from the same
completion event.

**Nothing in the observed history hung.** Two things only *looked* like a hang:

* a long-polling step emits no step transitions, so `updated_at` freezes for the
  whole 20-minute poll — it is a normal generation, not a stuck one;
* a cancelled run's pad keeps the concurrency group busy for up to ~3 minutes
  after the cancel, so the successor sits `pending`. That is the intended ≥5-min
  spacing, not a stall: it is what keeps the self-relaunch chain out of a hot
  loop. Do not "fix" it by removing the pad.

### 2.2 A cancelled run can look exactly like a hang (measured)

This was reproduced deliberately on 2026-09-25 (`#1451`, cancelled mid-flight at
06:59:34Z to prove the repair):

| Time (UTC) | What happened |
|---|---|
| 06:59:26 | the cancel lands *during* `pip install`; that step is `cancelled` |
| 06:59:26 | step 7 "Poll Telegram" → **`skipped`**, it never runs at all |
| 06:59:26 → **07:03:48** | step 8's anti-tight-loop pad runs to completion — GitHub honours a cancel only *after* an uninterruptible sleep |
| 07:03:49 | step 10 "Relaunch" → **`skipped`** (the chain loses its link) |
| 07:03:52 | the **run** finally settles `completed/cancelled` |

For those 4.3 minutes the run was `in_progress` with `updated_at` frozen — the
exact signature that gets reported as a stall — while its job was already dead
and its relaunch was never going to happen. `created_at` said nothing about it,
because the run was only 5 minutes old.

**Consequences, both encoded in the guard:**

* age alone is not enough to call a live-looking generation hung — so a frozen
  `updated_at` past the longest legitimate step is its own signal (§3.1);
* a *cancelled* generation is a normal event, not an emergency: the guard is
  kicked by that completion and repairs it in seconds (§3.2).

## 3. The guard (`scripts/ai_heartbeat_guard.py`)

Triggered **independently of the chain** by `.github/workflows/ai_heartbeat_guard.yml`
(6 cron lines, plus `workflow_run` on every poller completion, plus a push
self-check on the guard itself). It reads the poller's run history and decides:

| Situation | Action |
|---|---|
| an active generation younger than `STALE_MINUTES` | **healthy** — do nothing |
| active, `in_progress`, no step transition for `FROZEN_MINUTES` (**stuck**) | **dispatch** a fresh generation |
| active, `in_progress`, older than `STALE_MINUTES` (a **hung** job) | **dispatch** a fresh generation, which also cancels the hung one and frees the concurrency group |
| active but still `queued`/`pending` past `STALE_MINUTES` | **alert only** — a runner shortage, not a hung job; dispatching would just stack queued runs |
| nothing active, last run ended more than `GRACE_MINUTES` ago (**dead chain**) | **dispatch** |
| nothing active, last run ended seconds ago | **wait** — sleep out the remaining grace, then look again (below) |

Thresholds: `STALE_MINUTES = 40`, `FROZEN_MINUTES = 25`, `GRACE_MINUTES = 3` and
`ALERT_QUIET_MINUTES = 20`, all env-overridable (`AI_HEARTBEAT_STALE_MINUTES`,
`AI_HEARTBEAT_FROZEN_MINUTES`, `AI_HEARTBEAT_GRACE_MINUTES`,
`AI_HEARTBEAT_ALERT_QUIET_MINUTES`).

* **40 > the poller job's own `timeout-minutes: 30`.** A job past its timeout is
  orphaned, not slow, so a legitimately long generation can never be mistaken for
  a hung one. (A test asserts this invariant, reading the real YAML.)
* **25 > the Telegram long poll's own `--max-minutes 20`**, with slack, and
  `25 < 40` so the two hang signals are genuinely different rules — one about age,
  one about silence. Both are asserted against the real YAML by
  `test_frozen_threshold_sits_above_the_pollers_own_poll_window`.
* **3 > the seconds** a relaunch dispatch takes to appear, by an order of
  magnitude — small precisely so one run can sleep it out.

A `queued`/`pending` generation is deliberately **not** dispatched at, however
frozen its `updated_at` is: no runner is available, so a new run would only queue
behind the same shortage. It alerts (once) instead.

### 3.1 The freeze signal, and why it is not the age signal

`updated_at` freezing is *normal* — that is the long poll at work — so it is only
evidence when it outlasts what any step can do. In exchange it is
engine-independent: it names a stuck run whose `created_at` is minutes old, which
the age rule cannot see until 40 minutes, and it does not rely on the job timeout
being honoured (see §2.2, where it visibly was not).

### 3.2 The grace re-check (why recovery no longer needs a cron)

The guard is woken by the very event that breaks the chain — poller
`workflow_run: completed` fires for cancellations too. But at that instant the
replacement legitimately does not exist yet, so the first look can only say
**wait**. Deferring to the *next* trigger was the last hole, because the guard's
own crons deliver ~25% of the time here. So instead:

1. it sleeps the remaining grace (at most `RECHECK_CAP_SECONDS = 240`, inside
   its own 6-minute job timeout — a test pins that arithmetic), then
2. looks again, and
3. **if nothing is active it dispatches**, acting on that fact rather than on the
   rounded age arithmetic.

Step 3 matters: `age_minutes` is rounded to 0.1, so a re-check could otherwise
land a second inside the window and exit "cleanly" — silently re-opening the
exact hole. A test (`test_killed_generation_is_repaired_without_another_cron`)
caught that on the first run. Recovery is now bounded by `GRACE_MINUTES` from the
completion event, not by whether a cron lands.

### 3.3 Alert quieting

A runner shortage can leave a generation `queued` for a long time. Alerting on
every 10-minute check would be six messages an hour — the cry-wolf pattern that
trains you to ignore the real alerts. The guard alerts **once**, within
`ALERT_QUIET_MINUTES` of crossing `STALE_MINUTES`, then logs and stays quiet
until the situation changes.

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
* **names a live-but-hung generation** — an `in_progress` pulse older than
  `ai_poller.yml`'s own `timeout-minutes: 30` cannot be healthy, so the pulse
  line says `HUNG: in_progress past the poller job timeout, not slow` and the
  alert title becomes `AI BOT HEALTH ALERT — PULSE HUNG`. The number 30 is that
  job's declared timeout (a fact of the poller), not a policy threshold;
* kicks the guard when it alarms — belt and braces, not a second policy engine.
  The thresholds stay in one file: the guard. It has to: this patrol runs hourly
  and reads the journal, so on its own it would not have noticed a hung pulse
  for hours.

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

The same rehearsal, entirely in the cloud (Actions tab, or `gh workflow run`):

```
AI Bot Heartbeat Guard -> Run workflow
  stale_minutes  = 1     # classify the live generation as hung BY AGE
  frozen_minutes = 10    # ...or as STUCK, if it has been silent that long
  dispatch       = true  # force a repair
  dry_run        = true  # decide and report, change nothing
```

`frozen_minutes` is the one to use on a live pulse: a generation that is 5 minutes
into its poll has been silent for 5 minutes, so passing just under that makes the
real generation the subject of the rehearsal.

`--dry-run` (and `dry_run=true`) decides and reports without dispatching, and
skips the grace sleep, for a safe first look. `--no-recheck` skips only the
sleep-and-look-again. These inputs exist so the repair is **observed** — a new
poller generation must actually start — rather than asserted from tests.

See **§7** for the results of exactly this rehearsal against the live pulse.

## 6. Residual risks (accepted, and stated)

* **Actions disabled for inactivity** (60 days without repo activity) stops
  everything, guard included. Only a human can re-enable it. The watchdog's
  alert names this cause.
* **A repo-wide runner shortage** leaves generations `queued`; the guard alerts
  rather than stacking more work, which is the honest response.
* The guard's own crons are subject to the same ~25% delivery, so they are
  treated as a bonus, not as the mechanism. The two paths that matter do not
  depend on them: a generation that **completed without relaunching** is seen by
  the poller-completion trigger and repaired by the grace re-check in
  ~`GRACE_MINUTES`; a **hung** one is caught by whichever trigger fires next and
  replaced (which also cancels it).

## 7. Verified in the cloud (2026-09-25)

Everything in this section was exercised against the live pulse, not only in
tests.

### 7.1 A deliberately killed generation repairs itself in 7 seconds

The chain was broken on purpose by cancelling the running generation — the
documented way to stop the pulse, and precisely the "killed generation" case.

| Time (UTC) | Event |
|---|---|
| 06:59:34 | both active generations cancelled by hand (API) |
| 06:59:26 → 07:03:48 | #1451's job holds the run `in_progress` through its pad; `updated_at` frozen (§2.2) |
| 07:03:52 | #1451 settles `completed/cancelled`, relaunch step **skipped** — the chain is now genuinely dead |
| **07:03:54** | guard **#14** is created, `event=workflow_run` — kicked by that very completion, not by a cron |
| 07:03:59 | poller **#1453** created, `event=workflow_dispatch` — the repair dispatch |

Guard #14's own log, quoted from the run:

```text
GUARD: dispatch (dead) — no active generation and #1452 ended 4 min ago
(cancelled) with nothing to replace it — the self-relaunch chain is broken
Dispatched a fresh poller generation — the chain is recovering.
```

7 seconds from the dead run settling to a live successor, with no human and no
cron. The 4.3 minutes before it were GitHub holding the cancelled run through an
uninterruptible pad: nothing could have recovered inside that window, and the
guard's job was to be there the instant it closed.

### 7.2 The two states that look identical, told apart

| Rehearsal (real cloud runs) | Result |
|---|---|
| guard against a young active generation | `healthy (active) — generation #1451 is pending (0 min old)` |
| guard against a live long poll, **age rule disabled** (`stale_minutes=9999`) | `dispatch (hung) — generation #1453 is in_progress but has reported no step transition for 9 min (>= 8.72) — longer than the Telegram long poll can hold, so it is stuck, not polling` |
| forced repair (`dispatch=true`) | produced poller generation #1448, `event=workflow_dispatch` |

The second row is the point of §3.1: identical `status`, identical age question,
and the freeze is what answers it. All three ran against the live pulse.

### 7.3 What is still *not* proven

* A freeze longer than `FROZEN_MINUTES` has never been observed in production, so
  that rule has been rehearsed against a live generation and unit-tested with the
  threshold pinned to the real poll window — but it has not yet fired for real.
* The guard's crons still deliver a fraction of their firings. That no longer
  matters for the two cases above (both event-driven), but a pulse that dies
  *without* emitting a completion event would be waiting on them.
* Observed and accepted: a repaired generation starts while the previous one is
  still draining (the concurrency group serialises them), so the handoff is not
  instantaneous — it is bounded by the poller's own anti-tight-loop pad, which is
  what keeps this chain out of a hot loop.
