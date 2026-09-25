"""Tests for the heartbeat guard.

Two things are pinned here:

1. The DECISION, as a pure function: a young active generation is healthy, a
   generation past the poller's own job timeout is hung and must be replaced,
   a merely-queued one is a runner shortage (alert, do not stack), and a chain
   with nothing alive is dead once the grace window has passed.
2. The WIRING, from the actual YAML: the guard is triggered independently of
   the chain it repairs, holds the permissions it needs, and the poller does
   not kick it back (no mutual spin). The thresholds must stay outside the
   normal generation envelope, or the guard would cancel healthy generations.
3. The GRACE RE-CHECK, which is what makes a killed generation recoverable by
   the completion event that woke the guard instead of by a later cron: one
   run must sleep the remaining grace, look again, and dispatch only if the
   pulse is genuinely still dead.
"""

from __future__ import annotations

import importlib.util
import pathlib
from datetime import datetime, timedelta, timezone

import pytest
import yaml

NOW = datetime(2026, 9, 25, 6, 30, tzinfo=timezone.utc)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _hg():
    """Load scripts/ai_heartbeat_guard.py by absolute path (chdir-proof)."""
    script = ROOT / "scripts" / "ai_heartbeat_guard.py"
    spec = importlib.util.spec_from_file_location("ai_heartbeat_guard", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(
    number: int, *, minutes_ago: float, status: str,
    conclusion: str | None = None, ended_after: float | None = None,
    now: datetime = NOW,
) -> dict:
    """A GitHub run record, shaped like the API's.

    Tests that drive ``main()`` must pass ``now=datetime.now(timezone.utc)``:
    main() timestamps its decision with the real clock, so a fixture pinned to
    a fixed NOW would look hours dead to it.
    """
    created = now - timedelta(minutes=minutes_ago)
    if ended_after is None:
        ended = created + timedelta(minutes=21)
    else:
        ended = now - timedelta(minutes=ended_after)
    return {
        "run_number": number,
        "created_at": created.isoformat().replace("+00:00", "Z"),
        "updated_at": ended.isoformat().replace("+00:00", "Z"),
        "status": status,
        "conclusion": conclusion,
    }


class TestDecideHeartbeat:
    def test_young_active_generation_is_healthy(self):
        d = _hg().decide_heartbeat(NOW, [_run(1, minutes_ago=21, status="in_progress")])
        assert d["action"] == "healthy"
        assert d["run_number"] == 1

    def test_generation_past_the_job_timeout_is_hung_and_replaced(self):
        d = _hg().decide_heartbeat(NOW, [_run(7, minutes_ago=55, status="in_progress")])
        assert d["action"] == "dispatch"
        assert d["reason"] == "hung"
        assert "#7" in d["detail"] and "in_progress" in d["detail"]

    def test_stale_queued_run_alerts_instead_of_stacking_more(self):
        """No runner is picking it up — dispatching again would only stack."""
        d = _hg().decide_heartbeat(NOW, [_run(9, minutes_ago=70, status="queued")])
        assert d["action"] == "alert"
        assert d["reason"] == "queued"
        assert "runner" in d["detail"]

    def test_dead_chain_is_restarted_after_the_grace_window(self):
        d = _hg().decide_heartbeat(
            NOW, [_run(3, minutes_ago=40, status="completed",
                       conclusion="cancelled", ended_after=19)],
        )
        assert d["action"] == "dispatch"
        assert d["reason"] == "dead"
        assert "broken" in d["detail"]

    def test_recent_completion_waits_for_the_relaunch(self):
        d = _hg().decide_heartbeat(
            NOW, [_run(4, minutes_ago=5, status="completed",
                       conclusion="cancelled", ended_after=1)],
        )
        assert d["action"] == "wait"
        assert d["reason"] == "grace"

    def test_no_runs_at_all_is_dispatched(self):
        d = _hg().decide_heartbeat(NOW, [])
        assert d["action"] == "dispatch"
        assert d["reason"] == "missing"

    def test_newest_is_found_regardless_of_input_order(self):
        """The API's ordering is not assumed: the hung run may not be first."""
        runs = [
            _run(1, minutes_ago=200, status="completed", conclusion="cancelled"),
            _run(9, minutes_ago=80, status="in_progress"),
            _run(5, minutes_ago=120, status="completed", conclusion="cancelled"),
        ]
        d = _hg().decide_heartbeat(NOW, runs)
        assert d["action"] == "dispatch"
        assert d["run_number"] == 9

    @pytest.mark.parametrize("age,expected", [(39.0, "healthy"), (40.0, "dispatch")])
    def test_stale_boundary_is_inclusive(self, age, expected):
        d = _hg().decide_heartbeat(NOW, [_run(2, minutes_ago=age, status="in_progress")])
        assert d["action"] == expected

    def test_thresholds_default_outside_the_normal_envelope(self):
        """40 min > the poller job's own 30-min timeout (> the ~21-min cycle)."""
        hg = _hg()
        assert hg.STALE_MINUTES > 30
        assert 0 < hg.GRACE_MINUTES <= 15

    def test_decision_is_json_serialisable(self):
        import json

        d = _hg().decide_heartbeat(NOW, [_run(1, minutes_ago=1, status="in_progress")])
        assert json.loads(json.dumps(d, default=str))["action"] == "healthy"


class TestThresholdOverrides:
    """Threshold overrides exist so the recovery path can be rehearsed — so the
    empty and malformed cases must not be able to cancel a healthy pulse."""

    def test_blank_or_missing_env_uses_the_default(self, monkeypatch):
        hg = _hg()
        monkeypatch.setenv("X_THRESH", "")  # GitHub passes "" for unset inputs
        assert hg._env_float("X_THRESH", 40.0) == 40.0
        monkeypatch.delenv("X_THRESH")
        assert hg._env_float("X_THRESH", 40.0) == 40.0

    def test_override_is_honoured(self, monkeypatch):
        hg = _hg()
        monkeypatch.setenv("X_THRESH", "7.5")
        assert hg._env_float("X_THRESH", 40.0) == 7.5

    def test_garbage_warns_and_falls_back(self, monkeypatch, capsys):
        hg = _hg()
        monkeypatch.setenv("X_THRESH", "soon")
        assert hg._env_float("X_THRESH", 40.0) == 40.0
        assert "not a number" in capsys.readouterr().err

    def test_forced_dispatch_exercises_the_repair_even_when_healthy(
        self, monkeypatch, capsys
    ):
        hg = _hg()
        monkeypatch.setattr(hg, "REPO", "o/r")
        monkeypatch.setattr(hg, "TOKEN", "t")
        monkeypatch.setattr(
            hg, "list_poller_runs",
            lambda *a, **k: [_run(1, minutes_ago=1, status="in_progress")],
        )
        calls: list[str] = []
        monkeypatch.setattr(hg, "dispatch_poller", lambda *_: calls.append("x") or True)
        monkeypatch.setattr(hg, "alert", lambda *_: False)
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py", "--dispatch"])
        assert hg.main() == 0
        assert calls == ["x"]
        assert "forcing a repair" in capsys.readouterr().out


class TestGraceRecheck:
    """The hole this closes: a killed generation ends the chain, and the guard
    is kicked by that very completion — but at that instant its replacement has
    not appeared yet. Waiting for the NEXT trigger meant recovery depended on
    crons that deliver ~25% of the time. One run must be enough."""

    @staticmethod
    def _wire(monkeypatch, hg, looks):
        monkeypatch.setattr(hg, "REPO", "o/r")
        monkeypatch.setattr(hg, "TOKEN", "t")
        it = iter(looks)
        monkeypatch.setattr(hg, "list_poller_runs", lambda *a, **k: next(it))
        slept: list[float] = []
        monkeypatch.setattr(hg, "_sleep", lambda s: slept.append(s))
        dispatched: list[str] = []
        monkeypatch.setattr(
            hg, "dispatch_poller", lambda *_: dispatched.append("x") or True
        )
        alerted: list[str] = []
        monkeypatch.setattr(hg, "alert", lambda text: alerted.append(text) or False)
        return slept, dispatched, alerted

    def test_killed_generation_is_repaired_without_another_cron(
        self, monkeypatch, capsys
    ):
        hg = _hg()
        now = datetime.now(timezone.utc)
        killed = [_run(6, minutes_ago=6, status="completed",
                       conclusion="cancelled", ended_after=0.1, now=now)]
        slept, dispatched, alerted = self._wire(monkeypatch, hg, [killed, killed])
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py"])
        assert hg.main() == 0
        assert dispatched == ["x"], "one completion event must be enough"
        assert slept and slept[0] <= hg.RECHECK_CAP_SECONDS
        assert "after the grace" in capsys.readouterr().out
        assert any("RECOVERED" in t for t in alerted)

    def test_recheck_stands_down_when_the_relaunch_appears(self, monkeypatch, capsys):
        hg = _hg()
        now = datetime.now(timezone.utc)
        killed = [_run(6, minutes_ago=6, status="completed",
                       conclusion="cancelled", ended_after=0.1, now=now)]
        successor = [_run(7, minutes_ago=0.5, status="in_progress", now=now),
                     killed[0]]
        slept, dispatched, _alerted = self._wire(monkeypatch, hg, [killed, successor])
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py"])
        assert hg.main() == 0
        assert dispatched == [], "a normal handoff must never be doubled up"
        assert slept, "it should still have watched for it"
        assert "after the grace: healthy" in capsys.readouterr().out

    def test_dry_run_neither_sleeps_nor_dispatches(self, monkeypatch):
        hg = _hg()
        now = datetime.now(timezone.utc)
        killed = [_run(6, minutes_ago=6, status="completed",
                       conclusion="cancelled", ended_after=0.1, now=now)]
        slept, dispatched, _alerted = self._wire(monkeypatch, hg, [killed])
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py", "--dry-run"])
        assert hg.main() == 0
        assert slept == [] and dispatched == []

    def test_recheck_can_be_switched_off_for_rehearsals(self, monkeypatch):
        hg = _hg()
        now = datetime.now(timezone.utc)
        killed = [_run(6, minutes_ago=6, status="completed",
                       conclusion="cancelled", ended_after=0.1, now=now)]
        slept, dispatched, _alerted = self._wire(monkeypatch, hg, [killed])
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py", "--no-recheck"])
        assert hg.main() == 0
        assert slept == [] and dispatched == []

    def test_the_recheck_fits_inside_one_guard_run(self):
        """Grace, cap and the job timeout must be consistent, or the guard is
        killed mid-repair and the hole reopens."""
        hg = _hg()
        assert hg.GRACE_MINUTES * 60 + 5 <= hg.RECHECK_CAP_SECONDS
        timeout = _load("ai_heartbeat_guard.yml")["jobs"]["guard"]["timeout-minutes"]
        assert hg.RECHECK_CAP_SECONDS / 60 + 1 <= timeout


class TestAlertQuieting:
    """A runner shortage must not produce a Telegram message every ten minutes
    — that cry-wolf pattern is what trains you to ignore the real alerts."""

    @staticmethod
    def _run_guard(monkeypatch, hg, minutes_ago: float) -> list[str]:
        now = datetime.now(timezone.utc)
        monkeypatch.setattr(hg, "REPO", "o/r")
        monkeypatch.setattr(hg, "TOKEN", "t")
        monkeypatch.setattr(
            hg, "list_poller_runs",
            lambda *a, **k: [_run(9, minutes_ago=minutes_ago, status="queued", now=now)],
        )
        alerted: list[str] = []
        monkeypatch.setattr(hg, "alert", lambda text: alerted.append(text) or False)
        monkeypatch.setattr(hg, "dispatch_poller", lambda *_: False)
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py"])
        assert hg.main() == 0
        return alerted

    def test_first_detection_alerts(self, monkeypatch):
        hg = _hg()
        alerted = self._run_guard(monkeypatch, hg, hg.STALE_MINUTES + 5)
        assert len(alerted) == 1 and "DEGRADED" in alerted[0]

    def test_persistent_shortage_stops_re_alerting(self, monkeypatch, capsys):
        hg = _hg()
        alerted = self._run_guard(
            monkeypatch, hg, hg.STALE_MINUTES + hg.ALERT_QUIET_MINUTES + 20
        )
        assert alerted == []
        assert "staying quiet" in capsys.readouterr().out

    def test_quieted_shortage_still_dispatches_nothing(self, monkeypatch):
        """Quiet must not become passive: the guard still never stacks runs."""
        hg = _hg()
        called: list[str] = []
        now = datetime.now(timezone.utc)
        monkeypatch.setattr(hg, "REPO", "o/r")
        monkeypatch.setattr(hg, "TOKEN", "t")
        monkeypatch.setattr(
            hg, "list_poller_runs",
            lambda *a, **k: [_run(9, minutes_ago=200, status="queued", now=now)],
        )
        monkeypatch.setattr(hg, "dispatch_poller", lambda *_: called.append("x") or True)
        monkeypatch.setattr(hg, "alert", lambda *_: False)
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py"])
        assert hg.main() == 0
        assert called == []


class TestGuardEntryPoint:
    def test_without_repo_or_token_it_exits_cleanly(self, monkeypatch, capsys):
        hg = _hg()
        monkeypatch.setattr(hg, "REPO", "")
        monkeypatch.setattr(hg, "TOKEN", "")
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py"])
        assert hg.main() == 0
        assert "nothing to guard" in capsys.readouterr().err

    def test_blind_guard_fails_loudly(self, monkeypatch, capsys):
        """If it cannot read the pulse it must NOT call it healthy."""
        hg = _hg()
        monkeypatch.setattr(hg, "REPO", "o/r")
        monkeypatch.setattr(hg, "TOKEN", "t")

        def boom(*_a, **_k):
            raise RuntimeError("poller run list failed: HTTP 500 {}")

        monkeypatch.setattr(hg, "list_poller_runs", boom)
        monkeypatch.setattr(hg, "alert", lambda *_: False)
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py"])
        assert hg.main() == 1
        assert "HTTP 500" in capsys.readouterr().err

    def test_dry_run_repairs_nothing(self, monkeypatch, capsys):
        hg = _hg()
        monkeypatch.setattr(hg, "REPO", "o/r")
        monkeypatch.setattr(hg, "TOKEN", "t")
        monkeypatch.setattr(
            hg, "list_poller_runs",
            lambda *a, **k: [_run(8, minutes_ago=90, status="in_progress")],
        )
        called: list[str] = []
        monkeypatch.setattr(hg, "dispatch_poller", lambda *_: called.append("x") or True)
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py", "--dry-run"])
        assert hg.main() == 0
        assert called == []
        assert "dry-run: no dispatch" in capsys.readouterr().out

    def test_dispatch_failure_is_a_red_run(self, monkeypatch, capsys):
        hg = _hg()
        monkeypatch.setattr(hg, "REPO", "o/r")
        monkeypatch.setattr(hg, "TOKEN", "t")
        monkeypatch.setattr(
            hg, "list_poller_runs",
            lambda *a, **k: [_run(8, minutes_ago=90, status="in_progress")],
        )
        monkeypatch.setattr(hg, "dispatch_poller", lambda *_: False)
        monkeypatch.setattr(hg, "alert", lambda *_: False)
        monkeypatch.setattr("sys.argv", ["ai_heartbeat_guard.py"])
        assert hg.main() == 1
        assert "still down" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Wiring, read from the real YAML
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    path = ROOT / ".github" / "workflows" / name
    assert path.exists(), f"missing workflow: {name}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _steps(workflow: dict) -> str:
    """Every executable payload: ``run:`` strings AND github-script bodies.

    The health check has no ``run:`` steps at all — its logic is in
    ``with.script`` — so reading only ``run`` would silently see nothing.
    """
    out: list[str] = []
    for job in (workflow.get("jobs") or {}).values():
        for step in job.get("steps", []):
            out.append(str(step.get("run", "")))
            out.append(str((step.get("with") or {}).get("script", "")))
    return "\n".join(out)


def _step_names(workflow: dict) -> str:
    return "\n".join(
        str(step.get("name", ""))
        for job in (workflow.get("jobs") or {}).values()
        for step in job.get("steps", [])
    )


class TestHeartbeatWiring:
    def test_guard_workflow_exists_and_runs_the_guard(self):
        wf = _load("ai_heartbeat_guard.yml")
        assert wf["jobs"]
        assert "ai_heartbeat_guard.py" in _steps(wf)

    def test_guard_has_the_permissions_it_needs(self):
        perms = _load("ai_heartbeat_guard.yml").get("permissions", {})
        assert perms.get("actions") == "write"  # to dispatch a replacement
        assert perms.get("contents") in ("read", None)

    def test_guard_is_triggered_independently_of_the_chain(self):
        """It must run even when the chain is silent, so a cron AND a direct
        dispatch are both required."""
        on = _load("ai_heartbeat_guard.yml")[True]  # YAML parses `on:` as True
        assert "schedule" in on
        assert "workflow_dispatch" in on
        assert "AI Bot Telegram Poller" in on["workflow_run"]["workflows"]
        assert on["workflow_run"]["types"] == ["completed"]

    def test_cron_lines_are_dense_because_github_drops_them(self):
        on = _load("ai_heartbeat_guard.yml")[True]
        crons = [c["cron"] for c in on["schedule"]]
        assert len(crons) >= 4, "GitHub delivers only ~25% of scheduled firings"

    def test_no_mutual_spin_between_guard_and_poller(self):
        """The poller must NOT kick the guard; one direction only, or the pair
        spins. The guard's repair is an explicit dispatch, not an event."""
        poller = _load("ai_poller.yml")[True]
        kickers = (poller.get("workflow_run") or {}).get("workflows") or []
        assert "AI Bot Heartbeat Guard" not in kickers

    def test_poller_keeps_its_anti_tight_loop_pad_and_manual_stop(self):
        wf = _load("ai_poller.yml")
        assert "Anti-tight-loop pad" in _step_names(wf)
        # Cancelling must still be able to stop the chain (the guard is what
        # makes that safe to do), so the relaunch stays gated on !cancelled().
        relaunch = [
            s for job in wf["jobs"].values() for s in job.get("steps", [])
            if "Relaunch" in str(s.get("name", ""))
        ]
        assert relaunch and "!cancelled()" in str(relaunch[0].get("if"))

    def test_guard_thresholds_exceed_the_poller_job_timeout(self):
        """A legitimate long generation must never look hung."""
        hg = _hg()
        timeout = _load("ai_poller.yml")["jobs"]["poll"]["timeout-minutes"]
        assert hg.STALE_MINUTES > timeout

    def test_recovery_path_can_be_rehearsed_from_the_workflow(self):
        """The inputs are what make §5 of the heartbeat doc runnable, so the
        recovery path is exercised for real rather than asserted."""
        on = _load("ai_heartbeat_guard.yml")[True]
        inputs = on["workflow_dispatch"]["inputs"]
        assert {"stale_minutes", "dispatch", "dry_run"} <= set(inputs)

    def test_health_check_can_see_the_pulse(self):
        """It used to read only the journal, so a dead or hung pulse was
        invisible to it. It now reports the poller's state and kicks the
        guardian, while the guard keeps sole ownership of the thresholds."""
        wf = _load("ai_health_check.yml")
        assert wf.get("permissions", {}).get("actions") == "write"
        joined = _steps(wf)
        assert "ai_poller.yml" in joined
        assert "ai_heartbeat_guard.yml" in joined

    def test_health_check_can_NAME_a_live_but_hung_generation(self):
        """The watchdog's blind spot was a pulse that is alive but stuck. Its
        hour-long journal threshold would not have fired for hours, so a hung
        generation has to be named in the report itself."""
        joined = _steps(_load("ai_health_check.yml"))
        assert "HUNG" in joined
        assert "in_progress" in joined
        # ... while naming it in the alert is all it does: the repair stays with
        # the guard, whose own triggers fire far more often than this patrol.
        assert "createWorkflowDispatch" in joined
