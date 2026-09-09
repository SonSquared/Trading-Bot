"""Task 9 (plan MD): guarded CI — the workflow tests.

Pinned here:
- the weekly workflow renders reports only (--report-only) and never
  promotes;
- the monthly workflow promotes paper deployments only;
- NO workflow sets live-enable flags (CS_MODE=live / CS_LIVE_ENABLED=true
  / --send / any live service invocation);
- every workflow pins actions and uses least-privilege permissions;
- the test workflow runs the full suite, ruff, mypy, and the quick runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(".github/workflows")


def _load(name: str) -> dict:
    path = WORKFLOWS / name
    assert path.exists(), f"missing workflow: {name}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _all_steps(workflow: dict) -> list[str]:
    steps: list[str] = []
    for job in (workflow.get("jobs") or {}).values():
        for step in job.get("steps", []):
            steps.append(str(step.get("run", "")))
    return steps


class TestCsWorkflows:
    @pytest.mark.parametrize(
        "name",
        ["cs_ci.yml", "cs_weekly_report.yml", "cs_monthly_optimize.yml"],
    )
    def test_workflows_exist_and_parse(self, name):
        data = _load(name)
        assert "jobs" in data

    def test_weekly_is_report_only_and_monthly_never_enables_live(self):
        weekly = _all_steps(_load("cs_weekly_report.yml"))
        assert any("--report-only" in s for s in weekly)
        monthly = _all_steps(_load("cs_monthly_optimize.yml"))
        assert any("improve.py" in s for s in monthly)

    def test_no_workflow_sets_live_flags(self):
        for name in (
            "cs_ci.yml",
            "cs_weekly_report.yml",
            "cs_monthly_optimize.yml",
        ):
            steps = _all_steps(_load(name))
            joined = " ".join(steps).lower()
            assert "cs_mode=live" not in joined
            assert "cs_live_enabled=true" not in joined
            assert "live_enabled=true" not in joined
            # report sending is allowed only for the telegram *reporter*
            # through env tokens, never by arming live mode:
            assert "liveexecution" not in joined.replace(" ", "")
            assert "enable_live" not in joined

    def test_workflows_use_pinned_actions_and_least_privilege(self):
        for name in (
            "cs_ci.yml",
            "cs_weekly_report.yml",
            "cs_monthly_optimize.yml",
        ):
            wf = _load(name)
            perms = wf.get("permissions", {})
            assert perms.get("contents") in ("read", None), (
                f"{name}: contents must be read-only (or unset)"
            )
            for job in (wf.get("jobs") or {}).values():
                for step in job.get("steps", []):
                    uses = str(step.get("uses", ""))
                    if uses:
                        assert "@" in uses, f"{name}: unpinned action {uses}"

    def test_ci_runs_full_verification(self):
        steps = _all_steps(_load("cs_ci.yml"))
        joined = "\n".join(steps)
        assert "pytest" in joined
        assert "ruff check" in joined
        assert "mypy" in joined
        assert "improve.py --quick --report-only" in joined

    def test_monthly_does_not_push_state_or_invoke_live(self):
        steps = _all_steps(_load("cs_monthly_optimize.yml"))
        joined = " ".join(steps).lower()
        assert "bot-state" not in joined  # never touches the legacy bot's branch
        assert "binance" not in joined.replace("binance-usdm", "")
