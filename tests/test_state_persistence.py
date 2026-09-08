"""State-persistence regression tests.

Pins the two guards born from the 2026-09-08 incident chain:

1. **Cache-era failures that must never recur.** Run #114's watchdog
   finding caused the cache post-save to be SKIPPED — silently discarding
   the run's state. Run #116's restore-keys miss restarted the bot from a
   fresh $97 while trading was under way. The workflow now persists to a
   git branch: these tests pin that structural choice.

2. **The fresh-start tripwire.** If paper_state.json is missing while
   run_history.jsonl shows prior trading, the bot must refuse to trade —
   silently fabricating a second day zero would strand real positions and
   fork the ledger (exactly what run #116 did before the tripwire existed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.paper_trader as pt  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Workflow structure: git-backed state, gate placement, persist-on-failure
# ---------------------------------------------------------------------------

class TestWorkflowStatePersistence:
    @pytest.fixture(scope="class")
    def bot_yml(self):
        return yaml.safe_load(Path(".github/workflows/bot.yml").read_text())

    def _run_bot_steps(self, bot_yml):
        return bot_yml["jobs"]["run-bot"]["steps"]

    def _step(self, bot_yml, name):
        for s in self._run_bot_steps(bot_yml):
            if s.get("name") == name:
                return s
        raise AssertionError(f"step {name!r} not found in run-bot job")

    def test_no_actions_cache_for_state(self, bot_yml):
        caches = [s for s in self._run_bot_steps(bot_yml)
                  if str(s.get("uses", "")).startswith("actions/cache")]
        assert not caches, (
            "actions/cache must not carry trading state anymore: immutable "
            "entries, best-effort restore-keys, and post-save-skipped-on-"
            "failure lost runs #114/#116's state"
        )

    def test_state_restored_from_bot_state_branch(self, bot_yml):
        step = self._step(bot_yml, "Restore state from bot-state branch")
        assert "bot-state" in step["run"]

    def test_gate_runs_before_python_setup(self, bot_yml):
        names = [s.get("name") for s in self._run_bot_steps(bot_yml)]
        assert names.index("Schedule gate (dedupe redundant firings)") \
            < names.index("Set up Python"), (
            "the gate must run before pip setup so skipped redundant "
            "firings cost ~15s, not minutes of quota"
        )

    def test_gate_output_gates_trading_steps(self, bot_yml):
        for name in ("Run paper trading bot", "Watchdog (production anomalies)",
                     "Persist state to bot-state branch"):
            step = self._step(bot_yml, name)
            cond = str(step.get("if", ""))
            assert "gate.outputs.run != 'false'" in cond, (
                f"{name} must be skipped when the gate decides redundant"
            )

    def test_persist_runs_even_when_job_failing(self, bot_yml):
        step = self._step(bot_yml, "Persist state to bot-state branch")
        cond = str(step.get("if", ""))
        assert cond.strip().startswith("always()"), (
            "persist must run with always(): the #114 bug was state "
            "discarded because a later step failed the job"
        )

    def test_watchdog_failure_cannot_skip_persist(self, bot_yml):
        # Order matters: persist sits after watchdog but carries always(),
        # so a watchdog exit 1 still reaches it. This is the exact
        # regression that lost run #114's funding charge.
        names = [s.get("name") for s in self._run_bot_steps(bot_yml)]
        assert names.index("Watchdog (production anomalies)") \
            < names.index("Persist state to bot-state branch")
        persist = self._step(bot_yml, "Persist state to bot-state branch")
        assert "always()" in str(persist.get("if", ""))

    def test_persist_covers_all_history_files(self, bot_yml):
        step = self._step(bot_yml, "Persist state to bot-state branch")
        for required in ("paper_state.json", "paper_trades.jsonl",
                         "run_history.jsonl", "position_tracker.json",
                         "watchdog_last_ok.json"):
            assert required in step["run"], (
                f"{required} missing from persist file list"
            )

    def test_redundant_crons_exist(self, bot_yml):
        crons = bot_yml[True]["schedule"] if True in bot_yml \
            else bot_yml["on"]["schedule"]
        exprs = [c["cron"] for c in crons]
        assert len(exprs) >= 2, (
            "a single cron slot gets dropped wholesale (verified 5-7h gaps); "
            "redundant firings + gate are the mitigation"
        )

    def test_weekly_chart_reads_bot_state_branch(self):
        d = yaml.safe_load(
            Path(".github/workflows/weekly_chart.yml").read_text())
        steps = d["jobs"]["weekly-chart"]["steps"]
        restore = [s for s in steps if s.get("name") == "Restore paper trading state"]
        assert restore and "bot-state" in restore[0]["run"]
        caches = [s for s in steps
                  if str(s.get("uses", "")).startswith("actions/cache")]
        assert not caches


# ---------------------------------------------------------------------------
# 2. The fresh-start tripwire
# ---------------------------------------------------------------------------

class TestFreshStartTripwire:
    def test_missing_state_with_history_suspected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pt, "STATE_FILE", tmp_path / "paper_state.json")
        log = tmp_path / "run_history.jsonl"
        log.write_text('{"timestamp": "t", "status": "success"}\n',
                       encoding="utf-8")
        monkeypatch.setattr(pt, "RUN_LOG", log)
        assert pt.fresh_start_suspected() is True

    def test_missing_state_without_history_is_clean_fresh_start(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pt, "STATE_FILE", tmp_path / "paper_state.json")
        monkeypatch.setattr(pt, "RUN_LOG", tmp_path / "run_history.jsonl")
        assert pt.fresh_start_suspected() is False

    def test_existing_state_is_never_suspected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pt, "STATE_FILE", tmp_path / "paper_state.json")
        (tmp_path / "paper_state.json").write_text("{}", encoding="utf-8")
        (tmp_path / "run_history.jsonl").write_text("x\n", encoding="utf-8")
        monkeypatch.setattr(pt, "RUN_LOG", tmp_path / "run_history.jsonl")
        assert pt.fresh_start_suspected() is False

    def test_empty_history_file_is_clean(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pt, "STATE_FILE", tmp_path / "paper_state.json")
        log = tmp_path / "run_history.jsonl"
        log.write_text("", encoding="utf-8")
        monkeypatch.setattr(pt, "RUN_LOG", log)
        assert pt.fresh_start_suspected() is False

    def test_gate_skips_only_in_ci(self, monkeypatch):
        # The in-bot gate is defense-in-depth for redundant crons. It must
        # NEVER silence a local/manual run: only an explicit CI marker may
        # enable the skip path.
        called = {"n": 0}

        def fake_gate():
            called["n"] += 1
            return False, "would skip"

        monkeypatch.setattr(pt, "_gate_should_run", fake_gate)
        monkeypatch.setenv("GITHUB_ACTIONS", "false")
        assert pt._schedule_gate_allows() is True
        assert called["n"] == 0
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        assert pt._schedule_gate_allows() is False
        assert called["n"] == 1
