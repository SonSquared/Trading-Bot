"""Schedule-gate regression tests.

GitHub cron drops slots on free-tier private repos (verified 2026-09-07/08:
a 2-hourly schedule produced 5-7h gaps). The mitigation is redundant cron
firings plus this gate, which must guarantee exactly one thing: two bot
runs can never both trade within MIN_RUN_INTERVAL_MINUTES of each other.

The gate is the load-bearing piece of duplicate-open protection now, so
these tests pin its decision table, not its formatting.
"""

from __future__ import annotations

import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import schedule_gate  # noqa: E402

importlib.reload(schedule_gate)

NOW = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)


def _write_run_log(tmp_path: Path, entries: list[dict]) -> Path:
    log = tmp_path / "run_history.jsonl"
    log.write_text(
        "\n".join(
            __import__("json").dumps(e) for e in entries
        ) + "\n",
        encoding="utf-8",
    )
    return log


def _entry(hours_ago: float, status: str = "success") -> dict:
    ts = NOW - timedelta(hours=hours_ago)
    return {"timestamp": ts.isoformat(), "status": status,
            "duration_seconds": 12.0, "trades": 0, "errors": [],
            "equity": 97.0}


class TestGateDecisionTable:
    def test_no_history_means_run(self, tmp_path):
        run, reason = schedule_gate.should_run(
            now=NOW, run_log=tmp_path / "missing.jsonl")
        assert run is True
        assert "no successful run" in reason

    def test_recent_success_skips(self, tmp_path):
        # 30 minutes ago < 100-minute interval -> skip.
        log = _write_run_log(tmp_path, [_entry(0.5)])
        run, reason = schedule_gate.should_run(now=NOW, run_log=log)
        assert run is False
        assert "skip" in reason

    def test_old_success_runs(self, tmp_path):
        # 2 hours ago > 100-minute interval -> run.
        log = _write_run_log(tmp_path, [_entry(2.0)])
        run, reason = schedule_gate.should_run(now=NOW, run_log=log)
        assert run is True

    def test_failed_runs_do_not_block(self, tmp_path):
        # A failed run 5 minutes ago is NOT a success: the gate must still
        # allow a retry firing to trade (otherwise one failure would idle
        # the bot for the rest of the interval).
        log = _write_run_log(tmp_path, [_entry(5 / 60, status="failed")])
        run, _ = schedule_gate.should_run(now=NOW, run_log=log)
        assert run is True

    def test_latest_entry_wins(self, tmp_path):
        # Old success followed by recent success -> skip (newest counts).
        log = _write_run_log(tmp_path, [_entry(3.0), _entry(0.5)])
        run, _ = schedule_gate.should_run(now=NOW, run_log=log)
        assert run is False

    def test_recent_success_then_failure_still_sees_success(self, tmp_path):
        log = _write_run_log(tmp_path, [_entry(0.5, status="failed"),
                                        _entry(2.0, status="success")])
        # Latest entry is a failure; the gate scans for the newest SUCCESS,
        # which is 2h old -> run.
        run, _ = schedule_gate.should_run(now=NOW, run_log=log)
        assert run is True

    def test_corrupt_lines_are_skipped(self, tmp_path):
        log = tmp_path / "run_history.jsonl"
        log.write_text("not json\n{\"timestamp\": \"garbage\"}\n"
                       + __import__("json").dumps(_entry(0.5)) + "\n",
                       encoding="utf-8")
        run, _ = schedule_gate.should_run(now=NOW, run_log=log)
        assert run is False

    def test_custom_interval_respected(self, tmp_path):
        log = _write_run_log(tmp_path, [_entry(1.0)])  # 60 min ago
        # 60 < 90 -> skip; 60 >= 55 -> run.
        assert schedule_gate.should_run(now=NOW, run_log=log,
                                        min_interval_min=90)[0] is False
        assert schedule_gate.should_run(now=NOW, run_log=log,
                                        min_interval_min=55)[0] is True


class TestGateExitCodes:
    def test_main_returns_3_on_skip(self, tmp_path, capsys):
        log = _write_run_log(tmp_path, [_entry(0.5)])
        assert schedule_gate.main(run_log=log, now=NOW) == schedule_gate.SKIP_EXIT
        out = capsys.readouterr().out
        assert "skip" in out

    def test_main_returns_0_on_run(self, tmp_path):
        log = _write_run_log(tmp_path, [_entry(2.0)])
        assert schedule_gate.main(run_log=log, now=NOW) == 0

    def test_skip_exit_code_is_distinct(self):
        # 3 must not collide with success (0) or generic failure (1): the
        # workflow branch on it to keep the run green while doing nothing.
        assert schedule_gate.SKIP_EXIT == 3
