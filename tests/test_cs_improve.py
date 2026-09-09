"""Task 7 (plan MD): reporting, shared state, deterministic improvement runner.

Pinned here:
- StateLock is a lease: a second acquire fails closed while held, an expired
  lease can be taken over, and every holder has a run ID;
- reports render aggregates only — no secrets can appear in output;
- ``improve --quick --report-only`` is deterministic and network-FREE:
  patching socket.socket to explode does not break the run, and no
  promotion, state push, or Telegram send happens.
"""

from __future__ import annotations

import socket
import time
from unittest import mock

import pytest
from typer.testing import CliRunner

from crypto_system.reporting.dashboard import render_weekly_report
from crypto_system.state_sync import LockHeld, StateLock, new_run_id


class TestStateLock:
    def test_exclusive_acquire_fails_closed(self, tmp_path):
        lock_path = tmp_path / "state.lock"
        with StateLock(lock_path, owner="runner-A", ttl_seconds=60):
            with pytest.raises(LockHeld):
                StateLock(lock_path, owner="runner-B", ttl_seconds=60).acquire()

    def test_release_allows_reacquire(self, tmp_path):
        lock_path = tmp_path / "state.lock"
        lock = StateLock(lock_path, owner="runner-A", ttl_seconds=60)
        with lock:
            pass
        with StateLock(lock_path, owner="runner-B", ttl_seconds=60):
            pass  # no conflict after release

    def test_expired_lease_can_be_taken_over(self, tmp_path):
        lock_path = tmp_path / "state.lock"
        dead = StateLock(lock_path, owner="ghost", ttl_seconds=0)
        dead.acquire()
        time.sleep(0.05)  # lease TTL is 0 -> instantly expired
        with StateLock(lock_path, owner="new", ttl_seconds=60):
            pass

    def test_run_id_recorded_and_unique(self):
        a, b = new_run_id(), new_run_id()
        assert a != b and len(a) >= 16


class TestReportRedaction:
    def test_report_never_contains_secret_values(self, tmp_path):
        state = {
            "equity": 10_450.0,
            "start_equity": 10_000.0,
            "positions": {"BTCUSDT": {"qty": 0.1, "entry": 100.0}},
            "n_trades": 12,
            "api_key": "AKIA-SHOULD-NOT-APPEAR",
            "nested": {"telegram_bot_token": "SECRET-TOKEN"},
        }
        report = render_weekly_report(state)
        assert "AKIA-SHOULD-NOT-APPEAR" not in report
        assert "SECRET-TOKEN" not in report
        assert "10,450" in report  # aggregates still visible

    def test_report_sections(self):
        report = render_weekly_report(
            {"equity": 10_100.0, "start_equity": 10_000.0, "n_trades": 5,
             "positions": {}, "halts": [], "vetoes": []}
        )
        for section in ("Equity", "Trades", "Positions", "Halts", "Vetoes"):
            assert section in report


class TestQuickRunner:
    def _run(self, monkeypatch, *args):
        from scripts.improve import app

        def _no_network(*a, **kw):  # noqa: ANN002, ANN003
            raise AssertionError("network access attempted in quick mode")

        monkeypatch.setattr(socket, "socket", _no_network)
        sent: list[str] = []
        monkeypatch.setattr(
            "crypto_system.reporting.telegram.TelegramReporter.send",
            lambda self, text, **kw: sent.append(text) or True,
        )
        promoted: list[str] = []
        monkeypatch.setattr(
            "crypto_system.research.league.League.run",
            lambda self, candidates, df: promoted.append("LEAGUE-RAN") or mock.Mock(
                promoted=None, vetoed=True, rejected={}, fold_details={}
            ),
        )
        result = CliRunner().invoke(app, ["--quick", "--report-only"])
        return result, sent, promoted

    def test_quick_report_only_runs_without_network(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result, sent, promoted = self._run(monkeypatch)
        assert result.exit_code == 0, result.output
        assert "WEEKLY REPORT" in result.output

    def test_quick_mode_never_promotes_or_sends(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _result, sent, promoted = self._run(monkeypatch)
        assert sent == []  # no Telegram send
        assert promoted == []  # league promotion pipeline never executes

    def test_quick_mode_is_deterministic(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r1, _, _ = self._run(monkeypatch)
        out1 = r1.output
        r2, _, _ = self._run(monkeypatch)
        assert out1 == r2.output
