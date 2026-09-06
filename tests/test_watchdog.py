"""
Tests for the production watchdog (scripts/watchdog.py).

Every check must fire on its anomaly, stay silent on healthy data, and
never read production files — all paths are redirected into tmp dirs.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scripts.watchdog as wd

NOW = datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# check_missed_runs
# --------------------------------------------------------------------------

class TestMissedRuns:
    def test_no_history_alerts(self, tmp_path):
        msg = wd.check_missed_runs(NOW, run_log=tmp_path / "none.jsonl")
        assert msg and "never completed" in msg

    def test_recent_run_silent(self, tmp_path):
        rl = tmp_path / "run_history.jsonl"
        write_jsonl(rl, [{"timestamp": _iso(NOW - timedelta(minutes=10)),
                          "status": "success"}])
        assert wd.check_missed_runs(NOW, run_log=rl) is None

    def test_old_run_alerts(self, tmp_path):
        rl = tmp_path / "run_history.jsonl"
        # 15min interval * factor 6 = 90min allowed; 10h is way past.
        write_jsonl(rl, [{"timestamp": _iso(NOW - timedelta(hours=10)),
                          "status": "success"}])
        msg = wd.check_missed_runs(NOW, run_log=rl)
        assert msg and "Missed runs" in msg and "10.0h" in msg

    def test_unparseable_timestamp_alerts(self, tmp_path):
        rl = tmp_path / "run_history.jsonl"
        write_jsonl(rl, [{"timestamp": "not-a-date", "status": "success"}])
        assert wd.check_missed_runs(NOW, run_log=rl)

    def test_corrupt_lines_skipped(self, tmp_path):
        rl = tmp_path / "run_history.jsonl"
        rl.write_text("{broken\n" + json.dumps(
            {"timestamp": _iso(NOW - timedelta(minutes=5))}) + "\n", encoding="utf-8")
        assert wd.check_missed_runs(NOW, run_log=rl) is None


# --------------------------------------------------------------------------
# check_stale_state
# --------------------------------------------------------------------------

class TestStaleState:
    def test_missing_file_alerts(self, tmp_path):
        msg = wd.check_stale_state(NOW, state_file=tmp_path / "state.json")
        assert msg and "MISSING" in msg

    def test_corrupt_file_alerts(self, tmp_path):
        sf = tmp_path / "state.json"
        sf.write_text("{not json", encoding="utf-8")
        msg = wd.check_stale_state(NOW, state_file=sf)
        assert msg and "unreadable" in msg

    def test_no_last_run_at_alerts(self, tmp_path):
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps({"cash": 97.0}), encoding="utf-8")
        msg = wd.check_stale_state(NOW, state_file=sf)
        assert msg and "no last_run_at" in msg

    def test_ancient_state_alerts(self, tmp_path):
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(
            {"last_run_at": _iso(NOW - timedelta(hours=30))}), encoding="utf-8")
        msg = wd.check_stale_state(NOW, state_file=sf)
        assert msg and "30h" in msg

    def test_fresh_state_silent(self, tmp_path):
        sf = tmp_path / "state.json"
        sf.write_text(json.dumps(
            {"last_run_at": _iso(NOW - timedelta(minutes=10))}), encoding="utf-8")
        assert wd.check_stale_state(NOW, state_file=sf) is None


# --------------------------------------------------------------------------
# check_ledger_desync
# --------------------------------------------------------------------------

class TestLedgerDesync:
    def test_position_without_logged_open_alerts(self, tmp_path):
        sf = tmp_path / "state.json"
        tl = tmp_path / "trades.jsonl"
        sf.write_text(json.dumps(
            {"positions": {"ETH_USDT_USDT": {"side": 1}}}), encoding="utf-8")
        write_jsonl(tl, [{"action": "OPEN_LONG", "pair": "BTC_USDT_USDT"}])
        msg = wd.check_ledger_desync(NOW, trade_log=tl, state_file=sf)
        assert msg and "ETH_USDT_USDT" in msg

    def test_logged_open_silent(self, tmp_path):
        sf = tmp_path / "state.json"
        tl = tmp_path / "trades.jsonl"
        sf.write_text(json.dumps(
            {"positions": {"ETH_USDT_USDT": {"side": 1}}}), encoding="utf-8")
        write_jsonl(tl, [{"action": "OPEN_LONG", "pair": "ETH_USDT_USDT"}])
        assert wd.check_ledger_desync(NOW, trade_log=tl, state_file=sf) is None

    def test_flat_account_silent(self, tmp_path):
        sf = tmp_path / "state.json"
        tl = tmp_path / "trades.jsonl"
        sf.write_text(json.dumps({"positions": {}}), encoding="utf-8")
        assert wd.check_ledger_desync(NOW, trade_log=tl, state_file=sf) is None


# --------------------------------------------------------------------------
# check_desync_storm
# --------------------------------------------------------------------------

class TestDesyncStorm:
    def _log(self, tmp_path, n, age_h):
        tl = tmp_path / "trades.jsonl"
        write_jsonl(tl, [
            {"action": "CLOSE", "pair": "ETH_USDT_USDT",
             "reason": "stop_loss (ledger desync cleared)",
             "timestamp": _iso(NOW - timedelta(hours=age_h))}
            for _ in range(n)])
        return tl

    def test_storm_alerts(self, tmp_path):
        msg = wd.check_desync_storm(NOW, trade_log=self._log(tmp_path, 3, 1))
        assert msg and "storm" in msg and "3 ledger-desync" in msg

    def test_below_threshold_silent(self, tmp_path):
        assert wd.check_desync_storm(NOW, trade_log=self._log(tmp_path, 2, 1)) is None

    def test_old_events_do_not_count(self, tmp_path):
        # All 3 clears are 72h old — outside the 48h lookback.
        assert wd.check_desync_storm(NOW, trade_log=self._log(tmp_path, 3, 72)) is None


# --------------------------------------------------------------------------
# check_duplicate_opens
# --------------------------------------------------------------------------

class TestDuplicateOpens:
    def _tl(self, tmp_path, rows):
        tl = tmp_path / "trades.jsonl"
        write_jsonl(tl, rows)
        return tl

    def test_duplicates_within_window_alert(self, tmp_path):
        tl = self._tl(tmp_path, [
            {"action": "OPEN_LONG", "pair": "ETH_USDT_USDT",
             "timestamp": _iso(NOW - timedelta(days=2))},
            {"action": "OPEN_LONG", "pair": "ETH_USDT_USDT",
             "timestamp": _iso(NOW - timedelta(days=2) + timedelta(minutes=20))},
        ])
        msg = wd.check_duplicate_opens(NOW, trade_log=tl)
        assert msg and "Duplicate-open" in msg and "ETH_USDT_USDT" in msg

    def test_spaced_opens_silent(self, tmp_path):
        tl = self._tl(tmp_path, [
            {"action": "OPEN_LONG", "pair": "ETH_USDT_USDT",
             "timestamp": _iso(NOW - timedelta(days=3))},
            {"action": "OPEN_LONG", "pair": "ETH_USDT_USDT",
             "timestamp": _iso(NOW - timedelta(days=1))},
        ])
        assert wd.check_duplicate_opens(NOW, trade_log=tl) is None

    def test_old_duplicates_silent(self, tmp_path):
        tl = self._tl(tmp_path, [
            {"action": "OPEN_LONG", "pair": "ETH_USDT_USDT",
             "timestamp": _iso(NOW - timedelta(days=8))},
            {"action": "OPEN_LONG", "pair": "ETH_USDT_USDT",
             "timestamp": _iso(NOW - timedelta(days=8, minutes=-20))},
        ])
        assert wd.check_duplicate_opens(NOW, trade_log=tl) is None

    def test_different_pairs_silent(self, tmp_path):
        tl = self._tl(tmp_path, [
            {"action": "OPEN_LONG", "pair": "ETH_USDT_USDT",
             "timestamp": _iso(NOW - timedelta(hours=1))},
            {"action": "OPEN_LONG", "pair": "BTC_USDT_USDT",
             "timestamp": _iso(NOW - timedelta(hours=1) + timedelta(minutes=5))},
        ])
        assert wd.check_duplicate_opens(NOW, trade_log=tl) is None

    def test_consistency_repair_entries_ignored(self, tmp_path):
        tl = self._tl(tmp_path, [
            {"action": "OPEN_LONG", "pair": "ETH_USDT_USDT",
             "note": "paper-state-consistency repair",
             "timestamp": _iso(NOW - timedelta(hours=1))},
            {"action": "OPEN_LONG", "pair": "ETH_USDT_USDT",
             "note": "paper-state-consistency repair",
             "timestamp": _iso(NOW - timedelta(hours=1) + timedelta(minutes=1))},
        ])
        assert wd.check_duplicate_opens(NOW, trade_log=tl) is None


# --------------------------------------------------------------------------
# check_marker_age
# --------------------------------------------------------------------------

class TestMarkerAge:
    def test_stale_marker_with_fresh_runs_alerts(self, tmp_path):
        sf = tmp_path / "state.json"
        rl = tmp_path / "run_history.jsonl"
        sf.write_text(json.dumps(
            {"last_run_at": _iso(NOW - timedelta(hours=8))}), encoding="utf-8")
        write_jsonl(rl, [{"timestamp": _iso(NOW - timedelta(minutes=10)),
                          "status": "success"}])
        msg = wd.check_marker_age(NOW, state_file=sf, run_log=rl)
        assert msg and "stale" in msg.lower()

    def test_aligned_marker_silent(self, tmp_path):
        sf = tmp_path / "state.json"
        rl = tmp_path / "run_history.jsonl"
        sf.write_text(json.dumps(
            {"last_run_at": _iso(NOW - timedelta(minutes=10))}), encoding="utf-8")
        write_jsonl(rl, [{"timestamp": _iso(NOW - timedelta(minutes=10)),
                          "status": "success"}])
        assert wd.check_marker_age(NOW, state_file=sf, run_log=rl) is None


# --------------------------------------------------------------------------
# Aggregation, heartbeat, CLI
# --------------------------------------------------------------------------

class TestAggregationAndCli:
    def test_failing_check_isolated_not_fatal(self, monkeypatch, tmp_path):
        def boom(now, **kw):
            raise RuntimeError("boom")
        monkeypatch.setattr(wd, "ALL_CHECKS", [("boom", boom)])
        findings = wd.run_all_checks(NOW)
        assert len(findings) == 1 and "itself failed" in findings[0]

    def test_heartbeat_throttle(self, tmp_path):
        stamp = tmp_path / "ok.json"
        assert wd.should_send_heartbeat(NOW, stamp_file=stamp) is True
        wd.record_heartbeat(NOW, stamp_file=stamp)
        assert wd.should_send_heartbeat(NOW, stamp_file=stamp) is False
        # 13h later (past the 12h throttle): heartbeat allowed again.
        assert wd.should_send_heartbeat(NOW + timedelta(hours=13),
                                        stamp_file=stamp) is True

    def test_main_exit_1_on_findings(self, monkeypatch, tmp_path):
        monkeypatch.setattr(wd, "run_all_checks", lambda now=None: ["fake finding"])
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
        assert wd.main() == 1

    def test_main_exit_0_when_clean(self, monkeypatch, tmp_path):
        monkeypatch.setattr(wd, "run_all_checks", lambda now=None: [])
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
        monkeypatch.setattr(wd, "OK_STAMP", tmp_path / "ok.json")
        assert wd.main() == 0


# --------------------------------------------------------------------------
# check_equity_consistency
# --------------------------------------------------------------------------

class TestEquityConsistency:
    def test_missing_state_silent(self, tmp_path):
        assert wd.check_equity_consistency(NOW, state_file=tmp_path / "none.json") is None

    def test_flat_consistent_silent(self, tmp_path):
        st = tmp_path / "state.json"
        st.write_text(json.dumps({
            "cash": 95.0, "last_equity_marked": 95.0,
            "total_pnl": -2.0, "initial_capital": 97.0, "positions": {},
        }), encoding="utf-8")
        assert wd.check_equity_consistency(NOW, state_file=st) is None

    def test_flat_gap_alerts(self, tmp_path):
        # Cash-based loss is -2.00 but reported P&L says +0.50 — a cost was
        # charged to cash and hidden from P&L (the old funding bug class).
        st = tmp_path / "state.json"
        st.write_text(json.dumps({
            "cash": 95.0, "last_equity_marked": 95.0,
            "total_pnl": 0.5, "initial_capital": 97.0, "positions": {},
        }), encoding="utf-8")
        msg = wd.check_equity_consistency(NOW, state_file=st)
        assert msg and "mismatch" in msg

    def test_flat_equity_cash_divergence_alerts(self, tmp_path):
        st = tmp_path / "state.json"
        st.write_text(json.dumps({
            "cash": 95.0, "last_equity_marked": 120.0,
            "total_pnl": -2.0, "initial_capital": 97.0, "positions": {},
        }), encoding="utf-8")
        msg = wd.check_equity_consistency(NOW, state_file=st)
        assert msg and "cash" in msg

    def test_open_positions_gap_beyond_notional_alerts(self, tmp_path):
        # equity gain (+20) vs P&L (+0.5): gap 19.5 exceeds notional 5 + tol.
        st = tmp_path / "state.json"
        st.write_text(json.dumps({
            "cash": 92.0, "last_equity_marked": 117.0,
            "total_pnl": 0.5, "initial_capital": 97.0,
            "positions": {"ETH": {"size_usd": 5.0}},
        }), encoding="utf-8")
        msg = wd.check_equity_consistency(NOW, state_file=st)
        assert msg and "notional" in msg

    def test_open_positions_within_notional_silent(self, tmp_path):
        # Gap (3.0) is under notional (5.0) — plausible with open marks.
        st = tmp_path / "state.json"
        st.write_text(json.dumps({
            "cash": 92.0, "last_equity_marked": 100.0,
            "total_pnl": 0.5, "initial_capital": 97.0,
            "positions": {"ETH": {"size_usd": 5.0}},
        }), encoding="utf-8")
        assert wd.check_equity_consistency(NOW, state_file=st) is None

    def test_registered_in_all_checks(self):
        names = [n for n, _ in wd.ALL_CHECKS]
        assert "equity_consistency" in names
