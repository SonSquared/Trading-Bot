"""Telegram messaging correctness tests.

Every number a Telegram message shows must trace to the ledger, and every
message must actually send. These tests pin the messaging plumbing:

1. Workflow cache persists ALL history files (state, trade log, run history,
   tracker memory, watchdog stamp) — not just the state file. A cache that
   forgets history neuters /trades, /dashboard stats, and the watchdog's
   missed-run / duplicate-open / desync checks across runs.
2. /help tells the truth: real strategy names, real cadence. The old text
   advertised dead strategies and a 15-minute schedule that no longer exists.
3. /restart is honest: it cannot trigger a run on the Actions runner.
4. tg_send_message defaults to plain text (no parse_mode) so dynamic content
   containing "<" can never make Telegram reject the whole message with a
   400; HTML is opt-in for callers that use tags.
5. Position-tracker price alerts measure the real elapsed window between
   runs (~2h) instead of a mathematically impossible 1h-ago window, and
   console P&L uses the same gross-mark convention as the bot.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

import scripts.position_tracker as tracker_mod
import scripts.paper_trader as pt


# ---------------------------------------------------------------------------
# 1. Workflow cache persists every history file
# ---------------------------------------------------------------------------

class TestWorkflowCachePaths:
    def test_cache_covers_all_history_files(self):
        d = yaml.safe_load(Path(".github/workflows/bot.yml").read_text())
        steps = d["jobs"]["run-bot"]["steps"]
        cache_steps = [s for s in steps
                       if str(s.get("uses", "")).startswith("actions/cache")]
        assert len(cache_steps) == 1, "expected exactly one cache step"
        paths = cache_steps[0]["with"]["path"].split()
        for required in (
            "data/results/paper_state.json",
            "data/results/paper_trades.jsonl",
            "data/results/run_history.jsonl",
            "data/results/position_tracker.json",
            "data/results/watchdog_last_ok.json",
        ):
            assert required in paths, (
                f"{required} missing from cache path — it would reset to "
                f"empty every run, breaking /trades, /dashboard, or the "
                f"watchdog's history-dependent checks"
            )

    def test_cache_key_is_per_run_with_restore_prefix(self):
        d = yaml.safe_load(Path(".github/workflows/bot.yml").read_text())
        steps = d["jobs"]["run-bot"]["steps"]
        cache = [s for s in steps
                 if str(s.get("uses", "")).startswith("actions/cache")][0]
        key = cache["with"]["key"]
        assert "github.run_number" in key, (
            "stable cache keys are immutable-in-practice: the post-save is "
            "skipped and every run's trades are silently discarded"
        )
        assert cache["with"].get("restore-keys", "").strip()


# ---------------------------------------------------------------------------
# 2-3. /help and /restart tell the truth
# ---------------------------------------------------------------------------

def _stage_command(monkeypatch, command: str):
    updates = {"ok": True, "result": [{
        "update_id": 1,
        "message": {"text": command, "chat": {"id": 123}, "message_id": 1},
    }]}

    def fake_api(token, method, params=None):
        if method == "getUpdates":
            return updates
        return {"ok": True, "result": []}

    monkeypatch.setattr(pt, "tg_api_call", fake_api)


def _capture_sends(monkeypatch, texts):
    def fake_send(token, chat_id, text, parse_mode=None):
        texts.append(text)
        return True
    monkeypatch.setattr(pt, "tg_send_message", fake_send)


class TestHelpRestartViaHandler:
    def test_help_message_content(self, monkeypatch):
        texts: list[str] = []
        _capture_sends(monkeypatch, texts)
        _stage_command(monkeypatch, "/help")
        monkeypatch.setattr(pt, "load_state", lambda: pt._fresh_state())
        monkeypatch.setattr(pt, "fetch_live_prices", lambda pairs=None: {})
        monkeypatch.setattr(pt, "get_equity", lambda s, p=None: 97.0)
        pt.handle_telegram_commands("fake-token", "123")
        assert texts, "/help sent nothing"
        msg = texts[-1]
        strategies = ", ".join(sorted(pt.ACTIVE_STRATEGIES.keys()))
        assert strategies in msg
        assert "BB_RSI" not in msg
        assert "~2h" in msg
        assert "~15 min" not in msg

    def test_restart_message_content(self, monkeypatch):
        texts: list[str] = []
        _capture_sends(monkeypatch, texts)
        _stage_command(monkeypatch, "/restart")
        monkeypatch.setattr(pt, "load_state", lambda: pt._fresh_state())
        monkeypatch.setattr(pt, "fetch_live_prices", lambda pairs=None: {})
        monkeypatch.setattr(pt, "get_equity", lambda s, p=None: 97.0)
        pt.handle_telegram_commands("fake-token", "123")
        assert texts, "/restart sent nothing"
        msg = texts[-1]
        assert "acknowledged" in msg.lower()
        assert "~2h" in msg
        assert "running now" not in msg.lower()
        assert "~1 minute" not in msg.lower()


# ---------------------------------------------------------------------------
# 4. parse_mode defaults to plain text; HTML is opt-in
# ---------------------------------------------------------------------------

class TestParseMode:
    def test_default_send_has_no_parse_mode(self, monkeypatch):
        captured: dict = {}

        def fake_api(token, method, params=None):
            captured.update(params or {})
            return {"ok": True, "result": []}

        monkeypatch.setattr(pt, "tg_api_call", fake_api)
        ok = pt.tg_send_message("t", "c", "close < 50 candles is fine as text")
        assert ok is True
        assert "parse_mode" not in captured, (
            "default parse_mode must be absent — HTML parsing of dynamic "
            "content can 400-reject the whole message"
        )

    def test_html_opt_in(self, monkeypatch):
        captured: dict = {}

        def fake_api(token, method, params=None):
            captured.update(params or {})
            return {"ok": True, "result": []}

        monkeypatch.setattr(pt, "tg_api_call", fake_api)
        pt.tg_send_message("t", "c", "<b>bold</b>", parse_mode="HTML")
        assert captured.get("parse_mode") == "HTML"

    def test_watchdog_alert_requests_html(self):
        import inspect
        import scripts.watchdog as wd
        src = inspect.getsource(wd.main)
        assert 'parse_mode="HTML"' in src, (
            "watchdog alert uses <b> tags — it must request HTML explicitly"
        )
        # And the heartbeat (plain text) must NOT pass a parse mode.
        hb = inspect.getsource(wd.main).split("heartbeat")[1]
        assert "parse_mode" not in hb


# ---------------------------------------------------------------------------
# 5. Position-tracker price alerts at the real cadence
# ---------------------------------------------------------------------------

class TestPriceAlerts:
    def _tracker_with_sample(self, hours_ago: float, price: float) -> dict:
        now = datetime.now(timezone.utc)
        return {
            "price_history": {"ETH_USDT_USDT": [
                {"time": (now - timedelta(hours=hours_ago)).isoformat(),
                 "price": price},
            ]},
        }

    def test_no_alert_with_single_sample(self):
        # A single price sample has nothing to compare against — no alert.
        alerts = tracker_mod.check_price_alerts(
            {}, "ETH_USDT_USDT", "ETH/USDT", 110.0)
        assert alerts == []

    def test_alert_fires_on_2h_move_above_threshold(self):
        t = self._tracker_with_sample(2.0, 100.0)
        alerts = tracker_mod.check_price_alerts(
            t, "ETH_USDT_USDT", "ETH/USDT", 103.0)
        assert len(alerts) == 1
        assert "UP" in alerts[0]
        assert "+3.0%" in alerts[0]
        assert "2.0h change" in alerts[0]

    def test_down_move_alerts(self):
        t = self._tracker_with_sample(2.0, 100.0)
        alerts = tracker_mod.check_price_alerts(
            t, "ETH_USDT_USDT", "ETH/USDT", 97.0)
        assert len(alerts) == 1
        assert "DOWN" in alerts[0]

    def test_small_move_no_alert(self):
        t = self._tracker_with_sample(2.0, 100.0)
        alerts = tracker_mod.check_price_alerts(
            t, "ETH_USDT_USDT", "ETH/USDT", 101.0)
        assert alerts == []

    def test_short_window_never_fabricates_alert(self):
        # Samples minutes apart must never produce an alert regardless of
        # the move size — this is the restart-storm guard.
        t = self._tracker_with_sample(0.1, 100.0)
        alerts = tracker_mod.check_price_alerts(
            t, "ETH_USDT_USDT", "ETH/USDT", 150.0)
        assert alerts == []

    def test_cooldown_suppresses_repeat_alerts(self):
        now = datetime.now(timezone.utc)
        t = self._tracker_with_sample(2.0, 100.0)
        t["last_price_alert_ETH_USDT_USDT"] = now.isoformat()
        alerts = tracker_mod.check_price_alerts(
            t, "ETH_USDT_USDT", "ETH/USDT", 103.0)
        assert alerts == []


class TestTrackerGrossMarks:
    def test_check_positions_uses_gross_marks(self, monkeypatch, tmp_path):
        """Console summaries must match the bot's gross-mark convention."""
        state = {
            "cash": 60.0,
            "positions": {
                "ETH_USDT_USDT": {
                    "side": 1, "entry_price": 2000.0, "size_usd": 35.0,
                    "strategy": "test",
                },
            },
        }
        monkeypatch.setattr(tracker_mod, "load_positions", lambda: {
            "positions": state["positions"], "cash": state["cash"],
            "equity": 97.0})
        monkeypatch.setattr(tracker_mod, "load_tracker_state",
                            lambda: {"last_update": None, "position_data": {}})
        monkeypatch.setattr(tracker_mod, "save_tracker_state", lambda s: None)
        monkeypatch.setattr(tracker_mod, "fetch_price", lambda sym: 2100.0)
        monkeypatch.setattr(tracker_mod, "send_telegram", lambda text: None)

        summaries = tracker_mod.check_positions()
        assert len(summaries) == 1
        s = summaries[0]
        # Gross: qty=0.0175, move +100 -> +$1.75. Net would deduct 2x fee
        # (0.035) and report +1.715 — the old convention this test kills.
        assert abs(s["pnl_usd"] - 1.75) < 1e-9
        assert abs(s["pnl_pct"] - (100.0 / 2000.0 * 100)) < 1e-9
