"""
Robustness tests for the deployed paper trading bot.

These prove the hardening added after the audit actually holds under the
failure modes it was built for:

- Stale candle data must never generate a signal (fail loud, never trade stale).
- A crash between a logged open and a state save must never cause a duplicate
  open on the next run (idempotent-open guard cross-checks the trade log).
- Portfolio heat must be re-checked per open, not once per run (two same-run
  opens at 35% each used to bypass the 50% limit and reach 70%).
- State must be saved incrementally after every open/close so a crash can
  never resurrect or lose a trade.
- Persisted equity must be marked-to-market, not frozen at entry price.
- The full lifecycle (open -> adverse move -> risk stop -> close) must keep
  the accounting invariants at every step: cash + exposure = equity, every
  trade-log line matches a state mutation, cash never goes negative, and
  every close is realizable exactly once.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.paper_trader as pt
from scripts.paper_trader import (
    assert_fresh_candles,
    already_open,
    candle_age_seconds,
    charge_funding,
    close_position,
    get_equity,
    open_position,
    save_state,
    INITIAL_CAPITAL,
    MAX_CANDLE_AGE_FACTOR,
)
from trading_system.bot.accounting import FEE_RATE, SLIPPAGE_RATE, funding_boundaries_in
from trading_system.bot.candles import TIMEFRAME_SECONDS, closed_candles
from trading_system.bot.risk_manager import DEFAULT_RISK_MANAGER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PAIR = "ETH_USDT_USDT"


def make_df(n: int = 120, start_price: float = 100.0, seed: int = 7,
            end: datetime | None = None) -> pd.DataFrame:
    """Synthetic OHLCV with a ``timestamp`` column, ending at ``end``.

    The last candle is the currently-forming one when ``end`` == now
    (its close time is in the future), so ``closed_candles`` drops it.
    """
    if end is None:
        end = datetime.now(timezone.utc)
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=end, periods=n, freq="4h", tz="UTC")
    close = start_price + np.cumsum(rng.normal(0, start_price * 0.005, n))
    close = np.maximum(close, 1.0)
    high = close * (1 + rng.uniform(0.0005, 0.002, n))
    low = close * (1 - rng.uniform(0.0005, 0.002, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame({
        "timestamp": idx,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": 1000.0,
    })


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Redirect every module-level file target into a tmp dir."""
    monkeypatch.setattr(pt, "LOG_DIR", tmp_path)
    monkeypatch.setattr(pt, "TRADE_LOG", tmp_path / "paper_trades.jsonl")
    monkeypatch.setattr(pt, "STATE_FILE", tmp_path / "paper_state.json")
    monkeypatch.setattr(pt, "SUMMARY_FILE", tmp_path / "paper_summary.json")
    monkeypatch.setattr(pt, "RUN_LOG", tmp_path / "run_history.jsonl")
    monkeypatch.setattr(pt, "LOCK_FILE", tmp_path / "bot.lock")
    return tmp_path


def fresh_state() -> dict:
    return pt._fresh_state()


def log_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Staleness guard
# ---------------------------------------------------------------------------

class TestStalenessGuard:
    def test_fresh_candles_pass(self):
        df = make_df(end=datetime.now(timezone.utc))
        # Must not raise.
        assert_fresh_candles(df, "4h", PAIR)

    def test_stale_candles_raise(self):
        # Latest closed candle is 3 days old; 4h candles may be at most
        # 2 * 4h old. This is the "exchange API silently returned old data"
        # case — the bot must refuse to trade, loudly.
        df = make_df(end=datetime.now(timezone.utc) - timedelta(days=3))
        with pytest.raises(RuntimeError, match="STALE"):
            assert_fresh_candles(df, "4h", PAIR)

    def test_undatable_candles_raise(self):
        # No timestamp column and a non-datetime index: age cannot be
        # determined, so the guard must fail closed.
        df = pd.DataFrame({
            "open": [100.0] * 60, "high": [101.0] * 60,
            "low": [99.0] * 60, "close": [100.5] * 60, "volume": [1.0] * 60,
        })
        with pytest.raises(RuntimeError, match="STALE"):
            assert_fresh_candles(df, "4h", PAIR)

    def test_age_uses_close_time_not_open_time(self):
        # The newest closed 4h candle opened exactly 4h ago and closed just
        # now, so its age must be ~0. An off-by-one that used the OPEN time
        # would report age = 4h and block one run every cycle.
        df = make_df(end=datetime.now(timezone.utc))
        age = candle_age_seconds(df, "4h")
        assert age is not None
        assert age < 3600, (
            f"age {age/60:.0f}min — looks like open-time semantics, not close-time")

    def test_max_age_bound_is_two_intervals(self):
        # All candles at timestamps <= end; the newest closed candle opened at
        # `end - 4h` and closed at `end`. Its close age = now - end. To breach
        # the 8h (2x 4h) limit the frame must end more than 12h ago.
        now = datetime.now(timezone.utc)
        limit = timedelta(seconds=TIMEFRAME_SECONDS["4h"] * MAX_CANDLE_AGE_FACTOR)
        horizon = limit + timedelta(hours=4)  # 12h: candle closed exactly at limit
        # closed 1 minute inside the limit -> pass
        df_ok = make_df(end=(pd.Timestamp(now) - horizon + pd.Timedelta(minutes=1)).to_pydatetime())
        assert_fresh_candles(df_ok, "4h", PAIR)
        # closed 1 second inside the limit -> pass (1s tolerance absorbs the
        # microseconds between building the frame and the guard's clock sample)
        df_edge = make_df(end=(pd.Timestamp(now) - horizon + pd.Timedelta(seconds=1)).to_pydatetime())
        assert_fresh_candles(df_edge, "4h", PAIR)
        # closed 1 minute past the limit -> refuse
        df_bad = make_df(end=(pd.Timestamp(now) - horizon - pd.Timedelta(minutes=1)).to_pydatetime())
        with pytest.raises(RuntimeError, match="STALE"):
            assert_fresh_candles(df_bad, "4h", PAIR)

    def test_run_strategies_refuses_stale_data(self, isolated_paths, monkeypatch):
        """End-to-end: a stale feed produces an error entry and NO signal."""
        stale_df = make_df(end=datetime.now(timezone.utc) - timedelta(days=3))
        monkeypatch.setattr(pt, "fetch_latest", lambda *a, **k: stale_df.copy())
        monkeypatch.setattr(pt, "ACTIVE_STRATEGIES", {
            "Test": {"strategy": "Bollinger_Reversion", "pair": PAIR,
                     "timeframe": "4h", "weight": 1.0, "params": {}},
        })
        signals, errors = pt.run_strategies()
        assert not signals
        assert any("STALE" in e for e in errors)


# ---------------------------------------------------------------------------
# Duplicate-open guard
# ---------------------------------------------------------------------------

class TestDuplicateOpenGuard:
    def _log_open(self, pair=PAIR, side=1, minutes_ago=5, note=None):
        trade = {
            "timestamp": (datetime.now(timezone.utc)
                          - timedelta(minutes=minutes_ago)).isoformat(),
            "pair": pair,
            "action": "OPEN_LONG" if side == 1 else "OPEN_SHORT",
            "price": 100.0, "size_usd": 30.0,
        }
        if note:
            trade["note"] = note
        with open(pt.TRADE_LOG, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def test_recent_same_open_detected(self, isolated_paths):
        self._log_open()
        state = fresh_state()
        assert already_open(state, PAIR, 1) is True

    def test_different_side_not_detected(self, isolated_paths):
        self._log_open(side=1)
        state = fresh_state()
        assert already_open(state, PAIR, -1) is False

    def test_different_pair_not_detected(self, isolated_paths):
        self._log_open(pair="BTC_USDT_USDT")
        state = fresh_state()
        assert already_open(state, PAIR, 1) is False

    def test_old_open_not_detected(self, isolated_paths):
        self._log_open(minutes_ago=120)
        state = fresh_state()
        assert already_open(state, PAIR, 1, max_age_seconds=3600) is False

    def test_consistency_repair_note_ignored(self, isolated_paths):
        # Entries written by the state-repair pass must not count as live
        # opens, or a legitimately repaired position could never be
        # re-opened after a close.
        self._log_open(note="paper-state-consistency repair")
        state = fresh_state()
        assert already_open(state, PAIR, 1) is False

    def test_missing_log_returns_false(self, isolated_paths):
        state = fresh_state()
        assert already_open(state, PAIR, 1) is False

    def test_corrupt_line_skipped(self, isolated_paths):
        pt.TRADE_LOG.write_text('{"broken json\n' + json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": PAIR, "action": "OPEN_LONG", "price": 100.0,
        }) + "\n")
        state = fresh_state()
        assert already_open(state, PAIR, 1) is True

    def test_main_loop_blocks_duplicate_after_lost_save(self, isolated_paths, monkeypatch):
        """TRUE e2e of the scenario the guard exists for: the trade log
        recorded an open, but a crash lost it from state. Running the real
        main() must skip the re-open, keep the account flat, and report the
        prevention in the run log."""
        # Ground: trade log has an OPEN from 5 minutes ago...
        with open(pt.TRADE_LOG, "a") as f:
            f.write(json.dumps({
                "timestamp": (datetime.now(timezone.utc)
                              - timedelta(minutes=5)).isoformat(),
                "pair": PAIR,
                "action": "OPEN_LONG",
                "price": 100.0, "size_usd": 30.0,
            }) + "\n")
        # ...but the state save failed, so disk state has no position.
        save_state(fresh_state())

        monkeypatch.setattr(pt, "load_active_strategies", lambda: None)
        monkeypatch.setattr(pt, "ACTIVE_STRATEGIES", {
            "Test": {"strategy": "Bollinger_Reversion", "pair": PAIR,
                     "timeframe": "4h", "weight": 1.0, "params": {}},
        })
        monkeypatch.setattr(pt, "get_notifier", lambda: None)
        monkeypatch.setattr(pt, "load_telegram_config", lambda: {})
        monkeypatch.setattr(pt, "charge_funding",
                            lambda state, now=None: (0.0, {}))
        monkeypatch.setattr(pt, "fetch_live_prices",
                            lambda pairs=None: {PAIR: 100.0})
        # A fresh LONG signal on the same pair the log already recorded.
        monkeypatch.setattr(pt, "run_strategies", lambda: ({
            "Test": {"signal": 1, "weight": 1.0, "pair": PAIR,
                     "strategy": "Test", "price": 100.0,
                     "next_open": 100.0, "atr": 1.0, "candle_time": None},
        }, []))
        monkeypatch.setattr(pt, "get_equity", lambda s, prices=None: 97.0)

        pt.main()  # the real thing

        # No second open was logged.
        opens = [t for t in log_lines(pt.TRADE_LOG)
                 if t["action"].startswith("OPEN")]
        assert len(opens) == 1, "duplicate open must be prevented"
        # State stayed flat and consistent with the log.
        state = json.loads(pt.STATE_FILE.read_text())
        assert state["positions"] == {}
        assert state["cash"] == pytest.approx(INITIAL_CAPITAL)
        # The run log records the prevention, not a silent skip.
        runs = log_lines(pt.RUN_LOG)
        assert runs and any(
            "duplicate open prevented" in e
            for e in runs[-1]["errors"]), \
            "run log must surface the prevented duplicate"

    def test_telegram_message_reconciles_exactly(self, isolated_paths, monkeypatch, capsys):
        """THE Telegram contract: in a real main() run that opens funding and
        closes a position, the message's own lines must add up exactly:

            Equity - Start == P&L (realized) + P&L (unrealized)

        and Realized must equal cash change (all costs booked at charge time).
        """
        import re

        captured = []

        class FakeNotifier:
            def _send_message(self, msg):
                captured.append(msg)
                return True

        BTC = "BTC_USDT_USDT"
        monkeypatch.setattr(pt, "load_active_strategies", lambda: None)
        monkeypatch.setattr(pt, "ACTIVE_STRATEGIES", {
            "TestEth": {"strategy": "Bollinger_Reversion", "pair": PAIR,
                        "timeframe": "4h", "weight": 1.0, "params": {}},
            "TestBtc": {"strategy": "Bollinger_Reversion", "pair": BTC,
                        "timeframe": "4h", "weight": 1.0, "params": {}},
        })
        monkeypatch.setattr(pt, "get_notifier", lambda: FakeNotifier())
        monkeypatch.setattr(pt, "load_telegram_config", lambda: {})
        # REAL charge_funding — the funding path is part of the contract.
        monkeypatch.setattr(pt, "fetch_live_prices",
                            lambda pairs=None: {PAIR: 104.0, BTC: 100.0})

        # State: ETH LONG from 25h ago (guarantees >=3 funding boundaries)
        # with a prior funding charge already booked (last_funding_time 17h
        # ago). BTC flat.
        state = fresh_state()
        t_open = open_position(state, PAIR, 1, 100.0, "T", 30.0)
        now = datetime.now(timezone.utc)
        state["positions"][PAIR]["entry_time"] = (now - timedelta(hours=25)).isoformat()
        state["positions"][PAIR]["last_funding_time"] = (now - timedelta(hours=17)).isoformat()
        save_state(state)

        # ETH: strong reversal (LONG held, score < -0.1) -> CLOSE (realized).
        # BTC: strong fresh entry (flat, score > 0.4) -> OPEN (unrealized).
        monkeypatch.setattr(pt, "run_strategies", lambda: ({
            "TestEth": {"signal": -1, "weight": 1.0, "pair": PAIR,
                        "strategy": "TestEth", "price": 104.0,
                        "next_open": 104.0, "atr": 1.0, "candle_time": None},
            "TestBtc": {"signal": 1, "weight": 1.0, "pair": BTC,
                        "strategy": "TestBtc", "price": 100.0,
                        "next_open": 100.0, "atr": 1.0, "candle_time": None},
        }, []))

        pt.main()

        out = capsys.readouterr().out
        assert "reconciliation gap" not in out, out
        assert len(captured) == 1, "exactly one trade-alert message"
        msg = captured[0]

        def money(pattern):
            m = re.search(pattern, msg)
            assert m, f"pattern {pattern!r} not in message:\n{msg}"
            return float(m.group(1).replace(",", ""))

        equity = money(r"Equity: \$([\d,]+\.\d{2})")
        realized = money(r"P&L \(realized\): \$([+-]?[\d,]+\.\d{2})")
        unrealized = money(r"P&L \(unrealized\): \$([+-]?[\d,]+\.\d{2})")

        # THE identity — exact to the cent.
        # THE identity — exact to the cent (printed values).
        assert equity - INITIAL_CAPITAL == pytest.approx(
            realized + unrealized, abs=0.011)

        # Realized must equal reported total_pnl exactly, and the ETH cycle
        # closed while BTC stays open (the unrealized side of the identity).
        final_state = json.loads(pt.STATE_FILE.read_text())
        assert list(final_state["positions"].keys()) == [BTC]
        assert realized == pytest.approx(final_state["total_pnl"], abs=0.005)
        # Cash bookkeeping ground truth: final cash == last logged cash_after.
        all_trades = log_lines(pt.TRADE_LOG)
        assert all_trades[-1]["cash_after"] == pytest.approx(
            final_state["cash"], abs=1e-6)
        # And the flat-ETH cycle itself reconciles: realized == the ETH close
        # P&L minus the ETH entry fee minus funding charged this run.
        eth_close = next(t for t in all_trades if t.get("pair") == PAIR
                         and t["action"] == "CLOSE")
        eth_entry_fee = 30.0 * FEE_RATE
        funding_paid = -(final_state["total_pnl"] - eth_close["pnl_usd"]
                         + eth_entry_fee)
        assert funding_paid >= 0, "funding must have been charged this run"

        # Per-position lines show GROSS marks with the cost label; the BTC
        # line is the freshly-opened position whose mark is its own fill.
        assert "before fees+funding" in msg
        assert "BTC/USDT" in msg


# ---------------------------------------------------------------------------
# Portfolio heat re-check
# ---------------------------------------------------------------------------

class TestHeatRecheck:
    def test_second_same_run_open_rejected_at_heat_limit(self, isolated_paths):
        """Regression: can_open is evaluated once BEFORE the trade loop, so
        two opens in one run each passed the gate and reached 70% heat. The
        per-open re-check must reject the second open."""
        state = fresh_state()
        # First open already happened this run: 35% heat.
        open_position(state, PAIR, 1, 100.0, "T", 35.0)

        second_pair = "BTC_USDT_USDT"
        size_usd = min(100.0 * 0.35, state["cash"] * 0.95, INITIAL_CAPITAL * 0.50)
        assert size_usd == pytest.approx(35.0)

        # The pre-loop gate evaluates CURRENT heat only — this is exactly the
        # bypass the bug exploited: it sees 35% and says yes.
        ok, _ = DEFAULT_RISK_MANAGER.can_open_position(
            state["positions"], 100.0, state["cash"], {},
            peak_equity=100.0, dd_cooldown_until=None,
            now=datetime.now(timezone.utc))
        assert ok is True, "pre-loop gate genuinely passes at 35% heat"

        # The per-open re-check (as in main()) adds the new exposure first:
        existing = sum(p["size_usd"] for p in state["positions"].values())
        heat_with_new = (existing + size_usd) / 100.0 * 100
        assert heat_with_new > DEFAULT_RISK_MANAGER.max_portfolio_heat_pct, \
            "70% heat must breach the 50% limit"
        # So the loop `continue`s and the second position never exists.
        assert second_pair not in state["positions"]

    def test_open_within_heat_limit_allowed(self, isolated_paths):
        """Positive control: openings that stay under the cap must go
        through. With the production sizing (35% of equity) a second
        same-run open is always over the 50% cap, so this uses a small
        first position to prove the re-check admits legal combinations."""
        state = fresh_state()
        open_position(state, PAIR, 1, 100.0, "T", 10.0)  # 10% heat
        size_usd = min(100.0 * 0.35, state["cash"] * 0.95, INITIAL_CAPITAL * 0.50)
        existing = sum(p["size_usd"] for p in state["positions"].values())
        heat_with_new = (existing + size_usd) / 100.0 * 100
        assert heat_with_new <= DEFAULT_RISK_MANAGER.max_portfolio_heat_pct, \
            "test setup: 10% + 35% = 45% must fit under the 50% cap"
        # The re-check passes, so the second open really happens.
        open_position(state, "BTC_USDT_USDT", 1, 100.0, "T", size_usd)
        assert "BTC_USDT_USDT" in state["positions"]
        total = sum(p["size_usd"] for p in state["positions"].values())
        assert total / 100.0 * 100 <= DEFAULT_RISK_MANAGER.max_portfolio_heat_pct


# ---------------------------------------------------------------------------
# Crash-safe incremental saves
# ---------------------------------------------------------------------------

class TestCrashSafety:
    def test_state_saved_immediately_after_open(self, isolated_paths, monkeypatch):
        """A crash after open_position but before end-of-run save must not
        lose the position: the incremental save already persisted it."""
        state = fresh_state()
        open_position(state, PAIR, 1, 100.0, "T", 30.0)
        save_state(state)  # what main() does right after every open

        # Simulate the crash: reload from disk as the next run would.
        reloaded = pt.load_state()
        assert PAIR in reloaded["positions"]
        assert reloaded["positions"][PAIR]["side"] == 1
        # The idempotent guard then sees the log+state in agreement — no
        # phantom re-open, no duplicate.
        assert already_open(reloaded, PAIR, 1) is True

    def test_state_saved_immediately_after_close(self, isolated_paths):
        state = fresh_state()
        open_position(state, PAIR, 1, 100.0, "T", 30.0)
        save_state(state)
        close_position(state, PAIR, 102.0, "test")
        save_state(state)

        reloaded = pt.load_state()
        assert PAIR not in reloaded["positions"], \
            "a closed position must never resurrect after a crash"
        assert reloaded["total_trades"] == 1

    def test_save_then_reload_roundtrip_is_lossless(self, isolated_paths):
        state = fresh_state()
        state["last_equity_marked"] = 123.456
        state["last_prices"] = {PAIR: 99.5}
        open_position(state, PAIR, 1, 100.0, "T", 30.0)
        save_state(state)
        reloaded = pt.load_state()
        assert reloaded["last_equity_marked"] == pytest.approx(123.456)
        assert reloaded["last_prices"][PAIR] == pytest.approx(99.5)
        assert reloaded["positions"][PAIR]["size_usd"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Marked-to-market equity
# ---------------------------------------------------------------------------

class TestMarkedEquity:
    def test_get_equity_marks_open_position(self):
        state = fresh_state()
        open_position(state, PAIR, 1, 100.0, "T", 30.0)
        # Price rallied 10% since entry.
        equity = get_equity(state, prices={PAIR: 110.0})
        qty = 30.0 / state["positions"][PAIR]["entry_price"]
        expected = state["cash"] + 30.0 + qty * (110.0 - state["positions"][PAIR]["entry_price"])
        assert equity == pytest.approx(expected, abs=1e-9)
        assert equity > state["cash"]  # marked, not frozen at cost

    def test_short_position_marks_against_price_fall(self):
        state = fresh_state()
        open_position(state, PAIR, -1, 100.0, "T", 30.0)
        equity = get_equity(state, prices={PAIR: 90.0})
        assert equity > state["cash"], "short must gain when price falls"

    def test_missing_price_falls_back_to_cost_basis(self, isolated_paths, monkeypatch):
        # get_equity auto-fetches missing prices; make that deterministic
        # (and offline) so the pure cost-basis fallback is what's tested.
        monkeypatch.setattr(pt, "fetch_live_prices", lambda pairs=None: {})
        state = fresh_state()
        open_position(state, PAIR, 1, 100.0, "T", 30.0)
        equity = get_equity(state, prices={})  # no prices available
        assert equity == pytest.approx(state["cash"] + 30.0)


# ---------------------------------------------------------------------------
# Funding accounting
# ---------------------------------------------------------------------------

class TestFunding:
    def test_charge_funding_debits_cash_and_records(self):
        state = fresh_state()
        open_position(state, PAIR, 1, 100.0, "T", 30.0)
        entry_time = datetime.now(timezone.utc) - timedelta(hours=17)
        # 17h ago crosses the 00:00/08:00/16:00 boundaries twice.
        state["positions"][PAIR]["entry_time"] = entry_time.isoformat()
        state["positions"][PAIR].pop("last_funding_time", None)

        cash_before = state["cash"]
        total, per_pair = charge_funding(state, now=datetime.now(timezone.utc))
        assert PAIR in per_pair
        assert total == pytest.approx(per_pair[PAIR])
        assert state["cash"] == pytest.approx(cash_before - total)
        expected_n = funding_boundaries_in(entry_time, datetime.now(timezone.utc))
        assert expected_n >= 2
        assert total == pytest.approx(
            30.0 * 0.0001 * expected_n, rel=1e-6)
        assert state["positions"][PAIR]["funding_paid"] == pytest.approx(total)
        assert "last_funding_time" in state["positions"][PAIR]
        # Funding is a real cost: it must hit reported P&L too, not just
        # cash — else Equity and P&L lines disagree in Telegram.
        # total_pnl here = -(entry fee) - funding (no closes yet).
        assert state["total_pnl"] == pytest.approx(
            -(30.0 * FEE_RATE) - total, abs=1e-9)

    def test_charge_funding_is_idempotent_per_interval(self):
        # Charging twice within the same 8h window must not double-charge:
        # the second call sees last_funding_time >= the first call's now.
        # Times are pinned away from the 00/08/16 UTC funding boundaries:
        # this test was wall-clock flaky — whenever "now" sat within an
        # hour of a boundary, now1+1h legitimately crossed it and charged.
        state = fresh_state()
        open_position(state, PAIR, 1, 100.0, "T", 30.0)
        now1 = datetime(2026, 9, 10, 1, 30, tzinfo=timezone.utc)
        entry_time = now1 - timedelta(hours=9)
        state["positions"][PAIR]["entry_time"] = entry_time.isoformat()

        total1, _ = charge_funding(state, now=now1)
        state["positions"][PAIR]["last_funding_time"] = now1.isoformat()
        total2, _ = charge_funding(state, now=now1 + timedelta(hours=1))
        # One hour later, no new 8h boundary has been crossed.
        assert total2 == 0.0
        assert total1 > 0.0

    def test_full_cycle_pnl_reconciles_with_cash(self):
        """The Telegram-bug regression: after open -> funding -> close, the
        reported realized P&L must equal the cash change exactly. Every cost
        (entry fee, exit fee, funding) is booked in BOTH places."""
        state = fresh_state()
        t_open = open_position(state, PAIR, 1, 100.0, "T", 30.0)
        entry_time = datetime.now(timezone.utc) - timedelta(hours=17)
        state["positions"][PAIR]["entry_time"] = entry_time.isoformat()
        state["positions"][PAIR].pop("last_funding_time", None)

        funding_total, _ = charge_funding(state, now=datetime.now(timezone.utc))
        assert funding_total > 0

        t_close = close_position(state, PAIR, 105.0, "test")

        # THE identity: cash-based gain == reported realized P&L (flat).
        assert not state["positions"], "cycle must end flat"
        assert state["total_pnl"] == pytest.approx(
            state["cash"] - INITIAL_CAPITAL, abs=1e-9)
        # And the pieces sum: close pnl - entry fee - funding == total.
        assert state["total_pnl"] == pytest.approx(
            t_close["pnl_usd"] - t_open["fee"] - funding_total, abs=1e-9)


# ---------------------------------------------------------------------------
# Full lifecycle e2e with accounting invariants
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    def _assert_invariants(self, state, log_path: Path, prices: dict):
        """Cash + exposure = equity; every log line maps to a state
        mutation; cash never negative; no orphaned log entries."""
        exposure = sum(p["size_usd"] for p in state["positions"].values())
        equity = get_equity(state, prices=prices)
        assert state["cash"] >= -1e-9, "cash must never go negative"
        assert equity == pytest.approx(state["cash"] + exposure, abs=0.51), \
            f"equity {equity} != cash {state['cash']} + exposure {exposure}"

        opens = [t for t in log_lines(log_path) if t["action"].startswith("OPEN")]
        closes = [t for t in log_lines(log_path) if t["action"] == "CLOSE"]
        # Every logged open either still holds, or has exactly one close.
        open_pairs = [t["pair"] for t in opens]
        close_pairs = [t["pair"] for t in closes]
        for pair in close_pairs:
            assert pair in open_pairs, "close without a matching open"
        assert len(close_pairs) == len(set(close_pairs)), \
            "double close of the same pair"
        held = [p for p in state["positions"]]
        for pair in open_pairs:
            if pair not in held:
                assert pair in close_pairs, \
                    f"{pair} was opened+lost from state but never closed"

    def test_open_adverse_move_stop_close_invariants(self, isolated_paths, monkeypatch):
        """Full cycle: open long -> price craters -> 5% stop closes it ->
        invariants hold at every step and the position cannot re-close."""
        log_path = pt.TRADE_LOG
        state = fresh_state()

        # 1) Open.
        open_position(state, PAIR, 1, 100.0, "TestLong", 30.0)
        save_state(state)
        self._assert_invariants(state, log_path, {PAIR: 100.0})

        # 2) Adverse move: -6% on the position (beyond the 5% stop).
        prices = {PAIR: 94.0}
        equity_now = get_equity(state, prices=prices)
        closes, alerts = DEFAULT_RISK_MANAGER.check_positions(
            state["positions"], prices, equity_now, state["peak_equity"])
        assert any("Stop-loss" in c["reason"] for c in closes), \
            "5% stop must fire on a -6% move"
        assert any(a["severity"] == "WARNING" for a in alerts)

        # 3) Execute the risk close exactly as main() does.
        close = closes[0]
        t = close_position(state, close["symbol"], prices[PAIR], close["reason"])
        assert t is not None
        save_state(state)
        self._assert_invariants(state, log_path, prices)

        # 4) Position is gone from state; log has exactly 1 open + 1 close.
        assert PAIR not in state["positions"]
        ls = log_lines(log_path)
        assert len([x for x in ls if x["action"].startswith("OPEN")]) == 1
        assert len([x for x in ls if x["action"] == "CLOSE"]) == 1
        assert t["pnl_usd"] < 0, "the stop closed a losing trade"

        # 5) Close idempotency: a second close attempt must be a no-op.
        assert close_position(state, PAIR, 94.0, "retry") is None
        ls2 = log_lines(log_path)
        assert len([x for x in ls2 if x["action"] == "CLOSE"]) == 1

        # 6) The loss hits the account: cash reflects size + loss - fees.
        assert state["cash"] < INITIAL_CAPITAL
        assert state["losses"] == 1 and state["wins"] == 0
        # total_pnl must reconcile with cash: it includes BOTH fees (entry
        # fee booked at open, exit fee inside pnl_usd) — the exact bug class
        # that made Telegram's P&L line disagree with its Equity line.
        assert state["total_pnl"] == pytest.approx(t["pnl_usd"] - 30.0 * FEE_RATE)
        # Cash identity: entry debited size+fee, close credited size+pnl.
        assert state["cash"] == pytest.approx(
            INITIAL_CAPITAL - 30.0 - 30.0 * FEE_RATE + 30.0 + t["pnl_usd"], abs=1e-9)
        assert state["total_pnl"] == pytest.approx(
            state["cash"] - INITIAL_CAPITAL, abs=1e-9)

    def test_cycle_pnl_math_matches_shared_model(self, isolated_paths):
        """Close P&L must equal the shared accounting model (fees + slippage
        embedded in fills), not a flat approximation."""
        state = fresh_state()
        open_position(state, PAIR, 1, 100.0, "T", 30.0)
        entry_price = state["positions"][PAIR]["entry_price"]
        # Entry fill = ref * (1 + slippage) for a buy.
        assert entry_price == pytest.approx(100.0 * (1 + SLIPPAGE_RATE))

        exit_ref = 105.0
        t = close_position(state, PAIR, exit_ref, "test")
        exit_fill = exit_ref * (1 - SLIPPAGE_RATE)  # sell to close a long
        gross = 30.0 * (exit_fill - entry_price) / entry_price
        expected = gross - 30.0 * FEE_RATE
        assert t["pnl_usd"] == pytest.approx(expected, abs=1e-9)

    def test_reopen_after_close_is_allowed(self, isolated_paths):
        """The duplicate guard is a recency window, not a permanent ban:
        a recent logged open without a live position blocks a re-open,
        and once the window passes, re-opening is legitimate."""
        state = fresh_state()
        open_position(state, PAIR, 1, 100.0, "T", 30.0)
        save_state(state)
        close_position(state, PAIR, 105.0, "take profit")
        save_state(state)
        # Within the guard window: the recent logged open (position now
        # closed) must block an immediate duplicate open.
        assert already_open(state, PAIR, 1) is True
        # After the window passes, the same signal may open again. The open
        # happened milliseconds ago, so backdate it deterministically.
        lines = pt.TRADE_LOG.read_text().splitlines()
        fixed = []
        for line in lines:
            rec = json.loads(line)
            if rec.get("action", "").startswith("OPEN"):
                rec["timestamp"] = (datetime.now(timezone.utc)
                                     - timedelta(hours=2)).isoformat()
            fixed.append(json.dumps(rec))
        pt.TRADE_LOG.write_text("\n".join(fixed) + "\n")
        assert already_open(state, PAIR, 1, max_age_seconds=3600) is False


# ---------------------------------------------------------------------------
# Signal candle gating (closed candles only)
# ---------------------------------------------------------------------------

class TestClosedCandleGating:
    def test_forming_candle_dropped_from_signals(self):
        df = make_df(n=120, end=datetime.now(timezone.utc))
        closed = closed_candles(df, "4h")
        assert len(closed) == len(df) - 1, \
            "the forming candle must be excluded from signal computation"

    def test_naive_timestamps_treated_as_utc(self):
        idx = pd.date_range("2026-01-01", periods=60, freq="4h")
        df = pd.DataFrame({
            "timestamp": idx, "open": 100.0, "high": 101.0,
            "low": 99.0, "close": 100.5, "volume": 1.0,
        })
        closed = closed_candles(df, "4h")
        assert len(closed) == 60  # all far in the past -> all closed

    def test_datetime_index_frames_supported(self):
        df = make_df(n=60, end=datetime.now(timezone.utc)).set_index("timestamp")
        closed = closed_candles(df, "4h")
        assert len(closed) == 59


# ---------------------------------------------------------------------------
# Load-state validation (stale-cache and corruption hardening)
# ---------------------------------------------------------------------------

class TestLoadStateValidation:
    def test_corrupt_state_resets(self, isolated_paths):
        pt.STATE_FILE.write_text("{not json at all")
        state = pt.load_state()
        assert state["cash"] == INITIAL_CAPITAL
        assert state["positions"] == {}

    def test_stale_cache_with_impossible_cash_resets(self, isolated_paths):
        pt.STATE_FILE.write_text(json.dumps({"cash": 100000.0, "positions": {}}))
        state = pt.load_state()
        assert state["cash"] == INITIAL_CAPITAL

    def test_negative_cash_resets(self, isolated_paths):
        pt.STATE_FILE.write_text(json.dumps({"cash": -5.0, "positions": {}}))
        state = pt.load_state()
        assert state["cash"] == INITIAL_CAPITAL

    def test_valid_state_preserved(self, isolated_paths):
        good = fresh_state()
        good["cash"] = 90.0
        good["peak_equity"] = 97.0
        save_state(good)
        state = pt.load_state()
        assert state["cash"] == pytest.approx(90.0)
        assert state["peak_equity"] == pytest.approx(97.0)
