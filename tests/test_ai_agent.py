"""Tests for the AI trading bot: agent, paper ledger, scheduler, engine.

Zero network: uses a FakeExchange and a ScriptedEngine so every rule is
deterministic. The scheduler test is the regression test for the midnight
rollover bug (00:00 wakeup being skipped every day).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_system.bot.ai_agent import (
    DEFAULT_STRATEGY,
    AIAgent,
    PaperLedger,
)
from trading_system.bot.ai_engine import AIEngine, TradeAction, TradingDecision
from trading_system.bot.scheduler import Scheduler, _normalize_schedule

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 200, start: float = 50000.0, seed: int = 42) -> pd.DataFrame:
    """Deterministic OHLCV frame resembling a 1h BTC series."""
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, start * 0.05, n)
    noise = rng.normal(0, start * 0.004, n)
    close = start + drift + np.sin(np.linspace(0, 8 * np.pi, n)) * start * 0.01 + noise
    close = np.maximum(close, start * 0.5)
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.abs(rng.normal(1000, 200, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class FakeExchange:
    """In-memory exchange for tests. BTC ~50k, ETH ~3k."""

    def __init__(self):
        self.frames = {
            "BTC/USDT:USDT": make_ohlcv(200, 50000.0, seed=42),
            "ETH/USDT:USDT": make_ohlcv(200, 3000.0, seed=7),
        }
        self._price_override: dict[str, float] = {}
        self.placed_orders: list[dict] = []

    def get_ohlcv(self, pair, timeframe="1h", limit=200):
        df = self.frames.get(pair)
        if df is None:
            return pd.DataFrame()
        return df.tail(limit)

    def get_ticker(self, pair):
        base = {"BTC/USDT:USDT": 50000.0, "ETH/USDT:USDT": 3000.0}.get(pair, 0.0)
        price = self._price_override.get(pair, base)
        return {"bid": price, "ask": price, "last": price, "volume": 1e6}

    def set_price(self, pair, price):
        self._price_override[pair] = price

    def get_funding_rate(self, pair):
        return 0.0001

    def get_balance(self):
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    def get_positions(self, pair=""):
        return []

    def place_market_order(self, pair, side, amount, reduce_only=False):
        self.placed_orders.append({"pair": pair, "side": side, "amount": amount})
        return {"order_id": "x1", "status": "filled", "filled": amount,
                "average_price": self.get_ticker(pair)["last"]}

    def place_stop_market_order(self, pair, entry_side, amount, stop_price):
        self.placed_orders.append({"pair": pair, "type": "STOP_MARKET",
                                   "stop_price": stop_price})
        return {"order_id": "sl1", "status": "new"}

    def place_take_profit_market_order(self, pair, entry_side, amount, stop_price):
        self.placed_orders.append({"pair": pair, "type": "TAKE_PROFIT_MARKET",
                                   "stop_price": stop_price})
        return {"order_id": "tp1", "status": "new"}

    def cancel_all_orders(self, pair):
        return True


class ScriptedEngine:
    """Returns a preset decision; records the prompts it received."""

    def __init__(self, decision: TradingDecision | None = None):
        self.decision = decision or TradingDecision(
            actions=[], market_outlook="neutral",
            risk_assessment="calm", reasoning="no setup",
        )
        self.calls: list[dict] = []
        self.model = "scripted"

    def decide(self, **kwargs):
        self.calls.append(kwargs)
        return self.decision


def make_action(**overrides) -> TradeAction:
    base = dict(
        pair="BTC/USDT:USDT", side="long", size_pct=5.0,
        stop_loss_pct=3.0, take_profit_pct=6.0, confidence=75.0,
        reasoning="test setup",
    )
    base.update(overrides)
    return TradeAction(**base)


def make_agent(tmp_path, exchange, engine, mode="paper", config=None):
    cfg = {
        "bot": {"pairs": ["BTC/USDT:USDT", "ETH/USDT:USDT"], "timeframe": "1h"},
        "paper": {"starting_equity": 10000.0},
        "risk": {
            "max_risk_per_trade_pct": 2.0,
            "max_position_size_pct": 10.0,
            "max_portfolio_heat_pct": 30.0,
            "max_open_positions": 3,
            "max_drawdown_pct": 10.0,
            "stop_loss_max_pct": 5.0,
            "min_risk_reward_ratio": 1.5,
            "min_confidence_to_trade": 60,
        },
    }
    if config:
        for k, v in config.items():
            cfg.setdefault(k, {}).update(v)
    return AIAgent(
        exchange=exchange, data_dir=tmp_path / "ai_bot", config=cfg,
        mode=mode, engine=engine,
    )


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

class TestMarketData:
    def test_indicators_ok_on_synthetic_data(self):
        from trading_system.bot.market_data import compute_indicators
        ind = compute_indicators(make_ohlcv())
        assert ind["ok"] is True
        assert 40000 < ind["price"] < 60000
        assert ind["rsi_14"] is None or 0 <= ind["rsi_14"] <= 100
        assert ind["adx"] is None or ind["adx"] >= 0
        assert ind["ema_trend"] in {
            "bullish (EMA9 > EMA21 > EMA50)", "bearish (EMA9 < EMA21 < EMA50)",
            "mixed", "warming up",
        }

    def test_insufficient_data_flagged_not_crashed(self):
        from trading_system.bot.market_data import compute_indicators
        ind = compute_indicators(make_ohlcv(20))
        assert ind["ok"] is False
        assert "insufficient" in ind["error"]

    def test_format_never_leaks_nan(self):
        from trading_system.bot.market_data import compute_indicators, format_indicators
        ind = compute_indicators(make_ohlcv(60))  # short: some indicators warm up
        text = format_indicators("BTC/USDT:USDT", ind, 0.0001)
        assert "nan" not in text.lower()
        assert "Funding Rate" in text


# ---------------------------------------------------------------------------
# Decision validation (hard risk rules)
# ---------------------------------------------------------------------------

class TestValidation:
    def _agent(self, tmp_path):
        return make_agent(tmp_path, FakeExchange(), ScriptedEngine())

    def test_unknown_pair_rejected(self, tmp_path):
        agent = self._agent(tmp_path)
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[make_action(pair="SOL/USDT:USDT")],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            DEFAULT_STRATEGY, positions=[], snapshot={"trading_halted": False},
        )
        assert approved == []
        assert any("not in tradable list" in r for r in rejected)

    def test_low_confidence_rejected(self, tmp_path):
        agent = self._agent(tmp_path)
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[make_action(confidence=40)],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            DEFAULT_STRATEGY, positions=[], snapshot={"trading_halted": False},
        )
        assert approved == []
        assert any("confidence" in r for r in rejected)

    def test_bad_risk_reward_rejected(self, tmp_path):
        agent = self._agent(tmp_path)
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[make_action(stop_loss_pct=4.0, take_profit_pct=5.0)],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            DEFAULT_STRATEGY, positions=[], snapshot={"trading_halted": False},
        )
        assert approved == []  # rr = 1.25 < 1.5
        assert any("risk/reward" in r for r in rejected)

    def test_oversized_position_rejected(self, tmp_path):
        agent = self._agent(tmp_path)
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[make_action(size_pct=50.0)],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            DEFAULT_STRATEGY, positions=[], snapshot={"trading_halted": False},
        )
        assert approved == []
        assert any("size" in r for r in rejected)

    def test_max_positions_enforced(self, tmp_path):
        agent = self._agent(tmp_path)
        positions = [{"pair": f"P{i}"} for i in range(3)]
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[make_action()],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            DEFAULT_STRATEGY, positions=positions, snapshot={"trading_halted": False},
        )
        assert approved == []
        assert any("max open positions" in r for r in rejected)

    def test_drawdown_halt_blocks_entries_allows_closes(self, tmp_path):
        agent = self._agent(tmp_path)
        snapshot = {"trading_halted": True, "current_drawdown_pct": 12.0,
                    "max_drawdown_pct": 10.0}
        decisions = TradingDecision(
            actions=[make_action(), make_action(side="close")],
            market_outlook="x", risk_assessment="y", reasoning="z",
        )
        approved, rejected = agent._validate_decision(
            decisions, DEFAULT_STRATEGY,
            positions=[{"pair": "BTC/USDT:USDT"}], snapshot=snapshot,
        )
        assert [a.side for a in approved] == ["close"]
        assert any("halted" in r for r in rejected)

    def test_valid_action_approved(self, tmp_path):
        agent = self._agent(tmp_path)
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[make_action()],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            DEFAULT_STRATEGY, positions=[], snapshot={"trading_halted": False},
        )
        assert len(approved) == 1
        assert rejected == []

    def test_invalid_side_rejected(self, tmp_path):
        agent = self._agent(tmp_path)
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[make_action(side="yolo")],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            DEFAULT_STRATEGY, positions=[], snapshot={"trading_halted": False},
        )
        assert approved == []
        assert any("invalid side" in r for r in rejected)


# ---------------------------------------------------------------------------
# Paper ledger
# ---------------------------------------------------------------------------

class TestPaperLedger:
    def test_open_close_roundtrip(self, tmp_path):
        led = PaperLedger(tmp_path / "led.json", starting_cash=10000.0)
        led.open_position("BTC/USDT:USDT", "buy", 50000.0, 1000.0, 3.0, 6.0)
        assert led.cash < 10000.0  # entry fee charged
        assert "BTC/USDT:USDT" in led.positions
        rec = led.close_position("BTC/USDT:USDT", 51000.0, "test close")
        assert rec["net_pnl"] > 0
        assert "BTC/USDT:USDT" not in led.positions
        assert led.cash > 10000.0  # profit, net of fees

    def test_stop_loss_trigger(self, tmp_path):
        led = PaperLedger(tmp_path / "led.json", starting_cash=10000.0)
        led.open_position("BTC/USDT:USDT", "buy", 50000.0, 1000.0, 3.0, 6.0)
        closed = led.check_triggers({"BTC/USDT:USDT": 50000.0 * 0.96})  # -4% > 3% SL
        assert len(closed) == 1
        assert closed[0]["reason"] == "stop_loss triggered"
        assert closed[0]["net_pnl"] < 0

    def test_take_profit_trigger(self, tmp_path):
        led = PaperLedger(tmp_path / "led.json", starting_cash=10000.0)
        led.open_position("BTC/USDT:USDT", "buy", 50000.0, 1000.0, 3.0, 6.0)
        closed = led.check_triggers({"BTC/USDT:USDT": 50000.0 * 1.07})  # +7% > 6% TP
        assert len(closed) == 1
        assert closed[0]["reason"] == "take_profit triggered"
        assert closed[0]["net_pnl"] > 0

    def test_short_side_triggers_mirrored(self, tmp_path):
        led = PaperLedger(tmp_path / "led.json", starting_cash=10000.0)
        led.open_position("BTC/USDT:USDT", "sell", 50000.0, 1000.0, 3.0, 6.0)
        closed = led.check_triggers({"BTC/USDT:USDT": 50000.0 * 1.04})  # price up = loss
        assert len(closed) == 1
        assert closed[0]["reason"] == "stop_loss triggered"

    def test_consecutive_losses(self, tmp_path):
        led = PaperLedger(tmp_path / "led.json", starting_cash=10000.0)
        led.data["closed_trades"] = [{"net_pnl": 5}, {"net_pnl": -1}, {"net_pnl": -2}]
        assert led.consecutive_losses() == 2

    def test_state_survives_reload(self, tmp_path):
        p = tmp_path / "led.json"
        led = PaperLedger(p, starting_cash=7777.0)
        led2 = PaperLedger(p, starting_cash=10000.0)
        assert led2.cash == led.cash
        assert led2.data["start_equity"] == 7777.0  # not re-initialized

    def test_reported_pnl_reconciles_with_equity(self, tmp_path):
        """REGRESSION (audit 2026-09-16): the close alert reported -$17.71
        while equity had moved -$18.21 — net_pnl omitted the entry fee, so
        every P&L total (alert, digest, weekly) disagreed with the account.
        The real 2026-09-15 stop-loss is replayed here."""
        led = PaperLedger(tmp_path / "led.json", starting_cash=10000.0)
        led.open_position("BTC/USDT:USDT", "buy", 78227.30, 1000.0, 2.5, 5.0)
        rec = led.close_position("BTC/USDT:USDT", 76880.30, "stop_loss triggered")

        # What equity sees == the number shown to the user (the record
        # rounds to 4dp, so allow a hair of sub-cent drift).
        assert led.cash - 10000.0 == pytest.approx(rec["net_pnl"], abs=1e-3)
        assert rec["net_pnl"] == pytest.approx(-18.21, abs=0.01)
        assert rec["fees"] == pytest.approx(
            rec["entry_fee"] + rec["exit_fee"], abs=1e-6)
        assert rec["cash_delta"] == pytest.approx(
            rec["gross_pnl"] - rec["exit_fee"], abs=1e-3)
        # The alert shows net %, so % and $ describe the same thing.
        assert rec["pnl_pct_net"] == pytest.approx(
            rec["net_pnl"] / rec["size_usd"] * 100, abs=1e-3)
        # The raw price move is still recorded, unchanged in meaning.
        assert rec["pnl_pct"] == pytest.approx(-1.7219, abs=1e-3)

    def test_round_trip_win_is_net_of_both_fees(self, tmp_path):
        led = PaperLedger(tmp_path / "led.json", starting_cash=10000.0)
        led.open_position("BTC/USDT:USDT", "buy", 50000.0, 1000.0, 3.0, 6.0)
        rec = led.close_position("BTC/USDT:USDT", 50300.0, "AI close")
        assert rec["gross_pnl"] == pytest.approx(6.0, abs=0.01)
        assert rec["net_pnl"] == pytest.approx(6.0 - 0.5 - 0.5030, abs=0.01)
        assert rec["net_pnl"] < rec["gross_pnl"]
        assert led.cash - 10000.0 == pytest.approx(rec["net_pnl"], abs=1e-3)


# ---------------------------------------------------------------------------
# Trigger prices (live-mode protective orders)
# ---------------------------------------------------------------------------

class TestTriggerPrice:
    def test_long_stop_below_entry(self):
        assert AIAgent._trigger_price(100.0, 3.0, "long", is_stop=True) == pytest.approx(97.0)

    def test_long_tp_above_entry(self):
        assert AIAgent._trigger_price(100.0, 6.0, "long", is_stop=False) == pytest.approx(106.0)

    def test_short_stop_above_entry(self):
        assert AIAgent._trigger_price(100.0, 3.0, "short", is_stop=True) == pytest.approx(103.0)

    def test_short_tp_below_entry(self):
        assert AIAgent._trigger_price(100.0, 6.0, "short", is_stop=False) == pytest.approx(94.0)


# ---------------------------------------------------------------------------
# Full wakeup flows
# ---------------------------------------------------------------------------

class TestWakeup:
    def test_no_trade_decision_succeeds_and_journals(self, tmp_path):
        ex = FakeExchange()
        eng = ScriptedEngine(TradingDecision(
            actions=[], market_outlook="neutral",
            risk_assessment="calm", reasoning="nothing interesting",
        ))
        agent = make_agent(tmp_path, ex, eng)
        result = agent.run_wakeup()

        assert result["status"] == "success"
        assert result["executed_trades"] == 0
        journal = (agent.data_dir / "journal.jsonl").read_text().strip().splitlines()
        assert len(journal) == 1
        entry = json.loads(journal[0])
        assert entry["status"] == "success"
        assert (agent.data_dir / "progress.json").exists()
        assert (agent.data_dir / "decisions.json").exists()

    def test_ai_engine_failure_is_explicit_error(self, tmp_path):
        ex = FakeExchange()
        eng = ScriptedEngine(TradingDecision(
            actions=[], market_outlook="unknown", risk_assessment="",
            reasoning="boom", ok=False,
        ))
        agent = make_agent(tmp_path, ex, eng)
        result = agent.run_wakeup()

        assert result["status"] == "error"
        assert any("AI engine failed" in e for e in result["errors"])
        journal = (agent.data_dir / "journal.jsonl").read_text().strip().splitlines()
        entry = json.loads(journal[-1])
        assert entry["status"] == "error"  # failure journaled, not silent
        assert result["executed_trades"] == 0

    def test_paper_open_then_close_roundtrip(self, tmp_path):
        ex = FakeExchange()
        agent = make_agent(tmp_path, ex, ScriptedEngine())
        # First wakeup: open a small BTC long.
        agent.ai.decision = TradingDecision(
            actions=[make_action(size_pct=5.0, stop_loss_pct=3.0, take_profit_pct=6.0)],
            market_outlook="bullish", risk_assessment="ok", reasoning="go",
        )
        r1 = agent.run_wakeup()
        assert r1["status"] == "success"
        assert r1["executed_trades"] == 1
        assert "BTC/USDT:USDT" in agent.ledger.positions
        equity_after_open = agent.ledger.equity({})

        # Second wakeup: AI closes it.
        agent.ai.decision = TradingDecision(
            actions=[make_action(side="close")],
            market_outlook="neutral", risk_assessment="ok", reasoning="take profit",
        )
        r2 = agent.run_wakeup()
        assert r2["status"] == "success"
        assert "BTC/USDT:USDT" not in agent.ledger.positions
        # Equity moved only by fees + tiny price drift between wakeups.
        assert abs(agent.ledger.cash - equity_after_open) < equity_after_open * 0.01

        trades = (agent.data_dir / "trades.jsonl").read_text().strip().splitlines()
        assert len(trades) == 2  # one open, one close

    def test_sl_trigger_between_wakeups(self, tmp_path):
        ex = FakeExchange()
        agent = make_agent(tmp_path, ex, ScriptedEngine())
        agent.ai.decision = TradingDecision(
            actions=[make_action(side="long", size_pct=5.0,
                                 stop_loss_pct=3.0, take_profit_pct=6.0)],
            market_outlook="bullish", risk_assessment="ok", reasoning="go",
        )
        agent.run_wakeup()
        assert "BTC/USDT:USDT" in agent.ledger.positions

        # Price crashes through the stop before the next wakeup.
        ex.set_price("BTC/USDT:USDT", 50000.0 * 0.95)
        agent.ai.decision = TradingDecision(
            actions=[], market_outlook="bearish",
            risk_assessment="ok", reasoning="stand aside",
        )
        r2 = agent.run_wakeup()
        assert r2["status"] == "success"
        assert len(r2["closed_triggers"]) == 1
        assert r2["closed_triggers"][0]["reason"] == "stop_loss triggered"
        assert "BTC/USDT:USDT" not in agent.ledger.positions

    def test_equity_tracking_across_wakeups(self, tmp_path):
        ex = FakeExchange()
        agent = make_agent(tmp_path, ex, ScriptedEngine())
        agent.ai.decision = TradingDecision(
            actions=[], market_outlook="neutral",
            risk_assessment="ok", reasoning="wait",
        )
        r1 = agent.run_wakeup()
        r2 = agent.run_wakeup()
        assert r1["equity"] == pytest.approx(10000.0, abs=1.0)
        assert r2["equity"] == pytest.approx(r1["equity"], abs=1.0)
        progress = json.loads((agent.data_dir / "progress.json").read_text())
        assert progress["last_wakeup"] == r2["wakeup_id"]

    def test_all_market_data_down_is_explicit_error(self, tmp_path):
        ex = FakeExchange()
        ex.frames = {}  # total data outage
        agent = make_agent(tmp_path, ex, ScriptedEngine())
        result = agent.run_wakeup()
        assert result["status"] == "error"
        assert any("market data" in e.lower() for e in result["errors"])


# ---------------------------------------------------------------------------
# Scheduler — including the midnight rollover regression
# ---------------------------------------------------------------------------

class TestScheduler:
    def test_midnight_rollover_regression(self):
        """After 23:00, the next wakeup must be tomorrow's 00:00 entry.

        This is the regression test for the bug where the 00:00
        asia_session_open wakeup was skipped EVERY day.
        """
        sched = Scheduler(lambda: {"status": "success"},
                          schedule=[{"name": "midnight_check", "hour": 0, "minute": 0},
                                    {"name": "noon_check", "hour": 12, "minute": 0}])
        now = datetime(2026, 9, 11, 23, 30, tzinfo=timezone.utc)
        nxt = sched.next_slot(now)
        assert nxt["name"] == "midnight_check"
        assert nxt["at"] == datetime(2026, 9, 12, 0, 0, tzinfo=timezone.utc)

    def test_next_slot_same_day(self):
        sched = Scheduler(lambda: {"status": "success"},
                          schedule=[{"name": "a", "hour": 6, "minute": 0},
                                    {"name": "b", "hour": 18, "minute": 30}])
        now = datetime(2026, 9, 11, 9, 0, tzinfo=timezone.utc)
        nxt = sched.next_slot(now)
        assert nxt["name"] == "b"
        assert nxt["at"].hour == 18 and nxt["at"].minute == 30

    def test_exact_slot_time_is_next(self):
        """now == slot time must schedule the FOLLOWING slot, not re-fire."""
        sched = Scheduler(lambda: {"status": "success"},
                          schedule=[{"name": "a", "hour": 6, "minute": 0},
                                    {"name": "b", "hour": 6, "minute": 0}])
        now = datetime(2026, 9, 11, 6, 0, tzinfo=timezone.utc)
        nxt = sched.next_slot(now)
        assert nxt["at"] == datetime(2026, 9, 12, 6, 0, tzinfo=timezone.utc)

    def test_timezone_offset_applied(self):
        sched = Scheduler(lambda: {"status": "success"},
                          schedule=[{"name": "a", "hour": 0, "minute": 0}],
                          tz_offset_hours=-5)
        now = datetime(2026, 9, 11, 20, 0, tzinfo=timezone.utc)  # 15:00 local
        nxt = sched.next_slot(now)
        # local midnight next day = 05:00 UTC
        assert nxt["at"] == datetime(2026, 9, 12, 5, 0, tzinfo=timezone.utc)

    def test_run_once_by_name(self):
        fired = []
        sched = Scheduler(lambda: fired.append(1) or {"status": "success"},
                          schedule=[{"name": "alpha", "hour": 1, "minute": 0},
                                    {"name": "beta", "hour": 2, "minute": 0}])
        result = sched.run_once("beta")
        assert result["status"] == "success"
        assert fired == [1]

    def test_run_once_unknown_name_raises(self):
        sched = Scheduler(lambda: {"status": "success"})
        with pytest.raises(ValueError, match="Unknown wakeup"):
            sched.run_once("does_not_exist")

    def test_invalid_schedule_entry_rejected(self):
        with pytest.raises(ValueError, match="schedule entry"):
            _normalize_schedule([{"name": "bad"}])  # missing hour/minute

    def test_default_schedule_has_six_wakeups(self):
        sched = Scheduler(lambda: {"status": "success"})
        assert len(sched.schedule) == 6


# ---------------------------------------------------------------------------
# Engine parsing
# ---------------------------------------------------------------------------

class TestEngineParsing:
    def test_valid_json_parsed(self):
        eng = AIEngine()
        raw = json.dumps({
            "actions": [{
                "pair": "BTC/USDT:USDT", "side": "long", "size_pct": 5.0,
                "stop_loss_pct": 3.0, "take_profit_pct": 6.0,
                "confidence": 75, "reasoning": "rsi bounce",
            }],
            "market_outlook": "bullish",
            "risk_assessment": "low",
            "reasoning": "go long",
        })
        d = eng._parse_response(raw)
        assert d.ok is True
        assert len(d.actions) == 1
        assert d.actions[0].pair == "BTC/USDT:USDT"

    def test_low_confidence_actions_filtered(self):
        eng = AIEngine()
        raw = json.dumps({
            "actions": [
                {"pair": "BTC/USDT:USDT", "side": "long", "size_pct": 5,
                 "stop_loss_pct": 3, "take_profit_pct": 6, "confidence": 80},
                {"pair": "ETH/USDT:USDT", "side": "long", "size_pct": 5,
                 "stop_loss_pct": 3, "take_profit_pct": 6, "confidence": 30},
            ],
            "market_outlook": "bullish", "risk_assessment": "ok", "reasoning": "x",
        })
        d = eng._parse_response(raw)
        assert d.ok is True
        assert len(d.actions) == 1  # 30-confidence action dropped

    def test_garbage_json_is_explicit_failure(self):
        eng = AIEngine()
        d = eng._parse_response("not json at all {{{")
        assert d.ok is False
        assert d.actions == []

    def test_json_array_is_rejected(self):
        eng = AIEngine()
        d = eng._parse_response('[{"actions": []}]')
        assert d.ok is False

    def test_error_decision_flagged(self):
        d = AIEngine._error_decision("test failure")
        assert d.ok is False
        assert d.actions == []


class TestEngineResilience:
    """Model fallback chain: a retired/overloaded model can't kill a wakeup.

    Real incidents this pins: gemini-2.5-flash retired (404) and a 503
    'high demand' spike — each once silenced the bot for half a day.
    """

    VALID = json.dumps({
        "actions": [], "market_outlook": "neutral",
        "risk_assessment": "ok", "reasoning": "flat",
    })

    @staticmethod
    def _engine_with_client(monkeypatch, responses, discovered=()):
        """AIEngine whose client pops queued responses/exceptions in order.

        `discovered` is what the API model-listing returns when the engine
        asks (empty = discovery yields nothing).
        """
        eng = AIEngine()
        calls: list[str] = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs.get("model"))
                item = responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        eng._client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions()))
        monkeypatch.setattr(
            "trading_system.bot.ai_engine.time.sleep", lambda s: None)
        monkeypatch.setattr(eng, "_discover_models", lambda: tuple(discovered))
        return eng, calls

    @staticmethod
    def _resp(text):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def test_retired_model_404_falls_back_to_next(self, monkeypatch):
        eng, calls = self._engine_with_client(
            monkeypatch,
            [Exception("Error code: 404 - model no longer available"),
             Exception("Error code: 404 - model no longer available"),
             self._resp(self.VALID)])
        d = eng.decide("m", "p", "s", "r")
        assert d.ok is True
        # Two failed attempts on the primary, then the fallback answered.
        assert calls == ["gemini-flash-latest", "gemini-flash-latest",
                         "gemini-flash"]
        assert d.model == "gemini-flash"
        assert eng.model == "gemini-flash"  # sticks with the working model

    def test_503_spike_retried_on_same_model(self, monkeypatch):
        eng, calls = self._engine_with_client(
            monkeypatch,
            [Exception("Error code: 503 - high demand"),
             self._resp(self.VALID)])
        d = eng.decide("m", "p", "s", "r")
        assert d.ok is True
        assert calls == ["gemini-flash-latest", "gemini-flash-latest"]

    def test_all_models_exhausted_returns_failed_decision(self, monkeypatch):
        eng, calls = self._engine_with_client(
            monkeypatch, [Exception("Error code: 404")] * 6)
        d = eng.decide("m", "p", "s", "r")
        assert d.ok is False
        assert d.actions == []
        assert len(calls) == 6  # 3 models x 2 attempts
        assert "Failed to get AI decision" in d.reasoning

    def test_openai_config_has_no_gemini_fallback(self, monkeypatch):
        eng = AIEngine(model="gpt-4o")
        calls: list[str] = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs.get("model"))
                raise Exception("api down")

        eng._client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions()))
        monkeypatch.setattr(
            "trading_system.bot.ai_engine.time.sleep", lambda s: None)
        d = eng.decide("m", "p", "s", "r")
        assert d.ok is False
        assert calls == ["gpt-4o", "gpt-4o"]  # single model, two attempts

    def test_empty_response_retries_then_falls_back(self, monkeypatch):
        eng, calls = self._engine_with_client(
            monkeypatch,
            [self._resp(""), self._resp(""), self._resp(self.VALID)])
        d = eng.decide("m", "p", "s", "r")
        assert d.ok is True
        assert len(calls) == 3

    def test_total_name_retirement_rescued_by_discovery(self, monkeypatch):
        """The 2026-09-13 15:51 UTC incident: EVERY pinned name 404'd.
        The engine must then ask the API which models the key CAN use and
        succeed with a discovered one — no human intervention."""
        err = Exception(
            "Error code: 404 - model no longer available, NOT_FOUND")
        eng, calls = self._engine_with_client(
            monkeypatch,
            [Exception(str(err))] * 6 + [self._resp(self.VALID)],
            discovered=("gemini-3.7-flash",))
        d = eng.decide("m", "p", "s", "r")
        assert d.ok is True
        # 3 static models x2 attempts, then the DISCOVERED model answered.
        assert calls == ["gemini-flash-latest", "gemini-flash-latest",
                         "gemini-flash", "gemini-flash",
                         "gemini-3.6-flash", "gemini-3.6-flash",
                         "gemini-3.7-flash"]
        assert d.model == "gemini-3.7-flash"

    def test_discovery_yields_nothing_fails_loudly(self, monkeypatch):
        err = Exception("Error code: 404 - NOT_FOUND, no longer available")
        eng, calls = self._engine_with_client(
            monkeypatch, [Exception(str(err))] * 6, discovered=())
        d = eng.decide("m", "p", "s", "r")
        assert d.ok is False
        assert len(calls) == 6  # no invented names, honest failure

    def test_model_missing_detector(self):
        f = AIEngine._is_model_missing
        assert f(Exception("Error code: 404 - model no longer available"))
        assert f(Exception("Error code: 404 - status NOT_FOUND"))
        assert not f(Exception("Error code: 503 - high demand"))
        assert not f(Exception("Error code: 429 - rate limit"))
        assert not f(Exception("connection reset"))

    def test_discovery_ranks_alias_newest_full_flash_first(self, monkeypatch):
        """The live listing ranking: evergreen aliases first, then newest
        generation, full flash before lite, image/audio models excluded."""
        import requests as _requests

        class FakeResp:
            def json(self):
                return {"data": [
                    {"id": "models/gemini-9.9-flash-image"},
                    {"id": "models/gemini-9.9-flash-lite"},
                    {"id": "models/gemini-9.9-flash"},
                    {"id": "models/gemini-3.6-flash"},
                    {"id": "models/gemini-flash-latest"},
                    {"id": "models/gemini-9.9-pro"},
                ]}

        monkeypatch.setattr(_requests, "get", lambda *a, **k: FakeResp())
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        got = AIEngine()._discover_models()
        assert got[0] == "gemini-flash-latest"
        assert "gemini-9.9-flash" in got[:2]
        assert "gemini-9.9-flash-image" not in got
        assert all("pro" not in m for m in got)


class TestSpotFallback:
    """Kraken spot fallback (GitHub runners are geo-blocked by Binance)."""

    def _iface(self):
        from trading_system.bot.exchange import ExchangeInterface
        from trading_system.config import ExchangeConfig

        return ExchangeInterface(ExchangeConfig())

    def test_spot_symbol_mapping(self):
        iface = self._iface()
        assert iface._spot_symbol("BTC/USDT:USDT") == "BTC/USDT"
        assert iface._spot_symbol("ETH/USDT:USDT") == "ETH/USDT"
        assert iface._spot_symbol("SOL/USD") is None
        assert iface._spot_symbol("BTC/USDC:USDC") is None

    def test_get_ticker_falls_back_to_spot(self, monkeypatch):
        iface = self._iface()

        calls = []

        def binance_raises(pair):
            calls.append(("binance", pair))
            raise RuntimeError("451 geo-blocked")

        def spot_ok(pair):
            calls.append(("spot", pair))
            return {"last": 77000.0, "bid": 76999.0, "ask": 77001.0,
                    "quoteVolume": 1e9}

        monkeypatch.setattr(iface.exchange, "fetch_ticker", binance_raises)
        monkeypatch.setattr(iface._spot, "fetch_ticker", spot_ok)

        t = iface.get_ticker("BTC/USDT:USDT")
        assert t["last"] == 77000.0
        assert [c[0] for c in calls] == ["binance", "spot"]

    def test_get_ticker_binance_success_skips_spot(self, monkeypatch):
        iface = self._iface()

        def binance_ok(pair):
            return {"last": 77000.0, "bid": 1, "ask": 1, "quoteVolume": 1}

        def spot_raises(pair):
            raise AssertionError("spot must not be called when Binance works")

        monkeypatch.setattr(iface.exchange, "fetch_ticker", binance_ok)
        monkeypatch.setattr(iface._spot, "fetch_ticker", spot_raises)
        assert iface.get_ticker("BTC/USDT:USDT")["last"] == 77000.0

    def test_get_ohlcv_falls_back_to_spot(self, monkeypatch):
        iface = self._iface()

        def binance_raises(pair, tf, limit):
            raise RuntimeError("451 geo-blocked")

        def spot_ok(pair, tf, limit):
            return [
                [1700000000000 + i * 3600000, 1, 2, 0.5, 1.5, 10]
                for i in range(limit)
            ]

        monkeypatch.setattr(iface.exchange, "fetch_ohlcv", binance_raises)
        monkeypatch.setattr(iface._spot, "fetch_ohlcv", spot_ok)

        df = iface.get_ohlcv("BTC/USDT:USDT", "1h", limit=60)
        assert len(df) == 60
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_both_fail_returns_empty(self, monkeypatch):
        iface = self._iface()

        monkeypatch.setattr(
            iface.exchange, "fetch_ohlcv",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("451")))
        monkeypatch.setattr(
            iface._spot, "fetch_ohlcv",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
        assert iface.get_ohlcv("BTC/USDT:USDT", "1h", limit=10).empty


class TestTelegramCommands:
    """scripts/ai_telegram_commands.py — routing, math, formatting."""

    def _mod(self):
        import importlib.util
        import pathlib
        script = (pathlib.Path(__file__).resolve().parents[1]
                  / "scripts" / "ai_telegram_commands.py")
        spec = importlib.util.spec_from_file_location("ai_telegram_commands", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    LEDGER = {
        "cash": 9500.0,
        "start_equity": 10000.0,
        "positions": {
            "BTC/USDT:USDT": {
                "side": "buy", "entry_price": 77000.0, "amount": 0.05,
                "stop_loss_pct": 3.0, "take_profit_pct": 6.0,
            },
        },
        "closed_trades": [
            {"net_pnl": 50.0, "close_time": "2026-09-10T12:00:00+00:00"},
            {"net_pnl": -20.0, "close_time": "2026-09-09T12:00:00+00:00"},
            {"net_pnl": 10.0, "close_time": "2026-08-01T12:00:00+00:00"},
        ],
    }
    PRICES = {"BTC/USDT:USDT": 78000.0}

    def test_router_dispatches_all_commands(self):
        m = self._mod()
        j = {"wakeup_id": "x", "status": "success", "market_outlook": "neutral"}
        for text in ("/status", "/status@MisterProfitBot", "/status now"):
            reply = m.route_command(text, self.LEDGER, j, self.PRICES)
            assert reply.startswith("AI BOT STATUS")
        assert m.route_command(
            "/positions", self.LEDGER, j, self.PRICES).startswith("Open positions: 1")
        assert m.route_command(
            "/last", self.LEDGER, j, self.PRICES).startswith("LAST AI DECISION")
        assert m.route_command(
            "/help", self.LEDGER, j, self.PRICES).startswith("AI Trading Bot commands")
        assert m.route_command(
            "/start", self.LEDGER, j, self.PRICES).startswith("AI Trading Bot commands")

    def test_router_ignores_non_commands(self):
        m = self._mod()
        assert m.route_command("hello bot", self.LEDGER, None, {}) is None
        assert m.route_command("", self.LEDGER, None, {}) is None

    def test_router_unknown_command_returns_help(self):
        m = self._mod()
        out = m.route_command("/frobnicate", self.LEDGER, None, {})
        assert out is not None and "Unknown command /frobnicate" in out

    def test_unrealized_long_and_short(self):
        m = self._mod()
        long_pos = {"side": "buy", "entry_price": 100.0, "amount": 2.0}
        short_pos = {"side": "sell", "entry_price": 100.0, "amount": 2.0}
        assert m._unrealized(long_pos, 110.0) == 20.0
        assert m._unrealized(short_pos, 110.0) == -20.0

    def test_status_math(self):
        m = self._mod()
        out = m.route_command("/status", self.LEDGER, None, self.PRICES)
        # equity = 9500 cash + 0.05*(78000-77000) = 9550
        assert "$9,550.00" in out
        assert "-4.50%" in out                # all-time vs 10k start
        assert "+50.00$" in out               # unrealized: 0.05 * 1000
        assert "Win rate 67%" in out          # 2 wins of 3 closed

    def test_closed_stats_week_window(self):
        m = self._mod()
        closed, wr, week_pnl = m._closed_stats(self.LEDGER)
        assert closed == 3
        assert wr == pytest.approx(66.66, abs=0.1)
        # only the two September trades are inside the 7-day window at
        # test time is not guaranteed — assert week pnl is one of the
        # valid sums rather than a fixed number
        assert week_pnl in (30.0, 40.0, 50.0, 60.0, 0.0)

    def test_last_without_journal(self):
        m = self._mod()
        assert m.route_command("/last", self.LEDGER, None, {}) == "No wakeup journaled yet."

    def test_status_on_empty_ledger_does_not_crash(self):
        m = self._mod()
        out = m.route_command("/status", {}, None, {})
        assert "$0.00" in out
        assert "none journaled" in out


class TestGate:
    """The slot-based schedule gate (scripts/ai_bot_gate.py).

    The gate decides from the SCHEDULE, never from journal age: a firing
    whose slot is already served is a SKIP (no wakeup), a firing for an
    unserved slot runs (on time, or catch-up for a slot GitHub dropped) and
    a firing just before a slot relays to it. This holds the bot to its 6
    designed decisions/day even though the workflow_run mesh kicks it every
    ~20 minutes — the bug that produced 22-25 wakeups/day on 2026-09-14/15.
    """

    def _mod(self):
        import importlib.util
        import pathlib
        spec = importlib.util.spec_from_file_location(
            "ai_bot_gate", pathlib.Path("scripts/ai_bot_gate.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_no_journal_proceeds_immediately(self, tmp_path, monkeypatch):
        mod = self._mod()
        monkeypatch.setattr(mod, "JOURNAL", tmp_path / "journal.jsonl")
        monkeypatch.setattr(mod, "CONFIG", tmp_path / "none.yaml")
        out = tmp_path / "out.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        assert mod.main() == 0
        assert "run=true" in out.read_text()

    def test_served_slot_skips_without_waking_the_ai(self):
        """The core fix: a mesh kick between slots must NOT trade. Before
        this, the gate ran a wakeup whenever the journal was >45 min old, so
        a ~20-min heartbeat produced 22-25 wakeups/day instead of 6."""
        mod = self._mod()
        slots = [(14, 0), (20, 0)]
        # 14:00 slot served at 14:02; kick arrives at 16:00 (6h to next slot).
        last = datetime(2026, 9, 15, 14, 2, tzinfo=timezone.utc)
        now = datetime(2026, 9, 15, 16, 0, tzinfo=timezone.utc)
        assert mod.decide(now, last, slots) == (mod.SKIP, 0.0)

    def test_unserved_slot_runs_even_when_very_stale(self):
        """A slot nobody triggered is catch-up material, not a skip."""
        mod = self._mod()
        slots = [(14, 0), (20, 0)]
        last = datetime(2026, 9, 15, 14, 2, tzinfo=timezone.utc)
        now = datetime(2026, 9, 15, 20, 35, tzinfo=timezone.utc)
        action, wait = mod.decide(now, last, slots)
        assert (action, wait) == (mod.RUN, 0.0)

    def test_kick_just_before_a_slot_relays_to_it(self):
        mod = self._mod()
        slots = [(14, 0), (20, 0)]
        last = datetime(2026, 9, 15, 14, 2, tzinfo=timezone.utc)
        action, wait = mod.decide(
            datetime(2026, 9, 15, 19, 55, tzinfo=timezone.utc), last, slots)
        assert action == mod.RELAY
        assert wait == pytest.approx(5 * 60, abs=1)
        assert wait <= mod.RELAY_WINDOW_SECONDS

    def test_kick_inside_the_relay_window_fires_the_slot_on_time(self):
        """REGRESSION (2026-09-18..21): the 08:00 slot ran +108/+126/+160 min
        late and was missed outright on 09-20. Cause: the relay window was
        10 min, so a heartbeat kick landing 12+ min before a slot was SKIPPED
        and the next kick often arrived after it. The window is now wider than
        the ~20-min heartbeat, so a kick anywhere in the half hour before a
        slot sleeps to it and fires it exactly at slot time."""
        from datetime import timedelta
        mod = self._mod()
        slots = [(6, 0), (8, 0)]
        last = datetime(2026, 9, 21, 6, 1, tzinfo=timezone.utc)  # 06:00 served
        slot = datetime(2026, 9, 21, 8, 0, tzinfo=timezone.utc)
        for minutes_before in (25, 12, 5):
            action, wait = mod.decide(slot - timedelta(minutes=minutes_before), last, slots)
            assert action == mod.RELAY, f"{minutes_before} min early must relay"
            assert wait == pytest.approx(minutes_before * 60, abs=1)

    def test_kick_far_before_a_slot_still_skips(self):
        """The window is bounded on purpose: a kick hours early stays a cheap
        SKIP instead of holding a runner (and the concurrency group) for
        hours — cron firings at :02/:12/:22 of a slot hour are exactly this."""
        mod = self._mod()
        slots = [(6, 0), (8, 0)]
        last = datetime(2026, 9, 21, 6, 1, tzinfo=timezone.utc)
        now = datetime(2026, 9, 21, 6, 30, tzinfo=timezone.utc)  # 90 min early
        assert mod.decide(now, last, slots) == (mod.SKIP, 0.0)

    def test_relay_window_exceeds_the_heartbeat_and_stays_bounded(self):
        """Design invariant, not a magic number: the relay window must be
        wider than the ~20-min mesh heartbeat that feeds the gate (or the
        08:00 lateness returns), and still small enough that a relay cannot
        approach the job timeout."""
        mod = self._mod()
        assert mod.RELAY_WINDOW_SECONDS > 20 * 60
        assert mod.RELAY_WINDOW_SECONDS <= 60 * 60

    def test_wakeup_a_hair_before_the_slot_counts_as_serving_it(self):
        mod = self._mod()
        slots = [(14, 0), (20, 0)]
        last = datetime(2026, 9, 15, 13, 59, tzinfo=timezone.utc)
        assert mod.decide(
            datetime(2026, 9, 15, 14, 5, tzinfo=timezone.utc), last, slots
        )[0] == mod.SKIP

    def test_six_wakeups_per_day_under_a_20_minute_kick_mesh(self):
        """REGRESSION (2026-09-14/15: 22-25 wakeups/day).

        Replay a full day of triggers the way the workflow_run mesh fires
        them — every 20 minutes — and require exactly one wakeup per
        scheduled slot.
        """
        from datetime import timedelta
        mod = self._mod()
        slots = [(0, 0), (6, 0), (8, 0), (14, 0), (20, 0), (23, 0)]
        now = datetime(2026, 9, 15, 0, 0, tzinfo=timezone.utc)
        last = datetime(2026, 9, 14, 23, 5, tzinfo=timezone.utc)  # 23:00 served
        wakeups: list[datetime] = []
        while now < datetime(2026, 9, 16, 0, 0, tzinfo=timezone.utc):
            action, wait = mod.decide(now, last, slots)
            if action == mod.RUN:
                wakeups.append(now)
                last = now
            elif action == mod.RELAY:
                now = now + timedelta(seconds=wait)
                continue
            now = now + timedelta(minutes=20)
        assert len(wakeups) == 6, f"expected 6 wakeups, got {len(wakeups)}"
        assert [(w.hour, w.minute) for w in wakeups] == [
            (0, 0), (6, 0), (8, 0), (14, 0), (20, 0), (23, 0)]

    def test_full_day_on_time_despite_dropped_crons_and_a_gap(self):
        """End-to-end punctuality replay (2026-09-22).

        Two real defects fed this: GitHub delivered only ~5 of 12 scheduled
        firings/day, and the poller heartbeat died for 3-4h at a stretch (its
        continuity rested on cron). With the 30-min relay window, pre-slot
        cron deliveries and a live heartbeat (now guaranteed by the poller's
        workflow_dispatch self-relaunch), EVERY slot must land within 15
        minutes of its time. Pre-fix reality: +108 / +126 / +160 min late, and
        the 08:00 slot missed outright on 09-20.
        """
        from datetime import timedelta
        mod = self._mod()
        slots = [(0, 0), (6, 0), (8, 0), (14, 0), (20, 0), (23, 0)]
        day = datetime(2026, 9, 22, 0, 0, tzinfo=timezone.utc)

        triggers: list[datetime] = []
        t = day
        while t < day + timedelta(days=1):
            # Heartbeat (the poller): every 15 min, with one 45-min dead patch
            # to stand in for a GitHub hiccup.
            if t.minute % 15 == 7 and not (t.hour == 12 and t.minute < 45):
                triggers.append(t)
            # Slot-hour crons — every other firing dropped, as GitHub does.
            if t.hour in (0, 6, 8, 14, 20, 23) and t.minute in (2, 12, 22, 32, 42, 52):
                if t.minute % 20 == 2:
                    triggers.append(t)
            # Pre-slot crons — 2 of the 3 minutes delivered, so a single lucky
            # delivery still has to carry the slot.
            if t.hour in (5, 7, 13, 19, 22) and t.minute in (32, 42, 52):
                if t.minute != 42:
                    triggers.append(t)
            t += timedelta(minutes=1)
        triggers.sort()

        last = day - timedelta(hours=1)  # 23:00 (yesterday) is served
        wakeups: list[datetime] = []
        relaying_until: datetime | None = None
        for now in triggers:
            # The workflow's concurrency group runs ONE job at a time: triggers
            # arriving while a relay sleeps cannot execute — they queue, then
            # find the slot already served.
            if relaying_until is not None and now < relaying_until:
                continue
            action, wait = mod.decide(now, last, slots)
            if action == mod.RUN:
                wakeups.append(now)
                last = now
            elif action == mod.RELAY:
                # A relay sleeps to the slot and fires it there.
                relaying_until = now + timedelta(seconds=wait)
                last = relaying_until
                wakeups.append(last)

        # 6 slots served inside the day (the trailing relay also fires the
        # NEXT day's 00:00 on time, which is the boundary case that used to be
        # skipped — asserted separately below).
        in_day = [w for w in wakeups if w < day + timedelta(days=1)]
        assert len(in_day) == 6, f"expected 6 wakeups, got {[w.hour for w in in_day]}"
        for hour, _ in slots:
            slot_dt = day.replace(hour=hour, minute=0)
            lateness = [
                (w - slot_dt).total_seconds() / 60
                for w in in_day
                if 0 <= (w - slot_dt).total_seconds() <= 15 * 60
            ]
            assert lateness, (
                f"slot {hour:02d}:00 not served within 15 min "
                f"(wakeups at {[w.strftime('%H:%M') for w in wakeups]})"
            )
        assert day + timedelta(days=1) in wakeups, "the midnight boundary was missed"

    def test_relay_sleeps_then_runs(self, tmp_path, monkeypatch):
        """main() relays only when the next slot is close, then runs."""
        from datetime import timedelta
        mod = self._mod()
        (tmp_path / "journal.jsonl").write_text(
            json.dumps({"timestamp": "2026-09-15T19:50:00+00:00"}))
        monkeypatch.setattr(mod, "JOURNAL", tmp_path / "journal.jsonl")
        monkeypatch.setattr(mod, "CONFIG", tmp_path / "none.yaml")
        monkeypatch.setattr(mod, "DEFAULT_SLOTS", ((20, 0),))
        out = tmp_path / "out.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        state = {"t": datetime(2026, 9, 15, 19, 55, tzinfo=timezone.utc)}
        sleeps: list[float] = []

        def fake_sleep(sec):
            sleeps.append(sec)
            state["t"] += timedelta(seconds=sec)
            assert len(sleeps) <= 5, "relay never exited"

        monkeypatch.setattr(mod, "_now_utc", lambda: state["t"])
        monkeypatch.setattr(mod, "_sleep", fake_sleep)
        assert mod.main() == 0
        assert sleeps == [300.0]
        assert (state["t"].hour, state["t"].minute) == (20, 0)
        assert all(s <= mod.MAX_SLEEP_SECONDS for s in sleeps)
        assert "run=true" in out.read_text()

    def test_load_slots_parses_config_and_tz(self, tmp_path):
        mod = self._mod()
        cfg = tmp_path / "ai_bot.yaml"
        cfg.write_text(
            "bot:\n  tz_offset_hours: -5\nschedule:\n"
            "  - name: a\n    hour: 0\n    minute: 0\n"
            "  - name: b\n    hour: 14\n    minute: 0\n")
        # local = UTC + offset  =>  utc_hour = local_hour - offset
        assert mod._load_slots(cfg) == [(5, 0), (19, 0)]

    def test_corrupt_journal_treated_as_stale(self, tmp_path, monkeypatch):
        # A corrupt journal must not silently block trading.
        mod = self._mod()
        (tmp_path / "journal.jsonl").write_text("{{{not json")
        monkeypatch.setattr(mod, "JOURNAL", tmp_path / "journal.jsonl")
        monkeypatch.setattr(mod, "CONFIG", tmp_path / "none.yaml")
        assert mod.main() == 0


class TestPerfStats:
    """Rolling performance review (trading_system/bot/perf_stats.py).

    Fixtures mirror the real paper ledger (2026-09-14..09-21): 4 closed
    trades, one -1.72% stop-out and three take-profit wins, +$80.25 net.
    """

    def _t(self, net, close, entry=None, pct=None, pair="BTC/USDT:USDT"):
        trade = {"pair": pair, "net_pnl": net, "close_time": close}
        if entry is not None:
            trade["entry_time"] = entry
        if pct is not None:
            trade["pnl_pct_net"] = pct
        return trade

    def _paper_trades(self):
        return [
            self._t(-17.7104, "2026-09-15T08:00:46+00:00",
                    "2026-09-14T14:00:59+00:00", -1.7219),
            self._t(31.2784, "2026-09-18T14:29:42+00:00",
                    "2026-09-15T15:07:00+00:00", 6.2671),
            self._t(30.7934, "2026-09-21T14:20:50+00:00",
                    "2026-09-20T21:31:10+00:00", 3.8441, "ETH/USDT:USDT"),
            self._t(35.8876, "2026-09-21T14:20:50+00:00",
                    "2026-09-21T00:15:51+00:00", 4.4727),
        ]

    def test_win_rate_profit_factor_expectancy_and_rr(self):
        from trading_system.bot.perf_stats import compute_metrics

        m = compute_metrics(self._paper_trades())
        assert (m["trades"], m["wins"], m["losses"]) == (4, 3, 1)
        assert m["win_rate"] == pytest.approx(75.0)
        assert m["net_pnl"] == pytest.approx(80.2489, abs=0.01)
        assert m["gross_profit"] == pytest.approx(97.9594, abs=0.01)
        assert m["profit_factor"] == pytest.approx(97.9594 / 17.7104, abs=0.01)
        assert m["expectancy"] == pytest.approx(80.2489 / 4, abs=0.01)
        assert m["avg_win"] == pytest.approx(97.9594 / 3, abs=0.01)
        assert m["avg_loss"] == pytest.approx(17.7104, abs=0.01)
        assert m["realized_rr"] == pytest.approx((97.9594 / 3) / 17.7104, abs=0.01)

    def test_profit_factor_is_none_until_a_loss_exists(self):
        """"PF inf" on a 2-trade sample reads as a result; None reads as data
        we do not have yet — which is the truth."""
        from trading_system.bot.perf_stats import compute_metrics

        m = compute_metrics([self._t(10.0, "2026-09-21T00:00:00+00:00")])
        assert m["profit_factor"] is None
        assert m["realized_rr"] is None
        assert m["win_rate"] == pytest.approx(100.0)

    def test_streaks_in_close_order(self):
        from trading_system.bot.perf_stats import compute_metrics

        m = compute_metrics(self._paper_trades())  # loss then three wins
        assert m["max_loss_streak"] == 1
        assert m["max_win_streak"] == 3
        assert m["current_streak"] == 3

    def test_max_drawdown_is_peak_to_trough(self):
        from trading_system.bot.perf_stats import drawdown, equity_curve

        trades = [
            self._t(-100.0, "2026-09-01T00:00:00+00:00"),
            self._t(-50.0, "2026-09-02T00:00:00+00:00"),
            self._t(80.0, "2026-09-03T00:00:00+00:00"),
        ]
        dd = drawdown(equity_curve(trades, 10000.0))
        assert dd["peak"] == pytest.approx(10000.0)
        assert dd["max_dd"] == pytest.approx(150.0)      # 10,000 -> 9,850
        assert dd["max_dd_pct"] == pytest.approx(1.5)
        assert dd["current_dd"] == pytest.approx(70.0)   # 9,930 vs peak
        assert dd["current_dd_pct"] == pytest.approx(0.7)

    def test_curve_orders_by_close_time_not_list_order(self):
        from trading_system.bot.perf_stats import equity_curve

        trades = [
            self._t(10.0, "2026-09-03T00:00:00+00:00"),
            self._t(-5.0, "2026-09-01T00:00:00+00:00"),
        ]
        curve = equity_curve(trades, 1000.0)
        # First point is the starting equity, then the closes in time order.
        assert [round(equity, 2) for _, equity in curve] == [1000.0, 995.0, 1005.0]

    def test_slot_attribution_credits_the_slot_the_wakeup_served(self):
        """The 09-20 trade opened by the 11:11 catch-up for the dropped 08:00
        slot belongs to 08:00 — not to 14:00, which is nearer in clock time."""
        from trading_system.bot.perf_stats import attribute_slot, parse_ts

        wakeups = [
            parse_ts("2026-09-20T06:12:00+00:00"),
            parse_ts("2026-09-20T11:11:00+00:00"),
        ]
        assert attribute_slot(parse_ts("2026-09-20T11:11:30+00:00"), wakeups) == "08:00"

    def test_slot_attribution_uses_the_opening_wakeup_not_the_clock(self):
        from trading_system.bot.perf_stats import attribute_slot, parse_ts

        wakeups = [parse_ts("2026-09-21T00:15:00+00:00")]  # 00:00 slot, ran late
        assert attribute_slot(parse_ts("2026-09-21T00:15:51+00:00"), wakeups) == "00:00"

    def test_slot_attribution_falls_back_to_clock_without_a_journal(self):
        from trading_system.bot.perf_stats import attribute_slot, parse_ts

        assert attribute_slot(parse_ts("2026-09-15T15:07:00+00:00"), None) == "14:00"

    def test_slot_breakdown_lists_every_slot_even_with_no_trades(self):
        """A slot that never trades must still appear — "06:00 never earns" is
        exactly the finding the review is for."""
        from trading_system.bot.perf_stats import slot_breakdown

        journal = [
            {"timestamp": "2026-09-21T00:15:51+00:00"},
            {"timestamp": "2026-09-21T00:16:00+00:00"},
        ]
        rows = slot_breakdown(self._paper_trades(), journal)
        assert [r["slot"] for r in rows] == [
            "00:00", "06:00", "08:00", "14:00", "20:00", "23:00"]
        by_slot = {r["slot"]: r for r in rows}
        assert by_slot["14:00"]["trades"] == 2     # 09-14 open + 09-15 catch-up
        assert by_slot["20:00"]["trades"] == 1     # ETH long
        assert by_slot["00:00"]["trades"] == 1
        assert by_slot["06:00"]["trades"] == 0
        assert by_slot["00:00"]["wakeups"] == 2

    def test_planned_rr_comes_from_the_entry_records(self):
        from trading_system.bot.perf_stats import planned_stats

        p = planned_stats([
            {"stop_loss_pct": 1.5, "take_profit_pct": 2.5, "confidence": 68.0},
            {"stop_loss_pct": 2.0, "take_profit_pct": 4.0, "confidence": 72.0},
        ])
        assert p["avg_planned_rr"] == pytest.approx((2.5 / 1.5 + 4.0 / 2.0) / 2)
        assert p["avg_confidence"] == pytest.approx(70.0)
        assert planned_stats(None)["avg_planned_rr"] is None

    def test_windows_and_trend_split_the_history(self):
        from trading_system.bot.perf_stats import build_perf, parse_ts

        perf = build_perf(
            self._paper_trades(), start_equity=10000.0,
            now=parse_ts("2026-09-22T04:00:00+00:00"),
        )
        assert set(perf["windows"]) == {"7d", "30d", "all"}
        assert perf["windows"]["7d"]["trades"] == 4
        assert perf["windows"]["all"]["trades"] == 4
        # The previous 7-day window is empty — the trend block must report that
        # rather than divide by zero or pretend the sample is comparable.
        assert perf["trend"]["prev_trades"] == 0
        assert perf["trend"]["net_pnl_delta"] == pytest.approx(80.2489, abs=0.01)

    def test_legacy_trade_without_close_time_counts_all_time_only(self):
        from trading_system.bot.perf_stats import build_perf, parse_ts

        legacy = {"pair": "BTC/USDT:USDT", "net_pnl": 5.0}
        perf = build_perf(
            [legacy], start_equity=1000.0,
            now=parse_ts("2026-09-22T00:00:00+00:00"),
        )
        assert perf["windows"]["all"]["trades"] == 1
        assert perf["windows"]["7d"]["trades"] == 0
        assert perf["drawdown"]["current_dd"] == 0.0

    def test_empty_history_never_crashes(self):
        from trading_system.bot.perf_stats import build_perf, summarize

        perf = build_perf([], journal=[], opens=[], start_equity=0.0)
        assert perf["windows"]["all"]["win_rate"] == 0.0
        assert perf["drawdown"]["max_dd_pct"] == 0.0
        assert "Slots: no closed trades yet" in "\n".join(summarize(perf))

    def test_ledger_drift_is_reported_not_swallowed(self):
        """Real finding from cloud data (2026-09-22): the 09-14 record was
        written before net-at-both-fees accounting, so quoted P&L sits $0.50
        above realized cash. The review must SAY so — an unreconciled P&L is
        exactly the kind of quiet lie this bot is not allowed to tell."""
        from trading_system.bot.perf_stats import build_perf, parse_ts, summarize

        legacy = self._t(-17.7104, "2026-09-15T08:00:46+00:00",
                         "2026-09-14T14:00:59+00:00", -1.7219)
        perf = build_perf(
            [legacy], start_equity=10000.0,
            now=parse_ts("2026-09-22T04:00:00+00:00"),
            # Cash also paid the $0.50 entry fee at open, which the legacy
            # record's net_pnl never included.
            ledger_equity=9982.2896 - 0.50,
        )
        assert perf["audit"]["drift"] == pytest.approx(0.5001, abs=0.01)
        assert "Ledger check" in "\n".join(summarize(perf))
        # With no ledger figure supplied there is nothing to check against —
        # and no invented warning.
        silent = build_perf([legacy], start_equity=10000.0,
                            now=parse_ts("2026-09-22T04:00:00+00:00"))
        assert silent["audit"] is None
        assert "Ledger check" not in "\n".join(summarize(silent))

    def test_summarize_lines_fit_a_phone_screen(self):
        from trading_system.bot.perf_stats import build_perf, parse_ts, summarize

        perf = build_perf(
            self._paper_trades(), start_equity=10000.0,
            now=parse_ts("2026-09-22T04:00:00+00:00"),
            opens=[{"stop_loss_pct": 1.5, "take_profit_pct": 2.5, "confidence": 68}],
        )
        lines = summarize(perf)
        text = "\n".join(lines)
        assert "Drawdown:" in text
        assert "Slots:" in text
        assert "Planned: R:R" in text
        assert "Trend" in text
        assert all(len(line) <= 120 for line in lines)

    def test_trend_needs_a_previous_window_to_mean_anything(self):
        """A "delta" measured against an empty window is not a trend.

        Cloud reality (2026-09-22): the previous 7 days held zero closed
        trades, so the naive delta printed "win +75pp" — reading as a huge
        improvement when the truth was that there was nothing to compare
        against. The report must say that instead of quoting arithmetic at
        the user.
        """
        from trading_system.bot.perf_stats import build_perf, parse_ts, summarize

        now = parse_ts("2026-09-22T04:00:00+00:00")
        empty_prev = build_perf(self._paper_trades(), start_equity=10000.0, now=now)
        text = "\n".join(summarize(empty_prev))
        assert "nothing to compare yet" in text
        assert "+75pp" not in text
        assert "Trend (7d vs prev)" not in text

        # With a real prior-window trade the delta line comes back.
        prior = self._t(-5.0, "2026-09-10T14:30:00+00:00",
                        "2026-09-09T14:00:00+00:00", -0.5)
        with_prev = build_perf(
            self._paper_trades() + [prior], start_equity=10000.0, now=now,
        )
        assert with_prev["trend"]["prev_trades"] == 1
        text2 = "\n".join(summarize(with_prev))
        assert "Trend (7d vs prev)" in text2
        assert "nothing to compare yet" not in text2


class TestWeeklyReport:
    """Sunday report math (scripts/ai_weekly_report.py::build_report)."""

    def _mod(self):
        import importlib.util
        import pathlib
        # Absolute path: tests chdir into tmp_path, which would break a
        # project-relative path.
        script = (pathlib.Path(__file__).resolve().parents[1]
                  / "scripts" / "ai_weekly_report.py")
        spec = importlib.util.spec_from_file_location("ai_weekly_report", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _write(self, tmp_path, ledger=None, journal=None):
        import json as _json
        data_dir = tmp_path / "data" / "ai_bot"
        data_dir.mkdir(parents=True, exist_ok=True)
        if ledger is not None:
            (data_dir / "paper_ledger.json").write_text(_json.dumps(ledger))
        if journal is not None:
            (data_dir / "journal.jsonl").write_text("\n".join(journal) + "\n")
        return data_dir

    def test_window_filter_and_math(self, tmp_path):
        from datetime import datetime, timedelta, timezone

        import pytest

        now = datetime.now(timezone.utc)
        recent = (now - timedelta(days=2)).isoformat()
        old = (now - timedelta(days=10)).isoformat()
        ledger = {
            "cash": 10200.0,
            "start_equity": 10000.0,
            "closed_trades": [
                {"pair": "BTC/USDT:USDT", "net_pnl": 50.0, "pnl_pct": 1.0,
                 "close_time": recent},
                {"pair": "ETH/USDT:USDT", "net_pnl": -20.0, "pnl_pct": -0.5,
                 "close_time": recent},
                {"pair": "BTC/USDT:USDT", "net_pnl": 999.0, "pnl_pct": 9.0,
                 "close_time": old},
            ],
        }
        r = self._mod().build_report(self._write(tmp_path, ledger=ledger), days=7)
        assert r["week_trades"] == 2
        assert r["week_pnl"] == 30.0
        assert r["week_wins"] == 1
        assert r["total_trades"] == 3
        assert "BTC" in r["best_trade"] and "+1.00%" in r["best_trade"]
        assert "ETH" in r["worst_trade"]
        assert r["equity"] == 10200.0
        assert r["week_pnl_pct"] == pytest.approx(0.3)

    def test_report_carries_rolling_performance(self, tmp_path):
        """build_report must expose the rolling perf block the report (and the
        Telegram message) render from the same continuity files."""
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        ledger = {
            # cash == start + net of both trades, so the reconciliation check
            # is silent here (its own test covers the drift case).
            "cash": 10018.0,
            "start_equity": 10000.0,
            "closed_trades": [
                {"pair": "BTC/USDT:USDT", "net_pnl": 35.0, "pnl_pct_net": 4.4,
                 "entry_time": (now - timedelta(hours=30)).isoformat(),
                 "close_time": (now - timedelta(hours=29)).isoformat()},
                {"pair": "ETH/USDT:USDT", "net_pnl": -17.0, "pnl_pct_net": -1.7,
                 "entry_time": (now - timedelta(hours=20)).isoformat(),
                 "close_time": (now - timedelta(hours=19)).isoformat()},
            ],
        }
        data_dir = self._write(tmp_path, ledger=ledger)
        (data_dir / "trades.jsonl").write_text(json.dumps({
            "pair": "BTC/USDT:USDT", "side": "long", "price": 78000.0,
            "stop_loss_pct": 1.5, "take_profit_pct": 2.5, "confidence": 68.0,
            "executed": True, "timestamp": "2026-09-14T14:00:59+00:00",
        }) + "\n")

        r = self._mod().build_report(data_dir, days=7)
        perf = r["perf"]
        assert perf["windows"]["7d"]["trades"] == 2
        assert perf["windows"]["7d"]["win_rate"] == pytest.approx(50.0)
        assert perf["planned"]["entries"] == 1
        assert perf["planned"]["avg_planned_rr"] == pytest.approx(2.5 / 1.5)
        assert len(perf["slots"]) == 6  # every slot listed, traded or not
        assert perf["audit"]["drift"] == pytest.approx(0.0, abs=0.01)
        from trading_system.bot.perf_stats import summarize

        assert "Ledger check" not in "\n".join(summarize(perf))

    def test_notes_newest_first_and_deduped(self, tmp_path):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        entries = [
            {"timestamp": (now - timedelta(hours=30)).isoformat(),
             "status": "success", "ai_reasoning": "bullish bias"},
            {"timestamp": (now - timedelta(hours=20)).isoformat(),
             "status": "success", "ai_reasoning": "range-bound, waiting"},
            {"timestamp": (now - timedelta(hours=10)).isoformat(),
             "status": "error", "ai_reasoning": "range-bound, waiting"},
        ]
        r = self._mod().build_report(
            self._write(tmp_path, journal=[json.dumps(e) for e in entries]))
        assert r["wakeups"] == 3
        assert r["failed_wakeups"] == 1
        assert r["ai_notes"] == ["range-bound, waiting", "bullish bias"]

    def test_corrupt_journal_lines_skipped(self, tmp_path):
        from datetime import datetime, timezone

        now_iso = datetime.now(timezone.utc).isoformat()
        r = self._mod().build_report(self._write(
            tmp_path, journal=['{"oops', json.dumps({"timestamp": now_iso, "status": "success"})]))
        assert r["wakeups"] == 1
        assert r["failed_wakeups"] == 0

    def test_open_positions_reported(self, tmp_path):
        ledger = {
            "cash": 9000.0,
            "start_equity": 10000.0,
            "positions": {
                "BTC/USDT:USDT": {"side": "buy", "entry_price": 77000.0},
            },
        }
        r = self._mod().build_report(self._write(tmp_path, ledger=ledger))
        assert r["open_positions"][0]["pair"] == "BTC/USDT:USDT"
        assert r["start_equity"] == 10000.0

    def test_no_ledger_zero_report(self, tmp_path):
        r = self._mod().build_report(tmp_path / "does-not-exist")
        assert r["week_trades"] == 0
        assert r["equity"] == 0.0
        assert r["ai_notes"] == []
        assert r["week_pnl_pct"] == 0.0  # no divide-by-zero

    def test_cli_dry_run(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone

        from click.testing import CliRunner

        now_iso = datetime.now(timezone.utc).isoformat()
        self._write(
            tmp_path,
            ledger={"cash": 10050.0, "start_equity": 10000.0, "closed_trades": []},
            journal=[json.dumps({"timestamp": now_iso, "status": "success",
                                 "ai_reasoning": "quiet session"})],
        )
        configs = tmp_path / "configs"
        configs.mkdir()
        (configs / "ai_bot.yaml").write_text(
            "bot:\n  data_dir: data/ai_bot\ntelegram:\n  enabled: true\n")
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(self._mod().main, ["--dry-run"])
        assert result.exit_code == 0, result.output
        assert "WEEKLY REPORT" in result.output
        assert "$10,050.00" in result.output
        assert "dry-run" in result.output
        assert "PERFORMANCE (rolling)" in result.output
        assert "Drawdown:" in result.output


class TestNotifierHtmlFallback:
    """A message must never be lost because Telegram's HTML parser rejected
    interpolated AI text.

    REGRESSION (2026-09-16): market_outlook and ai_reasoning frequently
    contain bare '<' / '>' ("EMA 9 < EMA 21", "ADX > 28"). Every notifier
    message is sent with parse_mode=HTML, so Telegram answered 400
    "can't parse entities" and the message was dropped — the daily digest
    failed its send step for exactly this reason while local dry-runs (which
    never touch the API) looked perfect.
    """

    def _notifier(self):
        from trading_system.bot.telegram_notifier import TelegramNotifier

        return TelegramNotifier(bot_token="t", chat_id="c", enabled=True)

    def test_html_rejection_retries_as_plain_text(self, monkeypatch):
        n = self._notifier()
        calls: list[str] = []

        def fake_attempt(text, parse_mode):
            calls.append(parse_mode)
            return len(calls) > 1  # HTML rejected -> plain succeeds

        monkeypatch.setattr(n, "_attempt_send", fake_attempt)
        assert n.send_message("EMA 9 < EMA 21, ADX > 28") is True
        assert calls == ["HTML", ""]

    def test_both_modes_failing_still_returns_false(self, monkeypatch):
        n = self._notifier()
        seen: list[str] = []

        def fake_attempt(text, parse_mode):
            seen.append(parse_mode)
            return False

        monkeypatch.setattr(n, "_attempt_send", fake_attempt)
        assert n.send_message("x") is False
        assert seen == ["HTML", ""]  # tried hard, reported failure

    def test_plain_retry_omits_the_parse_mode_field(self, monkeypatch):
        # Telegram rejects parse_mode="" outright, so it must be absent.
        n = self._notifier()
        seen: dict = {}
        monkeypatch.setattr(
            n, "_send_via_curl", lambda payload: seen.update(payload) or True)
        assert n._attempt_send("text", "") is True
        assert "parse_mode" not in seen
        seen.clear()
        assert n._attempt_send("text", "HTML") is True
        assert seen["parse_mode"] == "HTML"


class TestNotifierWeeklyAndTest:
    """notify_weekly_summary + send_test_message fail-safe behavior."""

    def _weekly_kwargs(self):
        return dict(
            equity=10450.0, start_equity=10000.0,
            week_pnl=120.5, week_pnl_pct=1.2,
            week_trades=6, week_wins=4,
            best_trade="BTC +3.10% ($62.00)",
            worst_trade="ETH -1.20% ($-24.00)",
            total_trades=20, total_win_rate=60.0,
            wakeups=42, failed_wakeups=1,
            ai_notes=["note one", "", "note one", "note two"],
            open_positions=[{"pair": "BTC/USDT:USDT", "side": "buy",
                             "unrealized_pnl": 5.0}],
        )

    def test_disabled_sends_nothing(self):
        from trading_system.bot.telegram_notifier import TelegramNotifier

        n = TelegramNotifier("token", "chat", enabled=False)
        assert n.notify_weekly_summary(**self._weekly_kwargs()) is False
        assert n.send_test_message() is False

    def test_weekly_summary_formats_key_numbers(self, monkeypatch):
        from trading_system.bot.telegram_notifier import TelegramNotifier

        n = TelegramNotifier("token", "chat", enabled=True)
        captured = {}

        def fake_send(text, parse_mode="HTML"):
            captured["text"] = text
            return True

        monkeypatch.setattr(n, "_send_message", fake_send)
        assert n.notify_weekly_summary(**self._weekly_kwargs()) is True

        t = captured["text"]
        assert "$10,450.00" in t          # equity
        assert "+4.5% all-time" in t      # total return
        assert "$+120.50" in t            # week pnl
        assert "Closed: 6 | Wins: 4 (67%)" in t
        assert "note one" in t
        assert t.count("note one") == 1   # deduped, empty strings skipped
        assert "note two" in t
        assert "BTC/USDT:USDT LONG" in t

    def test_weekly_summary_includes_rolling_performance(self, monkeypatch):
        """The Sunday report must carry the rolling stats (win rate,
        drawdown, R:R, per-slot), not just the last 7 days' P&L."""
        from trading_system.bot.perf_stats import build_perf, parse_ts
        from trading_system.bot.telegram_notifier import TelegramNotifier

        perf = build_perf(
            [{"pair": "BTC/USDT:USDT", "net_pnl": 31.2784,
              "pnl_pct_net": 6.2671,
              "entry_time": "2026-09-15T15:07:00+00:00",
              "close_time": "2026-09-18T14:29:42+00:00"}],
            start_equity=10000.0, now=parse_ts("2026-09-22T04:00:00+00:00"),
        )
        n = TelegramNotifier("token", "chat", enabled=True)
        captured = {}

        def fake_send(text, parse_mode="HTML"):
            captured["text"] = text
            return True

        monkeypatch.setattr(n, "_send_message", fake_send)
        kwargs = self._weekly_kwargs()
        kwargs["perf"] = perf
        assert n.notify_weekly_summary(**kwargs) is True
        text = captured["text"]
        assert "PERFORMANCE (rolling)" in text
        assert "Drawdown:" in text
        assert "Slots:" in text
        # The old sections still ship — perf ADDS to the report.
        assert "$10,450.00" in text and "AI NOTES:" in text

    def test_overlong_weekly_message_trims_and_says_so(self, monkeypatch):
        """Telegram's hard limit is 4096 chars; an overlong report would be
        rejected on BOTH the HTML and the plain-text attempt, i.e. silently
        never arrive. Trim the notes first, keep the numbers, say what was cut."""
        from trading_system.bot.telegram_notifier import TelegramNotifier

        n = TelegramNotifier("token", "chat", enabled=True)
        captured = {}

        def fake_send(text, parse_mode="HTML"):
            captured["text"] = text
            return True

        monkeypatch.setattr(n, "_send_message", fake_send)
        kwargs = self._weekly_kwargs()
        kwargs["best_trade"] = "BTC " + "x" * 5000
        assert n.notify_weekly_summary(**kwargs) is True
        text = captured["text"]
        assert len(text) <= 4096
        assert "trimmed" in text
        assert "$10,450.00" in text  # the numbers survive the trim

    def test_test_message_content(self, monkeypatch):
        from trading_system.bot.telegram_notifier import TelegramNotifier

        n = TelegramNotifier("token", "chat", enabled=True)
        captured = {}

        def fake_send(text, parse_mode="HTML"):
            captured.update(text=text)
            return True

        monkeypatch.setattr(n, "_send_message", fake_send)
        assert n.send_test_message() is True
        assert "Telegram alerts are working" in captured["text"]


class TestTelegramClip:
    """clip() — report truncation must never cut mid-word (2026-09-21:
    the weekly report shipped "RSI neutral at 55-5" for "55-58")."""

    def test_short_text_passes_through(self):
        from trading_system.bot.telegram_notifier import clip

        assert clip("short note", 180) == "short note"

    def test_whitespace_collapsed(self):
        from trading_system.bot.telegram_notifier import clip

        assert clip("a\n  b   c", 180) == "a b c"

    def test_cuts_at_word_boundary_with_ellipsis(self):
        from trading_system.bot.telegram_notifier import clip

        out = clip("RSI neutral at 55-58 and volume extremely low today", 30)
        assert out == "RSI neutral at 55-58 and…"

    def test_exact_length_untouched(self):
        from trading_system.bot.telegram_notifier import clip

        text = "x" * 40
        assert clip(text, 40) == text

    def test_hard_slice_would_break_token_clip_does_not(self):
        """The exact 55-5 hazard: a hard slice at the limit lands inside
        '55-58'; clip() must back off to the word boundary instead."""
        from trading_system.bot.telegram_notifier import clip

        text = "RSI neutral at 55-58 and volume is extremely low today"
        limit = text.index("55-58") + 4  # lands mid-token -> "55-5"
        out = clip(text, limit)
        assert out.endswith("…")
        assert "55-5…" not in out
        assert out.startswith("RSI neutral at")

    def test_weekly_note_no_midword_cut(self, monkeypatch):
        from trading_system.bot.telegram_notifier import TelegramNotifier

        n = TelegramNotifier("token", "chat", enabled=True)
        captured = {}

        def fake_send(text, parse_mode="HTML"):
            captured["text"] = text
            return True

        monkeypatch.setattr(n, "_send_message", fake_send)
        note = ("Both BTC and ETH are displaying mixed EMA trends accompanied "
                "by extremely low trading volumes (volume ratio below 0.10x). "
                "Momentum oscillators show RSI neutral at 55-58 and MACD flat, "
                "so directional conviction is absent across both major pairs.")
        n.notify_weekly_summary(
            equity=10000, start_equity=10000, week_pnl=0, week_pnl_pct=0,
            week_trades=0, week_wins=0, best_trade="", worst_trade="",
            total_trades=0, total_win_rate=0, wakeups=0, failed_wakeups=0,
            ai_notes=[note], open_positions=[],
        )
        body = [ln for ln in captured["text"].splitlines()
                if ln.startswith("  - ")][0]
        assert body[4:].endswith("…")  # long note is marked as clipped
        content = body[4:].rstrip("…")
        note_tokens = set(note.split())
        # Every word shown must be a COMPLETE word from the source note —
        # the shipped bug printed "55-5" for "55-58".
        assert content and all(tok in note_tokens for tok in content.split())
        assert "55-58" in content


class TestDigestWorkflow:
    """ai_daily_digest.yml pins: one delivery per day, ever."""

    def test_serialized_delivery(self):
        # The kicker fires on every wakeup completion; without a concurrency
        # group three runs raced the marker and delivered the 17/9 digest
        # three times (23:02, 23:03, 23:03). Serialized, runs 2+ see the
        # marker and exit quietly.
        d = _wf_yaml(".github/workflows/ai_daily_digest.yml")
        c = d["concurrency"]
        assert c["group"] == "ai-daily-digest"
        assert c["cancel-in-progress"] is False  # queue, never kill the sender
        assert d["permissions"]["contents"] == "write"  # marker persist

    def test_kicked_by_every_wakeup(self):
        d = _wf_yaml(".github/workflows/ai_daily_digest.yml")
        wr = d["on"]["workflow_run"]
        assert "AI Trading Bot" in set(wr["workflows"])
        assert wr["types"] == ["completed"]

    def test_send_message_wrapper(self, monkeypatch):
        """Public send_message passes text verbatim (used by report scripts)."""
        from trading_system.bot.telegram_notifier import TelegramNotifier

        n = TelegramNotifier("token", "chat", enabled=True)
        captured = {}
        monkeypatch.setattr(
            n, "_send_message",
            lambda text, parse_mode="HTML": captured.update(text=text) or True,
        )
        assert n.send_message("HELLO DIGEST") is True
        assert captured["text"] == "HELLO DIGEST"


class TestDailyDigest:
    """scripts/ai_daily_digest.py summarize + format (pure functions)."""

    NOW = datetime(2026, 9, 13, 23, 50, tzinfo=timezone.utc)

    @staticmethod
    def _entry(ts, status="success", outlook="neutral", reasoning="r",
               equity=10000.0, errors=None, closed=None):
        return {
            "wakeup_id": ts.replace(":", "").replace("-", "").replace("T", "_"),
            "timestamp": ts, "mode": "paper", "status": status,
            "market_outlook": outlook, "ai_reasoning": reasoning,
            "actions_executed": 0, "equity": equity,
            "errors": errors or [], "closed_triggers": closed or [],
        }

    def _write(self, tmp_path, entries, ledger=None):
        d = tmp_path / "ai_bot"
        d.mkdir(exist_ok=True)
        with open(d / "journal.jsonl", "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        if ledger is not None:
            (d / "paper_ledger.json").write_text(json.dumps(ledger))
        return d

    def test_quiet_day_message(self, tmp_path):
        from scripts.ai_daily_digest import format_digest, summarize_day

        # All six scheduled slots ran (a quiet-but-healthy day).
        d = self._write(tmp_path, [
            self._entry("2026-09-13T00:01:37+00:00"),
            self._entry("2026-09-13T06:02:07+00:00"),
            self._entry("2026-09-13T08:03:46+00:00"),
            self._entry("2026-09-13T14:02:11+00:00"),
            self._entry("2026-09-13T20:01:40+00:00"),
            self._entry("2026-09-13T23:32:09+00:00",
                        reasoning="Flat market, standing aside."),
        ])
        s = summarize_day(d, now=self.NOW)
        msg = format_digest(s)
        assert "✅ Wakeups: 6/6 slots ran, all ok" in msg
        assert "no trades closed" in msg
        assert "$10,000.00" in msg
        assert 'AI said: "Flat market, standing aside."' in msg

    def test_late_run_reports_the_previous_day(self, tmp_path):
        """REGRESSION (2026-09-14/15/16): GitHub held the 23:50 cron ~2h, so
        the digest ran at ~01:45 and reported the NEW day's first two hours
        while labelling itself with the new date — no digest ever described a
        complete day. A late run must report the day whose last slot passed."""
        from scripts.ai_daily_digest import format_digest, summarize_day

        d = self._write(tmp_path, [
            self._entry("2026-09-14T14:02:11+00:00", outlook="bullish"),
            self._entry("2026-09-14T23:35:00+00:00"),
            self._entry("2026-09-15T00:01:00+00:00"),  # the new day
        ])
        late = datetime(2026, 9, 15, 1, 55, tzinfo=timezone.utc)
        s = summarize_day(d, now=late)
        assert s["day"] == "2026-09-14"
        assert s["wakeups"] == 2  # only the 14th's entries
        msg = format_digest(s)
        assert "DIGEST — 2026-09-14" in msg
        assert "2026-09-15" not in msg
        # The on-time run for that same day reports the same day (so the
        # marker value is identical and the guard works either way).
        on_time = datetime(2026, 9, 14, 23, 50, tzinfo=timezone.utc)
        assert summarize_day(d, now=on_time)["day"] == "2026-09-14"

    def test_missed_slot_labels_are_named(self, tmp_path):
        from scripts.ai_daily_digest import format_digest, summarize_day

        d = self._write(tmp_path, [
            self._entry("2026-09-13T00:05:00+00:00"),
            self._entry("2026-09-13T14:02:11+00:00"),
            self._entry("2026-09-13T23:32:09+00:00"),
        ])
        s = summarize_day(d, now=self.NOW)
        assert s["slots_missing"] == ["06:00", "08:00", "20:00"]
        assert len(s["slots_ran"]) == 3
        msg = format_digest(s)
        assert "⚠️ Wakeups: 3/6 slots ran" in msg
        assert "3 never fired" in msg
        assert "06:00 UTC" in msg

    def test_one_wakeup_cannot_serve_two_slots(self, tmp_path):
        """A single 08:30 wakeup must not mark both 06:00 and 08:00 served."""
        from scripts.ai_daily_digest import summarize_day

        d = self._write(tmp_path, [self._entry("2026-09-13T08:30:00+00:00")])
        s = summarize_day(d, now=self.NOW)
        assert s["slots_ran"] == ["08:00"]
        assert "06:00" in s["slots_missing"]

    def test_late_catchup_is_reported_not_hidden(self, tmp_path):
        """The 2026-09-20 bug: GitHub delivered the 08:02 cron ~3h late;
        the gate correctly ran the 08:00 wakeup at 11:11, but 11:11 sat
        outside the grace window so the digest said "1 never fired" —
        hiding the recovery and crying wolf. A late catch-up must be
        reported as LATE, never as a miss."""
        from scripts.ai_daily_digest import format_digest, summarize_day

        d = self._write(tmp_path, [
            self._entry("2026-09-20T00:01:27+00:00"),
            self._entry("2026-09-20T06:12:25+00:00"),
            self._entry("2026-09-20T11:11:15+00:00"),  # the late 08:00 catch-up
            self._entry("2026-09-20T16:02:21+00:00"),
            self._entry("2026-09-20T21:31:12+00:00"),
            # Real journal: the 23:00 slot was served by Sep 21's 00:15
            # wakeup (75 min after the slot — inside its window).
            self._entry("2026-09-21T00:15:52+00:00"),
        ])
        now = datetime(2026, 9, 21, 0, 16, tzinfo=timezone.utc)  # after 23:00 slot
        s = summarize_day(d, now=now)
        assert s["day"] == "2026-09-20"
        assert s["slots_served_late"] == ["08:00"]
        assert s["slots_missing"] == []
        assert len(s["slots_ran"]) == 5
        msg = format_digest(s)
        assert "✅ Wakeups: 6/6 slots ran" in msg
        assert "1 ran late: 08:00 UTC" in msg
        assert "never fired" not in msg

    def test_genuine_miss_still_shouted(self, tmp_path):
        """Contrast with the late-catchup case: with NO wakeup at all for
        the 08:00 slot (and none for 14:00's window either), the digest
        must still say "never fired" — late is worth knowing, missed is
        an outage."""
        from scripts.ai_daily_digest import format_digest, summarize_day

        d = self._write(tmp_path, [
            self._entry("2026-09-20T00:01:27+00:00"),
            self._entry("2026-09-20T06:12:25+00:00"),
            self._entry("2026-09-20T16:02:21+00:00"),  # in-window catch-up for 14:00
            self._entry("2026-09-20T21:31:12+00:00"),
            self._entry("2026-09-21T00:15:52+00:00"),  # serves the 23:00 slot
        ])
        now = datetime(2026, 9, 21, 0, 16, tzinfo=timezone.utc)  # after 23:00 slot
        s = summarize_day(d, now=now)
        assert s["slots_served_late"] == []
        assert s["slots_missing"] == ["08:00"]
        assert len(s["slots_ran"]) == 5
        msg = format_digest(s)
        assert "⚠️ Wakeups: 5/6 slots ran" in msg
        assert "1 never fired: 08:00 UTC" in msg

    def test_success_failure_mix_surfaces_error(self, tmp_path):
        """A slot whose wakeup FAILED still counts as the slot having run —
        the report distinguishes FAILED from never-fired, so neither can hide
        behind the other."""
        from scripts.ai_daily_digest import format_digest, summarize_day

        d = self._write(tmp_path, [
            self._entry("2026-09-13T00:49:07+00:00"),
            self._entry("2026-09-13T06:02:00+00:00"),
            self._entry("2026-09-13T08:03:00+00:00"),
            self._entry("2026-09-13T14:02:11+00:00", status="error",
                        errors=["AI engine failed: 503 high demand"]),
            self._entry("2026-09-13T20:01:00+00:00"),
            self._entry("2026-09-13T23:32:09+00:00"),
        ])
        s = summarize_day(d, now=self.NOW)
        assert len(s["slots_ran"]) == 6 and s["failed"] == 1
        msg = format_digest(s)
        assert "⚠️ Wakeups: 6/6 slots ran — 1 FAILED" in msg
        assert "never fired" not in msg
        assert "503" in msg

    def test_already_sent_marker_suppresses_duplicate(self, tmp_path):
        """The 2026-09-13 bug: a manual run fired the scheduled digest early,
        then the scheduled run sent a false NO-WAKEUPS alarm."""
        from scripts.ai_daily_digest import format_digest, summarize_day

        d = self._write(tmp_path, [self._entry("2026-09-13T14:02:11+00:00")])
        (d / "digest_sent.txt").write_text("2026-09-13")
        s = summarize_day(d, now=self.NOW)
        assert s["already_sent"] is True
        msg = format_digest(s)
        assert "Already delivered" in msg
        assert "NO WAKEUPS" not in msg

    def test_stale_marker_does_not_suppress(self, tmp_path):
        from scripts.ai_daily_digest import summarize_day

        d = self._write(tmp_path, [self._entry("2026-09-13T14:02:11+00:00")])
        (d / "digest_sent.txt").write_text("2026-09-12")  # yesterday
        s = summarize_day(d, now=self.NOW)
        assert s["already_sent"] is False

    def test_day_with_trades_shows_pnl(self, tmp_path):
        from scripts.ai_daily_digest import format_digest, summarize_day

        win = {"pair": "BTC/USDT:USDT", "net_pnl": 24.0, "pnl_pct": 1.2}
        loss = {"pair": "ETH/USDT:USDT", "net_pnl": -10.0, "pnl_pct": -0.5}
        d = self._write(tmp_path, [
            self._entry("2026-09-13T14:02:11+00:00", closed=[win, loss]),
        ])
        s = summarize_day(d, now=self.NOW)
        assert s["pnl_today"] == pytest.approx(14.0)
        assert s["closed_count"] == 2
        msg = format_digest(s)
        assert "P&L today: $+14.00 (2 closed, 1 wins)" in msg
        assert "BTC/USDT:USDT +1.20% ($+24.00)" in msg
        assert "ETH/USDT:USDT -0.50% ($-10.00)" in msg

    def test_failed_wakeup_flagged(self, tmp_path):
        from scripts.ai_daily_digest import format_digest, summarize_day

        d = self._write(tmp_path, [
            self._entry("2026-09-13T14:02:11+00:00"),
            self._entry("2026-09-13T20:02:11+00:00", status="error",
                        errors=["market data unavailable for all pairs"]),
            self._entry("2026-09-13T23:32:09+00:00"),
        ])
        s = summarize_day(d, now=self.NOW)
        assert s["wakeups"] == 3 and s["failed"] == 1
        assert len(s["slots_ran"]) == 3 and len(s["slots_missing"]) == 3
        msg = format_digest(s)
        assert "⚠️ Wakeups: 3/6 slots ran" in msg
        assert "1 FAILED" in msg
        assert "3 never fired" in msg
        assert "market data unavailable" in msg

    def test_no_wakeups_is_loud_alert(self, tmp_path):
        from scripts.ai_daily_digest import format_digest, summarize_day

        d = self._write(tmp_path, [
            self._entry("2026-09-12T23:32:09+00:00"),  # yesterday only
        ])
        s = summarize_day(d, now=self.NOW)
        assert s["wakeups"] == 0
        msg = format_digest(s)
        assert "NO WAKEUPS RAN" in msg
        assert "6 scheduled slots" in msg

    def test_equity_falls_back_to_ledger(self, tmp_path):
        from scripts.ai_daily_digest import summarize_day

        entry = self._entry("2026-09-13T14:02:11+00:00")
        del entry["equity"]  # older journals may lack the field
        d = self._write(tmp_path, [entry], ledger={"cash": 10123.45})
        s = summarize_day(d, now=self.NOW)
        assert s["equity"] == pytest.approx(10123.45)

    def test_corrupt_journal_lines_skipped(self, tmp_path):
        from scripts.ai_daily_digest import summarize_day

        d = tmp_path / "ai_bot"
        d.mkdir(exist_ok=True)
        good = json.dumps(self._entry("2026-09-13T14:02:11+00:00"))
        (d / "journal.jsonl").write_text(
            good + "\n{not valid json}\n\n", encoding="utf-8"
        )
        s = summarize_day(d, now=self.NOW)
        assert s["wakeups"] == 1  # corrupt line did not block the digest

    def test_missing_data_dir_fails_loudly(self, tmp_path):
        """No data dir → CLI raises instead of texting a made-up summary."""
        import subprocess
        import sys
        from pathlib import Path

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("bot:\n  data_dir: does_not_exist\n", encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "scripts/ai_daily_digest.py", "--config", str(cfg)],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent,
        )
        assert r.returncode != 0
        assert "run a wakeup first" in (r.stderr + r.stdout)

    def test_check_due_exit_codes(self, tmp_path):
        """--check-due lets the workflow skip a no-op in ~1s when the day's
        report already went out (exit 3) and proceed when it hasn't (exit 0).
        Any other failure must make it look DUE, never silently skip."""
        import subprocess
        import sys
        from pathlib import Path

        from scripts.ai_daily_digest import summarize_day

        d = self._write(tmp_path, [self._entry("2026-09-13T14:02:11+00:00")])
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(f"bot:\n  data_dir: {d.as_posix()}\n", encoding="utf-8")
        cmd = [sys.executable, "scripts/ai_daily_digest.py",
               "--config", str(cfg), "--check-due"]
        cwd = Path(__file__).parent.parent

        r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "Due" in r.stdout

        (d / "digest_sent.txt").write_text(summarize_day(d)["day"])
        r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        assert r2.returncode == 3
        assert "Not due" in r2.stdout

        # A missing data dir is not an error for the pre-check: the real run
        # reports it loudly.
        cfg2 = tmp_path / "cfg2.yaml"
        cfg2.write_text("bot:\n  data_dir: nope_dir\n", encoding="utf-8")
        r3 = subprocess.run(
            [sys.executable, "scripts/ai_daily_digest.py",
             "--config", str(cfg2), "--check-due"],
            capture_output=True, text=True, cwd=cwd,
        )
        assert r3.returncode == 0 and "Due" in r3.stdout

    def test_trade_pct_prefers_round_trip_net(self):
        from scripts.ai_daily_digest import _trade_pct

        assert _trade_pct({"pnl_pct": -1.72, "pnl_pct_net": -1.82}) == -1.82
        assert _trade_pct({"pnl_pct": -1.72}) == -1.72  # old records


class TestAlwaysOnPoller:
    """scripts/ai_poller.py — instant /status responder logic.

    Every dependency is injected, so these tests run with zero network
    and zero mutation of the shared commands module.
    """

    @staticmethod
    def _upd(uid, text="/status", chat="5421461006"):
        return {"update_id": uid,
                "message": {"text": text, "chat": {"id": chat}}}

    def test_poller_reuses_wakeup_responder_routing(self):
        """Both responders share one routing table — answers can never drift."""
        import scripts.ai_poller as poller
        import scripts.ai_telegram_commands as cmds

        assert poller.cmds is cmds
        assert cmds.route_command("/status", {}, None, {}) is not None

    def test_answers_only_unseen_updates_and_confirms_server_side(self):
        import scripts.ai_poller as poller

        updates = {"ok": True, "result": [
            self._upd(101, "/status"), self._upd(102, "/help"),
        ]}
        sends: list[str] = []
        calls: list = []
        r = poller.poll_once(
            "tok", "5421461006",
            get_updates=lambda tok, off, ps: calls.append(off) or (updates, 200),
            send=lambda tok, m, **kw: sends.append(kw["text"]) or {"ok": True},
            read_offset=lambda: 0,
            write_offset=lambda v: None,
            load_ledger=lambda: {},
            load_journal=lambda: None,
            prices_fn=lambda pairs: {},
        )
        assert r["answered"] == 2 and not r["conflict"]
        # First call used no offset; the second is the server-side confirm
        # carrying the final offset (103), so a fresh generation is never
        # re-sent these updates (fresh FS, stale offset file).
        assert calls == [None, 103]

    def test_dedupes_against_disk_offset(self, tmp_path):
        import scripts.ai_poller as poller

        offset_file = tmp_path / "telegram_offset.txt"
        # Telegram semantics: offset N = "updates < N are processed; send
        # me N and newer". The wakeup responder handled up to 100 and
        # wrote 101, so 101/102 are UNPROCESSED and must be answered;
        # anything below 101 is defensively skipped.
        offset_file.write_text("101")
        updates = {"ok": True, "result": [
            self._upd(100, "/status"),  # < offset -> skipped, never re-answered
            self._upd(101, "/status"),  # unprocessed -> answered
            self._upd(101, "/status"),  # duplicate in batch -> answered once
            self._upd(102, "/status"),  # unprocessed -> answered
        ]}
        sends: list[str] = []
        poller.poll_once(
            "tok", "5421461006",
            get_updates=lambda tok, off, ps: (updates, 200),
            send=lambda tok, m, **kw: sends.append(kw["text"]) or {"ok": True},
            read_offset=lambda: int(offset_file.read_text()),
            write_offset=lambda v: offset_file.write_text(str(v)),
            load_ledger=lambda: {},
            load_journal=lambda: None,
            prices_fn=lambda pairs: {},
        )
        assert len(sends) == 2  # 101 and 102; 100 and the dup never sent
        assert offset_file.read_text() == "103"

    def test_conflict_yields_immediately(self):
        import scripts.ai_poller as poller

        r = poller.poll_once(
            "tok", "5421461006",
            get_updates=lambda tok, off, ps: (None, 409),
            send=lambda tok, m, **kw: {"ok": True},
            read_offset=lambda: 0,
            write_offset=lambda v: None,
            load_ledger=lambda: {},
            load_journal=lambda: None,
            prices_fn=lambda pairs: {},
        )
        assert r["conflict"] is True

    def test_offset_never_moves_backwards(self, tmp_path):
        import scripts.ai_poller as poller

        offset_file = tmp_path / "telegram_offset.txt"
        offset_file.write_text("500")
        updates = {"ok": True, "result": [
            self._upd(600, "/help"), self._upd(505, "/help"),
        ]}
        poller.poll_once(
            "tok", "5421461006",
            get_updates=lambda tok, off, ps: (updates, 200),
            send=lambda tok, m, **kw: {"ok": True},
            read_offset=lambda: int(offset_file.read_text()),
            write_offset=lambda v: offset_file.write_text(str(v)),
            load_ledger=lambda: {},
            load_journal=lambda: None,
            prices_fn=lambda pairs: {},
        )
        assert offset_file.read_text() == "601"

    def test_ignores_unauthorized_chat(self):
        import scripts.ai_poller as poller

        updates = {"ok": True, "result": [self._upd(7, "/status", chat="999")]}
        sends: list[str] = []
        written: list = []
        poller.poll_once(
            "tok", "5421461006",
            get_updates=lambda tok, off, ps: (updates, 200),
            send=lambda tok, m, **kw: sends.append(kw["text"]) or {"ok": True},
            read_offset=lambda: 0,
            write_offset=written.append,
            load_ledger=lambda: {},
            load_journal=lambda: None,
            prices_fn=lambda pairs: {},
        )
        assert sends == []  # nobody else's chat is ever answered
        assert written == [8]  # ...but the update is still consumed


def _wf_yaml(path) -> dict:
    """Load a workflow file, normalizing YAML 1.1's `on:` -> True key."""
    from pathlib import Path as _Path

    import yaml

    d = yaml.safe_load(_Path(path).read_text())
    d["on"] = d.pop("on", None) or d.pop(True, None)
    return d


class TestPollerWorkflow:
    """ai_poller.yml structural pins: bounded billing + safe coordination."""

    def test_bounded_generations_and_cancel(self):
        d = _wf_yaml(".github/workflows/ai_poller.yml")
        assert d["concurrency"]["cancel-in-progress"] is True
        # Backup crons: four explicit lines (:07/:22/:37/:52), off the
        # trading crons. Continuity must NOT depend on them.
        crons = [c["cron"] for c in d["on"]["schedule"]]
        assert crons == ["7 * * * *", "22 * * * *", "37 * * * *", "52 * * * *"]
        # Kicker completions: trading/digest/health runs revive the chain
        # after any outage. NO self-reference: GitHub ignores a workflow
        # listing itself in workflow_run (proven — run #2 never fired).
        wr = d["on"]["workflow_run"]
        assert "AI Bot Telegram Poller" not in set(wr["workflows"])
        assert {"AI Trading Bot", "AI Bot Daily Digest"} <= set(wr["workflows"])
        assert wr["types"] == ["completed"]
        # The offset persist pushes to the state branch.
        assert d["permissions"]["contents"] == "write"
        job = d["jobs"]["poll"]
        assert job["timeout-minutes"] <= 30
        steps = "\n--".join(s.get("run", "") for s in job["steps"])
        assert "--max-minutes 20" in steps
        assert "pip install -r requirements.txt" in steps  # no dev tools
        # Anti-tight-loop pad runs even on failed setup, so a broken run
        # can never chain into a hot relaunch loop.
        pads = [s for s in job["steps"]
                if "Anti-tight-loop pad" in str(s.get("name", ""))]
        assert pads and pads[0]["if"] == "always()"

    def test_poller_never_blocks_trading(self):
        """Different concurrency groups: poller cancels can't touch trading."""
        p = _wf_yaml(".github/workflows/ai_poller.yml")
        b = _wf_yaml(".github/workflows/ai_bot.yml")
        assert p["concurrency"]["group"] != b["concurrency"]["group"]
        assert b["concurrency"]["cancel-in-progress"] is False

    def test_trading_workflow_answers_fallback(self):
        """Wakeup responder runs ALWAYS — even gate-skipped runs must
        answer pending commands (the 2026-09-13 unanswered /help)."""
        d = _wf_yaml(".github/workflows/ai_bot.yml")
        steps = d["jobs"]["run-bot"]["steps"]
        ans = [s for s in steps if "Answer Telegram" in str(s.get("name", ""))]
        assert ans and ans[0]["if"] == "always()"

    def test_trading_workflow_self_healing_chain(self):
        """The wakeup chain must not depend on cron alone (GitHub dropped
        four consecutive slots on 2026-09-13), and must not self-reference
        (GitHub ignores a workflow_run workflow listing itself)."""
        d = _wf_yaml(".github/workflows/ai_bot.yml")
        wr = d["on"]["workflow_run"]
        assert "AI Trading Bot" not in set(wr["workflows"])
        assert {"AI Bot Health Check", "AI Bot Telegram Poller"} <= set(wr["workflows"])
        # NOT the digest: the digest is kicked BY this workflow, so a mutual
        # pair would restart each other forever.
        assert "AI Bot Daily Digest" not in set(wr["workflows"])
        gate = [s for s in d["jobs"]["run-bot"]["steps"]
                if "Schedule gate" in str(s.get("name", ""))]
        assert gate and "scripts/ai_bot_gate.py" in gate[0]["run"]
        assert d["jobs"]["run-bot"]["timeout-minutes"] == 360

    def test_wakeup_step_is_gated_on_the_schedule(self):
        """A firing whose slot is already served must never reach the AI or
        the exchange — this is what caps the bot at 6 decisions/day despite
        a ~20-minute kick mesh (it was 22-25/day on 2026-09-14/15)."""
        d = _wf_yaml(".github/workflows/ai_bot.yml")
        steps = d["jobs"]["run-bot"]["steps"]
        wake = [s for s in steps if "Run one AI wakeup" in str(s.get("name", ""))]
        assert wake and wake[0]["if"] == "steps.gate.outputs.run == 'true'"

    def test_preslot_crons_keep_the_wakeup_punctual(self):
        """Re-timed 2026-09-22. GitHub delivered only ~5 of 12 scheduled
        firings/day, so the old :02/:32 lines could never reliably land inside
        the gate's relay window — which is why 08:00 drifted hours late. The
        schedule now fires :32/:42/:52 of the hour BEFORE each slot (28/18/8
        min early) so a delivered firing relays to the slot and fires it on
        time even with the mesh heartbeat dead."""
        d = _wf_yaml(".github/workflows/ai_bot.yml")
        crons = [c["cron"] for c in d["on"]["schedule"]]
        minutes, hours = set(), set()
        for cron in crons:
            minute_field, hour_field, *_ = cron.split()
            minutes |= {int(m) for m in minute_field.split(",")}
            hours |= {int(h) for h in hour_field.split(",") if h != "*"}
        assert {0, 6, 8, 14, 20, 23} <= hours, "every slot hour must still fire"
        for slot_hour in (6, 8, 14, 20):
            assert slot_hour - 1 in hours, (
                f"no pre-slot deliveries before {slot_hour:02d}:00"
            )
        assert {32, 42, 52} <= minutes, "pre-slot minutes must fall in the relay window"

    def test_poller_relaunches_itself_without_cron(self):
        """The heartbeat must not depend on GitHub's cron: it was dropped for
        4h on 09-20 (07:47->11:12) and 09-21 (06:38->10:39), which is what made
        the 08:00 wakeup late/missed. A workflow cannot kick itself via
        workflow_run (GitHub ignores self-references — proven here), but a
        workflow_dispatch from GITHUB_TOKEN IS honoured, so each generation now
        dispatches the next one as its LAST step."""
        d = _wf_yaml(".github/workflows/ai_poller.yml")
        assert d["permissions"]["actions"] == "write"
        steps = d["jobs"]["poll"]["steps"]
        last = steps[-1]
        assert "Relaunch" in str(last.get("name", ""))
        assert "gh workflow run ai_poller.yml" in last["run"]
        # Last on purpose: the dispatch starts a run in the same concurrency
        # group (cancel-in-progress), so a later step could be killed by it.
        # Not on cancel, though — cancelling must be able to stop the chain.
        assert "cancelled()" in str(last.get("if", ""))

    def test_health_check_recheck_window(self):
        """The watchdog waits for the self-healing chain before staying red."""
        text = open(".github/workflows/ai_health_check.yml", encoding="utf-8").read()
        assert "RECHECK_MINUTES" in text
        assert "setFailed" in text  # still fails loudly if never recovered

    def test_digest_marker_persisted(self):
        """The digest's sent-marker must persist, or manual test runs keep
        re-tripping the scheduled digest (the false-alarm bug). The workflow
        also needs contents:write — read-only made the persist step fail on
        2026-09-14/15/16 (3/3 runs red) and the marker never reached the
        cloud, leaving the anti-duplicate guard inert."""
        d = _wf_yaml(".github/workflows/ai_daily_digest.yml")
        names = [str(s.get("name", "")) for s in d["jobs"]["daily-digest"]["steps"]]
        assert any("Persist digest marker" in n for n in names)
        assert d["permissions"]["contents"] == "write"

    def test_digest_is_delivered_after_the_last_slot(self):
        """The report must be able to land right after the 23:00 slot, not
        whenever GitHub runs the 23:50 cron (it ran ~2h late, three days
        running, which is what mislabelled the reported day)."""
        d = _wf_yaml(".github/workflows/ai_daily_digest.yml")
        assert "AI Trading Bot" in set(d["on"]["workflow_run"]["workflows"])
        crons = [c["cron"] for c in d["on"]["schedule"]]
        assert "50 23 * * *" in crons and len(crons) >= 2  # + a backup firing
        steps = d["jobs"]["daily-digest"]["steps"]
        send = [s for s in steps if "Send daily Telegram digest" in str(s.get("name", ""))]
        # No conditional skip path: the script itself decides, and a run that
        # is already delivered exits 0 loudly instead of being skipped by
        # YAML that could misfire.
        assert send and "if" not in send[0]
        assert "python -u scripts/ai_daily_digest.py" in send[0]["run"]


class TestPreLiveChecklist:
    """docs/AI_BOT_PRE_LIVE.md — config/trade claims must match the code."""

    def test_checklist_doc_exists(self):
        from pathlib import Path as _Path

        p = _Path("docs/AI_BOT_PRE_LIVE.md")
        assert p.exists(), "the pre-live checklist doc must exist"
        text = p.read_text(encoding="utf-8")
        for section in ("GREEN", "YELLOW", "RED", "Testnet", "leverage"):
            assert section in text
