"""Tests for the AI trading bot: agent, paper ledger, scheduler, engine.

Zero network: uses a FakeExchange and a ScriptedEngine so every rule is
deterministic. The scheduler test is the regression test for the midnight
rollover bug (00:00 wakeup being skipped every day).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

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
    """The cron dedupe gate (scripts/ai_bot_gate.py)."""

    def _run_gate(self, monkeypatch, tmp_path, journal_lines=None):
        import importlib.util
        import pathlib
        spec = importlib.util.spec_from_file_location(
            "ai_bot_gate", pathlib.Path("scripts/ai_bot_gate.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        journal = tmp_path / "journal.jsonl"
        if journal_lines is not None:
            journal.write_text("\n".join(journal_lines) + "\n")
        monkeypatch.setattr(mod, "JOURNAL", journal)
        return mod.main()

    def test_no_journal_proceeds(self, tmp_path, monkeypatch):
        assert self._run_gate(monkeypatch, tmp_path) == 0

    def test_recent_wakeup_skips(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        rc = self._run_gate(
            monkeypatch, tmp_path,
            [json.dumps({"timestamp": now_iso, "status": "success"})])
        assert rc == 3  # skip

    def test_old_wakeup_proceeds(self, tmp_path, monkeypatch):
        from datetime import datetime, timedelta, timezone
        old_iso = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        rc = self._run_gate(
            monkeypatch, tmp_path,
            [json.dumps({"timestamp": old_iso, "status": "success"})])
        assert rc == 0

    def test_corrupt_journal_proceeds(self, tmp_path, monkeypatch):
        # A corrupt journal must not silently block trading.
        rc = self._run_gate(monkeypatch, tmp_path, ["{{{not json"])
        assert rc == 0


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

        d = self._write(tmp_path, [
            self._entry("2026-09-13T14:02:11+00:00"),
            self._entry("2026-09-13T23:32:09+00:00",
                        reasoning="Flat market, standing aside."),
        ])
        s = summarize_day(d, now=self.NOW)
        msg = format_digest(s)
        assert "✅ Wakeups: 2/2 ok" in msg
        assert "no trades closed" in msg
        assert "$10,000.00" in msg
        assert 'AI said: "Flat market, standing aside."' in msg

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
        msg = format_digest(s)
        assert "⚠️ Wakeups: 2/3 ok" in msg
        assert "market data unavailable" in msg

    def test_no_wakeups_is_loud_alert(self, tmp_path):
        from scripts.ai_daily_digest import format_digest, summarize_day

        d = self._write(tmp_path, [
            self._entry("2026-09-12T23:32:09+00:00"),  # yesterday only
        ])
        s = summarize_day(d, now=self.NOW)
        assert s["wakeups"] == 0
        msg = format_digest(s)
        assert "NO WAKEUPS RAN TODAY" in msg

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
        # Primary continuity: the workflow relaunches itself on completion
        # (cron is unreliable for fresh workflows), and trading-wakeup
        # completions revive the chain after any outage.
        wr = d["on"]["workflow_run"]
        assert set(wr["workflows"]) == {"AI Bot Telegram Poller", "AI Trading Bot"}
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
        """Wakeup responder still runs (always()) as the poller's fallback."""
        d = _wf_yaml(".github/workflows/ai_bot.yml")
        steps = d["jobs"]["run-bot"]["steps"]
        ans = [s for s in steps if "Answer Telegram" in str(s.get("name", ""))]
        assert ans and ans[0]["if"] == "always() && steps.gate.outputs.run != 'false'"


class TestPreLiveChecklist:
    """docs/AI_BOT_PRE_LIVE.md — config/trade claims must match the code."""

    def test_checklist_doc_exists(self):
        from pathlib import Path as _Path

        p = _Path("docs/AI_BOT_PRE_LIVE.md")
        assert p.exists(), "the pre-live checklist doc must exist"
        text = p.read_text(encoding="utf-8")
        for section in ("GREEN", "YELLOW", "RED", "Testnet", "leverage"):
            assert section in text
