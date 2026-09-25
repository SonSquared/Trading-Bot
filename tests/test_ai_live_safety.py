"""R1: a live position may not exist without a confirmed protective stop.

These tests drive the REAL ``AIAgent`` execution path (``_execute_open`` /
``_execute_close`` / ``run_wakeup``) against a stateful fake venue whose every
call can be made to fail. The defect R1 describes is an ordering bug: a stop
that failed to place used to fall through, and the agent logged the trade
anyway — leaving an open, unprotected position at a $97 account size, where
losing the stop is losing the whole loss budget.

Every failure mode asserts its ONE defined outcome:

  entry not accepted          -> refuse the entry (nothing opened)
  fill unconfirmed            -> refuse the entry (nothing to protect)
  stop rejected / timed out   -> flatten the filled size, refuse the entry
  partial fill                -> protect the FILLED size, not the request
  fill too small to protect   -> flatten, refuse
  take-profit rejected        -> keep the stop-protected position, alert
  close rejected / dropped    -> retry, then alert; never report a close
                                 that did not happen
  close only partially fills  -> re-close; if the remainder will not close,
                                 leave its protective orders in place
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tests.test_ai_agent import ScriptedEngine, make_ohlcv
from trading_system.bot.ai_agent import AIAgent
from trading_system.bot.ai_engine import TradeAction, TradingDecision
from trading_system.bot.venue_limits import MarketLimits, floor_to_step

PAIR = "ETH/USDT:USDT"
PRICE = 3000.0
STEP = 0.001
MIN_NOTIONAL = 20.0
# 30% of a $97 account at $3,000 ETH, quantised to the lot step.
REQUESTED = 0.009


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class LiveFakeExchange:
    """Stateful in-memory venue for the live execution path.

    Holds one real position (so protective orders have something to reduce)
    and exposes a switch for every fault R1 has to answer.
    """

    def __init__(self, price: float = PRICE, equity: float = 97.0):
        self.price = price
        self.equity = equity
        self.position = 0.0          # signed size of the one open position
        self.entry_price = 0.0
        self.orders: list[dict] = []
        self.open_protective: list[str] = []
        self._frames = {PAIR: make_ohlcv(200, price, seed=7)}

        # --- fault switches ---
        self.entry_returns_none = False
        self.entry_raises = False
        self.entry_raises_opens_position = False
        self.entry_partial_fraction = 1.0
        self.entry_reports_no_fill = False
        self.entry_opens_nothing = False
        self.stop_returns_none = False
        self.stop_raises = False
        self.tp_returns_none = False
        self.flatten_returns_none = False
        self.close_returns_none_count = 0
        self.close_raises = False
        self.close_fraction = 1.0          # every reduce-only close fills this much
        self.close_fill_fractions: list[float] = []  # ...or a per-call queue

    # --- market data ---
    def get_ohlcv(self, pair, timeframe="1h", limit=200):
        df = self._frames.get(pair)
        return df.tail(limit) if df is not None else pd.DataFrame()

    def get_ticker(self, pair):
        return {"bid": self.price, "ask": self.price, "last": self.price,
                "volume": 1e6}

    def get_funding_rate(self, pair):
        return 0.0001

    def get_market_limits(self, pair):
        return MarketLimits(
            pair=pair, min_notional=MIN_NOTIONAL, amount_step=STEP,
            min_amount=STEP, taker_fee_pct=0.05, source="exchange",
        )

    # --- account ---
    def get_balance(self):
        return {"total": self.equity, "free": self.equity, "used": 0.0}

    def get_positions(self, pair=""):
        if self.position == 0:
            return []
        return [{
            "pair": PAIR,
            "side": "long" if self.position > 0 else "short",
            "size": abs(self.position),
            "notional": abs(self.position) * self.price,
            "entry_price": self.entry_price,
            "unrealized_pnl": 0.0,
            "leverage": 1.0,
            "liquidation_price": 0.0,
        }]

    # --- orders ---
    def place_market_order(self, pair, side, amount, reduce_only=False):
        self.orders.append({"kind": "market", "side": side, "amount": amount,
                            "reduce_only": reduce_only})
        if reduce_only:
            return self._reduce_only_order(amount)
        if self.entry_raises:
            if self.entry_raises_opens_position:
                filled = floor_to_step(amount, STEP)
                self.position = filled
                self.entry_price = self.price
            raise RuntimeError("connection dropped mid-entry")
        if self.entry_returns_none:
            return None
        filled = floor_to_step(amount * self.entry_partial_fraction, STEP)
        if self.entry_opens_nothing:
            filled = 0.0
        else:
            self.position = filled
            self.entry_price = self.price
        return {
            "order_id": "e1", "status": "closed",
            "filled": 0.0 if self.entry_reports_no_fill else filled,
            "average_price": self.price, "fee": {},
        }

    def _reduce_only_order(self, amount):
        if self.close_raises:
            raise RuntimeError("connection dropped mid-close")
        if self.flatten_returns_none:
            return None
        if self.close_returns_none_count > 0:
            self.close_returns_none_count -= 1
            return None
        frac = (self.close_fill_fractions.pop(0) if self.close_fill_fractions
                else self.close_fraction)
        filled = floor_to_step(min(amount * frac, abs(self.position)), STEP)
        if self.position > 0:
            self.position = round(max(0.0, self.position - filled), 12)
        elif self.position < 0:
            self.position = round(min(0.0, self.position + filled), 12)
        return {"order_id": "c1", "status": "closed", "filled": filled,
                "average_price": self.price, "fee": {}}

    def place_stop_market_order(self, pair, entry_side, amount, stop_price):
        self.orders.append({"kind": "stop", "amount": amount,
                            "stop_price": stop_price})
        if self.stop_raises:
            raise RuntimeError("request timed out")
        if self.stop_returns_none:
            return None
        self.open_protective.append("stop")
        return {"order_id": "sl1", "status": "new"}

    def place_take_profit_market_order(self, pair, entry_side, amount, stop_price):
        self.orders.append({"kind": "tp", "amount": amount,
                            "stop_price": stop_price})
        if self.tp_returns_none:
            return None
        self.open_protective.append("tp")
        return {"order_id": "tp1", "status": "new"}

    def cancel_all_orders(self, pair):
        self.orders.append({"kind": "cancel_all"})
        self.open_protective.clear()
        return True


class CapturingNotifier:
    """Minimal notifier that records what the operator would have received."""

    def __init__(self):
        self.errors: list[tuple[str, str]] = []
        self.emergencies: list[str] = []
        self.opens: list[dict] = []

    def notify_error(self, error, context=""):
        self.errors.append((error, context))
        return True

    def notify_emergency_stop(self, reason):
        self.emergencies.append(reason)
        return True

    def notify_trade_open(self, **kwargs):
        self.opens.append(kwargs)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def live_agent(tmp_path, exchange, engine=None):
    cfg = {
        "bot": {"pairs": [PAIR], "timeframe": "1h", "paper_starting_equity": 97.0},
        "risk": {
            "max_risk_per_trade_pct": 2.0,
            "max_position_size_pct": 30.0,
            "max_portfolio_heat_pct": 30.0,
            "max_open_positions": 3,
            "max_drawdown_pct": 10.0,
            "stop_loss_max_pct": 5.0,
            "min_risk_reward_ratio": 1.5,
            "min_confidence_to_trade": 60,
        },
    }
    notifier = CapturingNotifier()
    agent = AIAgent(
        exchange=exchange, data_dir=tmp_path / "ai_bot", config=cfg,
        mode="live", engine=engine or ScriptedEngine(), notifier=notifier,
    )
    return agent, notifier


def eth_action(**overrides) -> TradeAction:
    base = dict(
        pair=PAIR, side="long", size_pct=30.0, stop_loss_pct=5.0,
        take_profit_pct=7.5, confidence=75.0, reasoning="r1 test entry",
    )
    base.update(overrides)
    return TradeAction(**base)


def close_action() -> TradeAction:
    return TradeAction(
        pair=PAIR, side="close", size_pct=0.0, stop_loss_pct=0.0,
        take_profit_pct=0.0, confidence=80.0, reasoning="r1 test exit",
    )


def no_sleep(monkeypatch):
    monkeypatch.setattr("trading_system.bot.ai_agent.time.sleep", lambda *_: None)


def stop_orders(ex):
    return [o for o in ex.orders if o["kind"] == "stop"]


def reduce_only_orders(ex):
    return [o for o in ex.orders if o["kind"] == "market" and o["reduce_only"]]


def trade_log_lines(agent) -> list[str]:
    if not agent.trades_path.exists():
        return []
    return [ln for ln in agent.trades_path.read_text().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Stop rejected / timed out -> flatten the position, refuse the entry
# ---------------------------------------------------------------------------

class TestStopFailureFlattens:
    def test_stop_rejected_flattens_and_refuses(self, tmp_path):
        ex = LiveFakeExchange()
        ex.stop_returns_none = True
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is None
        assert ex.position == 0.0                      # position taken off
        assert ex.open_protective == []                # nothing left naked
        flat = reduce_only_orders(ex)
        assert flat and flat[-1]["amount"] == pytest.approx(REQUESTED)
        assert flat[-1]["side"] == "sell"              # opposite of the buy
        assert any("FLATTENED" in e[0] for e in note.errors)
        assert trade_log_lines(agent) == []            # no fake trade recorded
        assert agent._entries_blocked_reason

    def test_stop_timeout_flattens_too(self, tmp_path):
        """A timed-out stop raises; it must end the same way as a rejection."""
        ex = LiveFakeExchange()
        ex.stop_raises = True
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is None
        assert ex.position == 0.0
        assert any("FLATTENED" in e[0] for e in note.errors)

    def test_flatten_rejected_escalates_to_emergency(self, tmp_path):
        ex = LiveFakeExchange()
        ex.stop_returns_none = True
        ex.flatten_returns_none = True
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is None
        assert ex.position == pytest.approx(REQUESTED)  # nothing could close it
        assert note.emergencies and "NAKED" in note.emergencies[-1]
        assert any("CRITICAL" in e[0] for e in note.errors)
        assert trade_log_lines(agent) == []

    def test_refusal_costs_fees_not_the_risk_budget(self, tmp_path):
        """The defined outcome at $97: a few cents of fees, not the $1.94."""
        ex = LiveFakeExchange()
        ex.stop_returns_none = True
        agent, note = live_agent(tmp_path, ex)

        agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        message = note.errors[-1][0]
        # $27.00 round trip at the 0.05% taker rate = $0.0270
        assert "round-trip cost of ~$0.0270" in message

    def test_block_prevents_a_second_entry_in_the_same_wakeup(self, tmp_path):
        ex = LiveFakeExchange()
        ex.stop_returns_none = True
        agent, _ = live_agent(tmp_path, ex)

        assert agent._execute_open(eth_action(), 97.0, {PAIR: PRICE}) is None
        orders_after_failure = len(ex.orders)

        # A venue that just refused a protective stop gets no more entries.
        assert agent._execute_open(eth_action(), 97.0, {PAIR: PRICE}) is None
        assert len(ex.orders) == orders_after_failure


# ---------------------------------------------------------------------------
# Partial fills
# ---------------------------------------------------------------------------

class TestPartialFill:
    def test_protective_orders_cover_only_the_filled_size(self, tmp_path):
        ex = LiveFakeExchange()
        ex.entry_partial_fraction = 0.8
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is not None
        filled = trade["amount"]
        assert filled == pytest.approx(0.007)          # 80% of 0.009
        assert filled < REQUESTED
        assert trade["size_usd"] == pytest.approx(0.007 * PRICE, abs=0.01)
        assert stop_orders(ex)[-1]["amount"] == pytest.approx(filled)
        assert [o for o in ex.orders if o["kind"] == "tp"][-1]["amount"] == pytest.approx(filled)
        assert not note.errors

    def test_partial_fill_too_small_to_protect_is_flattened(self, tmp_path):
        """40% of $27 is $10.80 — under the $20 venue minimum, so no stop can
        even be placed for it: flatten and refuse rather than hold it naked."""
        ex = LiveFakeExchange()
        ex.entry_partial_fraction = 0.4
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is None
        assert ex.position == 0.0
        assert stop_orders(ex) == []                   # never even attempted
        assert any("FLATTENED" in e[0] for e in note.errors)
        assert "venue minimum" in note.errors[-1][0] or "cannot carry" in note.errors[-1][0]

    def test_unreported_fill_reads_the_position(self, tmp_path):
        """When the venue omits the fill, the position itself is the truth."""
        ex = LiveFakeExchange()
        ex.entry_reports_no_fill = True
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is not None
        assert trade["amount"] == pytest.approx(REQUESTED)
        assert trade["stop_loss_order"] == "sl1"
        assert not note.errors

    def test_unconfirmed_fill_refuses_the_entry(self, tmp_path):
        ex = LiveFakeExchange()
        ex.entry_reports_no_fill = True
        ex.entry_opens_nothing = True
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is None
        assert stop_orders(ex) == []
        assert any("REFUSED" in e[0] for e in note.errors)
        assert trade_log_lines(agent) == []


# ---------------------------------------------------------------------------
# Entry order rejected / connection dropped on the entry
# ---------------------------------------------------------------------------

class TestEntryFailure:
    def test_entry_not_accepted_refuses(self, tmp_path):
        ex = LiveFakeExchange()
        ex.entry_returns_none = True
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is None
        assert stop_orders(ex) == []
        assert any("REFUSED" in e[0] for e in note.errors)

    def test_connection_drop_with_no_position_refuses(self, tmp_path):
        ex = LiveFakeExchange()
        ex.entry_raises = True
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is None
        assert ex.position == 0.0
        assert any("REFUSED" in e[0] for e in note.errors)

    def test_connection_drop_that_still_opened_gets_protected(self, tmp_path):
        """A dropped response can hide a real fill: read it back and protect it."""
        ex = LiveFakeExchange()
        ex.entry_raises = True
        ex.entry_raises_opens_position = True
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is not None
        assert ex.position == pytest.approx(REQUESTED)
        assert ex.open_protective == ["stop", "tp"]
        assert stop_orders(ex)[-1]["amount"] == pytest.approx(REQUESTED)


# ---------------------------------------------------------------------------
# Take-profit failure keeps a stopped position (the stop is what bounds loss)
# ---------------------------------------------------------------------------

class TestTakeProfitFailure:
    def test_missing_take_profit_is_alerted_not_flattened(self, tmp_path):
        ex = LiveFakeExchange()
        ex.tp_returns_none = True
        agent, note = live_agent(tmp_path, ex)

        trade = agent._execute_open(eth_action(), 97.0, {PAIR: PRICE})

        assert trade is not None
        assert ex.position == pytest.approx(REQUESTED)  # still held ...
        assert "stop" in ex.open_protective              # ... but protected
        assert trade["take_profit_order"] is None
        assert any("take-profit" in e[0] for e in note.errors)
        assert agent._entries_blocked_reason is None     # no need to block


# ---------------------------------------------------------------------------
# Close path
# ---------------------------------------------------------------------------

class TestCloseHardening:
    def _agent_with_position(self, tmp_path, ex):
        agent, note = live_agent(tmp_path, ex)
        ex.position = REQUESTED
        ex.entry_price = PRICE
        ex.open_protective = ["stop", "tp"]
        return agent, note

    def test_transient_close_failure_is_retried(self, tmp_path, monkeypatch):
        no_sleep(monkeypatch)
        ex = LiveFakeExchange()
        ex.close_returns_none_count = 1
        agent, note = self._agent_with_position(tmp_path, ex)

        trade = agent._execute_close(close_action(), ex.get_positions(PAIR), {PAIR: PRICE})

        assert trade is not None
        assert ex.position == 0.0
        assert len(reduce_only_orders(ex)) >= 2         # one failure, one fill
        assert not note.errors

    def test_persistent_close_failure_alerts_and_keeps_position(self, tmp_path, monkeypatch):
        no_sleep(monkeypatch)
        ex = LiveFakeExchange()
        ex.close_returns_none_count = 99
        agent, note = self._agent_with_position(tmp_path, ex)

        trade = agent._execute_close(close_action(), ex.get_positions(PAIR), {PAIR: PRICE})

        assert trade is None
        assert ex.position == pytest.approx(REQUESTED)  # still open
        assert ex.open_protective == ["stop", "tp"]     # stop NOT stripped
        assert not any(o["kind"] == "cancel_all" for o in ex.orders)
        assert any("FAILED" in e[0] for e in note.errors)
        assert trade_log_lines(agent) == []

    def test_close_connection_drop_alerts(self, tmp_path, monkeypatch):
        no_sleep(monkeypatch)
        ex = LiveFakeExchange()
        ex.close_raises = True
        agent, note = self._agent_with_position(tmp_path, ex)

        trade = agent._execute_close(close_action(), ex.get_positions(PAIR), {PAIR: PRICE})

        assert trade is None
        assert ex.position == pytest.approx(REQUESTED)
        assert any("FAILED" in e[0] for e in note.errors)

    def test_partial_close_recloses_the_remainder(self, tmp_path):
        ex = LiveFakeExchange()
        ex.close_fill_fractions = [0.5, 1.0]
        agent, note = self._agent_with_position(tmp_path, ex)

        trade = agent._execute_close(close_action(), ex.get_positions(PAIR), {PAIR: PRICE})

        assert trade is not None
        assert ex.position == 0.0
        assert any(o["kind"] == "cancel_all" for o in ex.orders)
        assert not note.errors

    def test_incomplete_close_leaves_protective_orders(self, tmp_path, monkeypatch):
        no_sleep(monkeypatch)
        ex = LiveFakeExchange()
        ex.close_fraction = 0.5      # every close only ever halves the position
        agent, note = self._agent_with_position(tmp_path, ex)

        trade = agent._execute_close(close_action(), ex.get_positions(PAIR), {PAIR: PRICE})

        assert trade is None
        assert ex.position > 0
        assert ex.open_protective == ["stop", "tp"]     # remainder still protected
        assert not any(o["kind"] == "cancel_all" for o in ex.orders)
        assert any("INCOMPLETE" in e[0] for e in note.errors)


# ---------------------------------------------------------------------------
# End to end through run_wakeup: the journal must show what happened
# ---------------------------------------------------------------------------

class TestWakeupRecordsFaults:
    def test_stop_failure_reaches_the_journal(self, tmp_path, monkeypatch):
        no_sleep(monkeypatch)
        ex = LiveFakeExchange()
        ex.stop_returns_none = True
        engine = ScriptedEngine(TradingDecision(
            actions=[eth_action()], market_outlook="bullish",
            risk_assessment="ok", reasoning="setup",
        ))
        agent, note = live_agent(tmp_path, ex, engine=engine)

        result = agent.run_wakeup()

        assert result["status"] == "success"
        assert result["executed_trades"] == 0
        assert any("FLATTENED" in e for e in result["errors"])
        journal = (agent.data_dir / "journal.jsonl").read_text().strip().splitlines()
        entry = json.loads(journal[-1])
        assert entry["status"] == "success"
        assert any("FLATTENED" in e for e in entry["errors"])
        assert entry["actions_executed"] == 0
        assert entry["equity"] == pytest.approx(97.0, abs=0.01)

    def test_block_does_not_leak_across_wakeups(self, tmp_path):
        ex = LiveFakeExchange()
        engine = ScriptedEngine(TradingDecision(
            actions=[eth_action()], market_outlook="bullish",
            risk_assessment="ok", reasoning="setup",
        ))
        agent, _ = live_agent(tmp_path, ex, engine=engine)
        agent._entries_blocked_reason = "left over from a previous wakeup"

        result = agent.run_wakeup()

        assert result["executed_trades"] == 1          # the reset let it through
        assert ex.position == pytest.approx(REQUESTED)
        assert ex.open_protective == ["stop", "tp"]

    def test_clean_entry_has_no_errors(self, tmp_path):
        ex = LiveFakeExchange()
        engine = ScriptedEngine(TradingDecision(
            actions=[eth_action()], market_outlook="bullish",
            risk_assessment="ok", reasoning="setup",
        ))
        agent, note = live_agent(tmp_path, ex, engine=engine)

        result = agent.run_wakeup()

        assert result["status"] == "success"
        assert result["errors"] == []
        assert not note.errors
        assert ex.open_protective == ["stop", "tp"]
