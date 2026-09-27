"""The drawdown halt must release — the deadlock the backtest found.

The halt blocked new entries on price alone, while the peak it measured against
only ever rose. A flat account 10% below its peak therefore could not trade,
could not recover, and stayed halted for good: the harness measured the trend
source making its last trade on 2022-02-04 and then refusing 12,588 entries for
the following four and a half years. That is a stalemate the mission forbids.

These tests drive the PRODUCTION wakeup path (``AIAgent.run_wakeup``) and the
committed harness (``run_backtest``) — not a re-implementation — and pin four
things:

  * the halt releases on a defined condition: the cool-off is SERVED, and then
    either the account is back inside the limit (release) or the baseline is
    re-armed (release from a new peak);
  * the cool-off cannot be cancelled by a price wiggle — a halt is a flat
    period, not a flag;
  * the risk, exposure and venue caps are untouched and still bind afterwards;
  * repeated re-arms are bounded without ever becoming permanent;
  * the halt's OWN three parameters are clamped where they are read, so a
    mistyped 0 or a negative window cannot disable the cool-off, spend the
    budget for free, or restore the deadlock;
  * every one of those corrections is reported — including the truncation of a
    fractional limit and a cool-off too large to become a date — and a value
    the reader cannot parse is a correction, not a wakeup-killing error.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.test_ai_live_safety import LiveFakeExchange, live_agent
from trading_system.bot.ai_agent import (
    HALT_MAX_COOLDOWN_HOURS,
    AIAgent,
    _halt_params,
)
from trading_system.bot.ai_engine import TradeAction, TradingDecision
from trading_system.bot.backtest import BacktestConfig, ScriptedSource, run_backtest

PAIR = "ETH/USDT:USDT"
OTHER = "BTC/USDT:USDT"
CONFIG = Path("configs/ai_bot.yaml")

# The account the tests start from: $100 of cash, a $125 recorded peak. That is
# a 20% drawdown with nothing open — the exact state the halt cannot escape.
EQUITY = 100.0
PEAK = 125.0


# ---------------------------------------------------------------------------
# Fakes: a deterministic market and an engine that always asks for an entry
# ---------------------------------------------------------------------------

def frame(price: float = 3000.0, n: int = 220) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": price, "high": price * 1.001, "low": price * 0.999,
         "close": price, "volume": [10.0] * n},
        index=idx,
    )


class FakeExchange:
    """ETH ~$3000; BTC is present so the venue floor can be exercised."""

    def __init__(self):
        self.frames = {PAIR: frame(), OTHER: frame(60000.0)}
        self.price = {PAIR: 3000.0, OTHER: 60000.0}

    def get_ohlcv(self, pair, timeframe="1h", limit=200):
        return self.frames.get(pair, pd.DataFrame()).tail(limit)

    def get_ticker(self, pair):
        price = self.price.get(pair, 0.0)
        return {"bid": price, "ask": price, "last": price, "volume": 1e6}

    def get_funding_rate(self, pair):
        return 0.0001

    def get_balance(self):
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    def get_positions(self, pair=""):
        return []


class ScriptedEngine:
    """One decision, repeated: the model always asks for the same entries."""

    def __init__(self, *actions: TradeAction):
        self.actions = list(actions)
        self.model = "scripted"
        self.calls = 0

    def decide(self, **kwargs):
        self.calls += 1
        return TradingDecision(
            actions=list(self.actions), market_outlook="neutral",
            risk_assessment="scripted", reasoning="scripted",
        )


def eth_entry(size_pct: float = 30.0, stop: float = 5.0,
              tp: float = 7.5, pair: str = PAIR) -> TradeAction:
    return TradeAction(pair=pair, side="long", size_pct=size_pct,
                       stop_loss_pct=stop, take_profit_pct=tp,
                       confidence=80.0, reasoning="test entry")


class Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, hours: float) -> None:
        self.now = self.now + timedelta(hours=hours)


def halted_agent(tmp_path, clock, engine, **risk) -> AIAgent:
    """A paper agent whose account sits 20% below its peak, flat."""
    cfg = {
        "bot": {"pairs": [PAIR, OTHER], "timeframe": "1h",
                "paper_starting_equity": EQUITY},
        "risk": {"max_risk_per_trade_pct": 2.0, "max_position_size_pct": 30.0,
                 "max_portfolio_heat_pct": 30.0, "max_open_positions": 3,
                 "max_drawdown_pct": 10.0, "stop_loss_max_pct": 5.0,
                 "min_risk_reward_ratio": 1.5, "min_confidence_to_trade": 60,
                 "dd_cooldown_hours": 24.0, "dd_rearm_limit": 2,
                 "dd_rearm_window_days": 30, **risk},
    }
    agent = AIAgent(
        exchange=FakeExchange(), data_dir=tmp_path / "ai_bot", config=cfg,
        mode="paper", engine=engine, clock=clock,
    )
    # The recorded high-water mark is ABOVE the current cash: the state the bot
    # wakes up in after a bad run.
    agent.ledger.data["peak_equity"] = PEAK
    agent.ledger._save()
    return agent


def rejections(agent_file: Path, last_wakeup_only: bool = False) -> list[str]:
    """The rejections the journal recorded (all of them, or just the last
    wakeup's when a test is asserting about a wakeup it just ran)."""
    lines = [ln for ln in agent_file.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    if last_wakeup_only:
        lines = lines[-1:]
    out: list[str] = []
    for line in lines:
        out.extend(json.loads(line).get("rejections") or [])
    return out


# ---------------------------------------------------------------------------
# On the production path: the deadlock, and its release
# ---------------------------------------------------------------------------

class TestTheHaltReleasesOnTheProductionPath:
    def test_a_halt_with_no_release_is_permanent(self, tmp_path):
        """The pre-fix behaviour, reproduced exactly by configuration.

        A cooldown longer than the run means the halt never releases, which is
        what the shipped code did structurally: nothing lowered ``peak_equity``
        and nothing else could clear the halt. Kept as a test so the release
        condition cannot be removed by accident.
        """
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(
            tmp_path, clock, ScriptedEngine(eth_entry()),
            dd_cooldown_hours=24.0 * 3650,
        )
        logger_dir = agent.data_dir

        for _ in range(40):                       # ~6.6 days of slots
            clock.advance(4)
            agent.run_wakeup()

        assert agent.ledger.open_positions_list() == []
        assert agent.ledger.cash == pytest.approx(EQUITY)   # frozen
        assert agent.ledger.data["peak_equity"] == pytest.approx(PEAK)
        all_rejections = rejections(logger_dir / "journal.jsonl")
        assert all_rejections, "the halt must journal why it refused"
        assert all("trading halted" in r for r in all_rejections)
        assert len(all_rejections) == 40
        assert not (logger_dir / "trades.jsonl").exists()

    def test_the_halt_releases_and_the_account_trades_again(self, tmp_path):
        """The fix: a cooldown, then a re-armed baseline, then business as usual."""
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()))
        started = clock.now

        # Inside the cooldown: refused, and told when entries come back.
        for _ in range(6):                        # 0h, 4h, ... 20h
            agent.run_wakeup()
            clock.advance(4)
        assert agent.ledger.open_positions_list() == []
        assert agent.ledger.cash == pytest.approx(EQUITY)
        first = rejections(agent.data_dir / "journal.jsonl")
        assert len(first) == 6
        assert all("flat until" in r for r in first), first

        # The cooldown elapses: the peak is re-armed here and the entry fills.
        assert clock.now - started == timedelta(hours=24)
        agent.run_wakeup()
        assert len(agent.ledger.open_positions_list()) == 1
        assert agent.ledger.data["peak_equity"] == pytest.approx(EQUITY, abs=0.05)

        progress = json.loads((agent.data_dir / "progress.json").read_text())
        assert progress["dd_cooldown_until"] is None
        assert progress["dd_rearm_count"] == 1
        assert progress["peak_equity"] == pytest.approx(EQUITY, abs=0.05)

    def test_the_release_is_not_a_price_recovery(self, tmp_path):
        """The whole point: equity never climbs back, and trading resumes anyway."""
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()))
        recovered_above_line = False

        for _ in range(7):                        # crosses the 24h mark
            agent.run_wakeup()
            recovered_above_line |= (
                agent.ledger.cash >= PEAK * 0.9        # 90% of the OLD peak
            )
            clock.advance(4)

        assert not recovered_above_line, "test premise: cash never recovers"
        assert len(agent.ledger.open_positions_list()) == 1

    def test_the_cooldown_survives_a_restart(self, tmp_path):
        """A halt that is not persisted is not a halt."""
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()))
        agent.run_wakeup()                        # engages the halt
        clock.advance(4)

        # A brand-new process on the same directory must still be halted.
        restarted = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()))
        restarted.run_wakeup()
        assert restarted.ledger.open_positions_list() == []
        assert all("trading halted" in r
                   for r in rejections(restarted.data_dir / "journal.jsonl"))

        # ...and it still releases on the same schedule.
        clock.advance(20)
        restarted.run_wakeup()
        assert len(restarted.ledger.open_positions_list()) == 1

    def test_a_recovery_above_the_line_does_not_shorten_the_cool_off(self,
                                                                   tmp_path):
        """The halt is a flat PERIOD, not a flag a price wiggle can clear.

        Clearing it on the first in-limit slot is what let the real-data replay
        serve no flat time at all after a breach — 10 engage/clear cycles in a
        single window.
        """
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()))
        progress_path = agent.data_dir / "progress.json"

        agent.run_wakeup()                        # 1. engages: 20% dd, 24h flat
        assert agent.ledger.open_positions_list() == []

        # 2. Hand a fresh peak back: the account is INSIDE the limit now, and
        #    still halted, because the cool-off has not been served.
        agent.ledger.data["peak_equity"] = EQUITY
        agent.ledger._save()
        clock.advance(4)
        agent.run_wakeup()
        assert agent.ledger.open_positions_list() == []
        progress = json.loads(progress_path.read_text())
        assert progress["dd_cooldown_until"] is not None
        assert progress["dd_rearm_count"] == 0
        why = rejections(agent.data_dir / "journal.jsonl", last_wakeup_only=True)
        assert any("trading halted" in r and "cool-off" in r for r in why), why

        # 3. Served: the same in-limit state releases on its own — no re-arm
        #    spent, no baseline moved.
        clock.advance(20)                         # past the 24h cool-off
        agent.run_wakeup()
        assert len(agent.ledger.open_positions_list()) == 1
        progress = json.loads(progress_path.read_text())
        assert progress["dd_rearm_count"] == 0
        assert progress["dd_cooldown_until"] is None
        assert progress["peak_equity"] == pytest.approx(EQUITY, abs=0.05)

    def test_a_release_leaves_no_usable_cooldown(self, tmp_path):
        """The bug the first harness re-run exposed on real data.

        A halt engaged, the account recovered above the line without the
        cooldown elapsing, and the spent cooldown stayed on the books — so the
        next drawdown skipped its own flat period and re-armed immediately. On
        real data that produced 7 re-arms, 2 engagements and not one flat
        period.
        """
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()))
        progress_path = agent.data_dir / "progress.json"

        # 1. Halt, hand back a peak, then serve the cool-off to release.
        agent.run_wakeup()
        agent.ledger.data["peak_equity"] = EQUITY
        agent.ledger._save()
        clock.advance(24)
        agent.run_wakeup()
        assert len(agent.ledger.open_positions_list()) == 1
        assert json.loads(progress_path.read_text())["dd_cooldown_until"] is None

        # 2. Flatten, then draw down a fresh 20%. The release must not have left
        #    a timestamp the next breach can jump straight through: this has to
        #    start a NEW cool-off, not re-arm on the old one.
        agent.ledger.data["positions"] = {}
        agent.ledger.data["peak_equity"] = PEAK
        agent.ledger._save()
        before_count = json.loads(progress_path.read_text())["dd_rearm_count"]
        clock.advance(4)
        agent.run_wakeup()
        assert agent.ledger.open_positions_list() == []
        progress = json.loads(progress_path.read_text())
        assert progress["dd_rearm_count"] == before_count, \
            "re-armed on a stale cooldown"
        assert progress["peak_equity"] == pytest.approx(PEAK), \
            "the peak was re-armed instead of starting a fresh cool-off"
        why = rejections(agent.data_dir / "journal.jsonl", last_wakeup_only=True)
        assert any("trading halted" in r and "flat until" in r for r in why), why

    def test_the_rearm_does_not_wait_for_an_open_position(self, tmp_path):
        """The re-arm is book-agnostic BY DESIGN, and heat is what bounds it.

        This halt blocks entries and never force-closes (a force-close is a
        separate risk decision), so requiring a flat book would hand the length
        of the halt to however long an unrelated position happens to live. The
        sibling bot can wait for flat because its drawdown stop CLOSES the book
        first. Pinned so the rule cannot silently become "wait for flat": the
        re-arm lands with the position still open, and the exposure cap — which
        counts that position — is what refuses the next entry.
        """
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock,
                             ScriptedEngine(eth_entry(size_pct=25.0)))
        # A position from before the halt. BTC's $50 floor means it was opened
        # when the account was at least $166.67; here it is half of equity.
        agent.ledger.open_position(OTHER, "buy", 60000.0, 50.0, 5.0, 7.5)
        assert len(agent.ledger.open_positions_list()) == 1

        agent.run_wakeup()                         # 1. the halt engages
        assert all("trading halted" in r
                   for r in rejections(agent.data_dir / "journal.jsonl"))
        assert len(agent.ledger.open_positions_list()) == 1

        clock.advance(24)                          # 2. the cool-off is served
        agent.run_wakeup()
        progress = json.loads((agent.data_dir / "progress.json").read_text())
        assert progress["dd_cooldown_until"] is None
        assert progress["dd_rearm_count"] == 1
        assert progress["peak_equity"] < PEAK, "the baseline must have moved"
        # The position was still open when the re-arm landed...
        assert len(agent.ledger.open_positions_list()) == 1
        why = rejections(agent.data_dir / "journal.jsonl", last_wakeup_only=True)
        assert not any("trading halted" in r for r in why), why
        # ...and the exposure cap, which counts it, refused the new entry.
        assert any("portfolio heat" in r for r in why), why

    def test_the_journal_reports_the_state_the_decisions_were_made_under(
            self, tmp_path):
        """A wakeup that BOUGHT must not be journaled as halted.

        The halt is evaluated twice per wakeup — before the decision, and again
        after execution, because a fill moves equity. Journaling the
        post-execution snapshot meant an entry's own fee could push the account
        a fraction over the line and the record then read "halted": true for a
        wakeup that had just bought. The journal carries the decision-time state
        now, and the engagement still governs the next wakeup.
        """
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock,
                             ScriptedEngine(eth_entry(size_pct=25.0)))
        # $100 of cash against an $111.11 baseline is 9.999% — inside the limit
        # by a cent. The entry's fee (~$0.012) lands equity at 10.01%, outside.
        agent.ledger.data["peak_equity"] = 111.11
        agent.ledger._save()

        agent.run_wakeup()

        # Premises: the wakeup really traded, and the halt really engaged.
        assert len(agent.ledger.open_positions_list()) == 1
        progress = json.loads((agent.data_dir / "progress.json").read_text())
        assert progress["dd_cooldown_until"] is not None

        last = json.loads(
            (agent.data_dir / "journal.jsonl").read_text().strip().splitlines()[-1])
        assert last["drawdown_halt"]["halted"] is False
        assert last["drawdown_halt"]["cooldown_until"] is None
        assert not [r for r in (last.get("rejections") or [])
                    if "trading halted" in r]

        # ...and the engagement governs the NEXT wakeup, as promised.
        clock.advance(4)
        agent.run_wakeup()
        why = rejections(agent.data_dir / "journal.jsonl", last_wakeup_only=True)
        assert any("trading halted" in r for r in why), why

    def test_the_live_peak_tracks_new_highs_and_still_trips(self, tmp_path):
        """The halt's baseline must be a high-water mark in BOTH modes.

        Live mode has no paper ledger, so the peak comes from ``progress.json``
        — where it used to be frozen at the first wakeup's equity. Measured on
        the production path before the fix: $97 → $120 → $150 left
        ``peak_equity`` pinned at 97.0. A halt measured from a stale baseline is
        not protection, and a re-armed baseline that cannot rise is not a new
        baseline either.
        """
        exchange = LiveFakeExchange(equity=97.0)
        agent, _ = live_agent(tmp_path, exchange, engine=ScriptedEngine())
        last = lambda: json.loads(  # noqa: E731 - one-line reader
            (agent.data_dir / "journal.jsonl").read_text().strip().splitlines()[-1]
        )

        for equity in (97.0, 120.0, 150.0):
            exchange.equity = equity
            agent.run_wakeup()

        progress = json.loads((agent.data_dir / "progress.json").read_text())
        assert progress["peak_equity"] == pytest.approx(150.0)
        assert progress["dd_cooldown_until"] is None      # a rise is not a halt
        assert last()["drawdown_halt"]["halted"] is False

        # ...and the guard still trips on a real drawdown from that peak.
        exchange.equity = 130.0                            # 13.3% below 150
        agent.run_wakeup()
        progress = json.loads((agent.data_dir / "progress.json").read_text())
        assert progress["dd_cooldown_until"] is not None
        assert last()["drawdown_halt"]["halted"] is True

    def test_a_calm_wakeup_re_arms_nothing(self, tmp_path):
        """A healthy account inside the limit must not touch its peak.

        Pins the halt's branch ORDER, which is load-bearing: with the re-arm
        test above the breach test, every ordinary wakeup spent a re-arm and
        rewrote the high-water mark to the current equity, quietly resetting
        the protection. Six calm wakeups must cost nothing.
        """
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()))
        agent.ledger.data["peak_equity"] = EQUITY      # equity == peak
        agent.ledger._save()

        for _ in range(6):
            agent.run_wakeup()
            clock.advance(4)

        progress = json.loads((agent.data_dir / "progress.json").read_text())
        assert progress["dd_rearm_count"] == 0
        assert progress["dd_cooldown_until"] is None
        assert progress["peak_equity"] == pytest.approx(EQUITY, abs=0.05)
        assert agent.ledger.open_positions_list() != []   # it simply traded

    def test_the_budget_bounds_repeats_without_ever_deadlocking(self, tmp_path):
        """Spending the re-arm budget holds the halt; it never ends it."""
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(
            tmp_path, clock, ScriptedEngine(eth_entry()),
            dd_rearm_limit=1, dd_rearm_window_days=30, dd_cooldown_hours=24.0,
        )

        agent.run_wakeup()                        # 1. halt
        clock.advance(24)
        agent.run_wakeup()                        # 2. re-armed, entry filled
        assert len(agent.ledger.open_positions_list()) == 1
        assert json.loads((agent.data_dir / "progress.json").read_text())[
            "dd_rearm_count"] == 1

        # Force a second 10% drawdown from the new baseline.
        agent.ledger.data["positions"] = {}
        agent.ledger.data["cash"] = EQUITY * 0.85
        agent.ledger._save()
        clock.advance(24)
        agent.run_wakeup()                        # 3. halted again
        clock.advance(24)
        agent.run_wakeup()                        # 4. budget spent: still halted
        assert agent.ledger.open_positions_list() == []
        assert any("trading halted" in r
                   for r in rejections(agent.data_dir / "journal.jsonl"))

        # The window rolls over, and the bot resumes on its own — never a
        # permanent shutdown waiting for a human.
        clock.advance(24 * 30)
        agent.run_wakeup()
        assert len(agent.ledger.open_positions_list()) == 1


# ---------------------------------------------------------------------------
# The halt's own config cannot silently turn it off
# ---------------------------------------------------------------------------

class RecordingLogger:
    """Records the halt module's own log calls.

    structlog writes to stdout here, so ``caplog`` cannot see them; the module
    logger itself is patched instead and every call is kept as
    ``(event, kwargs)``. A method this double does not know would be a bug in
    the double, so it raises rather than silently swallowing the call.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def _record(self, name):
        def record(event=None, **kw):
            self.events.append((event, kw))
        return record

    def __getattr__(self, name):
        if name in ("debug", "info", "warning", "error", "exception", "critical"):
            return self._record(name)
        raise AttributeError(name)


class TestTheHaltParametersCannotBeTypoedIntoOff:
    """Every value below was measured doing real damage before the guard, or is
    a boundary the guard must report instead of swallowing.

    ``dd_rearm_limit: 0`` re-created the unrecoverable deadlock the halt exists
    to remove (100 days, 200 wakeups, zero entries, the cooldown re-armed
    forever); ``dd_rearm_window_days: -1`` let the baseline ratchet DOWN
    (100 -> 85 -> 72.25 across three breaches) because the window rolled on
    every call, so the budget never bound; ``dd_cooldown_hours: 0`` engaged and
    re-armed inside one wakeup, serving no flat period at all. A non-numeric
    value used to raise out of the wakeup (``status=error`` on every wakeup),
    a cool-off too large to become a date died in the ``timedelta`` arithmetic,
    a fractional re-arm limit was truncated in silence, and the window floor
    admitted a value that rolled the budget away before the hold it spends
    could fire.
    """

    def _state(self, agent) -> dict:
        return json.loads((agent.data_dir / "progress.json").read_text())

    def _fresh_drawdown(self, agent) -> float:
        """Flatten and drop the account 15% below its current baseline."""
        baseline = self._state(agent)["peak_equity"]
        agent.ledger.data["positions"] = {}
        agent.ledger.data["cash"] = baseline * 0.85
        agent.ledger._save()
        return baseline

    def test_a_zero_rearm_limit_cannot_restore_the_deadlock(self, tmp_path):
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()),
                             dd_rearm_limit=0)          # the deadlock value

        agent.run_wakeup()                              # 1. engages
        clock.advance(24)
        agent.run_wakeup()                              # 2. cool-off served
        assert len(agent.ledger.open_positions_list()) == 1
        assert self._state(agent)["dd_rearm_count"] == 1

        # Clamped to exactly 1, not silently unlimited: the second fresh
        # breach in the same window must HOLD.
        baseline = self._fresh_drawdown(agent)
        clock.advance(24)
        agent.run_wakeup()                              # 3. engages again
        clock.advance(24)
        agent.run_wakeup()                              # 4. budget spent: held
        assert agent.ledger.open_positions_list() == []
        assert self._state(agent)["peak_equity"] == pytest.approx(baseline)

        # And never the old one-way door: the window rolls, and it trades.
        clock.advance(24 * 31)
        agent.run_wakeup()
        assert len(agent.ledger.open_positions_list()) == 1

    def test_a_negative_window_cannot_disable_the_budget(self, tmp_path):
        """The window is clamped UP until the budget can actually be spent:
        a window of one cool-off or less rolls before the hold can fire."""
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(
            tmp_path, clock, ScriptedEngine(eth_entry()),
            dd_rearm_limit=1, dd_rearm_window_days=-1,
        )
        agent.run_wakeup()                              # engages at T
        clock.advance(24)
        agent.run_wakeup()                              # cools off, re-arms
        assert len(agent.ledger.open_positions_list()) == 1
        assert self._state(agent)["dd_rearm_count"] == 1

        # A window that always rolled used to re-arm this breach for free AND
        # move the baseline down. Both must now be impossible.
        baseline = self._fresh_drawdown(agent)
        clock.advance(24)
        agent.run_wakeup()                              # engages again
        clock.advance(24)
        agent.run_wakeup()                              # budget spent: HELD
        assert agent.ledger.open_positions_list() == []
        assert self._state(agent)["peak_equity"] == pytest.approx(baseline)
        assert self._state(agent)["dd_rearm_count"] == 1

    def test_a_zero_cooldown_cannot_end_the_halt_inside_one_wakeup(self, tmp_path):
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()),
                             dd_cooldown_hours=0.0)
        agent.run_wakeup()
        state = self._state(agent)
        # It used to re-arm inside this very wakeup, discarding the peak.
        assert state["dd_rearm_count"] == 0
        assert state["dd_cooldown_until"] is not None
        assert agent.ledger.open_positions_list() == []
        assert any("trading halted" in r for r in
                   rejections(agent.data_dir / "journal.jsonl", True))

        # A REAL cool-off is served: half of it in, the account is still flat.
        clock.advance(12)
        agent.run_wakeup()
        assert agent.ledger.open_positions_list() == []
        assert any("trading halted" in r for r in
                   rejections(agent.data_dir / "journal.jsonl", True))

        # Then it resumes like any other served cool-off: not a permanent lock.
        clock.advance(12)
        agent.run_wakeup()
        assert len(agent.ledger.open_positions_list()) == 1

    def test_a_non_numeric_value_falls_back_instead_of_killing_the_wakeup(
            self, tmp_path):
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()),
                             dd_rearm_limit="two")
        result = agent.run_wakeup()
        assert result["status"] == "success"            # used to be status=error
        assert agent.ledger.open_positions_list() == []  # the halt still engages
        clock.advance(24)                                # the documented default
        agent.run_wakeup()
        assert len(agent.ledger.open_positions_list()) == 1

    # --- the guard's own read, and what it reports -------------------------

    @pytest.mark.parametrize("bad", [
        {"dd_cooldown_hours": "24h"},        # a string float() cannot read
        {"dd_cooldown_hours": []},           # a type float() refuses
        {"dd_rearm_limit": 10 ** 400},       # a number float() overflows on
    ])
    def test_a_value_the_guard_cannot_read_cannot_kill_the_wakeup(
            self, tmp_path, bad):
        """The read itself used to be the fragile part: ``float(default)`` ran
        outside any guard, so a bad value in the config's ``risk:`` block raised
        out of ``_halt_params`` and every wakeup ended ``status=error``."""
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()), **bad)
        result = agent.run_wakeup()
        assert result["status"] == "success"             # used to be status=error
        assert agent.ledger.open_positions_list() == []  # the halt still engages
        clock.advance(24)                                # the documented default
        agent.run_wakeup()
        assert len(agent.ledger.open_positions_list()) == 1

    def test_an_unrepresentably_long_cool_off_is_capped_not_raised(
            self, tmp_path):
        """A finite ``1e9`` hours is not a date: the engage branch used to die
        on ``now + timedelta(hours=1e9)`` with ``date value out of range``."""
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()),
                             dd_cooldown_hours=1e9)
        result = agent.run_wakeup()
        assert result["status"] == "success"             # used to be status=error
        resume = datetime.fromisoformat(self._state(agent)["dd_cooldown_until"])
        assert resume == clock.now + timedelta(hours=HALT_MAX_COOLDOWN_HOURS)

    def test_the_configured_cool_off_default_is_validated_and_used(self):
        """The caller's default is the agent's own config value, and it was
        dead: the dict literal's own key overrode it, so the fallback was always
        the literal 24h and a bad default raised instead of defaulting.

        Direct only: on the production path the config's ``risk:`` block is
        merged into the strategy's rules, so the key is always present.
        """
        assert _halt_params({}, 12.0)["cooldown_hours"] == 12.0
        assert _halt_params({}, 0)["cooldown_hours"] == 24.0
        assert _halt_params({}, "24h")["cooldown_hours"] == 24.0
        assert _halt_params({}, 10 ** 400)["cooldown_hours"] == 24.0
        corrected = _halt_params({"dd_cooldown_hours": -5}, 12.0)
        assert corrected["cooldown_hours"] == 12.0
        assert corrected["coerced"]["dd_cooldown_hours"] == (-5, 12.0)

    def test_a_fractional_limit_is_reported_as_a_correction(
            self, tmp_path, monkeypatch):
        """2.5 re-arms reads as 2, and used to do so with ``coerced == {}`` —
        the one correction this guard applied in silence. The log is the halt
        module's own, so the module logger is recorded."""
        import trading_system.bot.ai_agent as ai_agent_module
        recorder = RecordingLogger()
        monkeypatch.setattr(ai_agent_module, "logger", recorder)

        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()),
                             dd_rearm_limit=2.5)
        agent.run_wakeup()                               # engage (20% under peak)
        corrections = [kw for event, kw in recorder.events
                       if event == "drawdown_halt_config_coerced"]
        assert corrections, "2.5 -> 2 must be reported, not applied in silence"
        assert corrections[0]["coerced"]["dd_rearm_limit"] == [2.5, 2]

        # And it is enforced as the stricter whole number it reports: the third
        # fresh breach in the window is HELD, not re-armed.
        clock.advance(24)
        agent.run_wakeup()                               # serve -> re-arm(1)
        self._fresh_drawdown(agent)
        clock.advance(24)
        agent.run_wakeup()                               # engage again
        clock.advance(24)
        agent.run_wakeup()                               # serve -> re-arm(2)
        self._fresh_drawdown(agent)
        clock.advance(24)
        agent.run_wakeup()                               # engage again (count 2)
        clock.advance(24)
        agent.run_wakeup()                               # serve -> budget spent
        assert agent.ledger.open_positions_list() == []
        assert self._state(agent)["dd_rearm_count"] == 2

    def test_the_window_floor_admits_the_budget_hold(self, tmp_path):
        """The floor is ``limit + 2`` cool-offs: at ``limit + 1`` the window
        expires on the same instant the hold would start — the roll runs before
        the state branches — so the hold never fires. Measured before this fix
        at limit=1/24h: window 2.0 days re-armed, 2.5 days held."""
        assert _halt_params({"dd_rearm_limit": 1, "dd_rearm_window_days": 3.0},
                            24.0)["coerced"] == {}
        raised = _halt_params({"dd_rearm_limit": 1,
                               "dd_rearm_window_days": 2.0}, 24.0)
        assert raised["window_days"] == 3.0
        assert raised["coerced"]["dd_rearm_window_days"] == (2.0, 3.0)
        shipped = _halt_params({"dd_rearm_limit": 2,
                                "dd_rearm_window_days": 3.0}, 24.0)
        assert shipped["window_days"] == 4.0            # the shipped limit's floor

        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(eth_entry()),
                             dd_rearm_limit=1, dd_rearm_window_days=3.0)
        agent.run_wakeup()                               # engage
        clock.advance(24)
        agent.run_wakeup()                               # serve -> re-arm(1)
        self._fresh_drawdown(agent)
        agent.run_wakeup()                               # the fresh breach, at once
        clock.advance(24)
        agent.run_wakeup()                               # (limit+1)-th cool-off
        assert agent.ledger.open_positions_list() == []  # HELD, not re-armed
        assert self._state(agent)["dd_rearm_count"] == 1
        assert self._state(agent)["dd_cooldown_until"] is not None

    def test_a_long_value_is_the_flat_period_it_says(self):
        """The guard corrects what is unusable; it does not shorten what is
        merely long. A large cool-off or window is the operator's own flat
        period — the precondition docs/AI_BOT.md states — so it is accepted
        unchanged, and only the window floor it implies is applied (logged)."""
        params = _halt_params({"dd_cooldown_hours": 8760.0}, 24.0)
        assert "dd_cooldown_hours" not in params["coerced"]
        assert params["cooldown_hours"] == 8760.0
        assert params["coerced"]["dd_rearm_window_days"] == (30.0, 1460.0)
        large = _halt_params({"dd_rearm_window_days": 10000.0}, 24.0)
        assert large["window_days"] == 10000.0
        assert large["coerced"] == {}


# ---------------------------------------------------------------------------
# The caps the fix must NOT have touched
# ---------------------------------------------------------------------------

class TestTheCapsStillBindAfterARelease:

    def _rearmed(self, tmp_path, *actions):
        """An account that has just come OUT of the halt, ready to trade.

        The halt is engaged for real first (so the cooldown exists), then the
        clock is moved past it. Every caller asserts that the wakeup it then
        runs was NOT halted, so a cap test can never pass by accident.
        """
        clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
        agent = halted_agent(tmp_path, clock, ScriptedEngine(*actions))
        agent.run_wakeup()                        # engages the halt
        progress = json.loads((agent.data_dir / "progress.json").read_text())
        until = datetime.fromisoformat(progress["dd_cooldown_until"])
        clock.now = until + timedelta(minutes=1)  # the cooldown has elapsed
        return agent

    def _not_halted(self, agent) -> list[str]:
        """The last wakeup's rejections, having proved the halt was not in force.

        The halt's own engagement earlier in the test is deliberately excluded:
        these assertions are about what happened AFTER the release.
        """
        why = rejections(agent.data_dir / "journal.jsonl", last_wakeup_only=True)
        assert not any("trading halted" in r for r in why), why
        return why

    def test_a_legal_entry_after_the_release_is_executed(self, tmp_path):
        agent = self._rearmed(tmp_path, eth_entry())
        agent.run_wakeup()
        held = agent.ledger.open_positions_list()
        assert len(held) == 1
        # Inside the 30% exposure cap that also bounds a single position here.
        assert held[0]["notional"] <= EQUITY * 0.30 + 0.01
        assert held[0]["notional"] >= 20.0            # and above ETH's floor
        self._not_halted(agent)

    def test_the_size_cap_still_binds(self, tmp_path):
        agent = self._rearmed(tmp_path, eth_entry(size_pct=60.0))
        agent.run_wakeup()
        assert agent.ledger.open_positions_list() == []
        why = self._not_halted(agent)
        assert any("size" in r and "outside" in r for r in why), why

    def test_the_stop_and_risk_caps_still_bind(self, tmp_path):
        # 40% x 5% stop = 2.0% (at the cap); a 9% stop is refused by both the
        # stop bound and the implied risk.
        agent = self._rearmed(
            tmp_path, eth_entry(size_pct=40.0, stop=9.0, tp=13.5)
        )
        agent.run_wakeup()
        assert agent.ledger.open_positions_list() == []
        why = self._not_halted(agent)
        assert any("stop-loss" in r or "risk" in r for r in why), why

    def test_the_venue_floor_still_binds(self, tmp_path):
        agent = self._rearmed(tmp_path, eth_entry(pair=OTHER))
        agent.run_wakeup()
        assert agent.ledger.open_positions_list() == []
        why = [r for r in self._not_halted(agent) if "venue" in r]
        assert why, "BTC must still be refused with its arithmetic"
        assert "$50.00" in why[0] and "needs ~$166.67" in why[0]

    def test_the_heat_cap_still_binds_across_entries(self, tmp_path):
        """Two full-size entries cannot both open, halt or no halt."""
        agent = self._rearmed(
            tmp_path, eth_entry(pair=PAIR), eth_entry(pair=OTHER)
        )
        agent.run_wakeup()
        held = agent.ledger.open_positions_list()
        assert len(held) <= 1
        why = self._not_halted(agent)
        assert any("MIN_NOTIONAL" in r or "heat" in r for r in why), why


# ---------------------------------------------------------------------------
# The committed harness: a halted account that keeps trading
# ---------------------------------------------------------------------------

def declining_frame(per_hour: float = -0.004, hours: int = 900,
                    price: float = 3000.0) -> pd.DataFrame:
    """A market that only goes down, so every long stops out."""
    idx = pd.date_range("2022-01-01", periods=hours, freq="1h", tz="UTC")
    close = price * np.cumprod(np.full(hours, 1.0 + per_hour))
    return pd.DataFrame(
        {"open": close, "high": close * 1.0005, "low": close * 0.9995,
         "close": close, "volume": np.full(hours, 10.0)},
        index=idx,
    )


def always_long(slot) -> TradingDecision:
    """A decision that asks for a full-size long whenever it holds nothing."""
    held = {p.get("pair") for p in slot.positions}
    return TradingDecision(
        actions=[
            TradeAction(pair=pair, side="long", size_pct=30.0,
                        stop_loss_pct=5.0, take_profit_pct=7.5,
                        confidence=80.0, reasoning="always long")
            for pair in slot.prices if pair not in held
        ],
        market_outlook="n/a (test)", risk_assessment="always long",
        reasoning="always long", model="scripted",
    )


def config_with_cooldown(tmp_path, hours: float, name: str) -> Path:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw.setdefault("risk", {})["dd_cooldown_hours"] = hours
    raw["bot"]["pairs"] = [PAIR]
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


class TestTheHarnessShowsTheAccountComingBack:
    """Replays the committed harness: same candles, same rule, only the halt
    differs. One account goes quiet for good; the other comes back."""

    def _run(self, tmp_path, hours: float, name: str):
        cfg = BacktestConfig(
            pairs=[PAIR], source="scripted", folds=0, folds_enabled=False,
            regimes=False, config_path=config_with_cooldown(tmp_path, hours, name),
        )
        return run_backtest(
            cfg, frames={PAIR: declining_frame()}, funding={},
            source=ScriptedSource(builder=always_long),
        )

    def test_a_dead_account_comes_back(self, tmp_path):
        locked = self._run(tmp_path, 24.0 * 3650, "locked.yaml").window(
            "all data (continuous)")
        fixed = self._run(tmp_path, 24.0, "fixed.yaml").window(
            "all data (continuous)")

        def opens(window):
            return [t["t"] for t in window.samples if t["side"] in ("long", "short")]

        locked_opens, fixed_opens = opens(locked), opens(fixed)
        assert locked_opens, "the run must trade before the halt engages"
        assert "drawdown halt" in locked.rejections

        # The dead account: halted, then nothing but rejections ever after.
        assert fixed_opens[-1] > locked_opens[-1]
        assert (datetime.fromisoformat(fixed_opens[-1])
                - datetime.fromisoformat(locked_opens[-1])
                > timedelta(hours=24))
        assert len(fixed_opens) > len(locked_opens)

        # And the halt is what changed, not the money: both start from the
        # configured $97 account.
        assert locked.start_equity == fixed.start_equity == 97.0
