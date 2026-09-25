"""Tests for account scale and venue legality at the account's real size.

Two contracts are pinned here:

1. ``bot.paper_starting_equity`` is the ONE owner of the account's origin. A
   config 97 yields a 97 ledger, a missing key raises instead of inventing an
   account, and existing history is never rewritten.
2. An order the venue would refuse is refused here too, with a reason that
   says why and what it would take — never silently "filled" in paper.

Zero network: the fake exchange deliberately has no ``get_market_limits``, so
the agent falls back to the dated builtin table that mirrors real Binance
filters (MIN_NOTIONAL 50 BTC / 20 ETH, step 0.001, taker 0.05%).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
from datetime import datetime, timezone

import pytest

from trading_system.bot.account import (
    DEFAULT_STARTING_EQUITY,
    legacy_scale_notice,
    resolve_starting_equity,
    same_scale,
)
from trading_system.bot.ai_agent import DEFAULT_STRATEGY, AIAgent, PaperLedger
from trading_system.bot.ai_engine import TradeAction, TradingDecision
from trading_system.bot.venue_limits import (
    BUILTIN_LIMITS,
    MarketLimits,
    floor_to_step,
    leverage_for_margin,
    max_notional_for_risk,
    min_equity_required,
    min_legal_notional,
    order_problems,
    pair_feasibility,
    project_scale,
    resolve_limits,
    round_trip_fee,
    sizing_bounds,
)

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
SLOTS = [(0, 0), (6, 0), (8, 0), (14, 0), (20, 0), (23, 0)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(equity: float = 97.0, **risk_overrides) -> dict:
    risk = {
        "max_risk_per_trade_pct": 2.0,
        # The shipped config's value: the risk rules alone would allow
        # 2/5*100 = 40, but a single position cannot exceed the 30% exposure
        # cap either, so 30 is the ceiling that actually binds — and what makes
        # ETH placeable at $97.
        "max_position_size_pct": 30.0,
        "max_portfolio_heat_pct": 30.0,
        "max_open_positions": 3,
        "max_drawdown_pct": 10.0,
        "stop_loss_max_pct": 5.0,
        "min_risk_reward_ratio": 1.5,
        "min_confidence_to_trade": 60,
    }
    risk.update(risk_overrides)
    return {
        "bot": {
            "pairs": [BTC, ETH],
            "timeframe": "1h",
            "paper_starting_equity": equity,
        },
        "risk": risk,
        "telegram": {"enabled": False},
    }


def _strategy(**rule_overrides) -> dict:
    rules = dict(_cfg()["risk"])
    rules.update(rule_overrides)
    return {"name": "test", "pairs": [BTC, ETH], "timeframe": "1h", "rules": rules}


def _action(**overrides) -> TradeAction:
    base = dict(
        pair=ETH, side="long", size_pct=25.0, stop_loss_pct=3.0,
        take_profit_pct=6.0, confidence=75.0, reasoning="test",
    )
    base.update(overrides)
    return TradeAction(**base)


class _Notifier:
    """Records alerts; never touches the network."""

    def __init__(self):
        self.calls: list[tuple] = []

    def notify_error(self, msg, context=""):
        self.calls.append(("error", msg))

    def notify_trade_open(self, **kw):
        self.calls.append(("open", kw))

    def notify_trade_close(self, **kw):
        self.calls.append(("close", kw))


def _agent(tmp_path, cfg: dict, **kw) -> AIAgent:
    return AIAgent(
        # No get_market_limits on purpose -> dated builtin venue table.
        exchange=object(),
        data_dir=tmp_path / "ai_bot",
        config=cfg,
        mode="paper",
        engine=object(),
        notifier=_Notifier(),
        **kw,
    )


def _load_script(name: str):
    script = (
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# The config is the single owner of the account's origin
# ---------------------------------------------------------------------------

class TestStartingEquityOwnership:
    def test_config_value_is_read(self):
        assert resolve_starting_equity(_cfg(97.0)) == (97.0, "bot.paper_starting_equity")
        assert resolve_starting_equity(_cfg(250.0))[0] == 250.0

    def test_missing_key_raises_instead_of_defaulting(self):
        with pytest.raises(ValueError, match="bot.paper_starting_equity"):
            resolve_starting_equity({"bot": {"pairs": [BTC]}})

    def test_no_config_at_all_raises(self):
        with pytest.raises(ValueError, match="bot.paper_starting_equity"):
            resolve_starting_equity(None)

    def test_removed_legacy_key_is_not_a_fallback(self):
        """The shape of the original bug: a config whose ONLY equity key is the
        never-read top-level ``paper.starting_equity`` must fail loudly — it is
        how a $10,000 book ran silently for a $97 balance."""
        with pytest.raises(ValueError, match="bot.paper_starting_equity"):
            resolve_starting_equity({"paper": {"starting_equity": 97.0}})

    @pytest.mark.parametrize("bad", ["lots", None, 0, -5])
    def test_unusable_values_raise(self, bad):
        with pytest.raises(ValueError):
            resolve_starting_equity({"bot": {"paper_starting_equity": bad}})

    def test_agent_seeds_ledger_at_configured_97(self, tmp_path):
        """End to end: config 97 -> a 97 ledger, not 10000."""
        agent = _agent(tmp_path, _cfg(97.0))
        assert agent.ledger.cash == 97.0
        assert agent.ledger.data["start_equity"] == 97.0
        assert agent.ledger.data["peak_equity"] == 97.0

    def test_default_constant_is_the_users_real_capital(self):
        assert DEFAULT_STARTING_EQUITY == 97.0

    def test_existing_history_is_never_rewritten(self, tmp_path):
        """An existing ledger keeps its own origin; the config cannot reseed it."""
        dd = tmp_path / "ai_bot"
        dd.mkdir(parents=True)
        led = PaperLedger(dd / "paper_ledger.json", starting_cash=10000.0)
        led.data["cash"] = 10048.52
        led._save()

        agent = _agent(tmp_path, _cfg(97.0))

        assert agent.starting_equity == 97.0
        assert agent.ledger.data["start_equity"] == 10000.0
        assert agent.ledger.cash == 10048.52


class TestLegacyScaleNotice:
    def test_same_scale_is_quiet(self):
        assert legacy_scale_notice(97.0, 97.0) is None
        assert same_scale(10000.0, 10000.0)

    def test_legacy_10000_history_is_labelled(self):
        notice = legacy_scale_notice(10000.0, 97.0)
        assert notice and "LEGACY SCALE" in notice
        assert "10,000.00" in notice and "97.00" in notice

    def test_unknown_origin_is_not_accused(self):
        assert legacy_scale_notice(None, 97.0) is None
        assert legacy_scale_notice("not-a-number", 97.0) is None
        assert legacy_scale_notice(0, 97.0) is None


# ---------------------------------------------------------------------------
# Venue arithmetic
# ---------------------------------------------------------------------------

class TestVenueMath:
    def test_builtin_table_matches_real_binance_filters(self):
        btc, eth = BUILTIN_LIMITS[BTC], BUILTIN_LIMITS[ETH]
        assert (btc.min_notional, btc.amount_step) == (50.0, 0.001)
        assert (eth.min_notional, eth.amount_step) == (20.0, 0.001)
        assert btc.taker_fee_pct == eth.taker_fee_pct == 0.05

    def test_floor_to_step_never_rounds_up(self):
        assert floor_to_step(0.0080833, 0.001) == 0.008
        assert floor_to_step(0.000776, 0.001) == 0.0
        assert floor_to_step(0.02, 0.001) == 0.02

    def test_min_legal_notional_takes_the_tighter_floor(self):
        assert min_legal_notional(BUILTIN_LIMITS[ETH], 5.0) == 20.0
        assert min_legal_notional(
            BUILTIN_LIMITS[ETH], 100.0
        ) == 100.0

    def test_max_notional_for_risk(self):
        # 2% of 97 = $1.94 risked; a 5% stop allows a $38.80 notional.
        assert max_notional_for_risk(97.0, 2.0, 5.0) == pytest.approx(38.80)
        assert max_notional_for_risk(97.0, 0.0, 5.0) == 0.0

    def test_order_problems_names_both_failures(self):
        assert order_problems(BUILTIN_LIMITS[ETH], 24.0, 0.008) == []
        problems = order_problems(BUILTIN_LIMITS[ETH], 9.70, 0.000032)
        assert any("quantity" in p for p in problems)
        assert any("notional" in p for p in problems)

    def test_round_trip_fee_is_both_sides(self):
        assert round_trip_fee(1000.0, 0.05) == pytest.approx(1.0)

    def test_min_equity_accounts_for_the_size_cap_too(self):
        # Risk cap alone would say 20*3/2 = $30, but a 10% size cap needs $200.
        assert min_equity_required(20.0, 2.0, 3.0, 10.0) == pytest.approx(200.0)
        # At a 40% cap both demands agree (2/5*100 = 40 is risk-derived).
        assert min_equity_required(20.0, 2.0, 5.0, 40.0) == pytest.approx(50.0)
        # At the SHIPPED 30% cap the exposure rule is the binding demand.
        assert min_equity_required(20.0, 2.0, 5.0, 30.0) == pytest.approx(66.666, abs=1e-3)

    def test_sizing_bounds_reports_the_true_min_equity(self):
        b = sizing_bounds(
            ETH, BUILTIN_LIMITS[ETH], equity=97.0, risk_pct=2.0, stop_pct=5.0,
            max_position_size_pct=30.0, floor_usd=5.0,
        )
        assert (b.floor_usd, b.ceiling_usd) == (20.0, 29.1)
        assert b.min_equity == pytest.approx(66.666, abs=1e-3)
        assert b.feasible

    def test_leverage_is_margin_only(self):
        assert leverage_for_margin(38.8, 97.0, 50.0) == pytest.approx(0.8)
        assert leverage_for_margin(0.0, 97.0) == 0.0

    def test_project_scale_is_arithmetic_not_a_promise(self):
        p = project_scale(
            equity=97.0, notional=38.80, risk_pct=2.0, rr=1.5,
            trades_per_month=20.0, win_rate=0.5,
        )
        assert p["risk_usd"] == pytest.approx(1.94)
        assert p["win_net"] == pytest.approx(1.94 * 1.5 - p["fees_round_trip"])
        assert p["loss_net"] == pytest.approx(1.94 + p["fees_round_trip"])
        assert p["assumptions"]["win_rate"] == 0.5
        assert p["month_usd"] == pytest.approx(p["expectancy_per_trade"] * 20.0)


class TestFeasibilityAt97:
    def _row(self, pair, **over):
        kw = dict(
            equity=97.0, risk_pct=2.0, max_stop_pct=5.0,
            max_position_size_pct=30.0, floor_usd=5.0,
        )
        kw.update(over)
        return pair_feasibility(pair, BUILTIN_LIMITS[pair], **kw)

    def test_btc_is_not_tradable_at_97(self):
        row = self._row(BTC)
        assert row["feasible"] is False
        # 50 / 0.30: the exposure cap needs more equity than the risk cap's
        # $125, and the larger demand is the honest number to quote.
        assert row["min_equity"] == pytest.approx(166.666, abs=1e-3)
        assert "NOT tradable" in row["verdict"]

    def test_eth_is_tradable_at_97_with_the_effective_cap(self):
        row = self._row(ETH)
        assert row["feasible"] is True
        assert (row["floor_usd"], row["ceiling_usd"]) == (20.0, 29.1)

    def test_eth_is_not_tradable_under_the_old_10pct_cap(self):
        """The doc's claim, as a test: the old flat 10% cap allowed $9.70,
        under ETH's $20 minimum, so NO order was legal at $97."""
        row = self._row(ETH, max_position_size_pct=10.0)
        assert row["feasible"] is False
        assert row["min_equity"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# The agent refuses what the venue refuses
# ---------------------------------------------------------------------------

class TestAgentVenueHonesty:
    def test_btc_entry_rejected_with_the_reason_at_97(self, tmp_path):
        agent = _agent(tmp_path, _cfg(97.0))
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[_action(pair=BTC, size_pct=30.0)],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            _strategy(), positions=[], snapshot={"trading_halted": False, "equity": 97.0},
        )
        assert approved == []
        assert any("venue minimum $50.00" in r and "needs ~$166.67" in r for r in rejected)
        # ... and the reason says WHICH cap left the room.
        assert any("exposure cap 30%" in r for r in rejected)

    def test_undersized_entry_tells_the_ai_what_to_ask_for(self, tmp_path):
        agent = _agent(tmp_path, _cfg(97.0))
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[_action(pair=ETH, size_pct=5.0)],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            _strategy(), positions=[], snapshot={"trading_halted": False, "equity": 97.0},
        )
        assert approved == []
        assert any(
            "below the $20.00 venue minimum" in r and "at least 20.6%" in r
            for r in rejected
        )

    def test_old_10pct_cap_says_plainly_that_nothing_is_tradeable(self, tmp_path):
        agent = _agent(tmp_path, _cfg(97.0, max_position_size_pct=10.0))
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[_action(pair=ETH, size_pct=5.0)],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            _strategy(max_position_size_pct=10.0),
            positions=[], snapshot={"trading_halted": False, "equity": 97.0},
        )
        assert approved == []
        assert any("needs ~$200.00" in r for r in rejected)

    def test_legal_eth_entry_is_approved_and_opens(self, tmp_path):
        agent = _agent(tmp_path, _cfg(97.0))
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[_action(pair=ETH, size_pct=25.0)],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            _strategy(), positions=[], snapshot={"trading_halted": False, "equity": 97.0},
        )
        assert rejected == []
        assert len(approved) == 1

        trade = agent._execute_open(
            approved[0], equity=97.0, prices={ETH: 3000.0}
        )
        assert trade is not None
        # 25% of 97 = $24.25 -> 0.008083 ETH -> floored to the 0.001 step.
        assert trade["amount"] == pytest.approx(0.008)
        assert trade["size_usd"] == pytest.approx(24.0)
        assert ETH in agent.ledger.positions

    def test_execute_open_refuses_an_unplaceable_order(self, tmp_path):
        """Backstop: even reached directly, a sub-minimum order is not filled."""
        agent = _agent(tmp_path, _cfg(97.0))
        trade = agent._execute_open(_action(pair=ETH, size_pct=5.0), 97.0, {ETH: 3000.0})
        assert trade is None
        assert agent.ledger.positions == {}
        assert agent.ledger.cash == 97.0

    def test_execute_open_refuses_btc_below_one_step(self, tmp_path):
        agent = _agent(tmp_path, _cfg(97.0))
        trade = agent._execute_open(_action(pair=BTC, size_pct=40.0), 97.0, {BTC: 50000.0})
        assert trade is None
        assert agent.ledger.positions == {}

    def test_risk_context_states_the_venue_floor(self, tmp_path):
        """The AI cannot infer the venue floor from risk numbers, so state it."""
        agent = _agent(tmp_path, _cfg(97.0))
        strategy = _strategy()
        snapshot = agent._risk_snapshot(strategy, {}, 97.0, [])
        assert snapshot["equity"] == 97.0  # the snapshot carries the scale
        text = agent._build_risk_context(strategy, snapshot, {})
        assert "Venue Minimums" in text
        assert "$20.00" in text and "$50.00" in text
        assert "20.6% of current equity" in text
        assert "Leverage changes margin only" in text


class TestCapsAreEnforcedNotImplied:
    """The rescale is only honest if the caps it leans on are enforced directly:
    loss must not be implied by ``size cap x stop cap``, and exposure must be
    re-checked within a wakeup rather than once per run."""

    def test_implied_risk_over_the_cap_is_refused(self, tmp_path):
        # 30% of equity with a 9% stop would risk 2.7% in one trade.
        agent = _agent(tmp_path, _cfg(97.0))
        approved, rejected = agent._validate_decision(
            TradingDecision(
                actions=[_action(pair=ETH, size_pct=30.0, stop_loss_pct=9.0,
                                 take_profit_pct=20.0)],
                market_outlook="x", risk_assessment="y", reasoning="z"),
            _strategy(stop_loss_max_pct=10.0), positions=[],
            snapshot={"trading_halted": False, "equity": 97.0},
        )
        assert approved == []
        assert any("implied risk 2.70% of equity" in r for r in rejected)
        assert any("exceeds the 2% cap" in r for r in rejected)

    def test_second_entry_in_the_same_wakeup_cannot_breach_exposure(
        self, tmp_path
    ):
        """Each entry passes the per-trade gate; together they must not exceed
        the cap. (The main bot's replay caught exactly this shape — see
        tests/test_bot_robustness.py.)"""
        agent = _agent(tmp_path, _cfg(97.0))
        approved, rejected = agent._validate_decision(
            TradingDecision(
                actions=[_action(pair=ETH, size_pct=21.0),
                         _action(pair=BTC, size_pct=21.0)],
                market_outlook="x", risk_assessment="y", reasoning="z"),
            _strategy(), positions=[],
            snapshot={"trading_halted": False, "equity": 97.0},
        )
        assert [a.pair for a in approved] == [ETH]  # 21% >= ETH's $20 floor
        assert any("portfolio heat 42.0%" in r for r in rejected)
        assert any("exceed the 30% exposure cap" in r for r in rejected)

    def test_exposure_already_held_by_open_positions_counts(self, tmp_path):
        agent = _agent(tmp_path, _cfg(97.0))
        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[_action(pair=ETH, size_pct=20.0)],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            _strategy(), positions=[],
            snapshot={
                "trading_halted": False, "equity": 97.0,
                "portfolio_heat_pct": 20.0,
            },
        )
        assert approved == []
        assert any("open 20.0% + this 20%" in r for r in rejected)

    def test_a_full_size_entry_is_allowed_at_the_shipped_caps(self, tmp_path):
        """30% at the widest stop = 1.5% of equity risked, inside the 2% cap."""
        agent = _agent(tmp_path, _cfg(97.0))
        approved, rejected = agent._validate_decision(
            TradingDecision(
                actions=[_action(pair=ETH, size_pct=30.0, stop_loss_pct=5.0,
                                 take_profit_pct=10.0)],
                market_outlook="x", risk_assessment="y", reasoning="z"),
            _strategy(), positions=[],
            snapshot={"trading_halted": False, "equity": 97.0},
        )
        assert rejected == []
        assert len(approved) == 1


class TestConfigOwnsTheHardRules:
    """The config's ``risk:`` section must be the cap actually in force.

    Regression: strategy.json carried its own copy of every rule and silently
    shadowed the config, so a config of 40% was not the cap the agent applied.
    """

    def _stale_strategy_file(self, tmp_path, cap: float) -> None:
        dd = tmp_path / "ai_bot"
        dd.mkdir(parents=True, exist_ok=True)
        stale = json.loads(json.dumps(DEFAULT_STRATEGY))
        stale["rules"]["max_position_size_pct"] = cap
        (dd / "strategy.json").write_text(json.dumps(stale))

    def test_stale_strategy_cap_does_not_shadow_the_config(self, tmp_path):
        self._stale_strategy_file(tmp_path, 10.0)
        agent = _agent(tmp_path, _cfg(97.0))
        strategy = agent._read_strategy()
        assert strategy["rules"]["max_position_size_pct"] == 30.0

        approved, rejected = agent._validate_decision(
            TradingDecision(actions=[_action(pair=ETH, size_pct=25.0)],
                            market_outlook="x", risk_assessment="y", reasoning="z"),
            strategy, positions=[],
            snapshot={"trading_halted": False, "equity": 97.0},
        )
        assert rejected == []
        assert len(approved) == 1

    def test_seeded_strategy_file_carries_the_config_rules(self, tmp_path):
        agent = _agent(tmp_path, _cfg(97.0, max_position_size_pct=35.0))
        agent._read_strategy()  # seeds strategy.json on first read
        written = json.loads((tmp_path / "ai_bot" / "strategy.json").read_text())
        assert written["rules"]["max_position_size_pct"] == 35.0
        # Strategy-level fields still come from the document itself.
        assert written["pairs"] == [BTC, ETH]


class TestResolveLimits:
    def test_live_values_win_over_the_builtin_table(self):
        live = MarketLimits(
            pair=ETH, min_notional=15.0, amount_step=0.01,
            min_amount=0.01, taker_fee_pct=0.04, source="exchange",
        )
        assert resolve_limits(ETH, live).source == "exchange"
        assert resolve_limits(ETH, live).min_notional == 15.0

    def test_unknown_pair_has_no_limits(self):
        assert resolve_limits("SOL/USDT:USDT") is None


# ---------------------------------------------------------------------------
# Reports label the legacy scale instead of mixing it in
# ---------------------------------------------------------------------------

class TestReportLabelling:
    def _write(self, tmp_path, start_equity, journal=()):
        data_dir = tmp_path / "data" / "ai_bot"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "paper_ledger.json").write_text(json.dumps({
            "cash": start_equity, "start_equity": start_equity,
            "peak_equity": start_equity, "positions": {}, "closed_trades": [],
        }))
        if journal:
            (data_dir / "journal.jsonl").write_text("\n".join(journal) + "\n")
        return data_dir

    def _entry(self):
        return json.dumps({
            "wakeup_id": "20260924_230050",
            "timestamp": "2026-09-24T23:00:50+00:00",
            "status": "success", "market_outlook": "neutral",
            "ai_reasoning": "nothing to do", "closed_triggers": [],
            "errors": [], "equity": 97.0,
        })

    def test_weekly_report_flags_legacy_history(self, tmp_path):
        mod = _load_script("ai_weekly_report")
        dd = self._write(tmp_path, 10000.0)
        r = mod.build_report(dd, days=7, configured_equity=97.0)
        assert r["legacy_scale_notice"] and "LEGACY SCALE" in r["legacy_scale_notice"]

    def test_weekly_report_is_quiet_at_a_matching_scale(self, tmp_path):
        mod = _load_script("ai_weekly_report")
        dd = self._write(tmp_path, 97.0)
        r = mod.build_report(dd, days=7, configured_equity=97.0)
        assert r["legacy_scale_notice"] is None

    def test_weekly_report_without_a_configured_scale_stays_quiet(self, tmp_path):
        mod = _load_script("ai_weekly_report")
        dd = self._write(tmp_path, 10000.0)
        assert mod.build_report(dd, days=7)["legacy_scale_notice"] is None

    def test_digest_prints_the_legacy_warning(self, tmp_path):
        mod = _load_script("ai_daily_digest")
        dd = self._write(tmp_path, 10000.0, journal=[self._entry()])
        s = mod.summarize_day(
            dd, now=datetime(2026, 9, 25, 12, tzinfo=timezone.utc),
            slots=SLOTS, configured_equity=97.0,
        )
        assert s["legacy_scale_notice"]
        assert "LEGACY SCALE" in mod.format_digest(s)

    def test_digest_is_quiet_at_a_matching_scale(self, tmp_path):
        mod = _load_script("ai_daily_digest")
        dd = self._write(tmp_path, 97.0, journal=[self._entry()])
        s = mod.summarize_day(
            dd, now=datetime(2026, 9, 25, 12, tzinfo=timezone.utc),
            slots=SLOTS, configured_equity=97.0,
        )
        assert s["legacy_scale_notice"] is None
        assert "LEGACY SCALE" not in mod.format_digest(s)


def test_shipped_config_is_97_and_internally_consistent():
    """The real config must not drift back to a scale the user does not have,
    and the caps it ships must be consistent with each other."""
    import yaml

    cfg = yaml.safe_load(
        (pathlib.Path(__file__).resolve().parents[1] / "configs" / "ai_bot.yaml")
        .read_text()
    )
    equity, _ = resolve_starting_equity(cfg)
    assert equity == 97.0
    assert "paper" not in cfg  # the dead key must not come back
    risk = cfg["risk"]
    risk_derived = risk["max_risk_per_trade_pct"] / risk["stop_loss_max_pct"] * 100
    # The shipped per-trade cap is the smaller of what the risk rules allow and
    # what total exposure allows: a single position cannot be larger than
    # everything permitted, and a config naming a size no position can hold is
    # the kind of dead number this file exists to prevent.
    assert risk["max_position_size_pct"] == pytest.approx(
        min(risk_derived, risk["max_portfolio_heat_pct"])
    )
    # The rescale must not let one trade risk more than the risk cap...
    assert (
        risk["max_position_size_pct"] * risk["stop_loss_max_pct"] / 100
        <= risk["max_risk_per_trade_pct"] + 1e-9
    )
    # ... nor let the exposure cap sit below the per-trade cap, which would
    # make every entry impossible.
    assert risk["max_portfolio_heat_pct"] >= risk["max_position_size_pct"]
    # DEFAULT_STRATEGY is a document template, never the cap in force: the
    # config's risk section always overrides its rules (_with_config_rules).
    assert DEFAULT_STRATEGY["rules"]["max_position_size_pct"] == 10.0
