"""Tests for the AI bot backtest harness.

The harness's whole value is that it is the *same* code as production driven by
a simulated clock, so these tests check the things that could make that untrue:

  * the historical surface never serves a candle from after its cursor, and
    refuses to serve a different timeframe than it holds
  * the replay has no order path at all
  * the venue, fee, risk and exposure constraints are actually enforced on every
    executed entry, measured rather than assumed
  * the same seed replays to the same numbers
  * one wakeup happens per schedule slot, from the production schedule parser
  * the stop/target geometry and the two trigger models behave as declared
  * the LLM is labelled as not-testable and the report says so
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from trading_system.bot.ai_agent import AIAgent, PaperLedger
from trading_system.bot.ai_engine import TradeAction, TradingDecision
from trading_system.bot.backtest import (
    BacktestConfig,
    HistoricalExchange,
    JournalSource,
    NullSource,
    OrderPathTouched,
    RandomSource,
    ScriptedSource,
    build_source,
    run_backtest,
    run_suite,
    source_catalog,
    symbol_for,
)
from trading_system.bot.backtest.engine import (
    RecordingNotifier,
    SimClock,
    _intrabar_triggers,
    classify_rejection,
    iter_slots,
    load_slots,
)
from trading_system.bot.backtest.historical import load_ohlcv, timeframe_interval
from trading_system.bot.backtest.report import render
from trading_system.bot.backtest.synthetic import build_synthetic_frames
from trading_system.bot.venue_limits import resolve_limits

PAIR = "ETH/USDT:USDT"
OTHER = "BTC/USDT:USDT"
CONFIG = "configs/ai_bot.yaml"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def flat_frame(prices: list[float], start: str = "2022-01-01") -> pd.DataFrame:
    """OHLCV where every candle is one flat price (no intrabar range)."""
    idx = pd.date_range(start, periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices,
         "volume": [1.0] * len(prices)},
        index=idx,
    )


def small_frames(pairs=(PAIR,), days: int = 20, seed: int = 3):
    return build_synthetic_frames(list(pairs), "1h", days=days, seed=seed)


def tiny_config(tmp_path: Path, **overrides) -> BacktestConfig:
    """A fast, hermetic config: synthetic candles, one pair, no extra windows."""
    base = dict(
        pairs=[PAIR],
        synthetic=True,
        synthetic_days=overrides.pop("synthetic_days", 20),
        synthetic_seed=3,
        folds_enabled=False,
        folds=0,
        regimes=False,
        source="null",
    )
    base.update(overrides)
    return BacktestConfig(**base)


@pytest.fixture
def exchange() -> HistoricalExchange:
    return HistoricalExchange(small_frames(), timeframe="1h")


def scripted_round_trip() -> ScriptedSource:
    """Open once, then close once — exactly one round trip, deterministically."""
    state = {"opened": False}

    def build(slot):
        actions = []
        if slot.positions:
            actions.append(TradeAction(
                pair=PAIR, side="close", size_pct=0.0, stop_loss_pct=0.0,
                take_profit_pct=0.0, confidence=80.0, reasoning="scripted close",
            ))
        elif not state["opened"]:
            state["opened"] = True
            actions.append(TradeAction(
                pair=PAIR, side="long", size_pct=30.0, stop_loss_pct=5.0,
                take_profit_pct=7.5, confidence=80.0, reasoning="scripted open",
            ))
        return TradingDecision(
            actions=actions, market_outlook="x", risk_assessment="y", reasoning="z",
        )

    return ScriptedSource(builder=build)


# ---------------------------------------------------------------------------
# Historical surface: no look-ahead, no order path, no silent wrong data
# ---------------------------------------------------------------------------

class TestHistoricalSurface:
    def test_never_serves_a_candle_from_after_the_cursor(self):
        ex = HistoricalExchange(small_frames(), timeframe="1h")
        interval = timeframe_interval("1h")
        for day in range(1, 20):
            cursor = pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(days=day)
            ex.set_cursor(cursor)
            window = ex.get_ohlcv(PAIR, "1h", limit=200)
            assert not window.empty
            assert window.index[-1] + interval <= cursor
        assert ex.look_ahead_violations == 0

    def test_newest_candle_closes_exactly_at_the_cursor(self):
        """At an on-the-hour slot the last closed candle IS the slot price."""
        ex = HistoricalExchange(small_frames(), timeframe="1h")
        cursor = pd.Timestamp("2022-01-10 14:00", tz="UTC")
        ex.set_cursor(cursor)
        window = ex.get_ohlcv(PAIR, "1h", limit=200)
        assert window.index[-1] == cursor - pd.Timedelta(hours=1)
        assert ex.get_ticker(PAIR)["last"] == pytest.approx(window["close"].iloc[-1])

    def test_windows_advance_with_the_cursor_and_never_repeat_a_future_bar(self):
        ex = HistoricalExchange(small_frames(), timeframe="1h")
        seen: list[pd.Timestamp] = []
        for hour in range(200, 320):
            cursor = pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(hours=hour)
            ex.set_cursor(cursor)
            window = ex.get_ohlcv(PAIR, "1h", limit=200)
            seen.append(window.index[-1])
        assert seen == sorted(seen)
        assert len(set(seen)) == len(seen)
        assert ex.look_ahead_violations == 0

    def test_a_different_timeframe_is_refused_not_approximated(self):
        ex = HistoricalExchange(small_frames(), timeframe="1h")
        ex.set_cursor(pd.Timestamp("2022-01-10", tz="UTC"))
        with pytest.raises(ValueError, match="requested"):
            ex.get_ohlcv(PAIR, "4h")

    def test_forming_candle_option_stays_inside_the_cursor(self):
        ex = HistoricalExchange(small_frames(), timeframe="1h", forming_candle=True)
        cursor = pd.Timestamp("2022-01-10 14:00", tz="UTC")
        ex.set_cursor(cursor)
        window = ex.get_ohlcv(PAIR, "1h", limit=200)
        assert window.index[-1] == cursor
        assert ex.look_ahead_violations == 0

    def test_every_order_method_raises(self):
        ex = HistoricalExchange(small_frames(), timeframe="1h")
        ex.set_cursor(pd.Timestamp("2022-01-10", tz="UTC"))
        calls = [
            lambda: ex.place_market_order(PAIR, "buy", 1.0),
            lambda: ex.place_limit_order(PAIR, "buy", 1.0, 100.0),
            lambda: ex.place_stop_market_order(PAIR, "buy", 1.0, 95.0),
            lambda: ex.place_take_profit_market_order(PAIR, "buy", 1.0, 105.0),
            lambda: ex.cancel_order("x", PAIR),
            lambda: ex.cancel_all_orders(PAIR),
            lambda: ex.set_leverage(PAIR, 2),
        ]
        for call in calls:
            with pytest.raises(OrderPathTouched):
                call()
        assert ex.order_path_attempts == 7

    def test_missing_data_file_says_how_to_fetch_it(self, tmp_path):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_ohlcv(tmp_path, PAIR, "1h")
        assert "download_data.py" in str(excinfo.value)
        assert symbol_for(PAIR) in str(excinfo.value)

    def test_loader_reads_the_downloader_layout(self, tmp_path):
        frames = small_frames()
        target = tmp_path / symbol_for(PAIR)
        target.mkdir(parents=True)
        frames[PAIR].to_parquet(target / "klines_1h.parquet")
        loaded = load_ohlcv(tmp_path, PAIR, "1h")
        assert len(loaded) == len(frames[PAIR])
        assert str(loaded.index.tz) == "UTC"

    def test_venue_limits_fall_back_to_the_dated_builtin_table(self):
        """What the cloud runner gets when Binance is unreachable."""
        ex = HistoricalExchange(small_frames(), timeframe="1h")
        assert ex.get_market_limits(PAIR) is None
        eth = resolve_limits(PAIR, ex.get_market_limits(PAIR))
        btc = resolve_limits(OTHER, None)
        assert eth is not None and btc is not None
        assert (eth.min_notional, eth.amount_step) == (20.0, 0.001)
        assert (btc.min_notional, btc.amount_step) == (50.0, 0.001)
        assert eth.source == "builtin" and eth.taker_fee_pct == 0.05


# ---------------------------------------------------------------------------
# Synthetic data must be reproducible and must not drift
# ---------------------------------------------------------------------------

class TestSyntheticData:
    def test_same_seed_same_bytes(self):
        a = build_synthetic_frames([PAIR], days=10, seed=5)
        b = build_synthetic_frames([PAIR], days=10, seed=5)
        pd.testing.assert_frame_equal(a[PAIR], b[PAIR])

    def test_different_seed_differs(self):
        a = build_synthetic_frames([PAIR], days=10, seed=5)
        b = build_synthetic_frames([PAIR], days=10, seed=6)
        assert not a[PAIR]["close"].equals(b[PAIR]["close"])

    def test_frames_are_internally_consistent(self):
        frame = build_synthetic_frames([PAIR], days=5, seed=1)[PAIR]
        assert (frame["high"] >= frame[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (frame["low"] <= frame[["open", "close"]].min(axis=1) + 1e-9).all()
        assert (frame["volume"] > 0).all()
        assert str(frame.index.tz) == "UTC"


# ---------------------------------------------------------------------------
# The schedule comes from the production gate
# ---------------------------------------------------------------------------

class TestSchedule:
    def test_slots_match_the_production_gate(self):
        assert load_slots(CONFIG) == [(0, 0), (6, 0), (8, 0), (14, 0), (20, 0), (23, 0)]

    def test_exactly_six_slots_a_day_in_order(self):
        start = datetime(2022, 3, 1, tzinfo=timezone.utc)
        end = datetime(2022, 3, 4, tzinfo=timezone.utc)
        slots = list(iter_slots(start, end, load_slots(CONFIG)))
        assert len(slots) == 18  # three full days
        assert slots == sorted(slots)
        per_day: dict[str, int] = {}
        for slot in slots:
            per_day[slot.strftime("%Y-%m-%d")] = per_day.get(slot.strftime("%Y-%m-%d"), 0) + 1
        assert set(per_day.values()) == {6}
        assert all(start <= slot < end for slot in slots)

    def test_rejection_reasons_are_bucketed_by_cause(self):
        assert classify_rejection(
            f"{PAIR} long: venue minimum $50.00 for BTC/USDT:USDT exceeds the $29.10"
        ) == "venue: below the pair's MIN_NOTIONAL"
        assert classify_rejection(
            f"{PAIR} long: size 40.0% outside (0, 30.0%]"
        ) == "size cap"
        assert classify_rejection(
            f"{PAIR} long: implied risk 3.00% of equity exceeds the 2% cap"
        ) == "risk cap (2% of equity)"
        assert classify_rejection(
            f"{PAIR} long: trading halted: drawdown 11.0% >= limit 10.0%"
        ) == "drawdown halt"
        assert classify_rejection("something entirely new") == "other"

    def test_days_trims_the_span_not_just_the_window_count(self, tmp_path):
        """`--quick` promises a SHORT run, so the span has to shorten too."""
        full = run_backtest(
            tiny_config(tmp_path, synthetic_days=400, source="null")
        )
        tail = run_backtest(
            tiny_config(tmp_path, synthetic_days=400, days=10, source="null")
        )
        assert tail.evidence["effective_range"]["end"] == \
            full.evidence["effective_range"]["end"]
        assert tail.evidence["effective_range"]["start"] > \
            full.evidence["effective_range"]["start"]
        # Ten days at six slots, give or take the boundary slot.
        assert 54 <= tail.evidence["continuous_slots"] <= 60
        assert tail.evidence["continuous_slots"] < full.evidence["continuous_slots"]

    def test_days_cannot_start_the_replay_cold(self, tmp_path):
        """A window longer than the data must still wait for the warm-up."""
        result = run_backtest(
            tiny_config(tmp_path, synthetic_days=30, days=25, source="null")
        )
        cont = result.window("all data (continuous)")
        assert cont is not None and cont.slots > 0
        # 180 warm-up candles after 2022-01-01, not the raw 25-day cut.
        assert result.evidence["effective_range"]["start"].startswith("2022-01-08")
        assert cont.errors == []


# ---------------------------------------------------------------------------
# End-to-end: one wakeup per slot, real loop, real ledger
# ---------------------------------------------------------------------------

class TestReplayIsTheRealLoop:
    def test_one_wakeup_per_slot_and_nothing_more(self, tmp_path):
        cfg = tiny_config(
            tmp_path, synthetic_days=12, source="null",
            workdir=tmp_path / "work",
        )
        result = run_backtest(cfg)
        window = result.window("all data (continuous)")
        assert window is not None
        journal = (tmp_path / "work" / "continuous_all_data_(continuous)" / "journal.jsonl")
        lines = [ln for ln in journal.read_text().splitlines() if ln.strip()]
        assert len(lines) == window.slots
        entries = [json.loads(ln) for ln in lines]
        # every entry is a real wakeup of the production loop, one per slot
        assert all(e["status"] in {"success", "error"} for e in entries)
        assert entries[0]["wakeup_id"] <= entries[-1]["wakeup_id"]
        assert entries[0]["mode"] == "paper"

    def test_replay_never_reaches_the_order_path(self, tmp_path):
        cfg = tiny_config(tmp_path, source="random", source_params={"seed": 1})
        result = run_backtest(cfg)
        assert result.evidence["order_path_attempts"] == 0
        assert result.evidence["look_ahead_violations"] == 0

    def test_the_account_starts_at_the_configured_97(self, tmp_path):
        result = run_backtest(tiny_config(tmp_path, source="null"))
        window = result.window("all data (continuous)")
        assert window is not None
        assert window.start_equity == 97.0
        assert result.config["equity_from_config"] == 97.0
        assert result.config["bot"]["paper_starting_equity"] == 97


# ---------------------------------------------------------------------------
# Constraints, measured on what actually executed
# ---------------------------------------------------------------------------

class TestConstraintsAreEnforced:
    def test_no_trade_source_costs_nothing(self, tmp_path):
        result = run_backtest(tiny_config(tmp_path, source="null"))
        window = result.window("all data (continuous)")
        assert window is not None
        assert window.trades == 0 and window.opens == 0
        assert window.fees == 0.0
        assert window.end_equity == pytest.approx(97.0)

    def test_every_executed_entry_respected_the_caps_and_the_venue_floor(self, tmp_path):
        """Measured from the replay's own action log, not assumed."""
        cfg = tiny_config(
            tmp_path, synthetic_days=30, source="random",
            source_params={"seed": 4},
        )
        result = run_backtest(cfg)
        rules = result.config["risk_rules"]
        cap_pct = min(
            float(rules["max_position_size_pct"]),
            float(rules["max_portfolio_heat_pct"]),
        )
        risk_pct = float(rules["max_risk_per_trade_pct"])
        venues = result.config["venues"]

        opens = [t for t in result.trades if t["side"] in ("long", "short")]
        assert opens, "the random source should have taken at least one entry"
        for trade in opens:
            equity = float(trade["equity_at_slot"])
            size = float(trade["size_usd"])
            stop = float(trade["stop_loss_pct"])
            venue = venues[trade["pair"]]
            assert size <= equity * cap_pct / 100.0 + 1e-6, trade
            assert size * stop / 100.0 <= equity * risk_pct / 100.0 + 1e-9, trade
            assert size >= float(venue["min_notional"]) - 1e-6, trade
            assert float(trade["amount"]) >= float(venue["min_amount"]) - 1e-12, trade

    def test_two_full_size_proposals_in_one_wakeup_cannot_both_open(self, tmp_path):
        """Both pairs ask for the whole cap every slot; at most one may open.

        What is measured is the notional actually opened per wakeup against
        that wakeup's equity — the property the exposure cap exists to
        guarantee, independent of which pairs the venue allows.
        """
        cfg = tiny_config(
            tmp_path, pairs=[PAIR, OTHER], synthetic_days=30,
            source="random", source_params={"seed": 9, "trade_probability": 1.0},
        )
        result = run_backtest(cfg)
        rules = result.config["risk_rules"]
        cap_pct = min(float(rules["max_position_size_pct"]),
                      float(rules["max_portfolio_heat_pct"]))
        opens = [t for t in result.trades if t["side"] in ("long", "short")]
        assert opens
        opened_per_slot: dict[str, float] = {}
        equity_per_slot: dict[str, float] = {}
        for t in opens:
            opened_per_slot[t["t"]] = opened_per_slot.get(t["t"], 0.0) + float(t["size_usd"])
            equity_per_slot[t["t"]] = float(t["equity_at_slot"])
        for slot, notional in opened_per_slot.items():
            limit = equity_per_slot[slot] * cap_pct / 100.0
            assert notional <= limit + 1e-6, (slot, notional, limit)
        window = result.window("all data (continuous)")
        assert window is not None
        assert window.rejections.get("exposure cap", 0) > 0

    def test_fees_are_charged_on_both_sides_of_a_round_trip(self, tmp_path):
        cfg = tiny_config(tmp_path, synthetic_days=12, source="scripted")
        source = scripted_round_trip()
        result = run_backtest(cfg, source=source)
        window = result.window("all data (continuous)")
        assert window is not None
        assert window.trades == 1
        trade = window.closed_trades[0]
        size_usd = float(trade["size_usd"])
        notional_out = size_usd * (1 + float(trade["pnl_pct_net"]) / 100.0)
        expected_fees = size_usd * 0.0005 + notional_out * 0.0005
        assert float(trade["fees"]) == pytest.approx(expected_fees, rel=1e-3)
        assert float(trade["fees"]) > 0
        assert float(trade["net_pnl"]) < 0 or window.net_pnl != 0

    def test_venue_minimum_keeps_btc_out_at_97(self, tmp_path):
        """BTC needs ~$166.67; at $97 the refusal is arithmetic, not opinion."""
        cfg = tiny_config(
            tmp_path, pairs=[OTHER], synthetic_days=30, source="random",
            source_params={"seed": 2, "trade_probability": 1.0},
        )
        result = run_backtest(cfg)
        window = result.window("all data (continuous)")
        assert window is not None
        assert window.opens == 0
        assert window.rejections.get("venue: below the pair's MIN_NOTIONAL", 0) > 0
        example = window.rejection_examples["venue: below the pair's MIN_NOTIONAL"]
        assert "$50.00" in example and "$29.10" in example


# ---------------------------------------------------------------------------
# Stop / target geometry and the two trigger models
# ---------------------------------------------------------------------------

class TestTriggers:
    def _agent(self, tmp_path, exchange, ledger, clock):
        return AIAgent(
            exchange=exchange, data_dir=tmp_path, config={"bot": {
                "pairs": [PAIR], "timeframe": "1h", "paper_starting_equity": 97.0,
            }}, mode="paper", engine=NullSource(), ledger=ledger,
            notifier=RecordingNotifier(), clock=clock,
        )

    def test_slot_model_closes_at_the_observed_price_not_the_trigger(self, tmp_path):
        """Gaps through the stop cost more than the stop distance.

        This is the bot's real paper behaviour: the level decides WHETHER to
        exit, and the price the wakeup actually observes is the fill. Documented
        rather than smoothed away — 94.00 here is worse than the 95.00 stop.
        """
        frames = {PAIR: flat_frame([100.0, 100.0, 96.0, 94.0])}
        ex = HistoricalExchange(frames, timeframe="1h")
        clock = SimClock()
        ledger = PaperLedger(tmp_path / "l.json", starting_cash=97.0, clock=clock)
        agent = self._agent(tmp_path, ex, ledger, clock)
        ledger.open_position(PAIR, "buy", 100.0, 30.0, 5.0, 7.5)

        # at this slot the newest CLOSED candle is the 94.00 one (candle 3)
        clock.now = datetime(2022, 1, 1, 4, tzinfo=timezone.utc)
        ex.set_cursor(clock.now)
        closed = agent.process_triggers([PAIR])

        assert len(closed) == 1
        assert closed[0]["exit_price"] == pytest.approx(94.0)   # the observed price
        assert closed[0]["reason"] == "stop_loss triggered"
        assert closed[0]["gross_pnl"] == pytest.approx(-1.8)    # 0.3 x (94 - 100)
        # worse than the -5% the stop level alone would have cost
        assert closed[0]["gross_pnl"] < -30.0 * 0.05

    def test_slot_model_takes_profit_at_the_observed_price(self, tmp_path):
        frames = {PAIR: flat_frame([100.0, 100.0, 106.0, 108.0])}
        ex = HistoricalExchange(frames, timeframe="1h")
        clock = SimClock()
        ledger = PaperLedger(tmp_path / "l.json", starting_cash=97.0, clock=clock)
        agent = self._agent(tmp_path, ex, ledger, clock)
        ledger.open_position(PAIR, "buy", 100.0, 30.0, 5.0, 7.5)

        clock.now = datetime(2022, 1, 1, 4, tzinfo=timezone.utc)
        ex.set_cursor(clock.now)
        closed = agent.process_triggers([PAIR])

        assert len(closed) == 1
        assert closed[0]["exit_price"] == pytest.approx(108.0)  # the observed price
        assert closed[0]["reason"] == "take_profit triggered"
        assert closed[0]["gross_pnl"] > 30.0 * 0.075            # better than +7.5%

    def test_a_short_stop_is_above_entry(self, tmp_path):
        frames = {PAIR: flat_frame([100.0, 106.0])}
        ex = HistoricalExchange(frames, timeframe="1h")
        clock = SimClock()
        ledger = PaperLedger(tmp_path / "l.json", starting_cash=97.0, clock=clock)
        agent = self._agent(tmp_path, ex, ledger, clock)
        ledger.open_position(PAIR, "sell", 100.0, 30.0, 5.0, 7.5)

        clock.now = datetime(2022, 1, 1, 2, tzinfo=timezone.utc)
        ex.set_cursor(clock.now)
        closed = agent.process_triggers([PAIR])
        assert closed and closed[0]["exit_price"] == pytest.approx(106.0)
        assert closed[0]["gross_pnl"] < 0

    def test_intrabar_model_catches_a_stop_the_slot_model_misses(self, tmp_path):
        """A dip and recovery inside one candle: live stops out, paper does not."""
        idx = pd.date_range("2022-01-01", periods=4, freq="1h", tz="UTC")
        frames = {PAIR: pd.DataFrame(
            {"open": [100.0, 100.0, 100.0, 100.0],
             "high": [100.0, 100.0, 100.0, 100.0],
             # candle 2 dips to 93 (through a 95 stop) and closes back at 100
             "low": [100.0, 100.0, 93.0, 100.0],
             "close": [100.0, 100.0, 100.0, 100.0],
             "volume": [1.0, 1.0, 1.0, 1.0]},
            index=idx,
        )}
        ex = HistoricalExchange(frames, timeframe="1h")
        clock = SimClock()
        clock.now = datetime(2022, 1, 1, 2, tzinfo=timezone.utc)
        ledger = PaperLedger(tmp_path / "l.json", starting_cash=97.0, clock=clock)
        agent = self._agent(tmp_path, ex, ledger, clock)
        ledger.open_position(PAIR, "buy", 100.0, 30.0, 5.0, 7.5)

        ex.set_cursor(clock.now)
        assert agent.process_triggers([PAIR]) == []      # slot price never crossed
        assert ledger.positions                          # still open

        # the entry filled at the open of the 2h candle; the 2h-3h bar is the
        # first one a resting exchange stop would have seen
        intrabar = _intrabar_triggers(
            ex, ledger, clock.now, clock.now + pd.Timedelta(hours=1),
        )
        # a resting stop fills at its level (the standard convention), unlike
        # the slot model which fills at whatever price the wakeup observes
        assert intrabar and intrabar[0]["exit_price"] == pytest.approx(95.0)
        assert not ledger.positions


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_replays_to_the_same_numbers(self, tmp_path):
        def run(tag: str) -> dict:
            cfg = tiny_config(
                tmp_path / tag, synthetic_days=25, source="random",
                source_params={"seed": 5},
            )
            return run_backtest(cfg).as_dict()

        first, second = run("a"), run("b")
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    def test_a_different_seed_is_a_different_path(self, tmp_path):
        def run(tag: str, seed: int) -> dict:
            cfg = tiny_config(
                tmp_path / tag, synthetic_days=25, source="random",
                source_params={"seed": seed},
            )
            return run_backtest(cfg).as_dict()

        a = run("a", 5)
        b = run("b", 6)
        assert a["windows"] != b["windows"]

    def test_a_suite_shares_one_cache_without_changing_results(self, tmp_path):
        def configs(tag: str):
            return [
                tiny_config(tmp_path / tag, synthetic_days=20, source=name,
                            source_params={"seed": 5} if name == "random" else {})
                for name in ("null", "random")
            ]

        with_cache = run_suite(configs("cached"))
        without = [run_backtest(c) for c in configs("plain")]
        for a, b in zip(with_cache, without):
            assert a.as_dict() == b.as_dict()


# ---------------------------------------------------------------------------
# Decision sources: provenance, plumbing, and the LLM boundary
# ---------------------------------------------------------------------------

class TestDecisionSources:
    def test_every_source_declares_its_provenance(self):
        catalog = source_catalog()
        names = {entry["name"] for entry in catalog}
        assert {"null", "random", "trend", "rsi-reversion", "journal"} <= names
        for entry in catalog:
            assert entry["label"] in {"null", "mechanical", "recorded-LLM", "test"}
            assert entry["description"]
            assert entry["caveat"]
            assert entry["fitted"] is False

    def test_mechanical_sources_are_labelled_mechanical_not_llm(self):
        catalog = {entry["name"]: entry for entry in source_catalog()}
        assert catalog["trend"]["label"] == "mechanical"
        assert catalog["rsi-reversion"]["label"] == "mechanical"
        assert catalog["random"]["label"] == "null"
        assert catalog["journal"]["label"] == "recorded-LLM"
        for entry in catalog.values():
            if entry["label"] == "mechanical":
                assert "model" in entry["caveat"]

    def test_unknown_source_is_refused_with_the_list(self):
        with pytest.raises(ValueError, match="Unknown source"):
            build_source("magic")

    def test_random_source_is_reproducible_and_seed_dependent(self):
        from trading_system.bot.backtest.sources import SlotContext

        def decisions(seed: int) -> list[str]:
            src = RandomSource(seed=seed, trade_probability=1.0)
            out = []
            for i in range(40):
                slot = SlotContext(
                    now=datetime(2022, 1, 1, tzinfo=timezone.utc),
                    timeframe="1h", strategy={"rules": {
                        "max_position_size_pct": 30.0,
                        "max_portfolio_heat_pct": 30.0,
                        "max_risk_per_trade_pct": 2.0,
                    }},
                    equity=97.0, positions=[], prices={PAIR: 100.0},
                    indicators={PAIR: {"ok": True}},
                    snapshot={"portfolio_heat_pct": 0.0},
                )
                src.set_context(slot)
                out += [a.side for a in src.decide("", "", "", "").actions]
            return out

        assert decisions(3) == decisions(3)
        assert decisions(3) != decisions(4)

    def test_journal_source_replays_only_recorded_slots(self):
        records = [{
            "timestamp": "2022-01-02T06:00:00+00:00",
            "actions": [{"pair": PAIR, "side": "long", "size_pct": 30.0,
                         "stop_loss_pct": 5.0, "take_profit_pct": 7.5,
                         "confidence": 80.0, "reasoning": "recorded"}],
        }]
        source = JournalSource(records=records, tolerance_minutes=20)
        from trading_system.bot.backtest.sources import SlotContext

        def decide_at(hour: int):
            slot = SlotContext(
                now=datetime(2022, 1, 2, hour, tzinfo=timezone.utc),
                timeframe="1h", strategy={"rules": {}}, equity=97.0,
                positions=[], prices={PAIR: 100.0},
                indicators={PAIR: {"ok": True}}, snapshot={},
            )
            source.set_context(slot)
            return source.decide("", "", "", "")

        assert len(decide_at(6).actions) == 1
        assert len(decide_at(14).actions) == 0
        assert source.matched_slots == 1 and source.unmatched_slots == 1

    def test_journal_counts_records_it_cannot_replay(self):
        """A matched row with no readable actions must not read as a flat call.

        The live journal used to store ``actions_requested: 1`` — a COUNT — so
        those rows replay as nothing at all. If that were folded into
        "matched", a zero-trade result would look like the model deciding to
        sit flat. It is counted separately instead.
        """
        from trading_system.bot.backtest.sources import SlotContext

        records = [
            # Older format: only a count. Matched, but nothing to replay.
            {"timestamp": "2022-01-02T06:00:00+00:00", "actions_requested": 1,
             "equity": 10000.0},
            # This pass's format: the proposed actions themselves. The junk
            # entries are skipped rather than crashing the replay.
            {"timestamp": "2022-01-03T06:00:00+00:00", "actions_proposed": [
                {"pair": PAIR, "side": "long", "size_pct": 30.0,
                 "stop_loss_pct": 5.0, "take_profit_pct": 7.5,
                 "confidence": 80.0, "reasoning": "recorded"},
                "junk", None,
            ]},
            # An explicit empty list IS a replayable "do nothing".
            {"timestamp": "2022-01-04T06:00:00+00:00", "actions": []},
        ]
        source = JournalSource(records=records, tolerance_minutes=20)

        def decide_at(day: int):
            slot = SlotContext(
                now=datetime(2022, 1, day, 6, tzinfo=timezone.utc),
                timeframe="1h", strategy={"rules": {}}, equity=97.0,
                positions=[], prices={PAIR: 100.0},
                indicators={PAIR: {"ok": True}}, snapshot={},
            )
            source.set_context(slot)
            return source.decide("", "", "", "")

        assert decide_at(2).actions == []
        assert source.matched_slots == 1 and source.unmatched_slots == 0
        assert source.replayable_slots == 0
        assert source.records_without_actions == 1

        assert len(decide_at(3).actions) == 1
        decide_at(4)
        assert source.matched_slots == 3
        assert source.replayable_slots == 2
        assert source.records_without_actions == 1

        # No record at all is a different fact again: unmatched, not unreadable.
        assert decide_at(9).actions == []
        assert source.unmatched_slots == 1
        assert source.records_without_actions == 1

    def test_a_round_trip_runs_through_the_real_agent_loop(self, tmp_path):
        cfg = tiny_config(tmp_path, synthetic_days=12, source="scripted")
        result = run_backtest(cfg, source=scripted_round_trip())
        window = result.window("all data (continuous)")
        assert window is not None
        assert window.trades == 1
        assert window.closed_by_ai == 1
        assert window.open_positions_at_end == 0
        assert result.evidence["source_calls"] == window.slots


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------

class TestReport:
    def test_report_states_what_it_does_not_show(self, tmp_path):
        cfg = tiny_config(tmp_path, synthetic_days=12, source="null")
        results = [run_backtest(cfg)]
        text = render(results, command="python scripts/ai_backtest.py --synthetic")
        assert "WHAT THIS DOES NOT SHOW" in text
        assert "look-ahead" in text
        assert "NOT SUPPORTED" in text
        assert "not a claim about the" in text
        assert "python scripts/ai_backtest.py --synthetic" in text
        # the source's provenance travels with its numbers
        assert "[null]" in text
        assert "fitted to this data: no" in text

    def test_report_labels_a_synthetic_run_as_synthetic(self, tmp_path):
        cfg = tiny_config(tmp_path, synthetic_days=12, source="null")
        result = run_backtest(cfg)
        assert "synthetic" in result.evidence["data_source"]

    def test_report_carries_the_evidence_counters(self, tmp_path):
        cfg = tiny_config(tmp_path, synthetic_days=12, source="null")
        result = run_backtest(cfg)
        ev = result.evidence
        assert ev["slots_per_day"] == 6
        assert ev["slot_hours_utc"] == ["00:00", "06:00", "08:00", "14:00", "20:00", "23:00"]
        assert ev["look_ahead_violations"] == 0
        assert ev["order_path_attempts"] == 0
        assert ev["continuous_slots"] > 0

    def test_report_says_when_a_journal_could_not_be_replayed(self, tmp_path):
        """Zero trades from unreadable records must not read as "sat flat"."""
        journal = tmp_path / "journal.jsonl"
        journal.write_text(json.dumps({
            "timestamp": "2022-01-10T06:00:00+00:00",
            "actions_requested": 1,
            "equity": 10000.0,
        }) + "\n", encoding="utf-8")
        cfg = tiny_config(
            tmp_path, synthetic_days=14, source="journal",
            journal_path=journal,
        )
        result = run_backtest(cfg)
        assert result.evidence["journal_replayable_slots"] == 0
        assert result.evidence["journal_records_without_actions"] == 1
        text = render([result])
        assert "carried NO readable action" in text
        assert "NOT evidence that the" in text

    def test_the_venue_verdict_is_computed_from_the_config(self, tmp_path):
        """The $97 tradability claim must follow the venue helpers, not prose."""
        cfg = tiny_config(tmp_path, synthetic_days=12, pairs=[PAIR, OTHER])
        result = run_backtest(cfg)
        reach = result.evidence["venue_reach"]
        assert reach[PAIR]["tradable"] is True
        assert reach[OTHER]["tradable"] is False
        # BTC's floor is above what a $97 account may hold, and the report quotes
        # the same arithmetic the agent refuses entries with.
        assert reach[OTHER]["floor_usd"] == 50.0
        assert reach[OTHER]["ceiling_usd"] == 29.1
        assert reach[OTHER]["min_equity_usd"] == pytest.approx(166.67, abs=0.01)
        text = render([result], command="python scripts/ai_backtest.py")
        # The verdict is wrapped for the terminal, so compare it unwrapped.
        flat = " ".join(text.split())
        assert f"{PAIR} (venue floor $20.00, ceiling $29.10)" in flat
        assert f"It cannot on {OTHER}, whose $50.00 venue floor is above" in flat
        assert "needs about $166.67" in flat
        # Section numbering must not skip when optional sections are absent.
        numbers = _sections(text)
        assert numbers == set(range(1, len(numbers) + 1)), sorted(numbers)


def _sections(text: str) -> set[int]:
    import re

    return {int(m) for m in re.findall(r"^  (\d+)\. ", text, flags=re.M)}
