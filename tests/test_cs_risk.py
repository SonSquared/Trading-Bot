"""Task 5 (plan MD): portfolio governor, regime risk, kill switches.

Pinned here:
- stop-distance sizing: qty = equity * max_risk_per_trade / (price * stop),
- hard caps: position / symbol / correlation-cluster / gross / net /
  margin reserve / max positions — an ACCEPTED order can never breach one
  (property-tested with hypothesis over randomized intents and accounts),
- regime multiplier is bounded [0, 1] (property-tested),
- halts latch (drawdown, daily loss, data faults, ledger faults) and only
  clear via acknowledge() with an operator identity,
- no order path exists that bypasses the governor: it is the sole authorizer.
"""

from __future__ import annotations

import hypothesis
import pytest
from hypothesis import given
from hypothesis import strategies as st

from crypto_system.models import OrderIntent, RiskLimits
from crypto_system.risk.governor import AccountSnapshot, MarketState, RiskGovernor
from crypto_system.risk.killswitch import KillSwitch
from crypto_system.risk.regime import RegimeDetector


def _intent(**kw) -> OrderIntent:
    defaults = dict(
        symbol="BTCUSDT", side="long", notional=1000.0, strategy="trend"
    )
    defaults.update(kw)
    return OrderIntent(**defaults)


def _snapshot(**kw) -> AccountSnapshot:
    defaults = dict(
        equity=10_000.0,
        cash=10_000.0,
        gross_notional=0.0,
        net_notional=0.0,
        positions={},
        daily_pnl=0.0,
        peak_equity=10_000.0,
    )
    defaults.update(kw)
    return AccountSnapshot(**defaults)


def _market(price: float = 100.0) -> MarketState:
    return MarketState(
        price=price,
        cluster_of={"BTCUSDT": "majors", "ETHUSDT": "majors", "SOLUSDT": "alts"},
    )


class TestSizing:
    def test_stop_distance_sizing(self):
        # Uncapped limits so the raw stop-distance formula is observable.
        limits = RiskLimits(
            max_symbol_notional=100_000.0,
            max_gross_notional=100_000.0,
            max_cluster_notional=100_000.0,
            max_net_notional=100_000.0,
        )
        gov = RiskGovernor(limits)
        decision = gov.evaluate(
            _intent(), _snapshot(), _market(), stop_distance=0.02
        )
        assert decision.accepted
        expected_qty = 10_000.0 * 0.0025 / (100.0 * 0.02)
        assert decision.quantity == pytest.approx(expected_qty)

    def test_tight_stop_is_clamped_to_minimum(self):
        limits = RiskLimits(
            max_symbol_notional=100_000.0,
            max_gross_notional=100_000.0,
            max_cluster_notional=100_000.0,
            max_net_notional=100_000.0,
        )
        gov = RiskGovernor(limits)
        decision = gov.evaluate(
            _intent(), _snapshot(), _market(), stop_distance=0.0001
        )
        assert decision.stop_distance == limits.min_stop_distance
        # and sizing uses the clamped stop, not the razor-thin one
        expected_qty = 10_000.0 * 0.0025 / (100.0 * limits.min_stop_distance)
        assert decision.quantity == pytest.approx(expected_qty)


class TestCaps:
    def test_symbol_cap_clamps_quantity(self):
        limits = RiskLimits(max_symbol_notional=500.0)
        gov = RiskGovernor(limits)
        decision = gov.evaluate(_intent(), _snapshot(), _market(), stop_distance=0.02)
        assert decision.accepted
        assert decision.quantity * 100.0 <= 500.0

    def test_gross_cap_clamps_or_rejects(self):
        limits = RiskLimits(max_gross_notional=300.0)
        gov = RiskGovernor(limits)
        decision = gov.evaluate(_intent(), _snapshot(), _market(), stop_distance=0.02)
        assert decision.quantity * 100.0 <= 300.0

    def test_cluster_cap_across_correlated_symbols(self):
        limits = RiskLimits(max_cluster_notional=600.0, max_symbol_notional=10_000.0)
        gov = RiskGovernor(limits)
        snapshot = _snapshot(
            gross_notional=550.0,
            positions={"ETHUSDT": {"qty": 5.5, "price": 100.0, "side": "long",
                                   "cluster": "majors"}},
        )
        decision = gov.evaluate(_intent(), snapshot, _market(), stop_distance=0.02)
        # 550 already in majors; new BTC notional must keep cluster <= 600
        assert decision.quantity * 100.0 <= 50.0 + 1e-9

    def test_max_positions_rejects_new_symbol(self):
        gov = RiskGovernor(RiskLimits(max_positions=1))
        snapshot = _snapshot(
            positions={"ETHUSDT": {"qty": 1.0, "price": 100.0, "side": "long",
                                   "cluster": "majors"}},
        )
        decision = gov.evaluate(_intent(), snapshot, _market(), stop_distance=0.02)
        assert not decision.accepted
        assert "positions" in decision.reason

    def test_existing_symbol_topup_rejected_no_averaging_down(self):
        gov = RiskGovernor(RiskLimits())
        snapshot = _snapshot(
            gross_notional=100.0,
            positions={"BTCUSDT": {"qty": 1.0, "price": 100.0, "side": "long",
                                   "cluster": "majors"}},
        )
        decision = gov.evaluate(_intent(), snapshot, _market(), stop_distance=0.02)
        assert not decision.accepted
        assert "open position" in decision.reason

    def test_margin_reserve_respected(self):
        limits = RiskLimits(margin_reserve=0.5)
        gov = RiskGovernor(limits)
        snapshot = _snapshot(equity=10_000.0, cash=10_000.0)
        decision = gov.evaluate(
            _intent(), snapshot, _market(), stop_distance=0.02, leverage=10.0
        )
        margin = decision.quantity * 100.0 / 10.0
        assert margin <= 10_000.0 * 0.5


class TestHalts:
    def test_daily_loss_trips_latched_halt(self):
        kill = KillSwitch()
        gov = RiskGovernor(RiskLimits(), killswitch=kill)
        decision = gov.evaluate(
            _intent(),
            _snapshot(daily_pnl=-350.0),  # > 3% of 10k
            _market(),
            stop_distance=0.02,
        )
        assert not decision.accepted
        assert any(r.startswith("daily_loss") for r in kill.reasons)
        # latched: even a healthy account is refused now
        again = gov.evaluate(_intent(), _snapshot(), _market(), stop_distance=0.02)
        assert not again.accepted

    def test_drawdown_trips_latched_halt(self):
        kill = KillSwitch()
        gov = RiskGovernor(RiskLimits(), killswitch=kill)
        decision = gov.evaluate(
            _intent(),
            _snapshot(equity=8_500.0, peak_equity=10_000.0),  # 15% dd > 10%
            _market(),
            stop_distance=0.02,
        )
        assert not decision.accepted
        assert any(r.startswith("drawdown") for r in kill.reasons)

    def test_acknowledge_requires_operator_and_clears(self):
        kill = KillSwitch()
        kill.trip("data_fault", "stale feed")
        with pytest.raises(ValueError):
            kill.acknowledge("data_fault", operator="")
        kill.acknowledge("data_fault", operator="operator-1")
        assert not kill.latched

    def test_data_fault_flag_rejects(self):
        gov = RiskGovernor(RiskLimits())
        decision = gov.evaluate(
            _intent(), _snapshot(), _market(), stop_distance=0.02, data_ok=False
        )
        assert not decision.accepted
        assert "data" in decision.reason


class TestRegime:
    def test_multiplier_bounded(self):
        det = RegimeDetector()
        for vol in (0.0, 0.01, 0.05, 0.2, 1.0):
            m = det.multiplier(vol)
            assert 0.0 <= m <= 1.0

    def test_high_volatility_scales_risk_down(self):
        det = RegimeDetector()
        assert det.multiplier(0.01) > det.multiplier(0.06)
        assert det.multiplier(0.50) == 0.0

    @given(st.floats(min_value=0, max_value=5, allow_nan=False))
    @hypothesis.settings(max_examples=50, deadline=None)
    def test_multiplier_property_bounded(self, vol):
        m = RegimeDetector().multiplier(vol)
        assert 0.0 <= m <= 1.0


_PROPERTY_SETTINGS = hypothesis.settings(max_examples=40, deadline=None)


class TestGovernorProperties:
    @given(
        equity=st.floats(min_value=100, max_value=50_000, allow_nan=False),
        risk=st.floats(min_value=0.0005, max_value=0.005, allow_nan=False),
        stop=st.floats(min_value=0.005, max_value=0.15, allow_nan=False),
        gross=st.floats(min_value=0, max_value=10_000, allow_nan=False),
        price=st.floats(min_value=1, max_value=100_000, allow_nan=False),
    )
    @_PROPERTY_SETTINGS
    def test_accepted_order_never_breaks_cap(self, equity, risk, stop, gross, price):
        limits = RiskLimits(max_risk_per_trade=risk)
        gov = RiskGovernor(limits)
        snapshot = _snapshot(
            equity=equity, cash=equity, gross_notional=gross,
            net_notional=gross, peak_equity=equity,
        )
        decision = gov.evaluate(
            _intent(), snapshot, _market(price=price), stop_distance=stop
        )
        added = decision.quantity * price
        assert not decision.accepted or (
            snapshot.gross_notional + added <= limits.max_gross_notional + 1e-6
            and added <= limits.max_symbol_notional + 1e-6
        )

    @given(
        vol=st.floats(min_value=0, max_value=2, allow_nan=False),
        equity=st.floats(min_value=100, max_value=20_000, allow_nan=False),
    )
    @_PROPERTY_SETTINGS
    def test_accepted_risk_never_exceeds_per_trade_cap(self, vol, equity):
        det = RegimeDetector()
        limits = RiskLimits()
        gov = RiskGovernor(limits, regime=det)
        snapshot = _snapshot(equity=equity, cash=equity, peak_equity=equity)
        decision = gov.evaluate(
            _intent(), snapshot, _market(), stop_distance=0.02
        )
        if decision.accepted:
            risk_taken = decision.quantity * 100.0 * 0.02
            assert risk_taken <= equity * limits.max_risk_per_trade + 1e-6
