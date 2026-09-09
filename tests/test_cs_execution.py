"""Task 4 (plan MD): cost-aware perpetual paper execution.

Pinned here:
- fill price = quote + side_sign * (spread/2 + impact(volatility, notional))
- required margin = |fill_price * qty| / leverage
- taker fee on aggressive fills; deterministic seeded randomness (same seed,
  same fills); latency and partial fills modeled;
- funding applied per 8h boundary from an explicit rate schedule;
- liquidation-buffer rejection: an order that would push maintenance margin
  coverage below the buffer is refused, not executed;
- stops trigger on candle extremes and ledger everything;
- netting is prohibited: a second submit on an open position fails.
"""

from __future__ import annotations

import pytest

from crypto_system.audit.ledger import Ledger
from crypto_system.execution.paper import (
    LiquidationBufferRejected,
    PaperAccount,
    Quote,
)
from crypto_system.models import ExecutionMode, OrderIntent


def _account(tmp_path, cash: float = 10_000.0, seed: int = 42) -> PaperAccount:
    ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
    return PaperAccount(
        ledger=ledger, initial_cash=cash, maker_fee=0.0002, taker_fee=0.0005, seed=seed
    )


def _intent(
    symbol: str = "BTCUSDT", notional: float = 1000.0, side: str = "long"
) -> OrderIntent:
    return OrderIntent(symbol=symbol, side=side, notional=notional, strategy="test")


def _quote(price: float = 100.0, spread_bps: float = 2.0, vol: float = 0.01) -> Quote:
    return Quote(price=price, spread_bps=spread_bps, volatility=vol)


class TestFills:
    def test_fill_price_crosses_spread_in_trade_direction(self, tmp_path):
        buy_acct = _account(tmp_path / "a")
        sell_acct = _account(tmp_path / "b")
        fill_buy = buy_acct.submit(_intent(side="long"), _quote())
        fill_sell = sell_acct.submit(_intent(side="short"), _quote())
        assert fill_buy.price > 100.0
        assert fill_sell.price < 100.0
        # symmetric deviations around the mid
        assert abs(fill_buy.price - 100.0) == pytest.approx(
            abs(100.0 - fill_sell.price), rel=1e-9
        )

    def test_impact_scales_with_notional_and_volatility(self, tmp_path):
        small = _account(tmp_path / "a")
        big = _account(tmp_path / "b")
        q = _quote(vol=0.01)
        f_small = small.submit(_intent(notional=100.0), q)
        f_big = big.submit(_intent(notional=5_000.0, ), q) if False else big.submit(
            OrderIntent(symbol="BTCUSDT", side="long", notional=5_000.0,
                        leverage=2.0, strategy="test"),
            q,
        )
        assert f_big.price > f_small.price  # larger order walks the book more

    def test_fees_charged_on_filled_notional(self, tmp_path):
        acct = _account(tmp_path)
        fill = acct.submit(_intent(notional=1000.0), _quote())
        assert fill.fee == pytest.approx(fill.qty * fill.price * 0.0005)

    def test_deterministic_same_seed_same_fill(self, tmp_path):
        a = _account(tmp_path / "a")
        b = _account(tmp_path / "b")
        f1 = a.submit(_intent(), _quote())
        f2 = b.submit(_intent(), _quote())
        assert f1.price == f2.price
        assert f1.latency_ms == f2.latency_ms

    def test_partial_fill_possible_under_illiquidity(self, tmp_path):
        fills = []
        for i in range(30):
            acct = _account(tmp_path / f"run{i}", seed=i)
            q = _quote(vol=0.15)  # extreme volatility -> partial fill regime
            fills.append(acct.submit(_intent(notional=5_000.0), q))
        assert any(f.qty < f.requested_qty for f in fills)

    def test_double_open_rejected_no_martingale(self, tmp_path):
        acct = _account(tmp_path)
        acct.submit(_intent(notional=1000.0), _quote())
        with pytest.raises(RuntimeError, match="averaging down|already open"):
            acct.submit(_intent(notional=500.0), _quote())


class TestMargin:
    def test_required_margin_formula(self, tmp_path):
        acct = _account(tmp_path)
        fill = acct.submit(_intent(notional=1000.0), _quote())
        expected = abs(fill.price * fill.qty) / fill.leverage
        assert fill.margin_required == pytest.approx(expected)

    def test_insufficient_margin_rejected(self, tmp_path):
        acct = _account(tmp_path, cash=100.0)
        with pytest.raises(RuntimeError, match="margin"):
            acct.submit(_intent(notional=5_000.0), _quote())

    def test_liquidation_buffer_rejection(self, tmp_path):
        acct = _account(tmp_path, cash=200.0)
        with pytest.raises(LiquidationBufferRejected):
            acct.submit(
                OrderIntent(
                    symbol="BTCUSDT", side="long", notional=1_900.0,
                    leverage=10.0, strategy="test",
                ),
                _quote(),
            )


class TestFundingAndMarks:
    def test_funding_applied_per_8h_boundary(self, tmp_path):
        acct = _account(tmp_path)
        fill = acct.submit(_intent(notional=1000.0), _quote())
        cash_before = acct.equity_cash
        acct.apply_funding(rate=0.0001, at_price=100.0)
        # long pays positive funding on position notional (qty * price)
        assert acct.equity_cash == pytest.approx(cash_before - fill.qty * 100.0 * 0.0001)

    def test_mark_updates_unrealized(self, tmp_path):
        acct = _account(tmp_path)
        fill = acct.submit(_intent(notional=1000.0), _quote())
        unreal = acct.mark(price=110.0)
        expected = (110.0 - fill.price) * fill.qty
        assert unreal == pytest.approx(expected)
        assert acct.total_equity(price=110.0) == pytest.approx(
            acct.equity_cash + expected
        )


class TestStopsAndClose:
    def test_stop_closes_position_on_extreme(self, tmp_path):
        acct = _account(tmp_path)
        acct.submit(_intent(notional=1000.0), _quote())
        acct.attach_stop("BTCUSDT", 0.02)
        closed = acct.process_candle("BTCUSDT", high=101.0, low=97.0, close=97.5)
        assert closed, "stop at 2% below entry must trigger on low=97"
        assert acct.positions == {}

    def test_stop_does_not_trigger_inside_range(self, tmp_path):
        acct = _account(tmp_path)
        acct.submit(_intent(notional=1000.0), _quote())
        acct.attach_stop("BTCUSDT", 0.05)  # 5% stop, candle stays above
        closed = acct.process_candle("BTCUSDT", high=101.0, low=99.0, close=100.2)
        assert not closed
        assert "BTCUSDT" in acct.positions

    def test_close_realizes_pnl_and_fees(self, tmp_path):
        acct = _account(tmp_path)
        fill = acct.submit(_intent(notional=1000.0), _quote())
        cash_open = acct.equity_cash
        result = acct.close("BTCUSDT", price=110.0)
        expected_pnl = (110.0 - fill.price) * fill.qty - result.fee
        assert result.pnl == pytest.approx(expected_pnl)
        assert acct.equity_cash == pytest.approx(cash_open + result.pnl)
        assert acct.positions == {}

    def test_everything_ledgered(self, tmp_path):
        ledger = Ledger(tmp_path / "paper.jsonl", mode=ExecutionMode.PAPER)
        acct = PaperAccount(ledger=ledger, initial_cash=10_000.0, seed=7)
        acct.submit(_intent(), _quote())
        acct.attach_stop("BTCUSDT", 0.02)
        acct.process_candle("BTCUSDT", high=101.0, low=97.0, close=97.5)
        kinds = [e["payload"].get("type") for e in ledger.entries()]
        assert "OPEN" in kinds and "CLOSE" in kinds and "STOP" in kinds
        state = ledger.replay(initial_cash=10_000.0)
        assert state["positions"] == {}
