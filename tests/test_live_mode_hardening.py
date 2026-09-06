"""
Live-mode hardening tests for the portfolio bot.

The live path is where desync bugs become money: a ledger entry the
exchange doesn't agree with makes every future cycle retry a close that
can never succeed (an infinite loop against reality), and a close that
doesn't carry the real position size silently no-ops. These tests pin:

- Unfilled live opens must NOT register a ledger entry (no phantom SL).
- Partial fills must register exactly the filled amount.
- A close order that fails while the exchange no longer holds the
  position must clear the stale ledger entry (loop prevention).
- A close order that fails while the exchange STILL holds the position
  must keep the ledger entry (retry is correct there).
- Stop-loss closes in live mode must send the real size, reduce-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_system.bot.portfolio_bot import PortfolioTradingBot
from trading_system.bot.sltp_manager import SLTPManager
from trading_system.config import SystemConfig

PAIR = "ETH/USDT:USDT"
TICKER = {"bid": 99.5, "ask": 100.5, "last": 100.0}


def make_ohlcv(n: int = 120, start_price: float = 100.0, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV ending on the current 4h boundary."""
    rng = np.random.default_rng(seed)
    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(hours=4 * (n - 1))).floor("4h")
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    close = start_price + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)
    high = close * 1.01
    low = close * 0.99
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )


class FillControlledExchange:
    """Fake exchange whose fills and position book the tests control."""

    def __init__(self, filled_fraction: float = 1.0, position_book: list | None = None):
        self.orders = []
        self.filled_fraction = filled_fraction
        self.position_book = position_book if position_book is not None else []

    def get_positions(self, pair=""):
        return [p for p in self.position_book if p["pair"] == pair]

    def get_ticker(self, pair):
        return {"bid": 99.5, "ask": 100.5, "last": 100.0, "volume": 1e6}

    def get_balance(self):
        return {"total": 10000.0, "free": 9000.0, "used": 1000.0}

    def get_ohlcv(self, pair, timeframe, limit=250):
        return make_ohlcv()

    def place_market_order(self, pair, side, amount, reduce_only=False):
        self.orders.append({
            "pair": pair, "side": side, "amount": amount,
            "reduce_only": reduce_only,
        })
        filled = amount * self.filled_fraction
        return {
            "order_id": f"ord-{len(self.orders)}",
            "status": "closed" if filled > 0 else "rejected",
            "filled": filled,
            "average_price": 100.4 if filled > 0 else 0.0,
            "fee": {"cost": 0.01},
        }

    def set_leverage(self, pair, leverage):
        return True

    def connect(self):
        return True


class FakeState:
    def __init__(self):
        self._state = {}
        self.trades = []

    def get(self, key, default=None):
        return self._state.get(key, default)

    def set(self, key, value):
        self._state[key] = value

    def record_trade(self, trade):
        self.trades.append(trade)

    def record_order(self, order_id):
        pass

    def save(self):
        pass


class FakeNotifications:
    def send(self, msg):
        pass

    def notify_trade(self, t):
        pass


class FakeTelegram:
    def notify_trade_open(self, **kw):
        pass

    def notify_trade_close(self, **kw):
        pass

    def notify_bot_start(self, *a, **kw):
        pass

    def notify_bot_stop(self, *a, **kw):
        pass

    def notify_error(self, *a, **kw):
        pass


@pytest.fixture
def live_bot(tmp_path):
    cfg = SystemConfig.default()
    cfg.bot.state_file = str(tmp_path / "state.json")
    cfg.bot.mode = "live"
    cfg.exchange.pairs = ["ETH/USDT:USDT"]
    b = PortfolioTradingBot(cfg)
    b.exchange = FillControlledExchange()
    b.state = FakeState()
    b.notifications = FakeNotifications()
    b.telegram = FakeTelegram()
    b.sltp = SLTPManager(sl_atr_mult=3.0)
    b.signal_threshold = 0.3
    b.strategy_instances = []
    return b


# ---------------------------------------------------------------------------
# Open-path desync prevention
# ---------------------------------------------------------------------------

class TestLiveOpenHardening:
    def test_unfilled_open_registers_no_ledger_entry(self, live_bot):
        """A rejected/timeout order with zero fill must leave NO ledger
        entry — otherwise the close path fights a position that doesn't
        exist (the desync that made closes retry forever)."""
        live_bot.exchange.filled_fraction = 0.0
        live_bot._open_position(PAIR, True, 10000.0, TICKER, 0.8,
                                data={"4h": make_ohlcv()})
        assert live_bot.exchange.orders, "the order attempt must have been sent"
        assert live_bot.sltp.find_position(PAIR) is None, \
            "unfilled order must not register a stop/ledger entry"

    def test_partial_fill_registers_only_filled_amount(self, live_bot):
        """Only what actually filled may be tracked, or the close path
        tries to reduce more than the exchange holds."""
        live_bot.exchange.filled_fraction = 0.4
        live_bot._open_position(PAIR, True, 10000.0, TICKER, 0.8,
                                data={"4h": make_ohlcv()})
        pos = live_bot.sltp.find_position(PAIR)
        assert pos is not None
        requested = live_bot.exchange.orders[0]["amount"]
        assert pos.size == pytest.approx(requested * 0.4), \
            "ledger must track the FILLED amount, not the requested one"

    def test_full_fill_registers_full_amount(self, live_bot):
        live_bot._open_position(PAIR, True, 10000.0, TICKER, 0.8,
                                data={"4h": make_ohlcv()})
        pos = live_bot.sltp.find_position(PAIR)
        assert pos is not None
        assert pos.size == pytest.approx(live_bot.exchange.orders[0]["amount"])
        assert pos.stop_loss < pos.entry_price  # long SL below entry


# ---------------------------------------------------------------------------
# Close-path loop prevention
# ---------------------------------------------------------------------------

class TestLiveCloseHardening:
    def _open_tracked_position(self, live_bot):
        live_bot._open_position(PAIR, True, 10000.0, TICKER, 0.8,
                                data={"4h": make_ohlcv()})
        pos = live_bot.sltp.find_position(PAIR)
        assert pos is not None
        return pos

    def test_failed_close_with_gone_position_clears_ledger(self, live_bot):
        """Exchange says the position no longer exists (closed externally /
        liquidated) and the reduce-only order fails. The stale ledger entry
        must be cleared or every future cycle retries the close forever."""
        self._open_tracked_position(live_bot)
        # Order fails AND the position book is empty: position is gone.
        live_bot.exchange.filled_fraction = 0.0
        live_bot.exchange.position_book = []

        pos_dict = {"pair": PAIR, "side": "long", "size": 80.0,
                    "entry_price": 100.4}
        rec = live_bot._close_position(PAIR, pos_dict, "portfolio_exit")
        assert rec is None, "failed close must not fabricate a fill record"
        assert live_bot.sltp.find_position(PAIR) is None, \
            "stale ledger entry must be cleared when the exchange has no position"

        # Loop prevention: the NEXT cycle has nothing left to retry.
        assert live_bot.sltp.find_position(PAIR) is None
        # And repeating the close stays a safe no-op, not an error loop.
        rec2 = live_bot._close_position(PAIR, pos_dict, "portfolio_exit")
        assert rec2 is None

    def test_failed_close_with_live_position_keeps_ledger(self, live_bot):
        """The order failed but the exchange STILL holds the position:
        the ledger entry must be kept so the next cycle retries the
        close — clearing it here would orphan a real position."""
        self._open_tracked_position(live_bot)
        live_bot.exchange.filled_fraction = 0.0
        live_bot.exchange.position_book = [{"pair": PAIR, "size": 80.0}]

        pos_dict = {"pair": PAIR, "side": "long", "size": 80.0,
                    "entry_price": 100.4}
        rec = live_bot._close_position(PAIR, pos_dict, "portfolio_exit")
        assert rec is None
        assert live_bot.sltp.find_position(PAIR) is not None, \
            "retryable failure: ledger must stay in sync with the real position"

    def test_unfilled_close_order_keeps_ledger_for_retry(self, live_bot):
        """An order object that reports zero fill is a retryable failure,
        not proof the position is gone."""
        self._open_tracked_position(live_bot)
        live_bot.exchange.filled_fraction = 0.0
        live_bot.exchange.position_book = [{"pair": PAIR, "size": 80.0}]

        pos_dict = {"pair": PAIR, "side": "long", "size": 80.0,
                    "entry_price": 100.4}
        rec = live_bot._close_position(PAIR, pos_dict, "portfolio_exit")
        assert rec is None
        assert live_bot.sltp.find_position(PAIR) is not None

    def test_successful_live_close_clears_ledger_and_fills_reduce_only(self, live_bot):
        from trading_system.bot.accounting import FEE_RATE

        self._open_tracked_position(live_bot)
        size = live_bot.sltp.find_position(PAIR).size

        pos_dict = {"pair": PAIR, "side": "long", "size": size,
                    "entry_price": 100.4}
        # Reference exit 105 — but the REAL fill (fake exchange: 100.4) must
        # be what the record uses, never the reference.
        rec = live_bot._close_position(PAIR, pos_dict, "portfolio_exit",
                                       exit_price=105.0)
        assert rec is not None
        assert rec["amount"] == pytest.approx(size)
        assert rec["exit_price"] == pytest.approx(100.4), \
            "live P&L must use the real average_price fill"
        # Real fill == entry fill -> flat gross; P&L must be exactly the
        # shared-model round-trip fees (holding time ~0 -> no funding).
        notional = size * 100.4
        assert rec["pnl_usd"] == pytest.approx(-notional * FEE_RATE * 2, abs=1e-6)
        last = live_bot.exchange.orders[-1]
        assert last["side"] == "sell"
        assert last["reduce_only"] is True
        assert last["amount"] == pytest.approx(size)
        assert live_bot.sltp.find_position(PAIR) is None

    def test_disaster_stop_close_uses_real_size_in_live_mode(self, live_bot):
        """End-to-end disaster path: open live -> price craters through the
        SL -> SLTP action carries the REAL size -> live close sends exactly
        that, reduce-only. Guards the original size-0 no-op bug in the
        live path specifically."""
        self._open_tracked_position(live_bot)
        size = live_bot.sltp.find_position(PAIR).size

        actions = live_bot.sltp.check_price(PAIR, 1.0)  # crash price
        assert len(actions) == 1
        assert actions[0]["action"] == "close"
        assert actions[0]["size"] == pytest.approx(size), \
            "SLTP action must carry the real position size (not 0)"

        action = actions[0]
        fake_pos = {
            "pair": PAIR, "side": "long", "size": action["size"],
            "entry_price": action["entry"], "pnl_pct": action["pnl_pct"],
        }
        rec = live_bot._close_position(PAIR, fake_pos, "stop_loss",
                                       exit_price=action["exit"])
        assert rec is not None, "disaster stop must not silently no-op"
        assert rec["amount"] == pytest.approx(size)
        assert rec["pnl_usd"] < 0
        last = live_bot.exchange.orders[-1]
        assert last["reduce_only"] is True
        assert last["amount"] == pytest.approx(size)
        assert live_bot.sltp.find_position(PAIR) is None


# ---------------------------------------------------------------------------
# Close accounting matches the shared model
# ---------------------------------------------------------------------------

class TestLiveCloseAccounting:
    def test_live_close_pnl_uses_real_fill_and_estimated_costs(self, live_bot):
        """Live P&L: real average_price for the exit fill, fees + funding
        estimated at the shared-model rates so live reports are comparable
        to paper/backtest accounting."""
        from datetime import datetime, timedelta, timezone

        from trading_system.bot.accounting import FEE_RATE, funding_cost

        live_bot._open_position(PAIR, True, 10000.0, TICKER, 0.8,
                                data={"4h": make_ohlcv()})
        pos = live_bot.sltp.find_position(PAIR)
        amount, entry_price = pos.size, pos.entry_price
        # Held ~9h -> at least one 8h funding boundary.
        pos.entry_time = (datetime.now(timezone.utc)
                          - timedelta(hours=9)).isoformat()

        pos_dict = {"pair": PAIR, "side": "long", "size": amount,
                    "entry_price": entry_price, "entry_time": pos.entry_time}
        rec = live_bot._close_position(PAIR, pos_dict, "portfolio_exit",
                                       exit_price=105.0)
        assert rec is not None

        exit_fill = live_bot.exchange.orders[-1]
        # FakeExchange fills at 100.4 regardless of the reference price;
        # the record must use the REAL fill, not the reference.
        assert rec["exit_price"] == pytest.approx(100.4)
        notional = amount * entry_price
        gross = amount * (100.4 - entry_price)
        funding = funding_cost(notional, pos.entry_time, datetime.now(timezone.utc))
        assert rec["pnl_usd"] == pytest.approx(
            gross - notional * FEE_RATE * 2 - funding, abs=1e-6)
