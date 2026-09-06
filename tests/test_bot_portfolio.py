"""
Smoke tests for the portfolio bot's open/close execution paths.

These guard the two critical live-trading bugs found in audit:
1. NameError in ``_open_position`` (``strategies``/``data`` out of scope) —
   candle data is now threaded through explicitly.
2. Stop-loss closes silently no-oping because they passed ``size=0`` —
   SLTP actions now carry the real position size.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from trading_system.bot.candles import closed_candles
from trading_system.bot.portfolio_bot import PortfolioTradingBot
from trading_system.bot.sltp_manager import SLTPManager
from trading_system.config import SystemConfig


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


class FakeExchange:
    def __init__(self):
        self.orders = []

    def get_positions(self, pair=""):
        return []

    def get_ticker(self, pair):
        return {"bid": 99.5, "ask": 100.5, "last": 100.0, "volume": 1e6}

    def get_balance(self):
        return {"total": 10000.0, "free": 9000.0, "used": 1000.0}

    def get_ohlcv(self, pair, timeframe, limit=250):
        return make_ohlcv()

    def place_market_order(self, pair, side, amount, reduce_only=False):
        self.orders.append({
            "pair": pair, "side": side, "amount": amount, "reduce_only": reduce_only,
        })
        return {
            "order_id": f"ord-{len(self.orders)}",
            "status": "closed",
            "filled": amount,
            "average_price": 100.4,
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


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send(self, msg):
        self.messages.append(msg)

    def notify_trade(self, t):
        self.messages.append(t)

    def notify_error(self, e, context=""):
        self.messages.append(e)


class FakeTelegram:
    def __init__(self):
        self.calls = []

    def notify_trade_open(self, **kw):
        self.calls.append(("open", kw))

    def notify_trade_close(self, **kw):
        self.calls.append(("close", kw))

    def notify_bot_start(self, *a, **kw):
        pass

    def notify_bot_stop(self, *a, **kw):
        pass

    def notify_error(self, *a, **kw):
        pass


@pytest.fixture
def bot(tmp_path):
    cfg = SystemConfig.default()
    cfg.bot.state_file = str(tmp_path / "state.json")
    cfg.bot.mode = "paper"
    cfg.exchange.pairs = ["ETH/USDT:USDT"]
    b = PortfolioTradingBot(cfg)
    b.exchange = FakeExchange()
    b.state = FakeState()
    b.notifications = FakeNotifier()
    b.telegram = FakeTelegram()
    b.sltp = SLTPManager(sl_atr_mult=3.0)
    b.signal_threshold = 0.3
    # Direct-method tests don't need strategy instances.
    b.strategy_instances = []
    return b


TICKER = {"bid": 99.5, "ask": 100.5, "last": 100.0}
PAIR = "ETH/USDT:USDT"


def test_open_position_paper_registers_sl_no_nameerror(bot):
    """Regression: _open_position must not raise NameError, and must register the SL."""
    bot._open_position(PAIR, True, 10000.0, TICKER, 0.8, data={"4h": make_ohlcv()})
    pos = bot.sltp.find_position(PAIR)
    assert pos is not None
    assert pos.side == "long"
    assert pos.size > 0
    assert pos.stop_loss < pos.entry_price  # long SL is below entry
    assert bot.state.trades, "paper open must be recorded"


def test_open_position_live_places_order_and_registers_sl(bot):
    bot.bot_config.mode = "live"
    bot._open_position(PAIR, False, 10000.0, TICKER, 0.9, data={"4h": make_ohlcv()})
    assert len(bot.exchange.orders) == 1
    assert bot.exchange.orders[0]["side"] == "sell"
    pos = bot.sltp.find_position(PAIR)
    assert pos is not None and pos.side == "short"
    assert pos.stop_loss > pos.entry_price  # short SL is above entry


def test_open_position_dry_run_no_orders(bot):
    bot.bot_config.mode = "dry_run"
    bot._open_position(PAIR, True, 10000.0, TICKER, 0.8, data={"4h": make_ohlcv()})
    assert not bot.exchange.orders
    assert not bot.state.trades


def test_execute_signal_cycle_paper(bot):
    """Open on signal, no duplicate on same signal, close on flat."""
    agg_long = {"direction": 1, "weighted_score": 0.8, "signals": {}, "confidence": 0.8}
    data = {"4h": make_ohlcv()}

    bot._execute_portfolio_signal(PAIR, agg_long, TICKER, 10000.0, [], data=data)
    assert len(bot.sltp.positions) == 1

    # Same direction again -> no duplicate open
    bot._execute_portfolio_signal(PAIR, agg_long, TICKER, 10000.0, [], data=data)
    assert len(bot.sltp.positions) == 1

    # Flat signal -> close
    agg_flat = {"direction": 0, "weighted_score": 0.0, "signals": {}, "confidence": 0.0}
    bot._execute_portfolio_signal(PAIR, agg_flat, TICKER, 10000.0, [], data=data)
    assert len(bot.sltp.positions) == 0
    assert bot.state.trades, "opens + close must be recorded"


def test_sltp_close_executes_with_real_size(bot):
    """The old bug: SLTP closes passed size=0 and silently did nothing."""
    bot._open_position(PAIR, True, 10000.0, TICKER, 0.8, data={"4h": make_ohlcv()})
    size_before = bot.sltp.find_position(PAIR).size

    action = {
        "action": "close", "pair": PAIR, "side": "sell", "reason": "stop_loss",
        "pnl_pct": -0.04, "entry": 100.0, "exit": 96.0,
        "strategy": "portfolio", "size": size_before,
    }
    fake_pos = {
        "pair": PAIR, "side": "long", "size": action["size"],
        "entry_price": action["entry"], "pnl_pct": action["pnl_pct"],
    }
    rec = bot._close_position(PAIR, fake_pos, "stop_loss", exit_price=96.0)
    assert rec is not None, "SL close must not silently no-op"
    assert rec["amount"] == pytest.approx(size_before)
    assert rec["pnl_pct"] < 0
    assert len(bot.sltp.positions) == 0


def test_close_position_live_reduce_only(bot):
    bot.bot_config.mode = "live"
    bot._open_position(PAIR, True, 10000.0, TICKER, 0.8, data={"4h": make_ohlcv()})

    # Close via an exchange-known position (what _process_pair sees on a signal exit)
    pos = {"pair": PAIR, "side": "long", "size": 80.0, "entry_price": 100.0}
    rec = bot._close_position(PAIR, pos, "portfolio_exit")
    assert rec is not None
    last = bot.exchange.orders[-1]
    assert last["side"] == "sell"
    assert last["reduce_only"] is True
    assert last["amount"] == pytest.approx(80.0)


def test_sltp_check_price_actions_include_size():
    mgr = SLTPManager(sl_atr_mult=3.0)
    mgr.open_position(PAIR, "long", 100.0, 50.0, "portfolio", make_ohlcv())
    actions = mgr.check_price(PAIR, 1.0)  # crash price -> SL triggered
    assert len(actions) == 1
    assert actions[0]["size"] == pytest.approx(50.0)


def test_paper_close_uses_shared_accounting_model(bot):
    """Paper closes must net the shared model (fees + slippage + 8h funding)
    and report pnl_usd in dollars — not the old flat 0.1% / coin-units math."""
    from datetime import datetime, timedelta, timezone
    from trading_system.bot.accounting import FEE_RATE, SLIPPAGE_RATE, funding_cost

    bot._open_position(PAIR, True, 10000.0, TICKER, 0.8, data={"4h": make_ohlcv()})
    pos = bot.sltp.find_position(PAIR)
    assert pos is not None

    # Pin the entry ~9h ago so at least one 8h funding boundary accrues
    entry_dt = datetime.now(timezone.utc) - timedelta(hours=9)
    pos.entry_time = entry_dt.isoformat()

    entry_fill = pos.entry_price                       # ask + slippage at open
    amount = pos.size
    exit_ref = 105.0
    rec = bot._close_position(PAIR, {"pair": PAIR, "side": "long", "size": amount,
                                     "entry_price": entry_fill}, "portfolio_exit",
                                exit_price=exit_ref)
    assert rec is not None

    exit_fill = exit_ref * (1.0 - SLIPPAGE_RATE)       # selling a long
    notional = amount * entry_fill
    gross = amount * (exit_fill - entry_fill)
    funding = funding_cost(notional, pos.entry_time, datetime.now(timezone.utc))
    expected = gross - notional * FEE_RATE * 2 - funding

    assert rec["pnl_usd"] == pytest.approx(expected, abs=1e-6)
    assert rec["pnl_pct"] == pytest.approx((exit_fill - entry_fill) / entry_fill * 100.0, abs=1e-6)
    # Unit regression: a ~4-5% winner on a meaningful notional must be dollars,
    # not the old amount*pnl_pct coin-units result (~0.0002).
    assert rec["pnl_usd"] > 1.0
    assert len(bot.sltp.positions) == 0  # ledger entry removed


def test_closed_candles_drops_forming_candle():
    df = make_ohlcv(120)
    closed = closed_candles(df, "4h")
    assert len(closed) == len(df) - 1  # last candle is still forming
    assert len(closed) >= 100


def test_closed_candles_handles_timestamp_column():
    df = make_ohlcv(60).reset_index().rename(columns={"index": "timestamp"})
    closed = closed_candles(df, "4h")
    assert len(closed) == len(df) - 1