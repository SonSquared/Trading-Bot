"""The exchange surface ``AIAgent`` actually uses, served from stored OHLCV.

The point is fidelity, not a second simulator. This class implements exactly
the methods the production agent and ``market_data`` call — OHLCV, ticker,
funding, market limits — and nothing else. Candle windows end at the cursor and
are strictly closed, so a replayed wakeup can only see what a real wakeup at
that instant could have seen.

**No look-ahead, mechanically.** At a 1h timeframe and an on-the-hour slot, the
newest candle whose interval has fully elapsed at the cursor is the candle that
*closed* at the cursor, i.e. the price at the slot. Nothing after the cursor is
ever returned, and the number of times that rule could have been broken is
counted (``look_ahead_violations``) so a report can state zero rather than
assume it.

Order placement raises. The AI bot trades exclusively through its paper ledger,
so a backtest has no order path at all; making the order methods raise turns
"no orders were placed" from a claim into something a test can check.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Timeframes the config can name. Keys match Binance's timeframe strings, which
# is also what the downloader writes into the parquet filename.
TIMEFRAME_INTERVALS: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "6h": pd.Timedelta(hours=6),
    "12h": pd.Timedelta(hours=12),
    "1d": pd.Timedelta(days=1),
}

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class OrderPathTouched(RuntimeError):
    """Raised when replayed code tries to place or cancel a real order.

    Paper mode never does, so a replay that hits this has escaped the ledger —
    which is exactly the failure this guard exists to make loud.
    """


def timeframe_interval(timeframe: str) -> pd.Timedelta:
    try:
        return TIMEFRAME_INTERVALS[timeframe]
    except KeyError:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}. Known: "
            f"{', '.join(sorted(TIMEFRAME_INTERVALS))}"
        ) from None


def symbol_for(pair: str) -> str:
    """``BTC/USDT:USDT`` -> ``BTC_USDT_USDT`` (the downloader's directory name)."""
    return pair.replace("/", "_").replace(":", "_")


def candidate_paths(data_dir: Path, pair: str, timeframe: str) -> list[Path]:
    """Where the klines for ``pair`` might live, most likely first."""
    sym = symbol_for(pair)
    return [
        data_dir / sym / f"klines_{timeframe}.parquet",
        data_dir / pair / f"klines_{timeframe}.parquet",
        data_dir / f"klines_{sym}_{timeframe}.parquet",
    ]


def _normalize(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """UTC index, float OHLCV, sorted, de-duplicated, no unusable rows."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{pair}: OHLCV frame is missing columns {missing}")

    out = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    idx = pd.DatetimeIndex(out.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in REQUIRED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        raise ValueError(f"{pair}: OHLCV frame has no usable rows")
    return out


def load_ohlcv(data_dir: str | Path, pair: str, timeframe: str) -> pd.DataFrame:
    """Load stored klines for ``pair``, or explain how to get them."""
    data_dir = Path(data_dir)
    tried = candidate_paths(data_dir, pair, timeframe)
    for path in tried:
        if path.exists():
            df = _normalize(pd.read_parquet(path), pair)
            logger.info(
                "backtest_data_loaded", pair=pair, timeframe=timeframe,
                path=str(path), rows=len(df),
            )
            return df
    looked = "\n".join(f"    {p}" for p in tried)
    raise FileNotFoundError(
        f"No {timeframe} klines for {pair} under {data_dir}. Looked for:\n{looked}\n"
        "Fetch them with:\n"
        "    python scripts/download_data.py --timeframes "
        f"{timeframe} --pairs {pair}\n"
        "(or point --data-dir at a directory that already has them)"
    )


def load_funding(data_dir: str | Path, pair: str) -> pd.Series | None:
    """Funding-rate history as a float Series, or None when unavailable.

    Optional: the stored funding file only covers a subset of the OHLCV range
    here, and the paper ledger does not charge funding anyway. It is loaded so
    the report can quantify the omission instead of hand-waving it.
    """
    for name in ("funding_rates.parquet", f"funding_{symbol_for(pair)}.parquet"):
        path = Path(data_dir) / symbol_for(pair) / name
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except (OSError, ValueError) as e:  # pragma: no cover - corrupt file
            logger.warning("funding_load_failed", pair=pair, error=str(e))
            continue
        col = "funding_rate" if "funding_rate" in df.columns else df.columns[0]
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        series = pd.Series(pd.to_numeric(df[col], errors="coerce").values, index=idx)
        return series.dropna().sort_index()
    return None


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(REQUIRED_COLUMNS))


class HistoricalExchange:
    """Serves stored candles at a moving cursor; never places an order."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        timeframe: str = "1h",
        funding: dict[str, pd.Series] | None = None,
        forming_candle: bool = False,
    ):
        if not frames:
            raise ValueError("HistoricalExchange needs at least one pair's candles")
        self.frames = frames
        self.timeframe = timeframe
        self.interval = timeframe_interval(timeframe)
        self.funding = funding or {}
        self.forming_candle = forming_candle
        self._cursor: pd.Timestamp | None = None
        self._connected = False

        # Evidence counters, reported by the harness.
        self.ohlcv_calls = 0
        self.candles_served = 0
        self.look_ahead_violations = 0
        self.order_path_attempts = 0
        # Distinct (pair, newest closed candle) windows actually served. Both
        # the agent's fetch and the engine's indicator fetch go through
        # _closed(), so a call count would double-count the same window; this
        # is the number the no-look-ahead claim is about.
        self._windows: set[tuple[str, str]] = set()

    # -- the cursor ---------------------------------------------------------

    def set_cursor(self, when: datetime | pd.Timestamp) -> None:
        ts = pd.Timestamp(when)
        self._cursor = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

    @property
    def cursor(self) -> pd.Timestamp:
        if self._cursor is None:
            raise RuntimeError("cursor not set — call set_cursor() first")
        return self._cursor

    def data_range(self, pair: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        df = self.frames[pair]
        return df.index[0], df.index[-1] + self.interval

    # -- the surface AIAgent + market_data use ------------------------------

    def connect(self) -> bool:
        self._connected = True
        return True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _closed(self, pair: str) -> pd.DataFrame:
        """Candles fully closed at the cursor (never anything later)."""
        df = self.frames.get(pair)
        if df is None or df.empty:
            return empty_frame()
        cursor = self.cursor
        mask = df.index + self.interval <= cursor
        window = df.loc[mask]
        # The look-ahead check is about REAL candles, so it runs before any
        # synthetic current bar is appended below.
        if not window.empty:
            self._windows.add((pair, str(window.index[-1])))
            if window.index[-1] + self.interval > cursor:
                # Unreachable by construction; counted so a report can say so.
                self.look_ahead_violations += 1
        if self.forming_candle and not window.empty:
            # Production's fetch includes the candle still forming. Its open is
            # the price at the boundary the cursor sits on, so a synthetic bar
            # reproduces that row without reaching past the cursor.
            nxt = df.loc[df.index > window.index[-1]]
            price = float(nxt["open"].iloc[0]) if not nxt.empty else float(
                window["close"].iloc[-1]
            )
            stamp = window.index[-1] + self.interval
            window = pd.concat([
                window,
                pd.DataFrame(
                    {"open": [price], "high": [price], "low": [price],
                     "close": [price], "volume": [0.0]},
                    index=pd.DatetimeIndex([stamp], tz="UTC"),
                ),
            ])
        return window

    @property
    def distinct_windows(self) -> int:
        return len(self._windows)

    def get_ohlcv(self, pair: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        if timeframe != self.timeframe:
            raise ValueError(
                f"this replay only has {self.timeframe} candles, but {timeframe} "
                f"was requested for {pair} — a mismatch would silently change "
                "the strategy's inputs"
            )
        window = self._closed(pair)
        self.ohlcv_calls += 1
        tail = window.tail(limit)
        self.candles_served += len(tail)
        return tail.copy()

    def last_price(self, pair: str) -> float:
        window = self._closed(pair)
        return float(window["close"].iloc[-1]) if not window.empty else 0.0

    def get_ticker(self, pair: str) -> dict[str, float]:
        """Slot price: the close of the newest candle closed at the cursor."""
        window = self._closed(pair)
        if window.empty:
            return {"bid": 0.0, "ask": 0.0, "last": 0.0, "volume": 0.0}
        price = float(window["close"].iloc[-1])
        return {
            "bid": price, "ask": price, "last": price,
            "volume": float(window["volume"].iloc[-1]) * price,
        }

    def get_funding_rate(self, pair: str) -> float:
        series = self.funding.get(pair)
        if series is None or series.empty:
            return 0.0
        value = series.asof(self.cursor)
        return float(value) if pd.notna(value) else 0.0

    def get_market_limits(self, pair: str):
        """None: the agent then uses the dated builtin table.

        That is what the cloud runner does (Binance geo-blocks it), so the
        replay enforces the same MIN_NOTIONAL / LOT_SIZE / taker fee as
        production rather than an idealised version of them.
        """
        return None

    def get_balance(self) -> dict[str, float]:
        """Unused in paper mode; the ledger is the account."""
        return {"total": 0.0, "free": 0.0, "used": 0.0}

    def get_positions(self, pair: str = "") -> list[dict]:
        """Unused in paper mode; the ledger holds the positions."""
        return []

    def raw_slice(self, pair: str, start, end) -> pd.DataFrame:
        """Candles whose interval begins in ``[start, end)``.

        Deliberately NOT ``get_ohlcv``: this is what the intrabar trigger model
        walks, and looking forward between two wakeups is exactly what a resting
        exchange stop does — it fills whether or not the bot is awake. It is
        never used to build a decision's inputs.
        """
        df = self.frames.get(pair)
        if df is None or df.empty:
            return empty_frame()
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        return df.loc[(df.index >= lo) & (df.index < hi)]

    # -- the order path must not exist -------------------------------------

    def _no_orders(self, name: str):
        self.order_path_attempts += 1
        raise OrderPathTouched(
            f"{name}() was called during a replay. The AI bot trades only "
            "through its paper ledger; reaching an order method means the "
            "replay escaped the ledger and its numbers are not trustworthy."
        )

    def place_market_order(self, pair, side, amount, reduce_only=False):
        self._no_orders("place_market_order")

    def place_limit_order(self, pair, side, amount, price, reduce_only=False):
        self._no_orders("place_limit_order")

    def place_stop_market_order(self, pair, entry_side, amount, stop_price):
        self._no_orders("place_stop_market_order")

    def place_take_profit_market_order(self, pair, entry_side, amount, stop_price):
        self._no_orders("place_take_profit_market_order")

    def cancel_order(self, order_id, pair):
        self._no_orders("cancel_order")

    def cancel_all_orders(self, pair):
        self._no_orders("cancel_all_orders")

    def set_leverage(self, pair, leverage):
        self._no_orders("set_leverage")
