"""Exchange data adapter contract (plan MD Task 3).

All exchange-specific code is confined to this module behind the
``ExchangeDataAdapter`` interface. Nothing else in ``crypto_system`` imports
an exchange client directly, so the venue can be swapped without touching
research, execution, or risk code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ExchangeDataAdapter(Protocol):
    """Contract: canonical candles + funding, validated before return."""

    exchange_id: str

    def candles(
        self,
        symbol: str,
        interval: str,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Return a validated OHLCV frame with UTC ``open_time`` column."""
        ...

    def funding(self, symbol: str, *, since: datetime | None = None) -> pd.DataFrame:
        """Return a frame of funding events (time, rate)."""
        ...


class CandleFrame:
    """Binance USDⓈ-M adapter (ccxt backend). The only exchange-aware class."""

    def __init__(self, exchange_id: str = "binanceusdm") -> None:
        self.exchange_id = exchange_id
        self._client = None  # lazily constructed; tests never touch the network

    @classmethod
    def binanceusdm(cls) -> "CandleFrame":
        return cls("binanceusdm")

    def _ccxt(self):
        if self._client is None:
            import ccxt  # confined import: exchange code stays in this module

            self._client = getattr(ccxt, self.exchange_id)({"enableRateLimit": True})
        return self._client

    def candles(
        self,
        symbol: str,
        interval: str,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        from crypto_system.data.quality import validate_partition

        client = self._ccxt()
        params: dict[str, object] = {"limit": limit}
        if since is not None:
            params["since"] = int(since.timestamp() * 1000)
        raw = client.fetch_ohlcv(symbol, timeframe=interval, params=params)
        df = pd.DataFrame(
            raw, columns=["open_time_ms", "open", "high", "low", "close", "volume"]
        )
        df["open_time"] = pd.to_datetime(df.pop("open_time_ms"), unit="ms", utc=True)
        report = validate_partition(df, interval=interval)
        if not report.accepted:
            from crypto_system.data.quality import DataQualityError

            raise DataQualityError(f"partition rejected: {report.issues}")
        return df

    def funding(self, symbol: str, *, since: datetime | None = None) -> pd.DataFrame:
        client = self._ccxt()
        params: dict[str, object] = {}
        if since is not None:
            params["since"] = int(since.timestamp() * 1000)
        raw = client.fetch_funding_rate_history(symbol, params=params)
        df = pd.DataFrame(raw)
        if "timestamp" in df.columns:
            df["open_time"] = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
        return df
