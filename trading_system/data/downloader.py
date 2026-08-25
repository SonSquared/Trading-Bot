"""
Binance historical data downloader using ccxt.

Downloads OHLCV klines and funding rates for perpetual futures.
Handles rate limiting, pagination, and incremental updates.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ccxt
import pandas as pd
import structlog

from trading_system.config import SystemConfig

logger = structlog.get_logger(__name__)

# Binance rate limits: 1200 requests/min for futures
RATE_LIMIT_DELAY = 0.1  # seconds between requests (conservative)

TIMEFRAME_MS = {
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
}


class BinanceDataDownloader:
    """Downloads historical data from Binance Futures."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.exchange = ccxt.binanceusdm({
            "apiKey": config.exchange.api_key or None,
            "secret": config.exchange.api_secret or None,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if config.exchange.sandbox:
            self.exchange.set_sandbox_mode(True)

    def download_klines(
        self,
        pair: str,
        timeframe: str,
        start_date: str,
        end_date: str = "",
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Download OHLCV kline data for a pair and timeframe.

        Args:
            pair: Trading pair (e.g., "BTC/USDT:USDT")
            timeframe: Candle timeframe (e.g., "1h")
            start_date: Start date string "YYYY-MM-DD"
            end_date: End date string, empty for latest
            save: Whether to save to disk

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info("downloading_klines", pair=pair, timeframe=timeframe, start=start_date)

        since = int(datetime.strptime(start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp() * 1000)

        if end_date:
            until = int(datetime.strptime(end_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ).timestamp() * 1000)
        else:
            until = int(time.time() * 1000)

        all_candles: list[list] = []
        limit = 1000  # Binance max per request
        current_since = since

        while current_since < until:
            try:
                candles = self.exchange.fetch_ohlcv(
                    pair, timeframe, since=current_since, limit=limit
                )
                if not candles:
                    break

                all_candles.extend(candles)

                # Move past the last candle
                last_ts = candles[-1][0]
                current_since = last_ts + TIMEFRAME_MS.get(timeframe, 3600000)

                if len(candles) < limit:
                    break  # No more data

                time.sleep(RATE_LIMIT_DELAY)

            except ccxt.RateLimitExceeded:
                logger.warning("rate_limit_exceeded", pair=pair, timeframe=timeframe)
                time.sleep(5)
            except ccxt.NetworkError as e:
                logger.error("network_error", error=str(e), pair=pair)
                time.sleep(2)
            except Exception as e:
                logger.error("download_error", error=str(e), pair=pair)
                break

        if not all_candles:
            logger.warning("no_data_downloaded", pair=pair, timeframe=timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Filter to end_date if specified
        if end_date:
            end_dt = pd.Timestamp(end_date, tz="UTC")
            df = df[df.index <= end_dt]

        logger.info("download_complete", pair=pair, timeframe=timeframe, candles=len(df))

        if save:
            self._save_klines(pair, timeframe, df)

        return df

    def download_funding_rates(
        self,
        pair: str,
        start_date: str,
        end_date: str = "",
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Download historical funding rates for a perpetual futures pair.

        Funding rates are settled every 8 hours on Binance.
        """
        logger.info("downloading_funding_rates", pair=pair, start=start_date)

        since = int(datetime.strptime(start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp() * 1000)

        if end_date:
            until = int(datetime.strptime(end_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ).timestamp() * 1000)
        else:
            until = int(time.time() * 1000)

        all_rates: list[dict] = []
        limit = 1000
        current_since = since

        while current_since < until:
            try:
                rates = self.exchange.fetch_funding_rate_history(
                    pair, since=current_since, limit=limit
                )
                if not rates:
                    break

                for rate in rates:
                    all_rates.append({
                        "timestamp": pd.Timestamp(rate["timestamp"], unit="ms", tz="UTC"),
                        "funding_rate": rate["fundingRate"],
                        "funding_timestamp": pd.Timestamp(
                            rate["datetime"], tz="UTC"
                        ) if rate.get("datetime") else None,
                    })

                last_ts = rates[-1]["timestamp"]
                current_since = last_ts + 1

                if len(rates) < limit:
                    break

                time.sleep(RATE_LIMIT_DELAY)

            except ccxt.RateLimitExceeded:
                time.sleep(5)
            except Exception as e:
                logger.error("funding_download_error", error=str(e))
                break

        if not all_rates:
            logger.warning("no_funding_data", pair=pair)
            return pd.DataFrame()

        df = pd.DataFrame(all_rates)
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        if end_date:
            end_dt = pd.Timestamp(end_date, tz="UTC")
            df = df[df.index <= end_dt]

        logger.info("funding_download_complete", pair=pair, records=len(df))

        if save:
            self._save_funding_rates(pair, df)

        return df

    def download_all(
        self,
        start_date: str = "",
        end_date: str = "",
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Download all configured pairs and timeframes."""
        start = start_date or self.config.data.start_date
        end = end_date or self.config.data.end_date

        results: dict[str, dict[str, pd.DataFrame]] = {}

        for pair in self.config.exchange.pairs:
            results[pair] = {}

            # Download funding rates once per pair (not per timeframe)
            logger.info("downloading_pair", pair=pair)
            self.download_funding_rates(pair, start, end)

            for timeframe in self.config.exchange.timeframes:
                df = self.download_klines(pair, timeframe, start, end)
                results[pair][timeframe] = df

        return results

    def _save_klines(self, pair: str, timeframe: str, df: pd.DataFrame) -> None:
        """Save kline data to Parquet."""
        pair_name = self.config.get_pair_safe_name(pair)
        path = Path(self.config.data.data_dir) / pair_name / f"{timeframe}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info("saved_klines", path=str(path), rows=len(df))

    def _save_funding_rates(self, pair: str, df: pd.DataFrame) -> None:
        """Save funding rates to Parquet."""
        pair_name = self.config.get_pair_safe_name(pair)
        path = Path(self.config.data.data_dir) / pair_name / "funding_rates.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info("saved_funding_rates", path=str(path), rows=len(df))
