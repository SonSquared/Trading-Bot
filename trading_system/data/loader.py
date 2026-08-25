"""
Unified data loading interface.

Provides a single entry point for loading, cleaning, and splitting data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import structlog

from trading_system.config import SystemConfig
from trading_system.data.cleaner import DataCleaner
from trading_system.data.storage import DataStorage
from trading_system.data.validator import DataValidator

logger = structlog.get_logger(__name__)


class DataLoader:
    """Unified interface for loading and preparing data."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.storage = DataStorage(config.data.data_dir, config.data.results_dir)
        self.cleaner = DataCleaner()

    def load(
        self,
        pair: str,
        timeframe: str,
        validate: bool = True,
        clean: bool = True,
    ) -> pd.DataFrame | None:
        """Load data for a pair and timeframe, with optional validation and cleaning."""
        df = self.storage.load_data(pair, timeframe, "klines")

        if df is None:
            logger.warning("data_not_found", pair=pair, timeframe=timeframe)
            return None

        if df.empty:
            return df

        if validate:
            validator = DataValidator(timeframe=timeframe)
            is_valid, report = validator.validate_and_report(df, pair, timeframe)
            if not is_valid:
                logger.warning(
                    "data_validation_failed",
                    pair=pair,
                    timeframe=timeframe,
                    errors=report.error_count,
                )

        if clean:
            df = self.cleaner.clean(df, timeframe)

        return df

    def load_funding_rates(self, pair: str) -> pd.DataFrame | None:
        """Load funding rates for a pair."""
        df = self.storage.load_data(pair, "1h", "funding_rates")
        if df is not None and not df.empty:
            df = self.cleaner.clean_funding_rates(df)
        return df

    def split_data(
        self,
        df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Split data into in-sample, validation, out-of-sample, and holdout."""
        split = self.config.data_split

        in_sample_end = pd.Timestamp(split.in_sample_end, tz="UTC")
        validation_end = pd.Timestamp(split.validation_end, tz="UTC")
        oos_end = pd.Timestamp(split.out_of_sample_end, tz="UTC")

        result = {
            "in_sample": df[df.index <= in_sample_end],
            "validation": df[(df.index > in_sample_end) & (df.index <= validation_end)],
            "out_of_sample": df[(df.index > validation_end) & (df.index <= oos_end)],
            "holdout": df[df.index > oos_end],
        }

        for name, part in result.items():
            logger.info(
                "data_split",
                split=name,
                candles=len(part),
                start=str(part.index[0]) if len(part) > 0 else "N/A",
                end=str(part.index[-1]) if len(part) > 0 else "N/A",
            )

        return result

    def load_all_pairs(
        self,
        validate: bool = True,
        clean: bool = True,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Load data for all configured pairs and timeframes."""
        result = {}
        for pair in self.config.exchange.pairs:
            result[pair] = {}
            for timeframe in self.config.exchange.timeframes:
                df = self.load(pair, timeframe, validate=validate, clean=clean)
                if df is not None:
                    result[pair][timeframe] = df
        return result

    def get_data_summary(self) -> dict[str, dict[str, dict]]:
        """Get a summary of all available data."""
        summary = {}
        data_dir = Path(self.config.data.data_dir)

        for pair_dir in data_dir.iterdir():
            if pair_dir.is_dir():
                pair = pair_dir.name
                summary[pair] = {}
                for f in pair_dir.glob("*.parquet"):
                    df = pd.read_parquet(f)
                    tf = f.stem.replace("klines_", "").replace("funding_rates", "funding_rates")
                    summary[pair][tf] = {
                        "rows": len(df),
                        "start": str(df.index[0]) if len(df) > 0 else None,
                        "end": str(df.index[-1]) if len(df) > 0 else None,
                        "file_size_mb": f.stat().st_size / (1024 * 1024),
                    }

        return summary
