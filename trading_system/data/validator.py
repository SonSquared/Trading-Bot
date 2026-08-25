"""
Data quality validator for OHLCV candle data.

Detects: missing candles, duplicate timestamps, impossible OHLC relationships,
zero/negative prices, extreme anomalies, chronological ordering issues, gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from typing import Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

TIMEFRAME_MINUTES = {
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
}


@dataclass
class ValidationIssue:
    """A single data quality issue."""

    category: str  # "missing", "duplicate", "impossible_ohlc", "zero_price", "anomaly", "gap"
    severity: str  # "warning", "error", "critical"
    description: str
    affected_rows: int = 0
    affected_range: str = ""


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""

    pair: str
    timeframe: str
    total_candles: int
    date_range: str = ""
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity in ("error", "critical") for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary(self) -> str:
        lines = [
            f"Validation Report: {self.pair} {self.timeframe}",
            f"Total candles: {self.total_candles}",
            f"Date range: {self.date_range}",
            f"Errors: {self.error_count}, Warnings: {self.warning_count}",
        ]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.upper()}] {issue.category}: {issue.description}")
        return "\n".join(lines)


class DataValidator:
    """Validates OHLCV data quality."""

    def __init__(self, timeframe: str = "1h", max_price_change_pct: float = 0.5):
        """
        Args:
            timeframe: Expected timeframe for gap detection
            max_price_change_pct: Maximum allowed single-candle price change (50%)
        """
        self.timeframe = timeframe
        self.max_price_change_pct = max_price_change_pct

    def validate(self, df: pd.DataFrame, pair: str = "", timeframe: str = "") -> ValidationReport:
        """Run all validation checks on a DataFrame."""
        tf = timeframe or self.timeframe
        report = ValidationReport(
            pair=pair,
            timeframe=tf,
            total_candles=len(df),
        )

        if df.empty:
            report.issues.append(ValidationIssue(
                category="empty", severity="critical",
                description="DataFrame is empty",
            ))
            return report

        if len(df) > 0:
            report.date_range = f"{df.index[0]} to {df.index[-1]}"

        self._check_duplicates(df, report)
        self._check_chronological_order(df, report)
        self._check_ohlc_integrity(df, report)
        self._check_zero_negative_prices(df, report)
        self._check_extreme_moves(df, report)
        self._check_missing_candles(df, tf, report)
        self._check_gaps(df, tf, report)
        self._check_volume_anomalies(df, report)

        return report

    def _check_duplicates(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Check for duplicate timestamps."""
        dupes = df.index.duplicated()
        n_dupes = dupes.sum()
        if n_dupes > 0:
            report.issues.append(ValidationIssue(
                category="duplicate",
                severity="error",
                description=f"Found {n_dupes} duplicate timestamps",
                affected_rows=int(n_dupes),
            ))

    def _check_chronological_order(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Check timestamps are in chronological order."""
        if not df.index.is_monotonic_increasing:
            report.issues.append(ValidationIssue(
                category="ordering",
                severity="error",
                description="Timestamps are not in chronological order",
            ))

    def _check_ohlc_integrity(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Check impossible OHLC relationships."""
        # High must be >= Open, Close, Low
        bad_high = (
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["high"] < df["low"])
        )
        n_bad = bad_high.sum()
        if n_bad > 0:
            report.issues.append(ValidationIssue(
                category="impossible_ohlc",
                severity="error",
                description=f"High is less than Open/Close/Low in {n_bad} candles",
                affected_rows=int(n_bad),
            ))

        # Low must be <= Open, Close, High
        bad_low = (
            (df["low"] > df["open"]) |
            (df["low"] > df["close"]) |
            (df["low"] > df["high"])
        )
        n_bad_low = bad_low.sum()
        if n_bad_low > 0:
            report.issues.append(ValidationIssue(
                category="impossible_ohlc",
                severity="error",
                description=f"Low is greater than Open/Close/High in {n_bad_low} candles",
                affected_rows=int(n_bad_low),
            ))

        # Open and Close must be between Low and High
        bad_oc = (
            (df["open"] < df["low"]) |
            (df["open"] > df["high"]) |
            (df["close"] < df["low"]) |
            (df["close"] > df["high"])
        )
        n_bad_oc = bad_oc.sum()
        if n_bad_oc > 0:
            report.issues.append(ValidationIssue(
                category="impossible_ohlc",
                severity="error",
                description=f"Open/Close outside Low-High range in {n_bad_oc} candles",
                affected_rows=int(n_bad_oc),
            ))

    def _check_zero_negative_prices(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Check for zero or negative prices."""
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                bad = (df[col] <= 0).sum()
                if bad > 0:
                    report.issues.append(ValidationIssue(
                        category="zero_price",
                        severity="critical",
                        description=f"{bad} candles have zero/negative {col}",
                        affected_rows=int(bad),
                    ))

        if "volume" in df.columns:
            neg_vol = (df["volume"] < 0).sum()
            if neg_vol > 0:
                report.issues.append(ValidationIssue(
                    category="zero_price",
                    severity="error",
                    description=f"{neg_vol} candles have negative volume",
                    affected_rows=int(neg_vol),
                ))

    def _check_extreme_moves(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Check for extreme single-candle price changes."""
        if len(df) < 2:
            return

        returns = df["close"].pct_change().abs()
        extreme = returns > self.max_price_change_pct
        n_extreme = extreme.sum()
        if n_extreme > 0:
            max_move = returns.max()
            report.issues.append(ValidationIssue(
                category="anomaly",
                severity="warning",
                description=(
                    f"{n_extreme} candles with >{self.max_price_change_pct*100:.0f}% "
                    f"single-candle move (max: {max_move*100:.1f}%)"
                ),
                affected_rows=int(n_extreme),
            ))

    def _check_missing_candles(
        self, df: pd.DataFrame, timeframe: str, report: ValidationReport
    ) -> None:
        """Check for missing candles based on expected timeframe."""
        expected_minutes = TIMEFRAME_MINUTES.get(timeframe)
        if expected_minutes is None or len(df) < 2:
            return

        expected_delta = pd.Timedelta(minutes=expected_minutes)
        # Allow some tolerance (up to 2x expected interval)
        gaps = df.index.to_series().diff()
        missing = gaps[gaps > expected_delta * 2]
        n_missing = len(missing)

        if n_missing > 0:
            report.issues.append(ValidationIssue(
                category="missing",
                severity="warning",
                description=(
                    f"Found {n_missing} gaps larger than 2x expected interval "
                    f"({expected_minutes}min)"
                ),
                affected_rows=n_missing,
            ))

    def _check_gaps(
        self, df: pd.DataFrame, timeframe: str, report: ValidationReport
    ) -> None:
        """Report total gap duration."""
        expected_minutes = TIMEFRAME_MINUTES.get(timeframe)
        if expected_minutes is None or len(df) < 2:
            return

        expected_delta = pd.Timedelta(minutes=expected_minutes)
        gaps = df.index.to_series().diff()
        excess = gaps[gaps > expected_delta * 1.5]
        total_gap_minutes = excess.sum().total_seconds() / 60 if len(excess) > 0 else 0

        if total_gap_minutes > 0:
            report.issues.append(ValidationIssue(
                category="gap",
                severity="warning",
                description=(
                    f"Total gap duration: {total_gap_minutes:.0f} minutes "
                    f"({total_gap_minutes / 60:.1f} hours) across {len(excess)} gaps"
                ),
                affected_rows=len(excess),
            ))

    def _check_volume_anomalies(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Check for unusual volume patterns."""
        if "volume" not in df.columns or len(df) < 100:
            return

        # Zero volume candles
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            pct = zero_vol / len(df) * 100
            report.issues.append(ValidationIssue(
                category="volume_anomaly",
                severity="warning" if pct < 5 else "error",
                description=f"{zero_vol} candles with zero volume ({pct:.1f}%)",
                affected_rows=int(zero_vol),
            ))

    def validate_and_report(
        self, df: pd.DataFrame, pair: str = "", timeframe: str = ""
    ) -> tuple[bool, ValidationReport]:
        """Validate and return (is_valid, report)."""
        report = self.validate(df, pair, timeframe)
        is_valid = not report.has_errors

        if report.issues:
            logger.info(
                "validation_complete",
                pair=pair,
                timeframe=timeframe,
                errors=report.error_count,
                warnings=report.warning_count,
            )
        else:
            logger.info("validation_passed", pair=pair, timeframe=timeframe)

        return is_valid, report
