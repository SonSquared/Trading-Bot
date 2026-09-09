"""Data quality gates (plan MD Task 3).

Rule: reject or flag — never silently fill. A gap, duplicate, out-of-order
timestamp, clock-skewed future candle, impossible price, or non-positive
volume raises ``DataQualityError``. Staleness is a *flag* (the data is old
but not corrupt) so downstream consumers can decide fail-open or fail-closed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

_INTERVAL_RE = re.compile(r"^(\d+)([mhd])$")
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}

_REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


class DataQualityError(RuntimeError):
    """A partition failed a hard quality gate."""


def interval_to_timedelta(interval: str) -> timedelta:
    match = _INTERVAL_RE.match(interval)
    if not match:
        raise ValueError(f"unparseable interval: {interval!r}")
    return timedelta(seconds=int(match.group(1)) * _UNIT_SECONDS[match.group(2)])


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    issues: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    rows: int = 0


def validate_partition(
    df: pd.DataFrame,
    *,
    interval: str,
    now: datetime | None = None,
    max_staleness_intervals: int = 2,
) -> QualityReport:
    """Run all hard gates; return a report. Raises DataQualityError on any
    hard failure (gaps/duplicates/order/skew/impossible prices/volume)."""
    now = now or datetime.now(timezone.utc)
    step = interval_to_timedelta(interval)
    issues: list[str] = []
    flags: list[str] = []

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(f"missing required columns: {missing}")
    if df.empty:
        raise DataQualityError("partition is empty")

    times = pd.to_datetime(df["open_time"], utc=True)

    # Out-of-order detection on the RAW input (before any sorting).
    raw = times.to_list()
    if any(b < a for a, b in zip(raw, raw[1:])):
        raise DataQualityError("out-of-order timestamps detected")

    # Duplicates.
    dupes = times[times.duplicated()].tolist()
    if dupes:
        raise DataQualityError(f"duplicate timestamps: {dupes[:3]}")

    # Clock skew: anything in the (near) future is rejected.
    future = [t for t in raw if t > now + timedelta(minutes=5)]
    if future:
        raise DataQualityError(
            f"clock skew: {len(future)} timestamps in the future "
            f"(latest {max(future).isoformat()} vs now {now.isoformat()})"
        )

    # Gaps: consecutive opens must be exactly one interval apart.
    for a, b in zip(raw, raw[1:]):
        delta = b - a
        if delta != step:
            raise DataQualityError(
                f"gap/irregular spacing detected: {delta} between "
                f"{a.isoformat()} and {b.isoformat()} (expected {step})"
            )

    # Impossible prices.
    opens = df["open"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    closes = df["close"].astype(float)
    volumes = df["volume"].astype(float)

    bad_mask = (highs < lows) | (highs <= 0) | (lows <= 0)
    bad_mask |= (highs < pd.concat([opens, closes], axis=1).max(axis=1))
    bad_mask |= (lows > pd.concat([opens, closes], axis=1).min(axis=1))
    if bool(bad_mask.any()):
        raise DataQualityError(
            f"impossible prices at {int(bad_mask.sum())} row(s) "
            f"(low>high, non-positive, or open/close outside [low, high])"
        )

    if bool((volumes <= 0).any()):
        raise DataQualityError("non-positive volume present")

    # Staleness: flagged, not rejected.
    last = max(raw)
    staleness = now - last
    if staleness > step * max_staleness_intervals:
        flags.append(
            f"stale: last candle {last.isoformat()} is {staleness} old "
            f"(> {max_staleness_intervals}x interval)"
        )

    return QualityReport(accepted=True, issues=issues, flags=flags, rows=len(df))


def validate_and_describe(df: pd.DataFrame, **kwargs: Any) -> QualityReport:
    """Convenience wrapper raising on issues, returning flags otherwise."""
    return validate_partition(df, **kwargs)
