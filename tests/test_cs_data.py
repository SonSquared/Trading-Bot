"""Task 3 (plan MD): canonical data, quality gates, point-in-time catalogue.

Pinned here:
- quality gates REJECT gaps, duplicates, out-of-order timestamps, clock skew,
  impossible prices, and non-positive volumes — never silently fill;
- staleness and near-future timestamps are flagged (not silently accepted);
- the catalogue is point-in-time: resolve(as_of) can never see a revision
  created after as_of (no future leakage into research);
- catalogue revisions are immutable once registered.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from crypto_system.data.base import CandleFrame
from crypto_system.data.catalogue import DataCatalogue
from crypto_system.data.quality import DataQualityError, validate_partition


def _ts(hours_ago: int = 0, minutes: int = 0) -> datetime:
    return datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc) - timedelta(
        hours=hours_ago, minutes=minutes
    )


def _df(rows: list[tuple[datetime, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["open_time", "open", "high", "low", "close", "volume"],
    )


def _clean(n: int = 5, *, end: datetime | None = None) -> pd.DataFrame:
    end = end or _ts(minutes=0)
    step = timedelta(hours=4)
    rows = []
    t = end - step * (n - 1)
    for i in range(n):
        t_i = t + step * i
        rows.append((t_i, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0))
    return _df(rows)


class TestQualityGates:
    def test_clean_partition_passes(self):
        report = validate_partition(_clean(5), interval="4h", now=_ts())
        assert report.accepted and report.issues == []

    def test_gap_is_rejected_not_filled(self):
        df = _clean(5)
        df = df.drop(index=2).reset_index(drop=True)
        with pytest.raises(DataQualityError, match="gap"):
            validate_partition(df, interval="4h", now=_ts())

    def test_duplicate_timestamp_rejected(self):
        df = _clean(5)
        df.iloc[2] = df.iloc[1]
        with pytest.raises(DataQualityError, match="duplicate"):
            validate_partition(df, interval="4h", now=_ts())

    def test_out_of_order_rejected(self):
        df = _clean(5)
        df = df.iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)
        with pytest.raises(DataQualityError, match="order"):
            validate_partition(df, interval="4h", now=_ts())

    def test_future_timestamp_is_clock_skew(self):
        df = _clean(5, end=_ts() + timedelta(hours=1))
        with pytest.raises(DataQualityError, match="clock skew|future"):
            validate_partition(df, interval="4h", now=_ts())

    def test_impossible_price_rejected(self):
        df = _clean(5)
        df.loc[df.index[2], "low"] = 200.0  # low above high
        with pytest.raises(DataQualityError, match="impossible"):
            validate_partition(df, interval="4h", now=_ts())

    def test_negative_price_rejected(self):
        df = _clean(5)
        df.loc[df.index[2], "close"] = -1.0
        with pytest.raises(DataQualityError):
            validate_partition(df, interval="4h", now=_ts())

    def test_non_positive_volume_rejected(self):
        df = _clean(5)
        df.loc[df.index[1], "volume"] = 0.0
        with pytest.raises(DataQualityError, match="volume"):
            validate_partition(df, interval="4h", now=_ts())

    def test_stale_data_flagged_not_rejected(self):
        df = _clean(5, end=_ts() - timedelta(hours=20))
        report = validate_partition(df, interval="4h", now=_ts())
        assert report.accepted  # stale != corrupt
        assert any("stale" in f for f in report.flags)


class TestCatalogue:
    @pytest.fixture()
    def catalogue(self):
        cat = DataCatalogue()
        old = cat.register(
            dataset="BTCUSDT_4h",
            version="v1",
            created_at=_ts(hours_ago=48),
            rows=1000,
            source="exchange",
        )
        new = cat.register(
            dataset="BTCUSDT_4h",
            version="v2",
            created_at=_ts(hours_ago=4),
            rows=1012,
            source="exchange",
        )
        return cat, old, new

    def test_catalogue_as_of_excludes_future_revision(self, catalogue):
        cat, old, _new = catalogue
        resolved = cat.resolve("BTCUSDT_4h", as_of=_ts(hours_ago=24))
        assert resolved.version == old.version

    def test_resolve_latest_by_default(self, catalogue):
        cat, _old, new = catalogue
        assert cat.resolve("BTCUSDT_4h").version == new.version

    def test_resolve_unknown_dataset_fails_closed(self, catalogue):
        cat, _, _ = catalogue
        with pytest.raises(KeyError):
            cat.resolve("NOSUCH_4h")

    def test_revisions_are_immutable(self, catalogue):
        cat, old, _ = catalogue
        with pytest.raises(Exception):
            old.version = "v3"  # type: ignore[misc]

    def test_re_register_same_version_is_rejected(self, catalogue):
        cat, _, _ = catalogue
        with pytest.raises(ValueError, match="immutable"):
            cat.register(
                dataset="BTCUSDT_4h",
                version="v1",
                created_at=_ts(hours_ago=1),
                rows=999,
                source="exchange",
            )


class TestAdapterContract:
    def test_adapter_protocol_methods(self):
        adapter = CandleFrame.binanceusdm()
        assert hasattr(adapter, "candles")
        assert hasattr(adapter, "funding")
        assert adapter.exchange_id == "binanceusdm"
