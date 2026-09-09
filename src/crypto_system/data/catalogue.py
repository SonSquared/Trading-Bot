"""Point-in-time data catalogue (plan MD Task 3).

Every dataset revision is immutable and carries its creation timestamp.
``resolve(dataset, as_of)`` returns the newest revision *known at* ``as_of``
— a revision created after as_of is invisible. This is what makes research
reproducible: a backtest run at time T can never accidentally consume data
revisions that appeared later.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DatasetRevision(BaseModel):
    """One immutable version of a dataset."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    version: str
    created_at: datetime
    rows: int
    source: str
    content_hash: str = ""


class DataCatalogue:
    def __init__(self) -> None:
        self._revisions: dict[str, list[DatasetRevision]] = {}

    def register(
        self,
        *,
        dataset: str,
        version: str,
        created_at: datetime,
        rows: int,
        source: str,
        content_hash: str = "",
    ) -> DatasetRevision:
        """Register a new revision. (dataset, version) pairs are immutable."""
        existing = self._revisions.get(dataset, [])
        for rev in existing:
            if rev.version == version:
                raise ValueError(
                    f"revision {dataset}/{version} already registered "
                    f"at {rev.created_at.isoformat()} — catalogue entries "
                    f"are immutable"
                )
        revision = DatasetRevision(
            dataset=dataset,
            version=version,
            created_at=created_at,
            rows=rows,
            source=source,
            content_hash=content_hash,
        )
        self._revisions.setdefault(dataset, []).append(revision)
        self._revisions[dataset].sort(key=lambda r: r.created_at)
        return revision

    def resolve(
        self, dataset: str, *, as_of: Optional[datetime] = None
    ) -> DatasetRevision:
        """Newest revision created at or before as_of (default: latest)."""
        revisions = self._revisions.get(dataset)
        if not revisions:
            raise KeyError(f"unknown dataset: {dataset!r}")
        if as_of is None:
            return revisions[-1]
        eligible = [r for r in revisions if r.created_at <= as_of]
        if not eligible:
            raise KeyError(
                f"no revision of {dataset!r} existed at as_of={as_of.isoformat()}"
            )
        return eligible[-1]

    def history(self, dataset: str) -> list[DatasetRevision]:
        return list(self._revisions.get(dataset, []))
