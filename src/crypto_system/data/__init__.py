"""Data layer: adapter contract, quality gates, point-in-time catalogue."""

from crypto_system.data.base import CandleFrame, ExchangeDataAdapter
from crypto_system.data.catalogue import DataCatalogue, DatasetRevision
from crypto_system.data.quality import DataQualityError, QualityReport, validate_partition

__all__ = [
    "CandleFrame",
    "ExchangeDataAdapter",
    "DataCatalogue",
    "DatasetRevision",
    "DataQualityError",
    "QualityReport",
    "validate_partition",
]
