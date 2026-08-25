"""
Data storage using Parquet files for OHLCV data and SQLite for metadata/experiment results.

Provides versioned, reproducible data storage.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DataStorage:
    """Manages data storage in Parquet files with SQLite metadata."""

    @staticmethod
    def _safe_pair_name(pair: str) -> str:
        """Convert a trading pair to a filesystem-safe directory name."""
        return pair.replace("/", "_").replace(":", "_")

    def __init__(self, data_dir: str, results_dir: str):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._init_metadata_db()

    def _init_metadata_db(self) -> None:
        """Initialize the SQLite metadata database."""
        db_path = self.results_dir / "metadata.db"
        self.conn = sqlite3.connect(str(db_path))
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                data_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                n_rows INTEGER,
                file_hash TEXT,
                created_at TEXT,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT UNIQUE NOT NULL,
                strategy_name TEXT NOT NULL,
                parameters TEXT NOT NULL,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                data_start TEXT,
                data_end TEXT,
                results TEXT NOT NULL,
                created_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                validation_type TEXT NOT NULL,
                results TEXT NOT NULL,
                created_at TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)

        self.conn.commit()

    def save_data(
        self,
        df: pd.DataFrame,
        pair: str,
        timeframe: str,
        data_type: str = "klines",
        metadata: dict | None = None,
    ) -> str:
        """Save a DataFrame to Parquet and record metadata."""
        pair_name = self._safe_pair_name(pair)
        pair_dir = self.data_dir / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{data_type}_{timeframe}.parquet" if data_type != "funding_rates" else "funding_rates.parquet"
        file_path = pair_dir / filename

        df.to_parquet(file_path)

        # Calculate hash for reproducibility
        file_hash = self._hash_file(file_path)

        # Record metadata
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO data_versions
            (pair, timeframe, data_type, file_path, start_date, end_date,
             n_rows, file_hash, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pair,
            timeframe,
            data_type,
            str(file_path),
            str(df.index[0]) if len(df) > 0 else None,
            str(df.index[-1]) if len(df) > 0 else None,
            len(df),
            file_hash,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(metadata) if metadata else None,
        ))
        self.conn.commit()

        logger.info("data_saved", path=str(file_path), rows=len(df))
        return str(file_path)

    def load_data(
        self,
        pair: str,
        timeframe: str,
        data_type: str = "klines",
    ) -> pd.DataFrame | None:
        """Load data from Parquet."""
        pair_name = self._safe_pair_name(pair)
        pair_dir = self.data_dir / pair_name
        filename = f"{data_type}_{timeframe}.parquet" if data_type != "funding_rates" else "funding_rates.parquet"
        file_path = pair_dir / filename

        if not file_path.exists():
            return None

        df = pd.read_parquet(file_path)
        logger.info("data_loaded", path=str(file_path), rows=len(df))
        return df

    def save_experiment(
        self,
        experiment_id: str,
        strategy_name: str,
        parameters: dict,
        pair: str,
        timeframe: str,
        results: dict,
        data_start: str = "",
        data_end: str = "",
    ) -> None:
        """Save experiment results to the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO experiments
            (experiment_id, strategy_name, parameters, pair, timeframe,
             data_start, data_end, results, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            strategy_name,
            json.dumps(parameters),
            pair,
            timeframe,
            data_start,
            data_end,
            json.dumps(results, default=str),
            datetime.now(timezone.utc).isoformat(),
        ))
        self.conn.commit()

    def save_batch_experiments(self, experiments: list[dict]) -> None:
        """Batch save experiment results for performance."""
        cursor = self.conn.cursor()
        data = [
            (
                e["experiment_id"],
                e["strategy_name"],
                json.dumps(e["parameters"]),
                e["pair"],
                e["timeframe"],
                e.get("data_start", ""),
                e.get("data_end", ""),
                json.dumps(e["results"], default=str),
                datetime.now(timezone.utc).isoformat(),
            )
            for e in experiments
        ]
        cursor.executemany("""
            INSERT OR REPLACE INTO experiments
            (experiment_id, strategy_name, parameters, pair, timeframe,
             data_start, data_end, results, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        self.conn.commit()

    def save_validation(
        self,
        experiment_id: str,
        validation_type: str,
        results: dict,
    ) -> None:
        """Save validation results."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO validation_results
            (experiment_id, validation_type, results, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            experiment_id,
            validation_type,
            json.dumps(results, default=str),
            datetime.now(timezone.utc).isoformat(),
        ))
        self.conn.commit()

    def query_experiments(
        self,
        strategy_name: str | None = None,
        pair: str | None = None,
        timeframe: str | None = None,
        min_sharpe: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query experiments with optional filters."""
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        if pair:
            query += " AND pair = ?"
            params.append(pair)
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        results = []
        for row in rows:
            d = dict(zip(columns, row))
            d["results"] = json.loads(d["results"]) if d["results"] else {}
            d["parameters"] = json.loads(d["parameters"]) if d["parameters"] else {}
            results.append(d)

        return results

    def get_top_experiments(
        self,
        metric: str = "sharpe",
        n: int = 10,
        timeframe: str | None = None,
    ) -> list[dict]:
        """Get top N experiments by a metric."""
        query = "SELECT * FROM experiments"
        params = []
        if timeframe:
            query += " WHERE timeframe = ?"
            params.append(timeframe)
        query += " ORDER BY created_at DESC"

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        experiments = []
        for row in rows:
            d = dict(zip(columns, row))
            d["results"] = json.loads(d["results"]) if d["results"] else {}
            d["parameters"] = json.loads(d["parameters"]) if d["parameters"] else {}
            experiments.append(d)

        # Sort by metric
        def get_metric(exp: dict) -> float:
            results = exp.get("results", {})
            return results.get(metric, float("-inf"))

        experiments.sort(key=get_metric, reverse=True)
        return experiments[:n]

    def _hash_file(self, path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
