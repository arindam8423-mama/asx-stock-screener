from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


class MarketDataStore:
    """Small DuckDB wrapper for querying the local Parquet research dataset."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.database_path))

    def ingest_parquet_directory(self, prices_dir: Path) -> int:
        """Create/refresh the prices view over all locally downloaded Parquet files."""
        files = list(prices_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No Parquet files found in {prices_dir}")

        with self.connect() as con:
            con.execute(
                "CREATE OR REPLACE VIEW prices AS "
                "SELECT * FROM read_parquet(?)",
                [str(prices_dir / "*.parquet")],
            )
            count = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        return int(count)

    def query_prices(
        self,
        ticker: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Query the local prices view with optional ticker/date filters."""
        conditions: list[str] = []
        params: list[str] = []

        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.upper().removesuffix(".AX"))
        if start:
            conditions.append("date >= ?")
            params.append(start)
        if end:
            conditions.append("date < ?")
            params.append(end)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM prices {where} ORDER BY ticker, date"

        with self.connect() as con:
            return con.execute(sql, params).df()
