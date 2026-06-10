"""
DuckDB singleton connection with schema introspection.
All query helpers live here — no other module imports duckdb directly.
"""
from __future__ import annotations
import threading
from pathlib import Path
from typing import Optional
import duckdb
import pandas as pd
from app.core.config import get_settings
from app.monitoring.telemetry import get_logger

logger = get_logger(__name__)
_lock = threading.Lock()
_conn: Optional[duckdb.DuckDBPyConnection] = None


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:
                cfg = get_settings()
                Path(cfg.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
                try:
                    _conn = duckdb.connect(cfg.DB_PATH, read_only=read_only)
                    logger.info("duckdb_connected", path=cfg.DB_PATH, read_only=read_only)
                except duckdb.IOException:
                    # File locked by another process — fall back to read-only
                    logger.warning("duckdb_locked_fallback_readonly", path=cfg.DB_PATH)
                    _conn = duckdb.connect(cfg.DB_PATH, read_only=True)
    return _conn


def execute_query(sql: str) -> pd.DataFrame:
    return get_connection().execute(sql).fetchdf()


def execute_ddl(sql: str) -> None:
    get_connection().execute(sql)


def table_exists(name: str) -> bool:
    r = get_connection().execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name=?", [name]
    ).fetchone()
    return bool(r and r[0] > 0)


def get_schema_dict() -> dict[str, list[str]]:
    """Return {table_name: [col1, col2, ...]} for all tables."""
    con = get_connection()
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchdf()["table_name"].tolist()
    return {
        t: con.execute(f"PRAGMA table_info('{t}')").fetchdf()["name"].tolist()
        for t in tables
    }


def get_sample_values(table: str, column: str, n: int = 20) -> list[str]:
    df = execute_query(
        f"SELECT DISTINCT {column} FROM {table} "
        f"WHERE {column} IS NOT NULL ORDER BY {column} LIMIT {n}"
    )
    return df[column].astype(str).tolist()
