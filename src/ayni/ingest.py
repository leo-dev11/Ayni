"""Ingesta de los CSV crudos de Kaggle a Parquet con DuckDB.

Los CSV de Home Credit son grandes y se leen muchas veces durante el feature
engineering; convertirlos a Parquet una sola vez (compresión ZSTD) reduce el
tiempo de lectura y el espacio en disco. DuckDB hace la conversión leyendo el
archivo completo (``sample_size=-1``) para inferir tipos de forma estable.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import duckdb

from ayni.config import INTERIM_DIR, RAW_DIR, TABLES


def _lit(path) -> str:
    """Escapa una ruta para usarla como literal SQL."""
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def csv_path(table: str):
    return RAW_DIR / f"{table}.csv"


def parquet_path(table: str):
    return INTERIM_DIR / f"{table}.parquet"


def missing_raw(tables: Iterable[str] = TABLES) -> list[str]:
    """CSV que faltan en ``data/raw``."""
    return [t for t in tables if not csv_path(t).exists()]


def to_parquet(tables: Iterable[str] = TABLES, *, force: bool = False) -> dict[str, float]:
    """Convierte los CSV presentes a Parquet. Devuelve el tiempo por tabla."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    timings: dict[str, float] = {}

    for table in tables:
        src, dst = csv_path(table), parquet_path(table)
        if not src.exists():
            continue
        if dst.exists() and not force:
            continue
        start = time.perf_counter()
        con.execute(
            f"COPY (SELECT * FROM read_csv({_lit(src)}, header=true, sample_size=-1)) "
            f"TO {_lit(dst)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        timings[table] = time.perf_counter() - start

    con.close()
    return timings


def describe(tables: Iterable[str] = TABLES) -> list[dict[str, object]]:
    """Filas, columnas y tamaño en disco de cada Parquet disponible."""
    con = duckdb.connect()
    rows: list[dict[str, object]] = []
    for table in tables:
        path = parquet_path(table)
        if not path.exists():
            continue
        n_rows, n_cols = con.execute(
            f"SELECT count(*), count(*) FROM (SELECT * FROM read_parquet({_lit(path)}))"
        ).fetchone()
        n_cols = con.execute(
            f"SELECT count(*) FROM (DESCRIBE SELECT * FROM read_parquet({_lit(path)}))"
        ).fetchone()[0]
        rows.append(
            {
                "tabla": table,
                "filas": n_rows,
                "columnas": n_cols,
                "mb": round(path.stat().st_size / 1024**2, 1),
            }
        )
    con.close()
    return rows
