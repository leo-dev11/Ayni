"""Feature engineering: une las cinco tablas en un único `features.parquet`.

Pipeline:
1. Registra cada Parquet de `data/interim/` como vista de DuckDB.
2. Ejecuta los `.sql` de `sql/` (agregación por `SK_ID_CURR`) y trae un DataFrame por tabla.
3. Une todo sobre `application_train` y construye ratios a nivel solicitud.
4. Corrige el centinela `DAYS_EMPLOYED = 365243` (NaN + flag).
5. Escribe `data/processed/features.parquet` sin columnas constantes.

El target se conserva como etiqueta, nunca como feature: `TARGET` se excluye en el
entrenamiento, no aquí (hace falta para el split estratificado).
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

from ayni.config import (
    ID_COL,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SQL_DIR,
    TABLES,
    TARGET,
)
from ayni.schema import validate_features, validate_raw

BASE_TABLE = "application_train"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# Un `.sql` por tabla relacional: nombre de la vista de origen -> archivo.
SQL_FILES = {
    "bureau": "bureau_agg.sql",
    "previous_application": "previous_agg.sql",
    "installments_payments": "installments_agg.sql",
    "credit_card_balance": "credit_card_agg.sql",
}

EMPLOYED_SENTINEL = 365_243


def connect(interim_dir: Path = INTERIM_DIR, raw_dir: Path = RAW_DIR) -> duckdb.DuckDBPyConnection:
    """Conexión en memoria con una vista por tabla disponible."""
    con = duckdb.connect()
    for table in TABLES:
        path = interim_dir / f"{table}.parquet"
        if not path.exists():
            path = raw_dir / f"{table}.csv"
            if not path.exists():
                continue
            con.execute(
                f"CREATE OR REPLACE VIEW {table} AS "
                f"SELECT * FROM read_csv('{path.as_posix()}', header=true, sample_size=-1)"
            )
            continue
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{path.as_posix()}')"
        )
    return con


def run_sql(con: duckdb.DuckDBPyConnection, sql_file: str) -> pd.DataFrame:
    return con.execute((SQL_DIR / sql_file).read_text(encoding="utf-8")).df()


def add_ratios(app: pd.DataFrame) -> pd.DataFrame:
    """Ratios a nivel solicitud: lo que el EDA mostró que los montos sueltos no capturan."""
    df = app.copy()
    ingreso = df["AMT_INCOME_TOTAL"].replace(0, pd.NA)

    df["credit_income_ratio"] = df["AMT_CREDIT"] / ingreso
    df["annuity_income_ratio"] = df["AMT_ANNUITY"] / ingreso
    df["goods_credit_ratio"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, pd.NA)
    df["income_per_person"] = ingreso / df["CNT_FAM_MEMBERS"].replace(0, pd.NA)
    df["children_ratio"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"].replace(0, pd.NA)

    df["age_years"] = -df["DAYS_BIRTH"] / 365.25
    return df


def fix_employed(df: pd.DataFrame) -> pd.DataFrame:
    """`DAYS_EMPLOYED = 365243` es el código de jubilado/desempleado, no un valor real.

    Se convierte en nulo y se conserva la información con el flag `employed_anomaly`
    (ese grupo impaga 5.4 % frente al 8.7 % del resto, según el EDA).
    """
    df = df.copy()
    anomalo = df["DAYS_EMPLOYED"] == EMPLOYED_SENTINEL
    df["employed_anomaly"] = anomalo.astype("int8")
    df["employed_years"] = (-df["DAYS_EMPLOYED"] / 365.25).where(~anomalo)
    return df.drop(columns=["DAYS_EMPLOYED", "DAYS_BIRTH"])


def drop_constant(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Quita columnas con un único valor: no aportan y ensucian la importancia."""
    candidatas = [c for c in df.columns if c not in (ID_COL, TARGET)]
    constantes = [c for c in candidatas if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=constantes), constantes


def build_features(*, force: bool = False) -> Path:
    if FEATURES_PATH.exists() and not force:
        return FEATURES_PATH

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    con = connect()

    base = validate_raw(con.execute(f"SELECT * FROM {BASE_TABLE}").df())
    print(f"  base: {base.shape[0]:,} solicitudes × {base.shape[1]} columnas")

    agregados = []
    for archivo in SQL_FILES.values():
        t0 = time.perf_counter()
        tabla = run_sql(con, archivo)
        agregados.append(tabla)
        n_feat = tabla.shape[1] - 1
        print(f"  {archivo:<24} {n_feat:>2} features  ({time.perf_counter() - t0:.1f}s)")

    features = base
    for tabla in agregados:
        features = features.merge(tabla, on=ID_COL, how="left")

    features = add_ratios(features)
    features = fix_employed(features)
    features, constantes = drop_constant(features)
    if constantes:
        print(f"  columnas constantes descartadas: {len(constantes)}")
    features = validate_features(features)

    features.to_parquet(FEATURES_PATH, index=False, compression="zstd")

    sin_huerfanas = features[ID_COL].nunique() == base[ID_COL].nunique()
    print(f"\n  {features.shape[0]:,} filas × {features.shape[1]} columnas")
    print(f"  TARGET presente: {TARGET in features.columns}")
    print(f"  una fila por cliente con dato en el buró: {sin_huerfanas}")
    print(
        f"  {FEATURES_PATH.name}: {FEATURES_PATH.stat().st_size / 1024**2:.1f} MB "
        f"en {time.perf_counter() - start:.1f}s"
    )
    con.close()
    return FEATURES_PATH
