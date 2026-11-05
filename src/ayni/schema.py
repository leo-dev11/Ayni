"""Esquemas de validación con Pandera.

Dos contratos, uno por cada extremo del pipeline:

- `application_train_schema`: lo que debe cumplir la tabla cruda de solicitudes.
- `features_schema`: lo que debe cumplir `features.parquet` antes de entrenar.

La validación es *lazy*: se recogen todos los fallos y se reportan juntos, en lugar de
abortar en el primero. Un pipeline que falla por un dato malo tiene que decir cuál.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

from ayni.config import ID_COL, TARGET


class DataValidationError(Exception):
    """Los datos no cumplen el contrato y el pipeline debe detenerse."""


def _no_columnas_vacias(df: pd.DataFrame) -> bool:
    """Ninguna columna puede estar completamente nula: sería un error de join o una fuente caída."""
    return bool(df.notna().any().all())


def _no_constantes(df: pd.DataFrame) -> bool:
    """Una columna con un solo valor no aporta y suele delatar un merge mal hecho."""
    return bool((df.nunique(dropna=False) > 1).all())


# ── 1. Entrada cruda ────────────────────────────────────────────────────────────
# Solo se fijan las columnas que el pipeline usa o que definen la identidad de la fila.
# `strict=False`: la tabla real trae 122 columnas y las que sobren son bienvenidas.
application_train_schema = pa.DataFrameSchema(
    {
        ID_COL: pa.Column(int, checks=pa.Check.gt(0), unique=True, nullable=False),
        TARGET: pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
        "AMT_INCOME_TOTAL": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "AMT_CREDIT": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
        "AMT_ANNUITY": pa.Column(float, checks=pa.Check.ge(0), nullable=True),
        "AMT_GOODS_PRICE": pa.Column(float, checks=pa.Check.ge(0), nullable=True),
        "CNT_CHILDREN": pa.Column(int, checks=pa.Check.ge(0), nullable=False),
        # 2 filas de la tabla real traen nulo: se permite, pero nunca negativo.
        "CNT_FAM_MEMBERS": pa.Column(float, checks=pa.Check.ge(0), nullable=True),
        "DAYS_BIRTH": pa.Column(int, checks=pa.Check.lt(0), nullable=False),
        # Puede ser negativo (días trabajados) o 365243 (centinela): ambos son válidos aquí.
        "DAYS_EMPLOYED": pa.Column(int, nullable=False),
        "CODE_GENDER": pa.Column(str, checks=pa.Check.isin(["F", "M", "XNA"]), nullable=False),
        "NAME_CONTRACT_TYPE": pa.Column(str, nullable=False),
    },
    strict=False,
    name="application_train",
)

# ── 2. Features listas para modelar ─────────────────────────────────────────────
# Se enumeran las columnas que el entrenamiento da por hechas. Si falta una, el error
# aparece aquí y no tres celdas más abajo dentro de LightGBM.
REQUIRED_FEATURES = (
    ID_COL,
    TARGET,
    # ratios de la solicitud
    "credit_income_ratio",
    "annuity_income_ratio",
    "goods_credit_ratio",
    "income_per_person",
    "children_ratio",
    "age_years",
    # centinela corregido
    "employed_years",
    "employed_anomaly",
    # una por cada fuente agregada
    "bureau_n_credits",
    "bureau_debt_ratio",
    "bureau_max_dpd",
    "prev_n_applications",
    "prev_approval_rate",
    "prev_refusal_rate",
    "prev_repeater_share",
    "inst_n_payments",
    "inst_late_rate",
    "inst_payment_ratio",
    "cc_n_months",
    "cc_utilization_mean",
    "cc_delinquent_month_rate",
    "bureau_share_credit_card",
    "bureau_share_consumer",
)

# Columnas que son proporciones y por tanto deben vivir en [0, 1].
TASAS = (
    "prev_approval_rate",
    "prev_refusal_rate",
    "prev_repeater_share",
    "inst_late_rate",
    "bureau_share_credit_card",
    "bureau_share_consumer",
    "cc_delinquent_month_rate",
)

features_schema = pa.DataFrameSchema(
    {
        **{
            col: pa.Column(nullable=True)
            for col in REQUIRED_FEATURES
            if col not in (ID_COL, TARGET)
        },
        ID_COL: pa.Column(int, checks=pa.Check.gt(0), unique=True, nullable=False),
        TARGET: pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
        "age_years": pa.Column(float, checks=pa.Check.in_range(18, 100), nullable=False),
        "employed_anomaly": pa.Column(
            int, checks=pa.Check.isin([0, 1]), nullable=False, coerce=True
        ),
        "employed_years": pa.Column(float, checks=pa.Check.ge(0), nullable=True),
        "credit_income_ratio": pa.Column(float, checks=pa.Check.ge(0), nullable=True),
        "annuity_income_ratio": pa.Column(float, checks=pa.Check.ge(0), nullable=True),
        **{col: pa.Column(float, checks=pa.Check.in_range(0, 1), nullable=True) for col in TASAS},
    },
    checks=[
        pa.Check(_no_columnas_vacias, name="sin_columnas_100pct_nulas"),
        pa.Check(_no_constantes, name="sin_columnas_constantes"),
    ],
    strict=False,
    name="features",
)


def _resumen(error: SchemaErrors | SchemaError, maximo: int = 8) -> str:
    if isinstance(error, SchemaErrors):
        casos = error.failure_cases
        filas = [
            f"  - {fila['column']}: {fila['check']} (valor={fila['failure_case']!r})"
            for _, fila in casos.head(maximo).iterrows()
        ]
        extra = "" if len(casos) <= maximo else f"\n  … y {len(casos) - maximo} fallo(s) más"
        return "\n".join(filas) + extra
    return f"  - {error}"


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Valida contra el esquema. Lanza `DataValidationError` con el detalle de los fallos."""
    try:
        return schema.validate(df, lazy=True)
    except (SchemaErrors, SchemaError) as error:
        nombre = schema.name or "esquema"
        raise DataValidationError(
            f"Los datos no cumplen el esquema '{nombre}' "
            f"({df.shape[0]:,} filas × {df.shape[1]} columnas):\n"
            f"{_resumen(error)}"
        ) from error


# ── 3. Particiones ya separadas ────────────────────────────────────────────────
# Sobre train/test no se exige "ninguna columna constante": una muestra puede tener una
# columna de un solo valor por puro azar (le pasa a FLAG_MOBIL en el test de este dataset).
split_schema = pa.DataFrameSchema(
    {
        ID_COL: pa.Column(int, checks=pa.Check.gt(0), unique=True, nullable=False),
        TARGET: pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
        "age_years": pa.Column(float, checks=pa.Check.in_range(18, 100), nullable=False),
    },
    strict=False,
    name="split",
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    return validate(df, application_train_schema)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    faltantes = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if faltantes:
        raise DataValidationError(
            "Faltan columnas obligatorias en features: " + ", ".join(faltantes)
        )
    return validate(df, features_schema)


def validate_split(df: pd.DataFrame) -> pd.DataFrame:
    return validate(df, split_schema)
