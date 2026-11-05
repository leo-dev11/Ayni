"""Rutas y constantes compartidas del proyecto."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLFLOW_DB = ROOT / "mlflow.db"
# Permite aislar pruebas y CI del experimento real.
MLFLOW_URI = os.environ.get("AYNI_TRACKING_URI") or f"sqlite:///{MLFLOW_DB.as_posix()}"

RANDOM_STATE = 42
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
PREV_ID_COL = "SK_ID_PREV"

# Tablas del dataset de Kaggle, en orden de dependencia.
TABLES = (
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "credit_card_balance",
    "POS_CASH_balance",
)

# Subconjunto con el que se construye el modelo (application_test no trae TARGET).
MODEL_TABLES = (
    "application_train",
    "bureau",
    "previous_application",
    "installments_payments",
    "credit_card_balance",
)
