"""Exportación del modelo a un artefacto autónomo para servir.

MLflow guarda el modelo con rutas de artefacto que no viajan bien a un contenedor, así que
para el servicio se exporta:

- `models/booster.txt`   el Booster de LightGBM (incluye el mapeo de categorías)
- `models/manifest.json` versión, umbral, columnas en orden y vocabulario de categorías

Ventaja: la API no necesita MLflow ni el registro para arrancar, y la imagen queda más liviana.
El registro sigue siendo la fuente de verdad para entrenamiento y auditoría.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.lightgbm

from ayni.config import MLFLOW_URI, PROCESSED_DIR, REPORTS_DIR, ROOT

MODELS_DIR = ROOT / "models"
REGISTERED_MODEL = "ayni-lgbm"
CATEGORIAS_PATH = PROCESSED_DIR / "categorias.json"
UMBRAL_PATH = REPORTS_DIR / "umbral.json"


def exportar(version: str = "latest", destino: Path = MODELS_DIR) -> Path:
    """Descarga el modelo registrado y lo deja listo para `ayni.api`."""
    destino.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_URI)
    uri = f"models:/{REGISTERED_MODEL}/{version}"
    modelo = mlflow.lightgbm.load_model(uri)

    booster = destino / "booster.txt"
    modelo.booster_.save_model(str(booster))

    categorias = (
        json.loads(CATEGORIAS_PATH.read_text(encoding="utf-8")) if CATEGORIAS_PATH.exists() else {}
    )
    umbral_info = (
        json.loads(UMBRAL_PATH.read_text(encoding="utf-8")) if UMBRAL_PATH.exists() else {}
    )

    manifest = {
        "modelo": REGISTERED_MODEL,
        "version": version,
        "uri_origen": uri,
        "columnas": list(modelo.feature_name_),
        "categorias": categorias,
        "umbral": umbral_info.get("umbral_optimo"),
        "matriz_costos": umbral_info.get("matriz_costos"),
        "exportado": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (destino / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"modelo exportado a {destino.relative_to(ROOT)}")
    print(f"  columnas : {len(manifest['columnas'])}")
    print(f"  umbral   : {manifest['umbral']}")
    print(f"  categorías de texto: {len(categorias)}")
    return destino
