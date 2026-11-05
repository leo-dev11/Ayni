"""Explicabilidad: por qué el modelo aprueba o rechaza a cada cliente.

Tres piezas:

- **Global:** qué features mueven la predicción en promedio (`summary` de SHAP).
- **Local:** la descomposición exacta de una decisión concreta (`waterfall`).
- **Reason codes:** las cinco razones que se le dan a una persona cuando se le rechaza el
  crédito. Es lo que la banca real entrega y lo que consumirá `/explain` en la fase 7.

Los valores SHAP están en **log-odds** (la salida del modelo antes de la sigmoide): el signo
indica dirección del riesgo y la magnitud, cuánto empuja esa feature respecto de la media.
Las explicaciones se calculan sobre `train`, nunca sobre el test.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: se ejecuta desde el CLI
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import shap

from ayni.config import ID_COL, MLFLOW_URI, PROCESSED_DIR, REPORTS_DIR, ROOT, TARGET
from ayni.evaluate import aplicar_categorias
from ayni.split import TRAIN_PATH

# shap avisa en cada llamada de que LightGBM binario devuelve una lista; ya se normaliza.
warnings.filterwarnings("ignore", message=".*shap values output has changed.*")

CATEGORIAS_PATH = PROCESSED_DIR / "categorias.json"
UMBRAL_PATH = REPORTS_DIR / "umbral.json"
SUMMARY_PATH = ROOT / "reports" / "figures" / "09_shap_summary.png"
LOCAL_PATH = ROOT / "reports" / "figures" / "10_shap_local.png"
RAZONES_PATH = REPORTS_DIR / "razones_ejemplo.json"
REGISTERED_MODEL = "ayni-lgbm"
MUESTRA_GLOBAL = 5000


@dataclass
class Razon:
    """Una feature y su empuje sobre la decisión, en el formato que consume la API."""

    feature: str
    valor: object
    shap: float
    direccion: str  # "aumenta_riesgo" | "reduce_riesgo"

    def a_dict(self) -> dict:
        # mismos nombres que expone la API: `contribucion` en log-odds y `direccion`
        valor = self.valor
        if isinstance(valor, (int, float)) and not isinstance(valor, bool):
            valor = round(float(valor), 4)
        return {
            "feature": self.feature,
            "valor": valor,
            "contribucion": round(float(self.shap), 4),
            "direccion": self.direccion,
        }


def cargar_modelo(version: str = "latest") -> lgb.LGBMClassifier:
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.lightgbm.load_model(f"models:/{REGISTERED_MODEL}/{version}")


def cargar_umbral() -> float:
    if not UMBRAL_PATH.exists():
        raise SystemExit("No hay umbral. Ejecuta antes: python -m ayni evaluate")
    return json.loads(UMBRAL_PATH.read_text(encoding="utf-8"))["umbral_optimo"]


def preparar_matriz(df: pd.DataFrame) -> pd.DataFrame:
    """Deja el frame como lo espera el modelo: sin IDs ni target, sin género, categorías fijas."""
    X = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])
    X = X.drop(columns=["CODE_GENDER"], errors="ignore")
    vocabulario = (
        json.loads(CATEGORIAS_PATH.read_text(encoding="utf-8")) if CATEGORIAS_PATH.exists() else {}
    )
    for columna in X.columns:
        es_texto = X[columna].dtype == object or str(X[columna].dtype) in {
            "boolean",
            "category",
        }
        if es_texto and columna not in vocabulario:
            X[columna] = X[columna].astype("category")
    return aplicar_categorias(X, vocabulario)


def construir_explicador(modelo: lgb.LGBMClassifier) -> shap.TreeExplainer:
    """`tree_path_dependent` no necesita fondo de datos y es exacto para árboles."""
    return shap.TreeExplainer(modelo)


def valores_shap(explicador: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    valores = explicador.shap_values(X)
    if isinstance(valores, list):  # algunas versiones devuelven [clase0, clase1]
        valores = valores[-1]
    return np.asarray(valores)


def razones(explicador: shap.TreeExplainer, fila: pd.DataFrame, top: int = 5) -> list[Razon]:
    """Las `top` features que más empujan la decisión, ordenadas por magnitud absoluta."""
    valores = valores_shap(explicador, fila)[0]
    orden = np.argsort(-np.abs(valores))[:top]
    return [
        Razon(
            feature=str(fila.columns[i]),
            valor=fila.iloc[0, i],
            shap=float(valores[i]),
            direccion="aumenta_riesgo" if valores[i] > 0 else "reduce_riesgo",
        )
        for i in orden
    ]


def _ejemplos(modelo, X: pd.DataFrame, umbral: float) -> tuple[int, int]:
    """Un caso aprobado y uno rechazado **típicos** (probabilidad mediana de cada grupo).

    Elegir los casos pegados al corte haría que ambos rondaran 0.16 y la comparación no
    mostrara nada; se toman las medianas para que representen a su grupo.
    """
    proba = modelo.predict_proba(X)[:, 1]
    aprobados = np.sort(np.where(proba <= umbral)[0])
    rechazados = np.sort(np.where(proba > umbral)[0])
    if aprobados.size == 0 or rechazados.size == 0:
        raise SystemExit("La muestra no tiene casos de ambos lados del umbral")
    orden_aprobados = aprobados[np.argsort(proba[aprobados])]
    orden_rechazados = rechazados[np.argsort(proba[rechazados])]
    return int(orden_aprobados[len(orden_aprobados) // 2]), int(
        orden_rechazados[len(orden_rechazados) // 2]
    )


def _figura_global(valores: np.ndarray, X: pd.DataFrame) -> Path:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(valores, X, max_display=20, show=False, plot_size=(8, 6))
    plt.title("Qué mueve la predicción (SHAP, muestra de train)", fontsize=10, pad=12)
    plt.tight_layout()
    plt.savefig(SUMMARY_PATH, dpi=140, bbox_inches="tight")
    plt.close("all")
    return SUMMARY_PATH


def _figura_local(explicador, X: pd.DataFrame, indices: dict[str, int], proba) -> Path:
    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.5))

    for eje, (etiqueta, idx) in zip(ejes, indices.items(), strict=True):
        fila = X.iloc[[idx]]
        valores = valores_shap(explicador, fila)[0]
        orden = np.argsort(-np.abs(valores))[:8][::-1]
        nombres = [str(fila.columns[i])[:34] for i in orden]
        colores = ["#C44536" if valores[i] > 0 else "#2E6F95" for i in orden]
        eje.barh(nombres, valores[orden], color=colores, height=0.65)
        eje.axvline(0, color="#333333", lw=0.8)
        eje.set_title(f"{etiqueta} — P(impago) = {proba[idx]:.3f}", fontsize=9, fontweight="bold")
        eje.set_xlabel("empuje sobre el score (log-odds)")
        eje.spines[["top", "right"]].set_visible(False)
        eje.grid(axis="x", alpha=0.25)
        eje.tick_params(axis="y", labelsize=7)

    fig.text(
        0.5,
        -0.02,
        "rojo = aumenta el riesgo de impago · azul = lo reduce",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(LOCAL_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return LOCAL_PATH


def ejecutar(version: str = "latest", muestra: int = MUESTRA_GLOBAL) -> dict:
    umbral = cargar_umbral()
    modelo = cargar_modelo(version)

    train = pd.read_parquet(TRAIN_PATH)
    X = preparar_matriz(train)
    X_muestra = X.sample(n=min(muestra, len(X)), random_state=42)
    explicador = construir_explicador(modelo)

    print(f"calculando SHAP sobre {len(X_muestra):,} filas de train…")
    valores = valores_shap(explicador, X_muestra)
    fig_global = _figura_global(valores, X_muestra)

    media_abs = pd.Series(np.abs(valores).mean(axis=0), index=X_muestra.columns)
    top_global = media_abs.sort_values(ascending=False).head(20)
    top_global.round(4).to_csv(REPORTS_DIR / "shap_importancia_global.csv", header=["shap_abs"])

    proba = modelo.predict_proba(X)[:, 1]
    aprobado, rechazado = _ejemplos(modelo, X, umbral)
    fig_local = _figura_local(explicador, X, {"Aprobado": aprobado, "Rechazado": rechazado}, proba)

    razones_aprobado = [r.a_dict() for r in razones(explicador, X.iloc[[aprobado]])]
    razones_rechazado = [r.a_dict() for r in razones(explicador, X.iloc[[rechazado]])]

    detalle = {
        "version_modelo": version,
        "umbral": umbral,
        "n_muestra_shap": int(len(X_muestra)),
        "top_global": {k: round(float(v), 4) for k, v in top_global.head(10).items()},
        "aprobado": {
            "sk_id_curr": int(train.iloc[aprobado][ID_COL]),
            "probabilidad": round(float(proba[aprobado]), 4),
            "umbral": umbral,
            "decision": "approve" if proba[aprobado] <= umbral else "reject",
            "razones": razones_aprobado,
        },
        "rechazado": {
            "sk_id_curr": int(train.iloc[rechazado][ID_COL]),
            "probabilidad": round(float(proba[rechazado]), 4),
            "umbral": umbral,
            "decision": "approve" if proba[rechazado] <= umbral else "reject",
            "razones": razones_rechazado,
        },
        "figuras": [
            str(fig_global.relative_to(ROOT)),
            str(fig_local.relative_to(ROOT)),
        ],
    }
    RAZONES_PATH.write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\ntop 10 por |SHAP| medio:")
    for nombre, valor in top_global.head(10).items():
        print(f"  {nombre:<32} {valor:.4f}")
    print(f"\ncaso aprobado  (P={proba[aprobado]:.3f}, umbral {umbral}):")
    for razon in razones_aprobado:
        print(f"  {razon['feature']:<32} {razon['shap']:+.3f}  {razon['direccion']}")
    print(f"\ncaso rechazado (P={proba[rechazado]:.3f}, umbral {umbral}):")
    for razon in razones_rechazado:
        print(f"  {razon['feature']:<32} {razon['shap']:+.3f}  {razon['direccion']}")
    print(f"\n{RAZONES_PATH.relative_to(ROOT)}")
    return detalle
