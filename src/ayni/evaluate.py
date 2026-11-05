"""Fase 5: umbral por costo y evaluación final.

Dos momentos, separados a propósito:

1. **Elección del umbral** — con las probabilidades out-of-fold de `train`. Nunca con el test.
2. **Evaluación en test** — una sola vez, y con dos salvaguardas: hay que pedirlo explícitamente
   (`--confirmar-test`) y el archivo debe coincidir con el fingerprint guardado al crear el split.

El umbral se guarda en `reports/umbral.json` y se etiqueta en la versión del modelo registrado,
para que viaje con el artefacto y no viva solo en la cabeza de quien lo eligió.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: el pipeline corre en CLI
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ayni.config import ID_COL, MLFLOW_URI, PROCESSED_DIR, REPORTS_DIR, ROOT, TARGET
from ayni.export import resolver_version
from ayni.split import SPLIT_META, TEST_PATH, fingerprint
from ayni.threshold import (
    COSTO_FN,
    COSTO_FP,
    comparar,
    evaluar,
    sensibilidad,
    umbral_optimo,
)

OOF_DIR = PROCESSED_DIR / "oof"
UMBRAL_PATH = REPORTS_DIR / "umbral.json"
CURVA_PATH = REPORTS_DIR / "curva_costo.csv"
FIGURA_PATH = ROOT / "reports" / "figures" / "08_costo_vs_umbral.png"
EVALUACION_PATH = REPORTS_DIR / "evaluacion_test.json"
CATEGORIAS_PATH = PROCESSED_DIR / "categorias.json"
REGISTERED_MODEL = "ayni-lgbm"

# Suelo de calidad: si el modelo baja de aquí, el pipeline falla en vez de publicar.
ROC_AUC_MINIMO = 0.75


class CalidadInsuficiente(Exception):
    """El modelo no alcanza el mínimo exigido sobre el test congelado."""


class TestCongelado(Exception):
    """Se intentó leer el test sin cumplir las salvaguardas."""


def cargar_oof(corrida: str) -> pd.DataFrame:
    ruta = OOF_DIR / f"{corrida}.parquet"
    if not ruta.exists():
        raise SystemExit(f"No hay OOF para '{corrida}'. Ejecuta antes: python -m ayni train")
    return pd.read_parquet(ruta)


def aplicar_categorias(df: pd.DataFrame, vocabulario: dict[str, list]) -> pd.DataFrame:
    """Fija las categorías al vocabulario de train: si no, cada frame codifica distinto."""
    df = df.copy()
    for columna, categorias in vocabulario.items():
        if columna in df.columns:
            df[columna] = pd.Categorical(df[columna].astype(object), categories=categorias)
    return df


def _figura(curva: pd.DataFrame, optimo: float, sens: pd.DataFrame) -> Path:
    FIGURA_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 3.8))
    costo_min = curva["costo_por_solicitud"].min()

    ax1.plot(curva["umbral"], curva["costo_por_solicitud"], color="#C44536", lw=1.8)
    ax1.axvline(0.5, color="#333333", ls="--", lw=1)
    ax1.axvline(optimo, color="#2E6F95", lw=1.4)
    ax1.annotate(
        f"óptimo {optimo:.3f}\n{costo_min:.3f} por solicitud",
        xy=(optimo, costo_min),
        xytext=(optimo + 0.16, costo_min * 1.35),
        fontsize=8,
        color="#2E6F95",
        arrowprops={"arrowstyle": "->", "color": "#2E6F95"},
    )
    ax1.text(0.47, costo_min * 1.05, "0.5", ha="right", fontsize=8, color="#333333")
    ax1.set_title("Costo esperado por solicitud")
    ax1.set_xlabel("umbral: P(impago) máxima para aprobar")
    ax1.set_ylabel("costo / solicitud")

    ax2.plot(curva["umbral"], curva["tasa_aprobacion"] * 100, color="#2E6F95", lw=1.8)
    ax2.plot(
        curva["umbral"],
        curva["impago_entre_aprobados"] * 100,
        color="#C44536",
        lw=1.8,
    )
    ax2.text(curva["umbral"].iloc[5], 92, "tasa de aprobación", color="#2E6F95", fontsize=8)
    ax2.text(curva["umbral"].iloc[5], 18, "impago entre aprobados", color="#C44536", fontsize=8)
    ax2.axvline(optimo, color="#2E6F95", lw=1.4)
    ax2.axvline(0.5, color="#333333", ls="--", lw=1)
    ax2.set_title("Qué se aprueba en cada umbral")
    ax2.set_xlabel("umbral: P(impago) máxima para aprobar")
    ax2.set_ylabel("%")

    ax3.plot(sens["ratio_fn_fp"], sens["umbral_optimo"], marker="o", color="#2E6F95", lw=1.6)
    ax3.plot(
        sens["ratio_fn_fp"],
        sens["teorico_1/(ratio+1)"],
        ls=":",
        color="#333333",
        lw=1,
        label="teoría 1/(ratio+1)",
    )
    ax3.set_xscale("log")
    ax3.set_title("El umbral depende del costo, no del modelo")
    ax3.set_xlabel("cuántas veces cuesta más un falso negativo")
    ax3.set_ylabel("umbral óptimo")
    ax3.legend(frameon=False, fontsize=7)

    for ax in (ax1, ax2, ax3):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)

    fig.suptitle(f"Matriz de costos base: FN={COSTO_FN:g} · FP={COSTO_FP:g}", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURA_PATH, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return FIGURA_PATH


def elegir_umbral(corrida: str = "lgbm-ajustado") -> dict:
    """Paso 1: umbral sobre OOF. No toca el test."""
    oof = cargar_oof(corrida)
    optimo, curva = umbral_optimo(oof["y"], oof["proba"])
    sens = sensibilidad(oof["y"], oof["proba"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    curva.round(4).to_csv(CURVA_PATH, index=False)
    figura = _figura(curva, optimo, sens)
    tabla = comparar(oof["y"], oof["proba"])
    sens.round(4).to_csv(REPORTS_DIR / "sensibilidad_costos.csv", index=False)

    detalle = {
        "corrida": corrida,
        "matriz_costos": {"falso_negativo": COSTO_FN, "falso_positivo": COSTO_FP},
        "umbral_optimo": round(optimo, 4),
        "n_oof": int(len(oof)),
        "costo_por_solicitud_optimo": round(curva["costo_por_solicitud"].min(), 4),
        "costo_por_solicitud_umbral_050": round(
            curva.loc[(curva["umbral"] - 0.5).abs().idxmin(), "costo_por_solicitud"], 4
        ),
        "figura": figura.relative_to(ROOT).as_posix(),
        "sensibilidad_ratio_vs_umbral": sens.round(4).to_dict(orient="records"),
        "elegido": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    detalle["ahorro_relativo"] = round(
        1 - detalle["costo_por_solicitud_optimo"] / detalle["costo_por_solicitud_umbral_050"],
        4,
    )
    UMBRAL_PATH.write_text(
        json.dumps(detalle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"umbral elegido sobre {len(oof):,} predicciones OOF: {optimo:.4f}")
    print(tabla.to_string(index=False))
    print(f"  ahorro frente a 0.5: {detalle['ahorro_relativo']:.2%}")
    print(f"  {UMBRAL_PATH.relative_to(ROOT)}  ·  {figura.relative_to(ROOT)}")
    return detalle


def _verificar_test_congelado(test: pd.DataFrame) -> None:
    meta = json.loads(SPLIT_META.read_text(encoding="utf-8"))
    actual = fingerprint(test[ID_COL])
    if actual != meta["test_fingerprint"]:
        raise TestCongelado(
            "El test no es el que se congeló en la fase 2 "
            f"(fingerprint {actual} ≠ {meta['test_fingerprint']}). Se aborta la evaluación."
        )


def cargar_modelo(version: str = "latest"):
    mlflow.set_tracking_uri(MLFLOW_URI)
    return mlflow.lightgbm.load_model(f"models:/{REGISTERED_MODEL}/{resolver_version(version)}")


def evaluar_test(version: str = "latest") -> dict:
    """Paso 2: única lectura del test, con verificación de congelación y del modelo."""
    if not TEST_PATH.exists():
        raise SystemExit("No existe test.parquet. Ejecuta antes: python -m ayni split")
    if not UMBRAL_PATH.exists():
        raise SystemExit("No hay umbral elegido. Ejecuta: python -m ayni evaluate")

    umbral_info = json.loads(UMBRAL_PATH.read_text(encoding="utf-8"))
    optimo = umbral_info["umbral_optimo"]

    test = pd.read_parquet(TEST_PATH)
    _verificar_test_congelado(test)

    vocabulario = (
        json.loads(CATEGORIAS_PATH.read_text(encoding="utf-8")) if CATEGORIAS_PATH.exists() else {}
    )
    X = test.drop(columns=[c for c in (ID_COL, TARGET) if c in test.columns])
    X = X.drop(columns=["CODE_GENDER"], errors="ignore")
    for columna in X.columns:
        es_categorica = X[columna].dtype == object or str(X[columna].dtype) in {
            "boolean",
            "category",
        }
        if es_categorica and columna not in vocabulario:
            X[columna] = X[columna].astype("category")
    X = aplicar_categorias(X, vocabulario)

    modelo = cargar_modelo(version)
    version = resolver_version(version)
    proba = modelo.predict_proba(X)[:, 1]
    y = test[TARGET].to_numpy()

    filas = {}
    for nombre, corte in (("umbral_050", 0.5), ("umbral_optimo", optimo)):
        filas[nombre] = evaluar(y, proba, corte).a_dict()

    roc_auc = round(float(roc_auc_score(y, proba)), 4)
    resultado = {
        "version_modelo": version,
        "fingerprint_test": fingerprint(test[ID_COL]),
        "n_test": int(len(test)),
        "roc_auc": roc_auc,
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "umbral_optimo": optimo,
        **filas,
        "evaluado": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    EVALUACION_PATH.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nevaluación en test ({len(test):,} filas, fingerprint verificado)")
    print(f"  ROC-AUC {resultado['roc_auc']:.4f}  ·  PR-AUC {resultado['pr_auc']:.4f}")
    for nombre in ("umbral_050", "umbral_optimo"):
        fila = resultado[nombre]
        print(
            f"  {nombre:<14} umbral {fila['umbral']:.3f}  "
            f"aprobación {fila['tasa_aprobacion']:.1%}  "
            f"impago entre aprobados {fila['impago_entre_aprobados']:.1%}  "
            f"costo total {fila['costo']:,.0f}"
        )
    print(f"  {EVALUACION_PATH.relative_to(ROOT)}")

    if roc_auc < ROC_AUC_MINIMO:
        raise CalidadInsuficiente(
            f"ROC-AUC {roc_auc:.4f} por debajo del mínimo {ROC_AUC_MINIMO}: "
            "el modelo no se publica."
        )
    return resultado
