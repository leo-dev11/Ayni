"""Monitoreo de drift: ¿se movió la cartera respecto de lo que el modelo aprendió?

Se simula una cartera "de producción" a partir de `train` (nunca del test, que sigue congelado)
alterando a propósito algunas features, y se compara contra la referencia con Evidently.

El reporte se limita a las **features que el modelo realmente usa** (top 30 por |SHAP|): monitorear
185 columnas produce un HTML enorme y ruido en la lectura, y en un banco se monitorean las variables
del score, no todas las del dataset.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: se ejecuta desde el CLI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from ayni.config import REPORTS_DIR, ROOT
from ayni.split import TRAIN_PATH

DRIFT_HTML = REPORTS_DIR / "drift_reporte.html"
DRIFT_FIGURA = ROOT / "reports" / "figures" / "11_drift_distancias.png"
DRIFT_JSON = REPORTS_DIR / "drift_resumen.json"
IMPORTANCIA_CSV = REPORTS_DIR / "shap_importancia_global.csv"
TOP_MONITOREO = 30
# Evidently 0.7 compara **distancias** con un umbral (drift si la distancia lo supera),
# no p-valores. El umbral real se lee de cada métrica; este es solo el que usa por defecto.
UMBRAL_DRIFT = 0.1

# Columnas que se monitorizan siempre, aunque SHAP no las haya puesto arriba.
MONITOREO_MINIMO = (
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "age_years",
    "credit_income_ratio",
    "annuity_income_ratio",
    "income_per_person",
    "bureau_debt_ratio",
    "prev_refusal_rate",
    "inst_late_rate",
    "NAME_CONTRACT_TYPE",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
)

# Cambios que la alteración de una feature arrastra consigo (ratios derivados).
COLATERALES = ("credit_income_ratio", "annuity_income_ratio", "income_per_person")


def columnas_monitoreo() -> list[str]:
    """Top N por |SHAP| según la fase 6, ampliado con las columnas de negocio fijas."""
    columnas = set(MONITOREO_MINIMO)
    if IMPORTANCIA_CSV.exists():
        tabla = pd.read_csv(IMPORTANCIA_CSV)
        columna = "feature" if "feature" in tabla.columns else tabla.columns[0]
        columnas.update(tabla[columna].head(TOP_MONITOREO).tolist())
    return sorted(columnas)


def simular_produccion(df: pd.DataFrame, *, seed: int = 42) -> tuple[pd.DataFrame, dict[str, str]]:
    """Altera la cartera como lo haría un cambio de negocio o de mercado.

    Cada alteración es deliberada y está documentada: el objetivo es comprobar que el detector
    encuentra exactamente lo que se movió, y no inventa drift donde no lo hay.
    """
    rng = np.random.default_rng(seed)
    produccion = df.copy()
    cambios: dict[str, str] = {}

    produccion["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"] * 1.30
    cambios["AMT_INCOME_TOTAL"] = "ingresos +30 % (mejora salarial general)"

    produccion["EXT_SOURCE_2"] = (df["EXT_SOURCE_2"] - 0.15).clip(0, 1)
    cambios["EXT_SOURCE_2"] = "score externo castigado −0.15 (empeoró el buró)"

    produccion["age_years"] = df["age_years"] + 4
    cambios["age_years"] = "cartera 4 años mayor (envejeció la base de clientes)"

    # los ratios derivados se recalculan: si no, la producción quedaría inconsistente
    # (ingreso subido pero ratio con el ingreso viejo) y el drift de esas columnas no existiría.
    produccion["credit_income_ratio"] = produccion["AMT_CREDIT"] / produccion["AMT_INCOME_TOTAL"]
    produccion["annuity_income_ratio"] = produccion["AMT_ANNUITY"] / produccion["AMT_INCOME_TOTAL"]
    produccion["income_per_person"] = produccion["AMT_INCOME_TOTAL"] / produccion[
        "CNT_FAM_MEMBERS"
    ].replace(0, np.nan)
    cambios["credit_income_ratio"] = "derivado: se recalcula con el nuevo ingreso"
    cambios["annuity_income_ratio"] = "derivado: se recalcula con el nuevo ingreso"
    cambios["income_per_person"] = "derivado: se recalcula con el nuevo ingreso"

    # mezcla de producto: 30 % de los créditos en efectivo pasan a revolving
    es_cash = df["NAME_CONTRACT_TYPE"].astype(str).eq("Cash loans")
    elegidos = es_cash & (rng.random(len(df)) < 0.30)
    produccion["NAME_CONTRACT_TYPE"] = df["NAME_CONTRACT_TYPE"].astype(object)
    produccion.loc[elegidos, "NAME_CONTRACT_TYPE"] = "Revolving loans"
    cambios["NAME_CONTRACT_TYPE"] = "30 % de Cash loans → Revolving loans"

    return produccion, cambios


def _drift_por_columna(snapshot) -> dict[str, dict[str, object]]:
    """Distancia de drift por columna, con su umbral y el método usado.

    Evidently 0.7 reporta `ValueDrift` como **distancia** (Wasserstein normalizada para números,
    Jensen-Shannon para categóricas): hay drift cuando la distancia **supera** el umbral. Leerlo
    como p-valor invierte el diagnóstico entero.
    """
    resultados: dict[str, dict[str, object]] = {}
    for metric in snapshot.dict().get("metrics", []):
        config = metric.get("config", {})
        if config.get("type", "").endswith("ValueDrift"):
            resultados[config["column"]] = {
                "distancia": float(metric["value"]),
                "umbral": float(config.get("threshold", UMBRAL_DRIFT)),
                "metodo": str(config.get("method", "")),
            }
    return resultados


def _hay_drift(info: dict[str, object]) -> bool:
    return float(info["distancia"]) > float(info["umbral"])


def _figura(distancias: dict[str, dict], cambios: dict[str, str]) -> Path:
    """Barras de distancia por feature: la que supera la línea está en drift."""
    orden = sorted(distancias, key=lambda c: float(distancias[c]["distancia"]))
    valores = [float(distancias[c]["distancia"]) for c in orden]
    umbral = float(distancias[orden[0]]["umbral"]) if orden else UMBRAL_DRIFT
    alteradas = [c in cambios for c in orden]

    fig, ax = plt.subplots(figsize=(8, 0.28 * len(orden) + 1.6))
    colores = ["#C44536" if v > umbral else "#2E6F95" for v in valores]
    barras = ax.barh(
        [f"{c} ★" if a else c for c, a in zip(orden, alteradas, strict=True)],
        valores,
        color=colores,
        height=0.7,
    )
    for barra, valor in zip(barras, valores, strict=True):
        ax.text(
            valor + 0.01,
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.2f}",
            va="center",
            fontsize=7,
        )
    ax.axvline(umbral, color="#333333", ls="--", lw=1)
    ax.text(umbral + 0.01, -0.9, f"umbral {umbral:g}", fontsize=7, color="#333333")
    ax.set_xlabel("distancia entre referencia y producción")
    ax.set_title("Drift por feature · rojo = detectado, ★ = alterado a propósito", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    DRIFT_FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DRIFT_FIGURA, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return DRIFT_FIGURA


def ejecutar(*, muestra: int = 3000, seed: int = 42) -> dict:
    train = pd.read_parquet(TRAIN_PATH)
    columnas = columnas_monitoreo()

    rng = np.random.default_rng(seed)
    filas = rng.choice(len(train), size=min(muestra * 2, len(train)), replace=False)
    referencia = train.iloc[filas[:muestra]]
    base_produccion = train.iloc[filas[muestra : muestra * 2]]

    produccion, cambios = simular_produccion(base_produccion, seed=seed)

    referencia = referencia[columnas].reset_index(drop=True)
    produccion = produccion[columnas].reset_index(drop=True)

    reporte = Report(metrics=[DataDriftPreset()])
    snapshot = reporte.run(reference_data=referencia, current_data=produccion)
    DRIFT_HTML.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(DRIFT_HTML))

    distancias = _drift_por_columna(snapshot)
    detectadas = {col for col, info in distancias.items() if _hay_drift(info)}
    esperadas = set(cambios) | set(COLATERALES)
    acertadas = detectadas & esperadas
    no_detectadas = esperadas - detectadas
    falsos_positivos = detectadas - esperadas

    resumen = {
        "columnas_monitoreadas": len(columnas),
        "filas_referencia": int(len(referencia)),
        "filas_produccion": int(len(produccion)),
        "criterio": "distancia > umbral (Evidently 0.7)",
        "alteraciones": cambios,
        "columnas_con_drift": sorted(detectadas),
        "esperadas": sorted(esperadas),
        "detectadas_esperadas": sorted(acertadas),
        "no_detectadas": sorted(no_detectadas),
        "falsos_positivos": sorted(falsos_positivos),
        "share_columnas_con_drift": round(len(detectadas) / len(columnas), 4),
        "distancias": {
            c: {**info, "distancia": round(float(info["distancia"]), 4)}
            for c, info in sorted(distancias.items())
        },
        "figura": _figura(distancias, cambios).relative_to(ROOT).as_posix(),
        "reporte": DRIFT_HTML.relative_to(ROOT).as_posix(),
        "generado": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    DRIFT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_JSON.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"referencia: {len(referencia):,} filas · producción: {len(produccion):,}")
    print(f"columnas monitoreadas: {len(columnas)}")
    print(f"\nalteraciones plantadas ({len(cambios)}):")
    for columna, descripcion in cambios.items():
        info = distancias.get(columna, {})
        distancia = float(info.get("distancia", float("nan")))
        estado = "detectada" if columna in detectadas else "NO detectada"
        print(
            f"  {columna:<24} dist={distancia:>6.3f} (>{info.get('umbral', '?')}) "
            f"{estado:<12} {descripcion}"
        )
    if no_detectadas:
        print(f"\nesperadas no detectadas: {sorted(no_detectadas)}")
    if falsos_positivos:
        print(f"columnas marcadas sin haberlas tocado: {sorted(falsos_positivos)}")
    print(f"\nreporte: {DRIFT_HTML.relative_to(ROOT)}")
    print(f"resumen: {DRIFT_JSON.relative_to(ROOT)}")
    return resumen
