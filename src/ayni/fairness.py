"""Auditoría de sesgo: quién recibe el crédito y quién no, con la política real.

Se usan las predicciones **out-of-fold** de `train` (no el test, que sigue congelado): son las
mismas con las que se eligió el umbral, así que miden la política que se va a aplicar de verdad.

Se reportan dos cosas distintas, que se confunden con facilidad:

- **Tasa de aprobación** por grupo: si el modelo aprueba sistemáticamente menos a un grupo.
- **Mora entre aprobados** por grupo: si el riesgo que asume el banco es distinto. Una diferencia
  de aprobación solo es discriminatoria si el grupo no tiene peor riesgo que lo justifique.
- **Calibración por grupo** (mora predicha vs. observada): si a cada grupo el modelo le asigna el
  riesgo que de verdad tiene. Es la prueba de que una brecha "la explica el riesgo": un modelo que
  sobrestima el riesgo de un grupo lo penaliza aunque la mora media sea distinta.

Con tasas de impago distintas entre grupos, un modelo calibrado **no puede** aprobar igual a los
buenos pagadores de todos ellos (calibración e igualdad de oportunidades son incompatibles salvo
con un modelo perfecto). Por eso `aprobacion_pagadores` se reporta, pero no se lee sola.

La *regla del 80 %* (aprobación de cada grupo ≥ 80 % de la del más favorecido) se incluye
como referencia estándar de la industria, no como veredicto legal.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: se ejecuta desde el CLI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ayni.config import ID_COL, PROCESSED_DIR, REPORTS_DIR, ROOT, TARGET

OOF_DIR = PROCESSED_DIR / "oof"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
UMBRAL_PATH = REPORTS_DIR / "umbral.json"
SESGO_JSON = REPORTS_DIR / "sesgo_decision.json"
SESGO_FIGURA = ROOT / "reports" / "figures" / "12_sesgo_decision.png"
CORRIDA = "lgbm-ajustado"
TRAMOS_EDAD = [0, 25, 35, 45, 55, 65, 100]
# Grupos por debajo de este tamaño no entran en el ratio (XNA tiene 2 casos y lo distorsiona).
MIN_GRUPO = 30


def _grupo_por_edad(edades: pd.Series) -> pd.Series:
    etiquetas = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    return pd.cut(edades, TRAMOS_EDAD, right=False, labels=etiquetas)


def _metricas_grupo(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Aprobación y mora por grupo. La mora se mide **entre los aprobados**, no sobre el total."""
    trabajo = df.assign(aprobado=df["proba"] <= df["umbral"])

    resumen = trabajo.groupby(columna, observed=True).agg(
        n=(TARGET, "size"),
        tasa_aprobacion=("aprobado", "mean"),
        tasa_impago=(TARGET, "mean"),
        mora_predicha=("proba", "mean"),
    )
    mora = (
        trabajo[trabajo["aprobado"]]
        .groupby(columna, observed=True)[TARGET]
        .mean()
        .rename("mora_entre_aprobados")
    )
    # métrica más exigente: entre quienes SÍ habrían pagado, a quién se aprueba
    buenos = (
        trabajo[trabajo[TARGET] == 0]
        .groupby(columna, observed=True)["aprobado"]
        .mean()
        .rename("aprobacion_pagadores")
    )
    tabla = (
        resumen.join(mora)
        .join(buenos)
        .fillna({"mora_entre_aprobados": 0.0, "aprobacion_pagadores": 0.0})
        .reset_index()
    )

    # calibración por grupo: observada − predicha; > 0 = el modelo subestima el riesgo del grupo
    tabla["desvio_calibracion"] = tabla["tasa_impago"] - tabla["mora_predicha"]

    for clave in (
        "tasa_aprobacion",
        "tasa_impago",
        "mora_predicha",
        "desvio_calibracion",
        "mora_entre_aprobados",
        "aprobacion_pagadores",
    ):
        tabla[clave] = tabla[clave].round(4)
    validos = tabla[tabla["n"] >= MIN_GRUPO]["tasa_aprobacion"]
    mejor = float(validos.max()) if not validos.empty else 0.0
    tabla["ratio_vs_mejor"] = (tabla["tasa_aprobacion"] / mejor).round(4) if mejor > 0 else 0.0
    return tabla


def brecha_80(tabla: pd.DataFrame) -> float:
    """Regla del 80 %: aprobación del grupo menos favorecido sobre el más favorecido.

    Solo entran grupos con `n >= MIN_GRUPO`: con dos casos, el azar decide la métrica.
    """
    validos = tabla[tabla["n"] >= MIN_GRUPO]
    return round(float(validos["ratio_vs_mejor"].min()), 4)


def _figura(por_genero: pd.DataFrame, por_edad: pd.DataFrame) -> Path:
    """Solo grupos con tamaño suficiente: XNA (n=2) es ruido y aplasta la escala."""
    por_genero = por_genero[por_genero["n"] >= MIN_GRUPO]
    por_edad = por_edad[por_edad["n"] >= MIN_GRUPO]
    SESGO_FIGURA.parent.mkdir(parents=True, exist_ok=True)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.6))

    for eje, tabla, columna, titulo in (
        (ejes[0], por_genero, "CODE_GENDER", "Género"),
        (ejes[1], por_edad, "tramo_edad", "Tramo de edad"),
    ):
        x = np.arange(len(tabla))
        eje.bar(
            x - 0.2,
            tabla["tasa_aprobacion"] * 100,
            width=0.4,
            color="#2E6F95",
            label="tasa de aprobación",
        )
        eje.bar(
            x + 0.2,
            tabla["mora_entre_aprobados"] * 100,
            width=0.4,
            color="#C44536",
            label="mora entre aprobados",
        )
        eje.set_xticks(x)
        eje.set_xticklabels(tabla[columna].astype(str), fontsize=8)
        eje.set_title(titulo, fontsize=9, fontweight="bold")
        eje.set_ylabel("%")
        eje.spines[["top", "right"]].set_visible(False)
        eje.grid(axis="y", alpha=0.25)
        eje.legend(frameon=False, fontsize=7)

    fig.suptitle(
        "Política de decisión por grupo (umbral por costo sobre predicciones OOF)", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(SESGO_FIGURA, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return SESGO_FIGURA


def ejecutar(corrida: str = CORRIDA) -> dict:
    umbral = json.loads(UMBRAL_PATH.read_text(encoding="utf-8"))["umbral_optimo"]
    oof = pd.read_parquet(OOF_DIR / f"{corrida}.parquet")
    # sin TARGET: la etiqueta viene del OOF y duplicarla rompería el merge
    features = pd.read_parquet(FEATURES_PATH, columns=[ID_COL, "CODE_GENDER", "age_years"])

    datos = oof.merge(features, on=ID_COL, how="left")
    datos = datos.rename(columns={"y": TARGET})
    datos["umbral"] = umbral
    datos["tramo_edad"] = _grupo_por_edad(datos["age_years"])

    por_genero = _metricas_grupo(datos, "CODE_GENDER")
    por_edad = _metricas_grupo(datos, "tramo_edad")
    figura = _figura(por_genero, por_edad)

    resultado = {
        "corrida": corrida,
        "umbral": umbral,
        "n": int(len(datos)),
        "genero": por_genero.to_dict(orient="records"),
        "edad": por_edad.to_dict(orient="records"),
        "regla_80_por_genero": brecha_80(por_genero),
        "regla_80_por_edad": brecha_80(por_edad),
        "figura": figura.relative_to(ROOT).as_posix(),
        "nota": (
            "CODE_GENDER no entra en el modelo; se mide porque el sesgo puede llegar por "
            "variables correlacionadas."
        ),
    }
    SESGO_JSON.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"umbral {umbral} · {len(datos):,} predicciones OOF\n")
    print("por género:")
    print(
        por_genero.to_string(
            index=False,
            columns=[
                "CODE_GENDER",
                "n",
                "tasa_aprobacion",
                "aprobacion_pagadores",
                "tasa_impago",
                "mora_predicha",
                "mora_entre_aprobados",
                "ratio_vs_mejor",
            ],
        )
    )
    print("\npor tramo de edad:")
    print(
        por_edad.to_string(
            index=False,
            columns=[
                "tramo_edad",
                "n",
                "tasa_aprobacion",
                "aprobacion_pagadores",
                "tasa_impago",
                "mora_predicha",
                "mora_entre_aprobados",
                "ratio_vs_mejor",
            ],
        )
    )
    print(
        f"\nregla del 80 %: género {resultado['regla_80_por_genero']:.2f} · "
        f"edad {resultado['regla_80_por_edad']:.2f}"
    )
    print(f"{SESGO_JSON.relative_to(ROOT)} · {figura.relative_to(ROOT)}")
    return resultado
