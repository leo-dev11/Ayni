"""Tests de la decisión por costo: la aritmética que define la política del banco."""

import numpy as np
import pytest

from ayni.threshold import (
    COSTO_FN,
    COSTO_FP,
    curva_costo,
    evaluar,
    sensibilidad,
    umbral_optimo,
)


def toy():
    """4 casos: dos sanos, dos que impagan. Probabilidades ordenadas a mano.

    Se aprueba si `P(impago) <= umbral`, así que con umbral 0.5 se aprueban los dos
    de menor riesgo (índices 0 y 3: probabilidades 0.1 y 0.2).
    """
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.9, 0.2])
    return y, proba


def test_umbral_es_un_techo_de_riesgo_y_no_un_piso():
    y, proba = toy()
    e = evaluar(y, proba, 0.5)

    # aprobados: p=0.1 (paga) y p=0.2 (impaga) → 1 acierto y 1 falso negativo
    assert e.aprobados == 2
    assert e.falsos_negativos == 1
    # rechazados: p=0.6 (paga, falso positivo) y p=0.9 (impaga, acierto)
    assert e.falsos_positivos == 1


def test_costo_usa_la_matriz_documentada():
    y, proba = toy()
    e = evaluar(y, proba, 0.5)

    assert COSTO_FN == 5.0
    assert COSTO_FP == 1.0
    assert e.costo == COSTO_FN * 1 + COSTO_FP * 1
    assert e.costo_por_solicitud == pytest.approx(1.5)


def test_aprobar_a_un_moroso_cuesta_cinco_veces_mas_que_rechazar_a_uno_bueno():
    y = np.array([1, 0])
    proba = np.array([0.9, 0.9])

    aprueba_a_todos = evaluar(y, proba, 1.0)  # FN = 1
    rechaza_a_todos = evaluar(y, proba, 0.0)  # FP = 1

    assert aprueba_a_todos.falsos_negativos == 1
    assert rechaza_a_todos.falsos_positivos == 1
    assert aprueba_a_todos.costo == pytest.approx(5 * rechaza_a_todos.costo)


def test_impago_entre_aprobados():
    y, proba = toy()
    e = evaluar(y, proba, 0.5)

    # de los 2 aprobados, uno no paga
    assert e.impago_entre_aprobados == pytest.approx(0.5)


def test_sin_aprobados_no_hay_division_por_cero():
    y, proba = toy()
    e = evaluar(y, proba, 0.0)

    assert e.aprobados == 0
    assert e.tasa_aprobacion == 0.0
    assert e.impago_entre_aprobados == 0.0


def test_subir_el_umbral_aprueba_mas():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    proba = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.2, 2000), 0, 1)

    aprobaciones = [evaluar(y, proba, u).aprobados for u in np.linspace(0.05, 0.95, 20)]

    assert all(a <= b for a, b in zip(aprobaciones[:-1], aprobaciones[1:], strict=True))


def test_el_umbral_optimo_es_el_minimo_de_la_curva():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 5000)
    proba = np.clip(0.15 + 0.7 * y + rng.normal(0, 0.2, 5000), 0, 1)

    optimo, curva = umbral_optimo(y, proba)

    assert curva["costo_por_solicitud"].min() == pytest.approx(
        curva.loc[curva["umbral"] == optimo, "costo_por_solicitud"].iloc[0]
    )
    assert curva["umbral"].between(0, 1).all()
    assert curva["tasa_aprobacion"].between(0, 1).all()


def test_con_modelo_calibrado_el_optimo_sigue_1_sobre_ratio_mas_1():
    """Si el modelo está calibrado, el umbral que iguala costos es 1/(FN+FP por ratio)."""
    rng = np.random.default_rng(7)
    proba = rng.beta(1, 11, 60_000)  # ~8 % de mora, bien calibrada
    y = rng.binomial(1, proba)

    tabla = sensibilidad(y, proba, ratios=(5, 10))

    assert tabla.loc[0, "umbral_optimo"] == pytest.approx(1 / 6, abs=0.02)
    assert tabla.loc[1, "umbral_optimo"] == pytest.approx(1 / 11, abs=0.02)


def test_la_curva_lleva_todas_las_columnas_que_se_grafican():
    y, proba = toy()
    curva = curva_costo(y, proba, umbrales=[0.2, 0.5, 0.8])

    assert set(curva.columns) == {
        "umbral",
        "costo_por_solicitud",
        "tasa_aprobacion",
        "impago_entre_aprobados",
        "falsos_negativos",
        "falsos_positivos",
    }
    assert len(curva) == 3
