"""Tests de la auditoría de sesgo: métricas por grupo y regla del 80 %."""

import numpy as np
import pandas as pd
import pytest

from ayni.fairness import MIN_GRUPO, _grupo_por_edad, _metricas_grupo, brecha_80


def cartera() -> pd.DataFrame:
    """Tres grupos: uno grande con mora alta, otro grande con mora baja y uno diminuto."""
    filas = []
    # se aprueba con proba <= 0.16: 0.05 = aprobado, 0.9 = rechazado
    # grupo "A": 80 buenos y 20 morosos; se aprueba a 40 (los 20 morosos incluidos)
    for i in range(100):
        aprobado = i < 40
        filas.append(
            {
                "grupo": "A",
                "proba": 0.05 if aprobado else 0.9,
                "umbral": 0.16,
                "TARGET": int(i < 20),
            }
        )
    # grupo "B": 5 morosos, todos rechazados; se aprueba a 90 de los 95 buenos
    for i in range(100):
        rechazado = i < 5 or i >= 95
        filas.append(
            {
                "grupo": "B",
                "proba": 0.9 if rechazado else 0.05,
                "umbral": 0.16,
                "TARGET": int(i < 5),
            }
        )
    # grupo "C": 2 personas, ambas aprobadas y sin impago
    for _ in range(2):
        filas.append({"grupo": "C", "proba": 0.05, "umbral": 0.16, "TARGET": 0})
    return pd.DataFrame(filas)


def test_tasa_de_aprobacion_por_grupo():
    tabla = _metricas_grupo(cartera(), "grupo").set_index("grupo")

    assert tabla.loc["A", "tasa_aprobacion"] == pytest.approx(0.40)
    assert tabla.loc["B", "tasa_aprobacion"] == pytest.approx(0.90)
    assert tabla.loc["C", "n"] == 2


def test_mora_se_mide_entre_los_aprobados_no_sobre_el_total():
    tabla = _metricas_grupo(cartera(), "grupo").set_index("grupo")

    # en A se aprueba a 40: los 20 primeros (impagan) y 20 que pagan → mora 50 %
    assert tabla.loc["A", "mora_entre_aprobados"] == pytest.approx(0.5)
    # en B se aprueba a 90, de los cuales ninguno impaga (los 5 morosos quedan fuera)
    assert tabla.loc["B", "mora_entre_aprobados"] == pytest.approx(0.0)


def test_aprobacion_entre_quienes_habrian_pagado():
    tabla = _metricas_grupo(cartera(), "grupo").set_index("grupo")

    # A: de los 80 que pagan, aprueba a 20 → 25 %
    assert tabla.loc["A", "aprobacion_pagadores"] == pytest.approx(0.25)
    # B: de los 95 que pagan, aprueba a 90
    assert tabla.loc["B", "aprobacion_pagadores"] == pytest.approx(90 / 95, abs=1e-4)


def test_la_regla_del_80_ignora_grupos_diminutos():
    tabla = _metricas_grupo(cartera(), "grupo")
    # C tiene n=2 y ratio 1.11; si contara, el mínimo seguiría siendo 0.44 de A,
    # así que se comprueba con un caso donde el grupo chico sería el mínimo
    tabla_chica = pd.DataFrame(
        {
            "grupo": ["A", "B", "C"],
            "n": [100, 100, MIN_GRUPO - 1],
            "tasa_aprobacion": [0.9, 0.5, 0.1],
            "ratio_vs_mejor": [1.0, 0.55, 0.11],
        }
    )

    assert brecha_80(tabla_chica) == pytest.approx(0.55)
    assert brecha_80(tabla) == pytest.approx(0.40 / 0.90, abs=1e-4)


def test_tramos_de_edad_no_dejan_huecos():
    edades = pd.Series([20, 30, 40, 50, 60, 70])
    tramos = _grupo_por_edad(edades)

    assert tramos.tolist() == ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]


def test_un_grupo_sin_aprobados_no_rompe_la_mora():
    datos = pd.DataFrame(
        {"grupo": ["A", "A"], "proba": [0.9, 0.9], "umbral": 0.16, "TARGET": [0, 1]}
    )
    tabla = _metricas_grupo(datos, "grupo")

    assert tabla.loc[0, "tasa_aprobacion"] == 0.0
    assert tabla.loc[0, "mora_entre_aprobados"] == 0.0
    assert np.isfinite(tabla.loc[0, "ratio_vs_mejor"])
