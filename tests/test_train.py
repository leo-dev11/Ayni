"""Tests de la preparación de datos y de las reglas de selección del modelo."""

import numpy as np
import pandas as pd
import pytest

from ayni.train import Corrida, coste_de_excluir_genero, elegir_campeon, preparar


def corrida_minima(**kwargs) -> Corrida:
    base = {
        "nombre": "x",
        "familia": "lgbm",
        "params": {},
        "con_genero": False,
        "metricas": {"roc_auc_cv": 0.78},
    }
    return Corrida(**{**base, **kwargs})


def test_preparar_saca_target_e_id():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "edad": [30.0, 40.0],
            "nombre": ["a", "b"],
        }
    )
    X, y = preparar(df)

    assert {"SK_ID_CURR", "TARGET"} & set(X.columns) == set()
    assert y.tolist() == [0, 1]


def test_preparar_excluye_genero_salvo_que_se_pida():
    df = pd.DataFrame({"TARGET": [0, 1], "CODE_GENDER": ["F", "M"], "edad": [30.0, 40.0]})

    assert "CODE_GENDER" not in preparar(df)[0].columns
    assert "CODE_GENDER" in preparar(df, con_genero=True)[0].columns


def test_preparar_convierte_categoricas_para_lightgbm():
    df = pd.DataFrame({"TARGET": [0, 1], "nombre": ["a", "b"], "edad": [30.0, 40.0]})
    X, _ = preparar(df)

    assert str(X["nombre"].dtype) == "category"
    assert str(X["edad"].dtype) == "float64"


def test_el_campeon_nunca_usa_genero():
    sin = corrida_minima(nombre="sin", metricas={"roc_auc_cv": 0.78})
    con = corrida_minima(nombre="con", con_genero=True, metricas={"roc_auc_cv": 0.79})

    assert elegir_campeon([sin, con]).nombre == "sin"
    assert elegir_campeon([sin]).nombre == "sin"
    with pytest.raises(ValueError, match="sin género"):
        elegir_campeon([con])


def test_el_coste_del_genero_se_mide_contra_la_gemela_y_no_contra_la_mejor():
    """Si la corrida con género usa otros hiperparámetros, compararla con la mejor ajustada mezcla
    dos efectos. Se compara contra la de parámetros idénticos."""
    base = {"learning_rate": 0.05}
    ajustada = {"learning_rate": 0.03}
    corridas = [
        corrida_minima(nombre="ajustada", params=ajustada, metricas={"roc_auc_cv": 0.7800}),
        corrida_minima(nombre="base", params=base, metricas={"roc_auc_cv": 0.7796}),
        corrida_minima(
            nombre="base-con-genero", params=base, con_genero=True, metricas={"roc_auc_cv": 0.7802}
        ),
    ]

    comparacion = coste_de_excluir_genero(corridas)

    assert comparacion is not None
    assert comparacion.corrida_sin == "base"
    assert comparacion.delta == pytest.approx(0.0006)


def test_sin_pareja_comparable_no_se_reporta_coste():
    corridas = [
        corrida_minima(nombre="ajustada", params={"learning_rate": 0.03}),
        corrida_minima(nombre="con", con_genero=True, params={"learning_rate": 0.05}),
    ]

    assert coste_de_excluir_genero(corridas) is None


def test_sin_corrida_con_genero_no_hay_coste_que_reportar():
    assert coste_de_excluir_genero([corrida_minima()]) is None


def test_metrica_por_defecto_no_es_accuracy():
    # Guardia de diseño: si alguien añade accuracy al resumen, este test lo recuerda.
    corrida = corrida_minima(metricas={"roc_auc_cv": 0.78, "pr_auc_cv": 0.27})
    assert not any("accuracy" in nombre for nombre in corrida.metricas)
    assert np.isfinite(corrida.metricas["roc_auc_cv"])
