"""Tests de la simulación de producción y del criterio de drift."""

import numpy as np
import pandas as pd
import pytest

from ayni.drift import _hay_drift, simular_produccion


def cartera(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": rng.uniform(50_000, 400_000, n),
            "AMT_CREDIT": rng.uniform(50_000, 900_000, n),
            "AMT_ANNUITY": rng.uniform(5_000, 50_000, n),
            "CNT_FAM_MEMBERS": rng.integers(1, 5, n).astype(float),
            "EXT_SOURCE_2": rng.random(n),
            "age_years": rng.uniform(20, 70, n),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
        }
    )


def test_altera_lo_que_dice_alterar():
    base = cartera()
    produccion, cambios = simular_produccion(base)

    assert produccion["AMT_INCOME_TOTAL"].mean() == pytest.approx(
        base["AMT_INCOME_TOTAL"].mean() * 1.30
    )
    assert produccion["age_years"].mean() == pytest.approx(base["age_years"].mean() + 4)
    assert produccion["EXT_SOURCE_2"].mean() < base["EXT_SOURCE_2"].mean()
    assert set(cambios) >= {"AMT_INCOME_TOTAL", "EXT_SOURCE_2", "age_years", "NAME_CONTRACT_TYPE"}


def test_no_toca_lo_que_no_debe_tocar():
    base = cartera()
    produccion, _ = simular_produccion(base)

    for columna in ("AMT_CREDIT", "AMT_ANNUITY", "CNT_FAM_MEMBERS"):
        pd.testing.assert_series_equal(produccion[columna], base[columna], check_names=False)


def test_la_mezcla_de_producto_cambia():
    base = cartera(2000)
    produccion, _ = simular_produccion(base)

    antes = (base["NAME_CONTRACT_TYPE"] == "Cash loans").mean()
    despues = (produccion["NAME_CONTRACT_TYPE"] == "Cash loans").mean()

    assert antes - despues == pytest.approx(0.27, abs=0.05)


def test_los_ratios_se_recalculan_con_el_nuevo_ingreso():
    base = cartera()
    produccion, _ = simular_produccion(base)

    esperado = produccion["AMT_CREDIT"] / produccion["AMT_INCOME_TOTAL"]
    assert np.allclose(produccion["credit_income_ratio"], esperado)
    # si no se recalculara, el ratio se habría reducido un 30 %
    assert not np.allclose(
        produccion["credit_income_ratio"], base["AMT_CREDIT"] / base["AMT_INCOME_TOTAL"]
    )


def test_el_criterio_es_distancia_mayor_que_umbral():
    assert _hay_drift({"distancia": 0.2, "umbral": 0.1})
    assert not _hay_drift({"distancia": 0.05, "umbral": 0.1})
    assert not _hay_drift({"distancia": 0.1, "umbral": 0.1})


def test_ext_source_queda_recortado_entre_0_y_1():
    base = cartera()
    base["EXT_SOURCE_2"] = 0.05
    produccion, _ = simular_produccion(base)

    assert produccion["EXT_SOURCE_2"].between(0, 1).all()
    assert produccion["EXT_SOURCE_2"].max() == pytest.approx(0.0)
