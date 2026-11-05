"""Tests de las transformaciones de la fase 2 (no necesitan el dataset de Kaggle)."""

import pandas as pd
import pytest

from ayni.features import EMPLOYED_SENTINEL, add_ratios, drop_constant, fix_employed
from ayni.split import fingerprint


def test_centinela_pasa_a_nulo_y_deja_el_flag():
    df = pd.DataFrame(
        {
            "DAYS_EMPLOYED": [EMPLOYED_SENTINEL, -365.25],
            "DAYS_BIRTH": [-10_000, -10_000],
        }
    )
    out = fix_employed(df)

    assert "DAYS_EMPLOYED" not in out.columns
    assert "DAYS_BIRTH" not in out.columns
    assert out["employed_anomaly"].tolist() == [1, 0]
    assert out["employed_years"].isna().tolist() == [True, False]
    assert out["employed_years"].iloc[1] == pytest.approx(1.0)


def test_ratios_no_revientan_con_ingreso_cero():
    df = pd.DataFrame(
        {
            "AMT_CREDIT": [100.0],
            "AMT_INCOME_TOTAL": [0.0],
            "AMT_ANNUITY": [10.0],
            "AMT_GOODS_PRICE": [80.0],
            "CNT_FAM_MEMBERS": [0.0],
            "CNT_CHILDREN": [0],
            "DAYS_BIRTH": [-3652.5],
        }
    )
    out = add_ratios(df)

    assert out["credit_income_ratio"].isna().all()
    assert out["income_per_person"].isna().all()
    assert out["goods_credit_ratio"].iloc[0] == pytest.approx(0.8)
    assert out["age_years"].iloc[0] == pytest.approx(10.0)


def test_drop_constant_no_toca_id_ni_target():
    df = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1], "CONST": [5, 5], "VAR": [1, 2]})
    out, descartadas = drop_constant(df)

    assert descartadas == ["CONST"]
    assert {"SK_ID_CURR", "TARGET", "VAR"} == set(out.columns)


def test_fingerprint_del_test_ignora_el_orden_y_detecta_cambios():
    estable = fingerprint(pd.Series([1, 2, 3]))

    assert estable == fingerprint(pd.Series([3, 1, 2]))
    assert estable != fingerprint(pd.Series([1, 2, 4]))
    assert len(estable) == 16
