"""Tests de los esquemas: el pipeline tiene que detenerse ante datos imposibles."""

from pathlib import Path

import pandas as pd
import pytest

from ayni.schema import (
    REQUIRED_FEATURES,
    DataValidationError,
    validate_features,
    validate_raw,
    validate_split,
)

FIXTURES = Path(__file__).parent / "fixtures"


def features_minimas(**overrides) -> pd.DataFrame:
    """DataFrame de features que cumple el contrato, para romperlo a propósito."""
    base = {
        "SK_ID_CURR": [100001, 100002, 100003],
        "TARGET": [0, 1, 0],
        "credit_income_ratio": [2.5, 1.0, 3.0],
        "annuity_income_ratio": [0.14, 0.13, 0.12],
        "goods_credit_ratio": [0.9, 0.95, 1.0],
        "income_per_person": [90_000.0, 40_000.0, 50_000.0],
        "children_ratio": [0.0, 0.33, 0.0],
        "age_years": [34.0, 41.0, 52.0],
        "employed_years": [2.0, None, 10.0],
        "employed_anomaly": [0, 1, 0],
        "bureau_n_credits": [3, None, 1],
        "bureau_debt_ratio": [0.4, None, 0.0],
        "bureau_max_dpd": [0, None, 12],
        "prev_n_applications": [2, 5, 1],
        "prev_approval_rate": [1.0, 0.6, 0.0],
        "prev_refusal_rate": [0.0, 0.4, 1.0],
        "prev_repeater_share": [0.0, 1.0, 0.5],
        "inst_n_payments": [24, 60, 12],
        "inst_late_rate": [0.0, 0.25, 0.5],
        "inst_payment_ratio": [1.0, 0.98, 1.02],
        "cc_n_months": [None, 12, None],
        "cc_utilization_mean": [None, 0.6, None],
        "cc_delinquent_month_rate": [None, 0.0, None],
        "bureau_share_credit_card": [0.33, None, 0.0],
        "bureau_share_consumer": [0.66, None, 1.0],
    }
    return pd.DataFrame({**base, **overrides})


# ── entrada cruda ───────────────────────────────────────────────────────────────


def test_csv_valido_pasa():
    df = pd.read_csv(FIXTURES / "application_train_ok.csv")
    assert len(validate_raw(df)) == 6


def test_csv_corrupto_falla_y_dice_por_que():
    df = pd.read_csv(FIXTURES / "application_train_corrupt.csv")

    with pytest.raises(DataValidationError) as error:
        validate_raw(df)

    mensaje = str(error.value)
    assert "application_train" in mensaje
    # los cinco defectos plantados, todos reportados a la vez
    assert "TARGET" in mensaje
    assert "AMT_INCOME_TOTAL" in mensaje
    assert "DAYS_BIRTH" in mensaje
    assert "CODE_GENDER" in mensaje
    assert "SK_ID_CURR" in mensaje


# ── features ───────────────────────────────────────────────────────────────────


def test_features_minimas_pasan():
    assert len(validate_features(features_minimas())) == 3


def test_features_sin_columna_obligatoria_falla():
    df = features_minimas().drop(columns=["inst_late_rate"])

    with pytest.raises(DataValidationError, match="Faltan columnas obligatorias"):
        validate_features(df)


def test_edad_imposible_falla():
    with pytest.raises(DataValidationError, match="age_years"):
        validate_features(features_minimas(age_years=[34.0, -2.0, 52.0]))


def test_target_fuera_de_rango_falla():
    with pytest.raises(DataValidationError, match="TARGET"):
        validate_features(features_minimas(TARGET=[0, 2, 0]))


def test_columna_constante_falla():
    with pytest.raises(DataValidationError, match="sin_columnas_constantes"):
        validate_features(features_minimas(children_ratio=[0.0, 0.0, 0.0]))


def test_tasa_fuera_de_rango_falla():
    with pytest.raises(DataValidationError, match="prev_refusal_rate"):
        validate_features(features_minimas(prev_refusal_rate=[0.0, 1.4, 0.2]))


def test_todas_las_features_obligatorias_estan_en_el_esquema():
    assert set(REQUIRED_FEATURES) <= set(features_minimas().columns)


# ── particiones ────────────────────────────────────────────────────────────────


def test_split_no_exige_columnas_constantes():
    # FLAG_MOBIL es constante en el test real: eso no puede tumbar la validación.
    df = features_minimas(FLAG_MOBIL=[1, 1, 1])
    assert len(validate_split(df)) == 3
