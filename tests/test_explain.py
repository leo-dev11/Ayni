"""Tests de explicabilidad: las razones tienen que sumar, ordenarse y tener signo."""

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from ayni.explain import Razon, construir_explicador, preparar_matriz, razones, valores_shap


@pytest.fixture(scope="module")
def modelo_y_datos():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {f"f{i}": rng.normal(size=n) for i in range(5)} | {"ruido": rng.normal(size=n)}
    )
    logit = 2.0 * X["f0"] - 1.5 * X["f1"] + 0.5 * X["f2"]
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    modelo = LGBMClassifier(n_estimators=30, min_child_samples=10, verbose=-1)
    modelo.fit(X, y)
    return modelo, X, y


def test_preparar_matriz_quita_id_target_y_genero():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "CODE_GENDER": ["F", "M"],
            "f0": [0.1, 0.2],
        }
    )
    X = preparar_matriz(df)

    assert list(X.columns) == ["f0"]


def test_shap_es_aditivo_con_la_salida_del_modelo(modelo_y_datos):
    """propiedad central: base + suma de SHAP = score crudo del modelo."""
    modelo, X, _ = modelo_y_datos
    explicador = construir_explicador(modelo)

    valores = valores_shap(explicador, X)
    base = float(np.ravel(explicador.expected_value)[-1])
    score = modelo.predict(X, raw_score=True)

    assert np.allclose(valores.sum(axis=1) + base, score, atol=1e-5)


def test_razones_devuelve_cinco_ordenadas_por_impacto(modelo_y_datos):
    modelo, X, _ = modelo_y_datos
    explicador = construir_explicador(modelo)

    lista = razones(explicador, X.iloc[[0]], top=5)

    assert len(lista) == 5
    assert all(isinstance(r, Razon) for r in lista)
    magnitudes = [abs(r.shap) for r in lista]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_direccion_coincide_con_el_signo(modelo_y_datos):
    modelo, X, _ = modelo_y_datos
    explicador = construir_explicador(modelo)

    for razon in razones(explicador, X.iloc[[3]]):
        esperado = "aumenta_riesgo" if razon.shap > 0 else "reduce_riesgo"
        assert razon.direccion == esperado


def test_diccionario_serializable_para_la_api(modelo_y_datos):
    modelo, X, _ = modelo_y_datos
    explicador = construir_explicador(modelo)

    razon = razones(explicador, X.iloc[[1]], top=1)[0].a_dict()

    assert set(razon) == {"feature", "valor", "shap", "direccion"}
    assert isinstance(razon["shap"], float)
