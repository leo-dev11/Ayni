"""Tests de la API con un modelo diminuto, sin depender del artefacto real."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from lightgbm import LGBMClassifier

from ayni.api import COBERTURA_MINIMA, cargar_servicio, crear_app

COLUMNAS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "age_years",
    "EXT_SOURCE_2",
    *[f"extra_{i}" for i in range(15)],
]


@pytest.fixture(scope="module")
def cliente_api(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({c: rng.normal(0, 1, n) for c in COLUMNAS}).abs()
    X["EXT_SOURCE_2"] = rng.random(n)
    logit = -2 * X["EXT_SOURCE_2"] + 0.3 * X["AMT_CREDIT"] - 3
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    modelo = LGBMClassifier(n_estimators=40, min_child_samples=10, verbose=-1)
    modelo.fit(X, y)

    destino = Path(tmp_path_factory.mktemp("modelo"))
    modelo.booster_.save_model(str(destino / "booster.txt"))
    manifest = {
        "modelo": "ayni-lgbm",
        "version": "test",
        "columnas": COLUMNAS,
        "categorias": {},
        "umbral": 0.5,
        "matriz_costos": {"falso_negativo": 5.0, "falso_positivo": 1.0},
    }
    (destino / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    servicio = cargar_servicio(destino)
    with TestClient(crear_app(servicio)) as client:
        yield client


PAYLOAD = {
    "amt_income_total": 180000,
    "amt_credit": 450000,
    "amt_annuity": 24500,
    "age_years": 34,
}


def test_health_expone_version_y_umbral(cliente_api):
    datos = cliente_api.get("/health").json()

    assert datos["estado"] == "ok"
    assert datos["modelo_version"] == "test"
    assert datos["umbral"] == 0.5
    assert datos["n_features"] == len(COLUMNAS)


def test_predict_devuelve_probabilidad_y_decision(cliente_api):
    datos = cliente_api.post("/predict", json=PAYLOAD).json()

    assert 0 <= datos["probabilidad"] <= 1
    assert datos["decision"] in {"approve", "reject"}
    assert datos["umbral"] == 0.5


def test_la_decision_respeta_el_umbral(cliente_api):
    datos = cliente_api.post("/predict", json=PAYLOAD).json()
    esperado = "approve" if datos["probabilidad"] <= 0.5 else "reject"

    assert datos["decision"] == esperado


def test_explain_devuelve_cinco_razones_ordenadas(cliente_api):
    datos = cliente_api.post("/explain", json=PAYLOAD).json()

    assert 1 <= len(datos["razones"]) <= 5
    magnitudes = [abs(r["contribucion"]) for r in datos["razones"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for razon in datos["razones"]:
        assert razon["direccion"] in {"aumenta_riesgo", "reduce_riesgo"}
        assert set(razon) == {"feature", "valor", "contribucion", "direccion"}


def test_las_razones_son_coherentes_con_la_decision(cliente_api):
    datos = cliente_api.post("/explain", json=PAYLOAD).json()
    esperado = "aumenta_riesgo" if datos["decision"] == "reject" else "reduce_riesgo"

    assert datos["razones"]
    assert all(r["direccion"] == esperado for r in datos["razones"])


def test_las_razones_solo_hablan_de_campos_enviados(cliente_api):
    datos = cliente_api.post("/explain", json=PAYLOAD).json()
    enviados = {"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "age_years"}

    usadas = {r["feature"] for r in datos["razones"]}
    assert usadas <= enviados


def test_la_advertencia_salta_exactamente_cuando_la_cobertura_es_baja(cliente_api):
    datos = cliente_api.post("/explain", json=PAYLOAD).json()

    assert 0 < datos["cobertura"] < 1
    assert (datos["advertencia"] is not None) == (datos["cobertura"] < COBERTURA_MINIMA)


def test_entrada_invalida_se_rechaza(cliente_api):
    assert cliente_api.post("/predict", json={"amt_income_total": -1}).status_code == 422
    assert cliente_api.post("/predict", json={**PAYLOAD, "campo_inventado": 1}).status_code == 422
    assert cliente_api.post("/predict", json={**PAYLOAD, "age_years": 12}).status_code == 422
