"""Servicio de scoring: `/health`, `/predict` y `/explain`.

El contrato de entrada es deliberadamente **pequeño y de negocio**: los montos, la edad, el
empleo y los scores externos; los ratios que el modelo usa se derivan aquí. Las agregaciones
del buró y del historial (``bureau_n_credits``, ``prev_refusal_rate``…) son opcionales: en un
banco real ya estarían calculadas en el sistema de originación, y si faltan el modelo las trata
como ``NaN``, igual que en el entrenamiento.

Los nombres de los campos están en minúsculas y en inglés de negocio para que el `curl` del
README sea legible, aunque el modelo use los nombres originales de la competición.
"""

from __future__ import annotations

import json
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ayni.config import ROOT

# shap avisa en cada llamada de que LightGBM binario devuelve una lista; ya se normaliza.
warnings.filterwarnings("ignore", message=".*shap values output has changed.*")

MODELS_DIR = ROOT / "models"
TOP_RAZONES = 5
# El contrato HTTP cubre como máximo el 15.7 % de los 185 campos del modelo
# (montos, ratios derivados, scores y agregados que el banco ya tiene). Avisar por debajo
# del 50 % haría que la advertencia saltara siempre; 10 % es "solo mandaste lo obligatorio".
COBERTURA_MINIMA = 0.10


# ── contrato de entrada ────────────────────────────────────────────────────────
class Cliente(BaseModel):
    """Datos de una solicitud. Solo los montos y la edad son obligatorios."""

    model_config = {"extra": "forbid"}

    amt_income_total: Annotated[float, Field(gt=0, description="Ingreso total anual")]
    amt_credit: Annotated[float, Field(gt=0, description="Monto del crédito solicitado")]
    amt_annuity: Annotated[float, Field(ge=0, description="Cuota periódica")]
    age_years: Annotated[float, Field(ge=18, le=100, description="Edad del solicitante")]
    years_employed: Annotated[
        float | None, Field(ge=0, le=60, description="Años en el empleo actual; nulo = sin empleo")
    ] = None
    amt_goods_price: Annotated[float | None, Field(ge=0)] = None
    children: Annotated[int, Field(ge=0, le=20)] = 0
    family_members: Annotated[float, Field(ge=1, le=20)] = 1

    ext_source_1: Annotated[float | None, Field(ge=0, le=1)] = None
    ext_source_2: Annotated[float | None, Field(ge=0, le=1)] = None
    ext_source_3: Annotated[float | None, Field(ge=0, le=1)] = None

    contract_type: str | None = Field(default=None, description="Cash loans / Revolving loans")
    education: str | None = None
    income_type: str | None = None
    occupation: str | None = None
    organization: str | None = None
    family_status: str | None = None

    # Agregados que el banco ya tiene (opcionales)
    bureau_n_credits: Annotated[float | None, Field(ge=0)] = None
    bureau_debt_ratio: float | None = None
    prev_refusal_rate: Annotated[float | None, Field(ge=0, le=1)] = None
    prev_cnt_payment_max: Annotated[float | None, Field(ge=0, le=200)] = None
    inst_late_rate: Annotated[float | None, Field(ge=0, le=1)] = None
    cc_utilization_mean: float | None = None


class Prediccion(BaseModel):
    probabilidad: float
    decision: str
    umbral: float
    modelo_version: str


class Razon(BaseModel):
    feature: str
    valor: float | str | None
    contribucion: float
    direccion: str


class Explicacion(Prediccion):
    razones: list[Razon]
    cobertura: float = Field(
        description="Fracción de los 185 campos del modelo presentes en el payload"
    )
    advertencia: str | None = None


class Salud(BaseModel):
    estado: str
    modelo_version: str
    umbral: float
    n_features: int


# ── carga del artefacto ────────────────────────────────────────────────────────
@dataclass
class Servicio:
    booster: lgb.Booster
    explicador: shap.TreeExplainer
    manifest: dict

    @property
    def umbral(self) -> float:
        return float(self.manifest["umbral"])

    @property
    def version(self) -> str:
        return str(self.manifest["version"])

    @property
    def columnas(self) -> list[str]:
        return list(self.manifest["columnas"])


def cargar_servicio(directorio: Path = MODELS_DIR) -> Servicio:
    booster_path = Path(directorio) / "booster.txt"
    manifest_path = Path(directorio) / "manifest.json"
    if not booster_path.exists() or not manifest_path.exists():
        raise RuntimeError(
            f"No hay modelo en {directorio}. Ejecuta antes: python -m ayni export-model"
        )
    booster = lgb.Booster(model_file=str(booster_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Servicio(booster=booster, explicador=shap.TreeExplainer(booster), manifest=manifest)


# ── preparación de la fila ─────────────────────────────────────────────────────
def a_fila(cliente: Cliente, servicio: Servicio) -> pd.DataFrame:
    """Convierte el payload en una fila con las 185 columnas que espera el modelo."""
    fila: dict[str, object] = dict.fromkeys(servicio.columnas, np.nan)

    fila.update(
        {
            "AMT_INCOME_TOTAL": cliente.amt_income_total,
            "AMT_CREDIT": cliente.amt_credit,
            "AMT_ANNUITY": cliente.amt_annuity,
            "CNT_CHILDREN": cliente.children,
            "CNT_FAM_MEMBERS": cliente.family_members,
            "age_years": cliente.age_years,
            "credit_income_ratio": cliente.amt_credit / cliente.amt_income_total,
            "annuity_income_ratio": cliente.amt_annuity / cliente.amt_income_total,
            "income_per_person": cliente.amt_income_total / cliente.family_members,
            "children_ratio": cliente.children / cliente.family_members,
        }
    )

    if cliente.years_employed is None:
        fila["employed_anomaly"] = 1
    else:
        fila["employed_years"] = cliente.years_employed
        fila["employed_anomaly"] = 0

    if cliente.amt_goods_price:
        fila["AMT_GOODS_PRICE"] = cliente.amt_goods_price
        fila["goods_credit_ratio"] = cliente.amt_goods_price / cliente.amt_credit

    for i, valor in enumerate(
        (cliente.ext_source_1, cliente.ext_source_2, cliente.ext_source_3), start=1
    ):
        if valor is not None:
            fila[f"EXT_SOURCE_{i}"] = valor

    texto = {
        "NAME_CONTRACT_TYPE": cliente.contract_type,
        "NAME_EDUCATION_TYPE": cliente.education,
        "NAME_INCOME_TYPE": cliente.income_type,
        "OCCUPATION_TYPE": cliente.occupation,
        "ORGANIZATION_TYPE": cliente.organization,
        "NAME_FAMILY_STATUS": cliente.family_status,
    }
    for columna, valor in texto.items():
        if valor is not None:
            fila[columna] = valor

    historial = {
        "bureau_n_credits": cliente.bureau_n_credits,
        "bureau_debt_ratio": cliente.bureau_debt_ratio,
        "prev_refusal_rate": cliente.prev_refusal_rate,
        "prev_cnt_payment_max": cliente.prev_cnt_payment_max,
        "inst_late_rate": cliente.inst_late_rate,
        "cc_utilization_mean": cliente.cc_utilization_mean,
    }
    for columna, valor in historial.items():
        if valor is not None:
            fila[columna] = valor

    X = pd.DataFrame([fila], columns=servicio.columnas)

    for columna, categorias in servicio.manifest.get("categorias", {}).items():
        if columna in X.columns:
            X[columna] = pd.Categorical(X[columna].astype(object), categories=categorias)
    return X


def _probabilidad(servicio: Servicio, X: pd.DataFrame) -> float:
    return float(np.ravel(servicio.booster.predict(X))[0])


def _decision(servicio: Servicio, probabilidad: float) -> str:
    return "approve" if probabilidad <= servicio.umbral else "reject"


def _razones(
    servicio: Servicio, X: pd.DataFrame, decision: str, top: int = TOP_RAZONES
) -> list[Razon]:
    """Hasta `top` features **con dato** y **coherentes con la decisión**.

    Al rechazar se devuelven solo las que suben el riesgo; al aprobar, solo las que lo bajan. Dar
    a un rechazado un motivo que dice "esto juega a tu favor" es confuso y, en banca, legalmente
    insostenible. Si el filtro deja la lista vacía, se devuelve la mejor disponible sin filtrar.


    Un `NaN` sí empuja el score, pero "no nos diste tu antigüedad de registro" no es una razón
    defendible ante un cliente. El campo ausente se reporta aparte, como advertencia.
    """
    valores = np.asarray(servicio.explicador.shap_values(X))
    if valores.ndim == 3:  # (n, features, clases)
        valores = valores[:, :, -1]
    valores = valores[0]

    presentes = ~X.iloc[0].isna().to_numpy()
    candidatas = np.where(presentes)[0]
    if candidatas.size == 0:  # sin ningún dato: se explica con el modelo entero
        candidatas = np.arange(len(valores))
    signo = 1 if decision == "reject" else -1
    con_signo = candidatas[signo * valores[candidatas] > 0]
    if con_signo.size:
        candidatas = con_signo
    orden = candidatas[np.argsort(-np.abs(valores[candidatas]))][:top]

    return [
        Razon(
            feature=str(X.columns[i]),
            valor=_valor(X.iloc[0, i]),
            contribucion=round(float(valores[i]), 4),
            direccion="aumenta_riesgo" if valores[i] > 0 else "reduce_riesgo",
        )
        for i in orden
    ]


def _valor(valor: object) -> float | str | None:
    """Deja el valor legible: número redondeado, texto tal cual, nulo como `None`."""
    if pd.isna(valor):
        return None
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return round(float(valor), 4)
    return str(valor)


def _cobertura(X: pd.DataFrame) -> float:
    return round(float(X.iloc[0].notna().mean()), 3)


# ── aplicación ────────────────────────────────────────────────────────────────
def crear_app(servicio: Servicio | None = None, directorio: Path = MODELS_DIR) -> FastAPI:
    """Fábrica: permite inyectar un servicio ya cargado (tests) o cargarlo al arrancar."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.servicio = servicio or cargar_servicio(directorio)
        yield

    app = FastAPI(
        title="Ayni · scoring de riesgo crediticio",
        description=(
            "Predice la probabilidad de impago y decide con un umbral elegido por costo.\n\n"
            "`/explain` añade hasta cinco razones (*reason codes*), solo de los campos informados."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def servicio_actual() -> Servicio:
        cargado = getattr(app.state, "servicio", None)
        if cargado is None:
            raise HTTPException(status_code=503, detail="Modelo no cargado")
        return cargado

    @app.get("/health", response_model=Salud, tags=["sistema"])
    def health() -> Salud:
        actual = servicio_actual()
        return Salud(
            estado="ok",
            modelo_version=actual.version,
            umbral=actual.umbral,
            n_features=len(actual.columnas),
        )

    @app.post("/predict", response_model=Prediccion, tags=["scoring"])
    def predict(cliente: Cliente) -> Prediccion:
        actual = servicio_actual()
        X = a_fila(cliente, actual)
        probabilidad = _probabilidad(actual, X)
        return Prediccion(
            probabilidad=round(probabilidad, 4),
            decision=_decision(actual, probabilidad),
            umbral=actual.umbral,
            modelo_version=actual.version,
        )

    @app.post("/explain", response_model=Explicacion, tags=["scoring"])
    def explain(cliente: Cliente) -> Explicacion:
        actual = servicio_actual()
        X = a_fila(cliente, actual)
        probabilidad = _probabilidad(actual, X)
        cobertura = _cobertura(X)
        advertencia = (
            None
            if cobertura >= COBERTURA_MINIMA
            else (
                f"Solo se informó el {cobertura:.0%} de los campos del modelo: la probabilidad "
                "puede diferir de la que se obtendría con el expediente completo."
            )
        )
        return Explicacion(
            probabilidad=round(probabilidad, 4),
            decision=_decision(actual, probabilidad),
            umbral=actual.umbral,
            modelo_version=actual.version,
            razones=_razones(actual, X, _decision(actual, probabilidad)),
            cobertura=cobertura,
            advertencia=advertencia,
        )

    return app


app = crear_app()
