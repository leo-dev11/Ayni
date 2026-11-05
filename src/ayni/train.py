"""Entrenamiento, validación cruzada y registro de experimentos.

Cada corrida deja en MLflow: parámetros, métricas por fold y fuera de fold, importancia
de features y el modelo. El mejor por ROC-AUC de validación se registra como `ayni-lgbm`.

Métricas principales: ROC-AUC (comparable con el estándar de la competición) y PR-AUC
(la lectura honesta con 8 % de positivos). La accuracy no se reporta: no distingue un
modelo útil de uno que aprueba a todo el mundo.

El test no se toca aquí. Todo se mide con `StratifiedKFold` sobre `train.parquet`.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ayni.config import ID_COL, RANDOM_STATE, REPORTS_DIR, ROOT, TARGET
from ayni.split import TRAIN_PATH

EXPERIMENT = "ayni"
REGISTERED_MODEL = "ayni-lgbm"
TRACKING_DB = ROOT / "mlflow.db"
# Permite aislar pruebas y CI sin ensuciar el experimento real.
TRACKING_URI = os.environ.get("AYNI_TRACKING_URI") or f"sqlite:///{TRACKING_DB.as_posix()}"
MLRUNS_DIR = ROOT / "mlruns"
FOLDS = 5

# El modelo que se elige como campeón se decide con `roc_auc_cv`, nunca con el test.
METRICAS = ("roc_auc", "pr_auc")


@dataclass
class Corrida:
    """Resultado de una configuración evaluada con validación cruzada."""

    nombre: str
    familia: str
    params: dict
    con_genero: bool
    metricas: dict[str, float] = field(default_factory=dict)
    metricas_fold: list[dict[str, float]] = field(default_factory=list)
    oof: np.ndarray | None = None
    modelo: object | None = None
    run_id: str | None = None
    segundos: float = 0.0
    best_iterations: list[int] = field(default_factory=list)


def preparar(df: pd.DataFrame, *, con_genero: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X/y, aplica la decisión de la fase 1 y deja las categóricas como `category`."""
    y = df[TARGET].astype(int)
    X = df.drop(columns=[c for c in (ID_COL, TARGET) if c in df.columns])
    if not con_genero and "CODE_GENDER" in X.columns:
        X = X.drop(columns=["CODE_GENDER"])
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) in {"boolean", "category"}:
            X[col] = X[col].astype("category")
    return X, y


NUMERICAS = ("float64", "int64", "int32", "int8", "float32")
CATEGORICAS = ("object", "category", "boolean")


def _categorica_a_texto(df: pd.DataFrame) -> pd.DataFrame:
    """OneHotEncoder no acepta `category` con nulos: a texto, y el nulo pasa a ser una categoría."""
    return df.astype(object).where(df.notna(), "SIN_DATO").astype(str)


def crear_logreg() -> Pipeline:
    """Baseline honesto: imputación + escalado + one-hot, sin nada específico de árboles."""
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("imputar", SimpleImputer(strategy="median")),
                                    ("escalar", StandardScaler()),
                                ]
                            ),
                            lambda X: X.select_dtypes(include=NUMERICAS).columns,
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    ("texto", FunctionTransformer(_categorica_a_texto)),
                                    (
                                        "onehot",
                                        OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                                    ),
                                ]
                            ),
                            lambda X: X.select_dtypes(include=CATEGORICAS).columns,
                        ),
                    ]
                ),
            ),
            (
                "modelo",
                LogisticRegression(max_iter=300, C=0.5, class_weight="balanced"),
            ),
        ]
    )


def crear_lgbm(params: dict) -> lgb.LGBMClassifier:
    base = {
        "objective": "binary",
        # El early stopping se mide con AUC: con `scale_pos_weight` la logloss de
        # validación deja de ser comparable y corta en la primera iteración.
        "metric": "auc",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }
    return lgb.LGBMClassifier(**{**base, **params})


def _ajustar(modelo, Xt, yt, Xv, yv):
    if isinstance(modelo, lgb.LGBMClassifier):
        modelo.fit(
            Xt,
            yt,
            eval_X=Xv,
            eval_y=yv,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        modelo.fit(Xt, yt)
    return modelo


def validar_cruzado(
    crear: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    folds: int = FOLDS,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, list[dict[str, float]], list[int]]:
    """StratifiedKFold out-of-fold: probabilidades OOF, métricas por fold y nº de árboles."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    por_fold: list[dict[str, float]] = []
    iteraciones: list[int] = []

    for k, (idx_tr, idx_val) in enumerate(skf.split(X, y), start=1):
        modelo = crear()
        _ajustar(modelo, X.iloc[idx_tr], y.iloc[idx_tr], X.iloc[idx_val], y.iloc[idx_val])
        proba = modelo.predict_proba(X.iloc[idx_val])[:, 1]
        oof[idx_val] = proba

        fold = {
            "roc_auc": roc_auc_score(y.iloc[idx_val], proba),
            "pr_auc": average_precision_score(y.iloc[idx_val], proba),
        }
        por_fold.append(fold)
        if isinstance(modelo, lgb.LGBMClassifier):
            iteraciones.append(int(modelo.best_iteration_ or modelo.n_estimators_))
        print(
            f"    fold {k}: ROC-AUC {fold['roc_auc']:.4f}  PR-AUC {fold['pr_auc']:.4f}"
            + (f"  árboles {iteraciones[-1]}" if iteraciones else "")
        )

    return oof, por_fold, iteraciones


def importancia(modelo: lgb.LGBMClassifier, columnas) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": columnas, "importancia": modelo.feature_importances_})
        .sort_values("importancia", ascending=False)
        .reset_index(drop=True)
    )


def elegir_campeon(corridas: list[Corrida]) -> Corrida:
    """Mejor corrida de LightGBM que respeta la política de sesgo (sin CODE_GENDER).

    La variante con género existe solo para medir su coste: no puede ganar el registro.
    """
    candidatas = [c for c in corridas if c.familia == "lgbm" and not c.con_genero]
    if not candidatas:
        raise ValueError("No hay corridas de LightGBM sin género entre las que elegir")
    return max(candidatas, key=lambda c: c.metricas["roc_auc_cv"])


def coste_de_excluir_genero(corridas: list[Corrida]) -> float | None:
    """Diferencia de ROC-AUC entre la mejor corrida sin género y la que sí lo usa."""
    sin_genero = [c for c in corridas if c.familia == "lgbm" and not c.con_genero]
    con_genero = [c for c in corridas if c.familia == "lgbm" and c.con_genero]
    if not sin_genero or not con_genero:
        return None
    referencia = max(c.metricas["roc_auc_cv"] for c in sin_genero)
    return con_genero[0].metricas["roc_auc_cv"] - referencia


def configurar_mlflow() -> None:
    TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)


def evaluar(nombre, familia, params, con_genero, *, X, y, folds=FOLDS) -> Corrida:
    corrida = Corrida(nombre=nombre, familia=familia, params=params, con_genero=con_genero)
    crear = crear_logreg if familia == "logreg" else (lambda: crear_lgbm(params))

    print(f"\n[{nombre}] familia={familia} con_genero={con_genero}")
    inicio = time.perf_counter()
    oof, por_fold, iteraciones = validar_cruzado(crear, X, y, folds=folds)
    corrida.segundos = time.perf_counter() - inicio
    corrida.oof = oof
    corrida.metricas_fold = por_fold
    corrida.best_iterations = iteraciones

    for metrica in METRICAS:
        valores = [f[metrica] for f in por_fold]
        corrida.metricas[f"{metrica}_cv"] = float(np.mean(valores))
        corrida.metricas[f"{metrica}_std"] = float(np.std(valores))
    corrida.metricas["roc_auc_oof"] = float(roc_auc_score(y, oof))
    corrida.metricas["pr_auc_oof"] = float(average_precision_score(y, oof))
    corrida.metricas["segundos"] = round(corrida.segundos, 1)

    print(
        f"    ROC-AUC CV {corrida.metricas['roc_auc_cv']:.4f} "
        f"± {corrida.metricas['roc_auc_std']:.4f}  |  "
        f"PR-AUC CV {corrida.metricas['pr_auc_cv']:.4f}  |  "
        f"{corrida.segundos:.0f}s"
    )
    return corrida


def reentrenar_final(corrida: Corrida, X: pd.DataFrame, y: pd.Series):
    """Reentrena con todo el train usando el número de árboles mediano de la CV."""
    modelo = crear_logreg() if corrida.familia == "logreg" else crear_lgbm(corrida.params)
    if isinstance(modelo, lgb.LGBMClassifier) and corrida.best_iterations:
        modelo.set_params(n_estimators=int(np.median(corrida.best_iterations)))
    return modelo.fit(X, y)


def registrar(corrida: Corrida, X: pd.DataFrame, y: pd.Series) -> str:
    """Guarda la corrida en MLflow y devuelve el run_id."""
    modelo = reentrenar_final(corrida, X, y)
    corrida.modelo = modelo

    with mlflow.start_run(run_name=corrida.nombre) as run:
        mlflow.log_params(
            {
                "familia": corrida.familia,
                "con_genero": corrida.con_genero,
                "n_features": X.shape[1],
                "n_folds": len(corrida.metricas_fold),
                **{f"param_{k}": v for k, v in corrida.params.items()},
            }
        )
        mlflow.log_metrics(corrida.metricas)
        for i, fold in enumerate(corrida.metricas_fold, start=1):
            mlflow.log_metrics({f"roc_auc_fold{i}": fold["roc_auc"]}, step=i)

        if isinstance(modelo, lgb.LGBMClassifier):
            tabla = importancia(modelo, X.columns)
            destino = REPORTS_DIR / f"importancia_{corrida.nombre}.csv"
            destino.parent.mkdir(parents=True, exist_ok=True)
            tabla.head(40).to_csv(destino, index=False)
            mlflow.log_artifact(str(destino), artifact_path="importancia")
            mlflow.log_dict(
                {fila.feature: int(fila.importancia) for fila in tabla.head(15).itertuples()},
                "importancia_top15.json",
            )
            mlflow.lightgbm.log_model(modelo, name="model")
        else:
            # skops no serializa selectores lambda: el baseline viaja con cloudpickle.
            mlflow.sklearn.log_model(modelo, name="model", serialization_format="cloudpickle")

        corrida.run_id = run.info.run_id

    return corrida.run_id


def main(
    *, folds: int = FOLDS, registrar_modelo: bool = True, solo: str | None = None
) -> list[Corrida]:
    configurar_mlflow()
    train = pd.read_parquet(TRAIN_PATH)
    ratio = float((train[TARGET] == 0).sum() / train[TARGET].sum())
    print(f"train: {len(train):,} filas · ratio 0:1 = {ratio:.2f}")

    lgbm_base = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 50,
        "colsample_bytree": 0.8,
    }
    lgbm_ajustado = {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 100,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
    }

    # (nombre, familia, params, con_genero)
    plan = [
        ("baseline-logreg", "logreg", {}, False),
        ("lgbm-base", "lgbm", lgbm_base, False),
        ("lgbm-pos-weight", "lgbm", {**lgbm_base, "scale_pos_weight": ratio}, False),
        ("lgbm-con-genero", "lgbm", lgbm_base, True),
        ("lgbm-ajustado", "lgbm", lgbm_ajustado, False),
    ]
    if solo:
        plan = [c for c in plan if c[0] == solo]
        if not plan:
            raise SystemExit(f"Corrida desconocida: {solo}")

    corridas = []
    for nombre, familia, params, con_genero in plan:
        X, y = preparar(train, con_genero=con_genero)
        corrida = evaluar(nombre, familia, params, con_genero, X=X, y=y, folds=folds)
        registrar(corrida, X, y)
        corridas.append(corrida)

    if registrar_modelo:
        mejor = elegir_campeon(corridas)
        version = mlflow.register_model(f"runs:/{mejor.run_id}/model", REGISTERED_MODEL)
        print(f"\nRegistrado {REGISTERED_MODEL} v{version.version} ← {mejor.nombre}")

    resumen = pd.DataFrame(
        [
            {
                "corrida": c.nombre,
                "familia": c.familia,
                "con_genero": c.con_genero,
                "roc_auc_cv": round(c.metricas["roc_auc_cv"], 4),
                "pr_auc_cv": round(c.metricas["pr_auc_cv"], 4),
                "roc_auc_oof": round(c.metricas["roc_auc_oof"], 4),
                "segundos": c.metricas["segundos"],
            }
            for c in corridas
        ]
    ).sort_values("roc_auc_cv", ascending=False)

    print("\n" + resumen.to_string(index=False))

    sin_genero = [c for c in corridas if c.familia == "lgbm" and not c.con_genero]
    con_genero = [c for c in corridas if c.familia == "lgbm" and c.con_genero]
    if sin_genero and con_genero:
        referencia = max(c.metricas["roc_auc_cv"] for c in sin_genero)
        delta = con_genero[0].metricas["roc_auc_cv"] - referencia
        print(
            f"\ncoste de excluir CODE_GENDER: {delta:+.4f} AUC "
            f"(sin género {referencia:.4f} · con género "
            f"{con_genero[0].metricas['roc_auc_cv']:.4f})"
        )

    destino = REPORTS_DIR / "resumen_corridas.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(
            {c.nombre: {**c.metricas, "run_id": c.run_id} for c in corridas},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nresumen en {destino.relative_to(ROOT)}")
    return corridas
