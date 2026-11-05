"""Decisión por costo: el umbral no es 0.5, se calcula.

Un modelo devuelve probabilidades; el banco necesita una decisión binaria. El punto de corte
óptimo no es el que maximiza el acierto, sino el que minimiza el costo esperado:

- **Falso negativo (aprobar a quien no paga):** se pierde el principal prestado. **Costo 5.**
- **Falso positivo (rechazar a quien sí pagaría):** se pierde el margen de interés. **Costo 1.**

La matriz es un supuesto de negocio explícito, no un dato del banco: queda como parámetro para
poder recalcular el umbral con costos reales sin reentrenar nada.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

COSTO_FN = 5.0
COSTO_FP = 1.0
UMBRALES = np.round(np.arange(0.01, 0.96, 0.005), 3)


@dataclass
class Evaluacion:
    """Resultado de aplicar un umbral a un conjunto de probabilidades."""

    umbral: float
    n: int
    aprobados: int
    rechazados: int
    # Nombres de negocio, no de matriz de confusión: el "falso negativo" es el cliente
    # al que se le dio el crédito y no pagó.
    falsos_negativos: int  # aprobados que impagan → costo alto
    falsos_positivos: int  # rechazados que habrían pagado → costo bajo
    aprobados_pagadores: int
    rechazados_impagos: int
    costo: float
    costo_fn: float = COSTO_FN
    costo_fp: float = COSTO_FP

    @property
    def tasa_aprobacion(self) -> float:
        return self.aprobados / self.n if self.n else 0.0

    @property
    def impago_entre_aprobados(self) -> float:
        """De los aprobados, qué fracción no paga. Es el riesgo que asume el banco."""
        return self.falsos_negativos / self.aprobados if self.aprobados else 0.0

    @property
    def costo_por_solicitud(self) -> float:
        return self.costo / self.n if self.n else 0.0

    def a_dict(self) -> dict[str, float]:
        datos = asdict(self)
        datos.update(
            {
                "tasa_aprobacion": round(self.tasa_aprobacion, 4),
                "impago_entre_aprobados": round(self.impago_entre_aprobados, 4),
                "costo_por_solicitud": round(self.costo_por_solicitud, 4),
            }
        )
        return datos


def evaluar(
    y_true,
    proba,
    umbral: float,
    *,
    costo_fn: float = COSTO_FN,
    costo_fp: float = COSTO_FP,
) -> Evaluacion:
    """Aplica la regla `aprobar si P(impago) <= umbral` y cuenta la matriz con costos.

    El umbral es la **probabilidad máxima de impago que se acepta** para conceder el crédito:
    cuanto más alto, más clientes aprobados y más mora asumida. Aprobar con `proba >= umbral`
    sería conceder el crédito precisamente a los que más riesgo tienen.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    aprueba = p <= umbral

    fn = int(np.sum(aprueba & (y == 1)))
    fp = int(np.sum(~aprueba & (y == 0)))
    aprobados_pagadores = int(np.sum(aprueba & (y == 0)))
    rechazados_impagos = int(np.sum(~aprueba & (y == 1)))

    return Evaluacion(
        umbral=float(umbral),
        n=int(y.size),
        aprobados=int(aprueba.sum()),
        rechazados=int((~aprueba).sum()),
        falsos_negativos=fn,
        falsos_positivos=fp,
        aprobados_pagadores=aprobados_pagadores,
        rechazados_impagos=rechazados_impagos,
        costo=costo_fn * fn + costo_fp * fp,
        costo_fn=costo_fn,
        costo_fp=costo_fp,
    )


def curva_costo(
    y_true,
    proba,
    *,
    umbrales=UMBRALES,
    costo_fn: float = COSTO_FN,
    costo_fp: float = COSTO_FP,
) -> pd.DataFrame:
    """Costo, aprobación y morosidad resultante para cada umbral de la rejilla."""
    filas = [
        {
            "umbral": u,
            "costo_por_solicitud": e.costo_por_solicitud,
            "tasa_aprobacion": e.tasa_aprobacion,
            "impago_entre_aprobados": e.impago_entre_aprobados,
            "falsos_negativos": e.falsos_negativos,
            "falsos_positivos": e.falsos_positivos,
        }
        for u in umbrales
        for e in [evaluar(y_true, proba, u, costo_fn=costo_fn, costo_fp=costo_fp)]
    ]
    return pd.DataFrame(filas)


def umbral_optimo(
    y_true,
    proba,
    *,
    umbrales=UMBRALES,
    costo_fn: float = COSTO_FN,
    costo_fp: float = COSTO_FP,
) -> tuple[float, pd.DataFrame]:
    """Umbral que minimiza el costo total y la curva completa (para graficar y auditar)."""
    curva = curva_costo(y_true, proba, umbrales=umbrales, costo_fn=costo_fn, costo_fp=costo_fp)
    mejor = curva.loc[curva["costo_por_solicitud"].idxmin(), "umbral"]
    return float(mejor), curva


def sensibilidad(
    y_true,
    proba,
    ratios: tuple[float, ...] = (1, 2, 5, 10, 20, 50),
    *,
    umbrales=UMBRALES,
) -> pd.DataFrame:
    """Cómo cambia el umbral óptimo al cambiar la relación de costos FN:FP.

    Con un modelo calibrado, el umbral que iguala costos marginales vale 1/(ratio+1):
    es la probabilidad de impago a la que da lo mismo aprobar que rechazar.
    """
    filas = []
    for ratio in ratios:
        optimo, _ = umbral_optimo(y_true, proba, umbrales=umbrales, costo_fn=ratio, costo_fp=1.0)
        e = evaluar(y_true, proba, optimo, costo_fn=ratio, costo_fp=1.0)
        filas.append(
            {
                "ratio_fn_fp": ratio,
                "umbral_optimo": round(optimo, 4),
                "teorico_1/(ratio+1)": round(1 / (ratio + 1), 4),
                "tasa_aprobacion": round(e.tasa_aprobacion, 4),
                "impago_entre_aprobados": round(e.impago_entre_aprobados, 4),
                "costo_por_solicitud": round(e.costo_por_solicitud, 4),
            }
        )
    return pd.DataFrame(filas)


def comparar(y_true, proba, umbrales: dict[str, float] | None = None, **kwargs) -> pd.DataFrame:
    """Tabla comparativa de varios umbrales (por defecto 0.5 contra el óptimo)."""
    if umbrales is None:
        optimo, _ = umbral_optimo(y_true, proba, **kwargs)
        umbrales = {"0.50 (por defecto)": 0.5, f"óptimo ({optimo:.3f})": optimo}

    filas = []
    for nombre, umbral in umbrales.items():
        evaluacion = evaluar(y_true, proba, umbral, **kwargs)
        filas.append(
            {
                "umbral": nombre,
                "tasa_aprobacion": round(evaluacion.tasa_aprobacion, 4),
                "impago_entre_aprobados": round(evaluacion.impago_entre_aprobados, 4),
                "costo_total": round(evaluacion.costo, 1),
                "costo_por_solicitud": round(evaluacion.costo_por_solicitud, 4),
            }
        )
    return pd.DataFrame(filas)
