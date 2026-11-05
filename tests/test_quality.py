"""Tests del suelo de calidad y de que el paquete es importable sin datos de Kaggle.

La compuerta completa (ROC-AUC real) vive en `ayni evaluate --confirmar-test`, donde sí hay test
congelado. Aquí se comprueba la regla, que es lo que el resto de la lógica puede romper sin avisar.
"""

import pytest

from ayni import __version__, config
from ayni.evaluate import ROC_AUC_MINIMO, CalidadInsuficiente


def verificar(roc_auc: float, minimo: float = ROC_AUC_MINIMO) -> None:
    """Misma regla que aplica `evaluar_test` antes de dar por bueno el modelo."""
    if roc_auc < minimo:
        raise CalidadInsuficiente(f"ROC-AUC {roc_auc:.4f} por debajo del mínimo {minimo}")


def test_el_minimo_es_el_documentado():
    assert ROC_AUC_MINIMO == 0.75


def test_un_modelo_por_debajo_del_suelo_se_detiene():
    with pytest.raises(CalidadInsuficiente, match="por debajo del mínimo"):
        verificar(0.70)


def test_el_suelo_es_inclusivo():
    verificar(0.75)
    verificar(0.85)


def test_el_champion_medido_supera_el_suelo_con_margen():
    # 0.7829 medido en test en la fase 5; el margen es la señal de que el suelo no está al filo.
    assert 0.7829 - ROC_AUC_MINIMO > 0.02


def test_el_paquete_es_importable_sin_datos():
    """En CI no hay data/: el paquete y sus rutas deben funcionar igual."""
    assert __version__
    for ruta in (config.RAW_DIR, config.INTERIM_DIR, config.PROCESSED_DIR):
        assert ruta.is_absolute()
        assert ruta.parent == config.DATA_DIR
    assert (config.ROOT / "pyproject.toml").exists()
