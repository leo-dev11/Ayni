"""Sanity checks del esqueleto: paquete importable, rutas y tablas declaradas."""

from ayni import __version__, config


def test_version_expuesta():
    assert __version__


def test_root_apunta_al_repositorio():
    assert (config.ROOT / "requirements.txt").exists()
    assert config.ROOT.name.lower() == "ayni"


def test_tablas_del_modelo_son_subconjunto():
    assert set(config.MODEL_TABLES) <= set(config.TABLES)
    assert "application_test" not in config.MODEL_TABLES


def test_los_directorios_de_datos_cuelgan_de_data():
    # En CI no hay datos: solo se comprueba la estructura de rutas, no su contenido.
    for path in (config.RAW_DIR, config.INTERIM_DIR, config.PROCESSED_DIR):
        assert path.parent == config.DATA_DIR
        assert path.is_absolute()
