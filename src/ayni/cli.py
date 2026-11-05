"""CLI del proyecto: ``python -m ayni <comando>``.

Sustituye a ``make`` en Windows, donde no hay make instalado. El Makefile
delegará aquí para que el comportamiento sea idéntico en local y en CI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile

import pandas as pd

from ayni.config import RAW_DIR, TABLES
from ayni.features import FEATURES_PATH, build_features, connect
from ayni.ingest import describe, missing_raw, to_parquet
from ayni.schema import (
    DataValidationError,
    validate_features,
    validate_raw,
    validate_split,
)
from ayni.split import TEST_PATH, TRAIN_PATH, make_split

COMPETITION = "home-credit-default-risk"
ZIP_NAME = f"{COMPETITION}.zip"

RULES_HELP = f"""
Kaggle respondió 403: el token es válido, pero la cuenta no ha aceptado las reglas
de la competición. Ábrela y pulsa "I Understand and Accept":

  https://www.kaggle.com/competitions/{COMPETITION}/rules

Después vuelve a ejecutar:  python -m ayni download
"""

AUTH_HELP = """
No se pudo autenticar contra Kaggle. Cualquiera de estas tres opciones sirve:

  1. OAuth (sin archivos):
       .venv\\Scripts\\kaggle.exe auth login

  2. Token de API (Settings > API > Generate New Token):
       guarda el token en  C:\\Users\\<usuario>\\.kaggle\\access_token
       o exporta  KAGGLE_API_TOKEN=<token>

  3. kaggle.json clásico con usuario y clave:
       {"username": "<usuario-de-kaggle>", "key": "<token>"}
       en  C:\\Users\\<usuario>\\.kaggle\\kaggle.json

Además hay que entrar una vez a la página de la competición y aceptar las reglas:
si no, Kaggle responde 403 aunque el token sea correcto.
"""

PENDING: dict[str, str] = {}


def cmd_download(args: argparse.Namespace) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / ZIP_NAME

    if zip_path.exists() and not args.force:
        print(f"Ya existe {zip_path.name}; se salta la descarga (usa --force para repetirla).")
    else:
        print(f"Descargando {COMPETITION} desde Kaggle… (~1.3 GB)")
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "kaggle",
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-p",
                str(RAW_DIR),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        output = f"{result.stdout}{result.stderr}"
        if result.returncode != 0 or not zip_path.exists():
            print(output.strip())
            print(RULES_HELP if "403" in output else AUTH_HELP)
            return 1
        print(output.strip())

    print(f"Descomprimiendo en {RAW_DIR}…")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)

    return cmd_data(argparse.Namespace())


def cmd_data(_: argparse.Namespace) -> int:
    missing = missing_raw(TABLES)
    if not missing:
        print(f"OK: los {len(TABLES)} CSV están en data/raw")
        return 0
    print("Faltan CSV en data/raw:")
    for name in missing:
        print(f"  - {name}.csv")
    print("\nDescárgalos de https://www.kaggle.com/competitions/home-credit-default-risk/data")
    print("o con:  python -m ayni download   (requiere aceptar las reglas de la competencia)")
    return 1


def cmd_parquet(args: argparse.Namespace) -> int:
    timings = to_parquet(force=args.force)
    if not timings:
        print("Nada que convertir: revisa que los CSV estén en data/raw y usa --force.")
        return 0
    for table, seconds in timings.items():
        print(f"{table:<24} {seconds:6.1f}s")
    print(f"\n{len(timings)} tabla(s) en data/interim/")
    return 0


def cmd_check(_: argparse.Namespace) -> int:
    rows = describe()
    if not rows:
        print("No hay Parquet en data/interim. Ejecuta primero: python -m ayni parquet")
        return 1
    print(f"{'tabla':<24}{'filas':>10}{'columnas':>10}{'MB':>8}")
    for row in rows:
        print(f"{row['tabla']:<24}{row['filas']:>10,}{row['columnas']:>10}{row['mb']:>8}")
    return 0


def cmd_validate(_: argparse.Namespace) -> int:
    """Pasa los esquemas de Pandera sobre los datos que haya en disco."""
    etapas: list[tuple[str, object, object]] = []

    con = connect()
    if con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'application_train'"
    ).fetchone()[0]:
        raw = con.execute("SELECT * FROM application_train").df()
        etapas.append(("application_train (cruda)", validate_raw, raw))

    if FEATURES_PATH.exists():
        features = pd.read_parquet(FEATURES_PATH)
        etapas.append(("features.parquet", validate_features, features))

    for nombre, archivo, validador in (
        ("train.parquet", TRAIN_PATH, validate_split),
        ("test.parquet", TEST_PATH, validate_split),
    ):
        if archivo.exists():
            etapas.append((nombre, validador, pd.read_parquet(archivo)))

    if not etapas:
        print("No hay datos que validar. Ejecuta antes: python -m ayni features")
        return 1

    fallos = 0
    for nombre, validador, datos in etapas:
        try:
            validador(datos)
            print(f"  OK    {nombre:<24} {datos.shape[0]:>7,} filas × {datos.shape[1]} columnas")
        except DataValidationError as error:
            fallos += 1
            print(f"  FALLO {nombre}\n{error}\n")

    print(f"\n{len(etapas) - fallos} de {len(etapas)} etapas cumplen el esquema.")
    return 1 if fallos else 0


def cmd_train(args: argparse.Namespace) -> int:
    # Import perezoso: mlflow y lightgbm tardan segundos y el resto de comandos no los usa.
    from ayni.train import main as entrenar  # noqa: PLC0415

    entrenar(folds=args.folds, registrar_modelo=not args.no_register, solo=args.solo)
    return 0


def cmd_fairness(args: argparse.Namespace) -> int:
    from ayni.fairness import ejecutar  # noqa: PLC0415

    ejecutar(corrida=args.corrida)
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    from ayni.drift import ejecutar  # noqa: PLC0415

    ejecutar(muestra=args.muestra)
    return 0


def cmd_export_model(args: argparse.Namespace) -> int:
    from ayni.export import exportar  # noqa: PLC0415

    exportar(version=args.version)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from ayni.explain import ejecutar  # noqa: PLC0415

    ejecutar(version=args.version, muestra=args.muestra)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from ayni.evaluate import elegir_umbral, evaluar_test  # noqa: PLC0415

    elegir_umbral(args.corrida)
    if not args.confirmar_test:
        print(
            "\nEl test sigue cerrado. Para la evaluación final (una sola vez):\n"
            "  python -m ayni evaluate --confirmar-test"
        )
        return 0
    evaluar_test(args.version)
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    if FEATURES_PATH.exists() and not args.force:
        print(f"Ya existe {FEATURES_PATH.name}; usa --force para regenerarlo.")
        return 0
    print("Construyendo features desde los .sql de sql/…")
    build_features(force=True)
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    if not FEATURES_PATH.exists():
        print("No hay features.parquet. Ejecuta primero: python -m ayni features")
        return 1
    if TRAIN_PATH.exists() and not args.force:
        print(f"Ya existe el split ({TRAIN_PATH.name} / {TEST_PATH.name}); usa --force.")
        return 0
    meta = make_split()
    print(f"  train: {meta['n_train']:,} filas ({meta['tasa_train']:.2%} de impago)")
    print(f"  test : {meta['n_test']:,} filas ({meta['tasa_test']:.2%} de impago)")
    print(f"  fingerprint del test: {meta['test_fingerprint']}")
    print("\nEl test queda congelado: no se usa hasta la evaluación final de la fase 5.")
    return 0


def cmd_pending(args: argparse.Namespace) -> int:
    print(f"'{args.command}' todavía no existe: {PENDING[args.command]}")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ayni", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("data", help="verifica que los CSV crudos estén en data/raw").set_defaults(
        func=cmd_data
    )

    p_download = sub.add_parser("download", help="descarga y descomprime el dataset de Kaggle")
    p_download.add_argument("--force", action="store_true", help="vuelve a descargar el zip")
    p_download.set_defaults(func=cmd_download)

    p_parquet = sub.add_parser("parquet", help="convierte los CSV a Parquet en data/interim")
    p_parquet.add_argument("--force", action="store_true", help="reconvierte aunque ya exista")
    p_parquet.set_defaults(func=cmd_parquet)

    sub.add_parser("check", help="lista filas y columnas de cada Parquet").set_defaults(
        func=cmd_check
    )

    sub.add_parser("validate", help="valida los datos contra los esquemas de Pandera").set_defaults(
        func=cmd_validate
    )

    p_train = sub.add_parser("train", help="entrena con CV y registra las corridas en MLflow")
    p_train.add_argument("--folds", type=int, default=5, help="nº de folds (defecto: 5)")
    p_train.add_argument(
        "--solo", default=None, help="ejecuta solo una corrida (baseline-logreg, lgbm-base, ...)"
    )
    p_train.add_argument(
        "--no-register", action="store_true", help="no registra el mejor modelo en el Registry"
    )
    p_train.set_defaults(func=cmd_train)

    p_fair = sub.add_parser("fairness", help="aprobación y mora por género y tramo de edad")
    p_fair.add_argument("--corrida", default="lgbm-ajustado", help="OOF a usar")
    p_fair.set_defaults(func=cmd_fairness)

    p_drift = sub.add_parser("drift", help="simula producción y mide drift con Evidently")
    p_drift.add_argument("--muestra", type=int, default=3000, help="filas por conjunto")
    p_drift.set_defaults(func=cmd_drift)

    p_export = sub.add_parser("export-model", help="deja el modelo listo para la API")
    p_export.add_argument("--version", default="latest", help="versión del modelo")
    p_export.set_defaults(func=cmd_export_model)

    p_explain = sub.add_parser("explain", help="SHAP global, local y reason codes")
    p_explain.add_argument("--version", default="latest", help="versión del modelo")
    p_explain.add_argument("--muestra", type=int, default=5000, help="filas para el SHAP global")
    p_explain.set_defaults(func=cmd_explain)

    p_eval = sub.add_parser("evaluate", help="umbral por costo (y evaluación final en test)")
    p_eval.add_argument("--corrida", default="lgbm-ajustado", help="OOF a usar")
    p_eval.add_argument(
        "--confirmar-test", action="store_true", help="lee el test congelado (una sola vez)"
    )
    p_eval.add_argument("--version", default="latest", help="versión del modelo registrado")
    p_eval.set_defaults(func=cmd_evaluate)

    p_features = sub.add_parser("features", help="ejecuta los .sql y genera features.parquet")
    p_features.add_argument("--force", action="store_true", help="regenera aunque ya exista")
    p_features.set_defaults(func=cmd_features)

    p_split = sub.add_parser("split", help="split estratificado 80/20 con test congelado")
    p_split.add_argument("--force", action="store_true", help="vuelve a generar el split")
    p_split.set_defaults(func=cmd_split)

    for name, phase in PENDING.items():
        sub.add_parser(name, help=f"pendiente — {phase}").set_defaults(func=cmd_pending)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows: evita que los acentos se rompan cuando la salida va por pipe.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
