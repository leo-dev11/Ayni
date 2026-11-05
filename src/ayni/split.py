"""Split estratificado train/test con semilla fija.

El test se genera **una sola vez** y no se vuelve a tocar hasta la evaluación final de
la fase 5. Para que eso sea verificable y no una promesa, se guarda un hash de los
`SK_ID_CURR` del test: si el archivo cambia, el hash cambia.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from ayni.config import ID_COL, PROCESSED_DIR, RANDOM_STATE, TARGET
from ayni.schema import validate_split

TEST_SIZE = 0.2
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
SPLIT_META = PROCESSED_DIR / "split_meta.json"


def fingerprint(ids: pd.Series) -> str:
    """Hash estable de los IDs del test, para detectar que el test fue modificado."""
    ordenados = sorted(str(int(i)) for i in ids)
    return hashlib.sha256("\n".join(ordenados).encode()).hexdigest()[:16]


def make_split(
    features_path=None, *, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE
) -> dict:
    origen = features_path or (PROCESSED_DIR / "features.parquet")
    df = pd.read_parquet(origen)

    train, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)

    train = validate_split(train)
    test = validate_split(test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False, compression="zstd")
    test.to_parquet(TEST_PATH, index=False, compression="zstd")

    meta = {
        "seed": seed,
        "test_size": test_size,
        "n_total": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "positivos_train": int(train[TARGET].sum()),
        "positivos_test": int(test[TARGET].sum()),
        "tasa_train": round(float(train[TARGET].mean()), 6),
        "tasa_test": round(float(test[TARGET].mean()), 6),
        "n_features": int(df.shape[1] - 2),
        "test_fingerprint": fingerprint(test[ID_COL]),
        "generado": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    SPLIT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
