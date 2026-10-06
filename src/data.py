"""Carga, objetivo binario, exclusión por disponibilidad temporal y split congelado."""
import hashlib
import json
import datetime as dt

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as C


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw() -> pd.DataFrame:
    """Lee data.csv (sep=';'), limpia espacios/tabuladores en nombres y agrega row_id."""
    df = pd.read_csv(C.DATA_RAW, sep=";")
    df.columns = [c.strip() for c in df.columns]  # 'Daytime/evening attendance\t' -> sin tab
    df.insert(0, C.ID_COL, range(len(df)))
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Objetivo binario (ADR-04) y exclusión de columnas del 2.º semestre (ADR-01).

    Es una regla fija de diseño (no depende de los valores), por eso puede ir antes del split.
    """
    out = df.drop(columns=C.SECOND_SEM_COLS)
    out[C.TARGET] = (out[C.TARGET_RAW] == "Dropout").astype(int)
    return out


def make_split(force: bool = False) -> dict:
    """Crea el split 80/20 estratificado por el objetivo binario, UNA vez.

    Si el test congelado ya existe, no lo regenera: solo verifica su hash contra el manifiesto.
    """
    if C.TEST_PATH.exists() and C.SPLIT_MANIFEST.exists() and not force:
        man = json.loads(C.SPLIT_MANIFEST.read_text(encoding="utf-8"))
        assert sha256_file(C.TEST_PATH) == man["test_sha256"], "El test congelado fue modificado"
        assert sha256_file(C.TRAIN_PATH) == man["train_sha256"], "train.csv fue modificado"
        return man

    df = prepare(load_raw())
    n_dup = int(df.drop(columns=[C.ID_COL]).duplicated().sum())  # duplicados ANTES de partir
    train, test = train_test_split(df, test_size=C.TEST_SIZE, random_state=C.SEED, stratify=df[C.TARGET])
    C.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train.sort_values(C.ID_COL).to_csv(C.TRAIN_PATH, index=False)
    test.sort_values(C.ID_COL).to_csv(C.TEST_PATH, index=False)
    feat = [c for c in df.columns if c not in (C.ID_COL, C.TARGET, C.TARGET_RAW)]
    man = {
        "fecha": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": C.SEED, "test_size": C.TEST_SIZE, "estratificado_por": C.TARGET,
        "n_total": len(df), "n_train": len(train), "n_test": len(test),
        "prevalencia_train": round(float(train[C.TARGET].mean()), 4),
        "prevalencia_test": round(float(test[C.TARGET].mean()), 4),
        "duplicados_exactos_antes_split": n_dup,
        "filas_identicas_train_test": int(len(train[feat].merge(test[feat], how="inner"))),
        "raw_sha256": sha256_file(C.DATA_RAW),
        "train_sha256": sha256_file(C.TRAIN_PATH),
        "test_sha256": sha256_file(C.TEST_PATH),
    }
    C.SPLIT_MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    return man


def load_train():
    """Devuelve (X, y, df) del conjunto de entrenamiento. Única fuente para EDA y CV."""
    df = pd.read_csv(C.TRAIN_PATH)
    X = df.drop(columns=[C.TARGET, C.TARGET_RAW])
    return X, df[C.TARGET], df


def load_test_for_final_evaluation():
    """Solo debe llamarlo src/evaluate_test.py (evaluación única registrada)."""
    df = pd.read_csv(C.TEST_PATH)
    X = df.drop(columns=[C.TARGET, C.TARGET_RAW])
    return X, df[C.TARGET], df
