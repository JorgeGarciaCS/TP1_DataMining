"""Entrena el pipeline final con TODO train y lo guarda con metadatos (joblib + JSON).

MLflow es opcional: si se pasa `mlflow_experiment` se loguea el run automáticamente.
"""
import datetime as dt
import json
import platform

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from src import config as C
from src.data import load_train, sha256_file
from src.models import make_model

VERSION = "0.1-tp1"


def model_paths(name):
    stem = f"dropout_{name}_v{VERSION}"
    return C.MODELS_DIR / f"{stem}.joblib", C.MODELS_DIR / f"{stem}.json"


def train_and_save(
    name: str,
    threshold: float,
    cv_summary: dict,
    extra: dict | None = None,
    mlflow_experiment: str | None = None,
):
    """Entrena el modelo `name`, lo guarda en disco y opcionalmente loguea en MLflow.

    Parámetros
    ----------
    name               : clave del modelo ('logreg', 'hist_gb', etc.)
    threshold          : umbral calibrado en OOF
    cv_summary         : dict con métricas OOF (pr_auc_media, roc_auc_media, …)
    extra              : dict extra que se agrega al JSON de metadatos
    mlflow_experiment  : si se provee, crea un run en ese experimento MLflow
    """
    X, y, _ = load_train()
    pipe = make_model(name).fit(X.drop(columns=[C.ID_COL]), y)
    C.MODELS_DIR.mkdir(exist_ok=True)
    pkl, meta_path = model_paths(name)
    joblib.dump(pipe, pkl)

    meta = {
        "modelo": name,
        "version": VERSION,
        "fecha_entrenamiento": dt.datetime.now().isoformat(timespec="seconds"),
        "semilla": C.SEED,
        "umbral_decision": threshold,
        "clase_positiva": "Dropout (1)",
        "momento_prediccion": "fin del 1.er semestre (sin variables del 2.º semestre)",
        "n_train": int(len(y)),
        "prevalencia_train": float(y.mean()),
        "features_entrada": [c for c in X.columns if c != C.ID_COL],
        "train_sha256": sha256_file(C.TRAIN_PATH),
        "raw_sha256": sha256_file(C.DATA_RAW),
        "modelo_sha256": sha256_file(pkl),
        "versiones": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "cv": cv_summary,
        **(extra or {}),
    }
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    # ── MLflow (opcional) ────────────────────────────────────────────────────
    if mlflow_experiment:
        from src.mlflow_utils import init_run, log_cv_results, log_model_artifact
        with init_run(experiment=mlflow_experiment, run_name=name,
                      tags={"modelo": name, "version": VERSION, "fase": "tp1"}):
            log_cv_results(cv_summary)
            log_model_artifact(pipe, model_name=name, threshold=threshold, metadata=meta)

    return pipe, meta
