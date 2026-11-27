"""Utilidades para logging de experimentos con MLflow.

Uso típico
----------
from src.mlflow_utils import init_run, log_cv_results, log_model_artifact

with init_run(experiment="dropout_modelos", run_name="logreg_baseline"):
    log_cv_results(cv_summary)
    log_model_artifact(pipe, model_name="logreg", threshold=0.242)
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import joblib
import mlflow

ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = ROOT / "mlflow.db"
EXPERIMENT_DEFAULT = "dropout_tp1"


def setup_tracking(uri: str | None = None) -> None:
    """Configura el tracking URI (por defecto: base SQLite del proyecto)."""
    tracking_uri = uri or f"sqlite:///{MLFLOW_DB.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)


@contextmanager
def init_run(
    experiment: str = EXPERIMENT_DEFAULT,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
):
    """Context manager que crea un run de MLflow y lo cierra al salir.

    Parámetros
    ----------
    experiment : nombre del experimento (se crea si no existe)
    run_name   : nombre descriptivo del run
    tags       : dict de tags extra (ej. {"modelo": "logreg", "fase": "tp1"})
    tracking_uri : URI de tracking; None → mlflow.db en la raíz del proyecto
    """
    setup_tracking(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        yield run


def log_cv_results(cv_summary: dict[str, Any]) -> None:
    """Registra métricas OOF de validación cruzada.

    Espera un dict con claves como:
        pr_auc_media, pr_auc_sd, roc_auc_media, roc_auc_sd
    """
    metric_map = {
        "pr_auc_media":  "cv_pr_auc_mean",
        "pr_auc_sd":     "cv_pr_auc_std",
        "roc_auc_media": "cv_roc_auc_mean",
        "roc_auc_sd":    "cv_roc_auc_std",
    }
    for src_key, mlflow_key in metric_map.items():
        if src_key in cv_summary:
            mlflow.log_metric(mlflow_key, float(cv_summary[src_key]))


def log_threshold_metrics(threshold: float, recall: float, precision: float,
                           f1: float, tp: int, fp: int, fn: int, tn: int) -> None:
    """Registra métricas del punto de operación (umbral calibrado)."""
    mlflow.log_metric("threshold", threshold)
    mlflow.log_metric("op_recall", recall)
    mlflow.log_metric("op_precision", precision)
    mlflow.log_metric("op_f1", f1)
    mlflow.log_metric("op_tp", tp)
    mlflow.log_metric("op_fp", fp)
    mlflow.log_metric("op_fn", fn)
    mlflow.log_metric("op_tn", tn)


def log_ablation(ablation: dict[str, dict]) -> None:
    """Registra resultados de ablación como métricas."""
    for group, vals in ablation.items():
        prefix = f"ablation_{group}"
        for k, v in vals.items():
            mlflow.log_metric(f"{prefix}_{k}", float(v))


def log_model_artifact(
    pipeline,
    model_name: str,
    threshold: float,
    metadata: dict | None = None,
) -> None:
    """Serializa el pipeline con joblib y lo sube como artefacto al run activo.

    También registra el threshold y el metadata JSON como parámetros/artefacto.
    """
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("threshold", threshold)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Pipeline serializado
        pkl_path = tmp_path / f"{model_name}.joblib"
        joblib.dump(pipeline, pkl_path)
        mlflow.log_artifact(str(pkl_path), artifact_path="model")

        # Metadata JSON (opcional)
        if metadata:
            meta_path = tmp_path / "metadata.json"
            meta_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            mlflow.log_artifact(str(meta_path), artifact_path="model")


def log_figure(fig_path: str | Path, subfolder: str = "figures") -> None:
    """Sube una figura PNG/SVG como artefacto."""
    mlflow.log_artifact(str(fig_path), artifact_path=subfolder)


def log_params_from_pipeline(pipeline) -> None:
    """Extrae y registra hiperparámetros del estimador final del pipeline."""
    try:
        estimator = pipeline.named_steps.get("clf") or pipeline.steps[-1][1]
        params = estimator.get_params()
        # Solo parámetros serializables y no muy largos
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                mlflow.log_param(k, v)
    except Exception:
        pass  # no romper si el pipeline tiene otra estructura
