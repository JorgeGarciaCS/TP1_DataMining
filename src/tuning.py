"""Tuning bayesiano con Optuna para HistGradientBoostingClassifier.

Cada trial Optuna se loguea automáticamente en MLflow (experimento 'dropout_tf1').
El estudio busca maximizar PR-AUC en validación cruzada OOF (mismos folds que TP1).

Uso desde el notebook 05:
    from src.tuning import run_study
    study = run_study(n_trials=50)
"""
from __future__ import annotations

import warnings
from typing import Any

import mlflow
import numpy as np
import optuna
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score

from src import config as C
from src.data import load_train
from src.evaluation import cross_validate_oof, choose_threshold, get_cv
from src.mlflow_utils import setup_tracking
from src.pipelines.preprocessor import build_pipeline, n_categorical

# Silenciar warnings de versión de sklearn al importar
warnings.filterwarnings("ignore", category=UserWarning)

EXPERIMENT_TF1 = "dropout_tf1"


def _build_hist_gb(trial: optuna.Trial) -> HistGradientBoostingClassifier:
    """Define el espacio de búsqueda de hiperparámetros para HistGradientBoosting."""
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 500, step=50),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 7, 63),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        early_stopping=False,
        categorical_features=list(range(n_categorical())),
        random_state=C.SEED,
    )


def _objective(trial: optuna.Trial, X, y) -> float:
    """Objetivo Optuna: PR-AUC promedio en CV estratificada (mismos folds que TP1).

    Cada trial se loguea como un run hijo en MLflow.
    """
    clf = _build_hist_gb(trial)
    pipe = build_pipeline(clf, "tree")

    # CV rápida (5×1) para el objective — más trials en menos tiempo
    cv = get_cv()
    scores = cross_val_score(
        pipe, X.drop(columns=[C.ID_COL]), y,
        cv=cv, scoring="average_precision", n_jobs=1,
    )
    pr_auc_mean = float(scores.mean())
    pr_auc_std  = float(scores.std(ddof=1))

    # Log en MLflow (run hijo dentro del run del estudio)
    with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
        for k, v in trial.params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("cv_pr_auc_mean", pr_auc_mean)
        mlflow.log_metric("cv_pr_auc_std",  pr_auc_std)
        mlflow.log_metric("trial_number", trial.number)

    return pr_auc_mean


def run_study(
    n_trials: int = 50,
    timeout_seconds: int | None = None,
    tracking_uri: str | None = None,
    show_progress: bool = True,
) -> optuna.Study:
    """Lanza el estudio Optuna y lo encapsula en un run MLflow padre.

    Parámetros
    ----------
    n_trials         : número máximo de trials Optuna
    timeout_seconds  : límite de tiempo alternativo (None = sin límite)
    tracking_uri     : URI de MLflow (None → sqlite local)
    show_progress    : muestra barra de progreso en notebook

    Retorna
    -------
    study : objeto optuna.Study con todos los trials y el mejor resultado
    """
    setup_tracking(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_TF1)

    X, y, _ = load_train()

    optuna.logging.set_verbosity(
        optuna.logging.INFO if show_progress else optuna.logging.WARNING
    )
    study = optuna.create_study(
        direction="maximize",
        study_name="hist_gb_tuning",
        sampler=optuna.samplers.TPESampler(seed=C.SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    with mlflow.start_run(run_name="optuna_hist_gb", tags={"fase": "tf1", "modelo": "hist_gb"}):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("optimizer", "optuna_tpe")

        study.optimize(
            lambda trial: _objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=show_progress,
        )

        # Loguear el mejor trial en el run padre
        best = study.best_trial
        mlflow.log_metric("best_cv_pr_auc", best.value)
        for k, v in best.params.items():
            mlflow.log_param(f"best_{k}", v)

    return study


def evaluate_best_trial(study: optuna.Study) -> dict[str, Any]:
    """Re-entrena con los mejores params en CV completa (5×5) y retorna métricas OOF.

    Usar para comparar con LogReg baseline antes de decidir si hay nuevo campeón.
    """
    X, y, _ = load_train()
    best_params = study.best_trial.params

    clf = HistGradientBoostingClassifier(
        **{k: v for k, v in best_params.items()},
        early_stopping=False,
        categorical_features=list(range(n_categorical())),
        random_state=C.SEED,
    )
    pipe = build_pipeline(clf, "tree")

    fold_df, oof_preds = cross_validate_oof(pipe, X.drop(columns=[C.ID_COL]), y)
    pr_auc_mean = fold_df["pr_auc"].mean()
    pr_auc_std  = fold_df["pr_auc"].std(ddof=1)
    roc_auc_mean = fold_df["roc_auc"].mean()

    threshold, prec, rec = choose_threshold(y.to_numpy(), oof_preds, min_recall=0.85)

    return {
        "pr_auc_media":  pr_auc_mean,
        "pr_auc_sd":     pr_auc_std,
        "roc_auc_media": roc_auc_mean,
        "umbral":        threshold,
        "precision_oof": prec,
        "recall_oof":    rec,
        "fold_df":       fold_df,
        "oof_preds":     oof_preds,
        "best_params":   best_params,
    }
