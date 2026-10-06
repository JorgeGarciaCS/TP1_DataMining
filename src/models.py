"""Baselines y modelos preliminares (sin tuning pesado; Optuna queda para el TF1)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV

from src import config as C
from src.pipelines.preprocessor import build_pipeline, n_categorical

APPROVED = "Curricular units 1st sem (approved)"


class ApprovedZeroRule(BaseEstimator, ClassifierMixin):
    """Regla de dominio: 'unidades aprobadas en el 1.er semestre = 0' -> Dropout.

    No aprende nada del entrenamiento. Su 'probabilidad' es binaria (0/1), por eso su PR-AUC
    equivale a un único punto de operación (precisión y recall de la regla).
    """

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = (X[APPROVED].to_numpy() == 0).astype(float)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_model(name: str, drop=(), add_flags=True):
    """Fábrica de estimadores. Mismos objetos para CV, ablación y modelo final.

    Decisión ADR-07: sin class_weight. El desbalance (≈32 % Dropout) se maneja con la métrica
    (PR-AUC) y con el umbral elegido en OOF; así las probabilidades quedan calibradas (Brier).
    """
    s = C.SEED
    if name == "dummy_prior":      # predice la prevalencia de train: referencia de PR-AUC y Brier
        return DummyClassifier(strategy="prior")
    if name == "dummy_todos_dropout":  # alerta a todos: referencia de recall=1 / F1
        return DummyClassifier(strategy="constant", constant=1)
    if name == "regla_aprobadas0":
        return ApprovedZeroRule()
    if name == "logreg":
        # Grilla pequeña de C (regularización L2) con CV interna -> CV anidada dentro de cada fold
        m = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=5, scoring="average_precision",
                                 max_iter=5000, n_jobs=1)
        return build_pipeline(m, "linear", drop, add_flags)
    if name == "random_forest":
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                   n_jobs=1, random_state=s)
        return build_pipeline(m, "tree", drop, add_flags)
    if name == "hist_gb":
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                                           l2_regularization=1.0, early_stopping=False,
                                           categorical_features=list(range(n_categorical(drop))),
                                           random_state=s)
        return build_pipeline(m, "tree", drop, add_flags)
    raise ValueError(name)


BASELINES = ["dummy_prior", "dummy_todos_dropout", "regla_aprobadas0"]
MODELS = ["logreg", "random_forest", "hist_gb"]
# Orden de preferencia por simplicidad/interpretabilidad si no hay diferencia real (ADR-08)
SIMPLICITY_ORDER = ["logreg", "random_forest", "hist_gb"]
