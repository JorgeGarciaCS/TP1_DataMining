"""Transformadores de features (sin estado y sin acceso a la variable objetivo)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config as C

EVALS = "Curricular units 1st sem (evaluations)"
APPROVED = "Curricular units 1st sem (approved)"
GRADE = "Curricular units 1st sem (grade)"


class FirstSemesterFeatures(BaseEstimator, TransformerMixin):
    """Corrige la semántica del 0 en el 1.er semestre (ADR-03 revisado).

    Evidencia (src/data + auditoría ronda 1b): en todo el dataset `grade == 0` <=> `approved == 0`
    y la nota mínima distinta de 0 es 9.8. Por tanto la nota es el promedio de las unidades
    APROBADAS: con 0 aprobadas la nota no está definida (no es "sacó 0").

    - sin_evaluaciones_1s = 1 si no rindió ninguna evaluación en el 1.er semestre.
    - sin_aprobadas_1s    = 1 si no aprobó ninguna unidad (nota no definida).
    - La nota se convierte a NaN cuando no está definida; el imputador del pipeline la rellena
      con la mediana de TRAIN. Así el modelo lineal no interpreta 0 como "nota muy baja".

    Solo usa columnas X; nunca la variable objetivo. `fit` no aprende nada (stateless).
    """

    def __init__(self, add_flags: bool = True, nan_undefined_grade: bool = True):
        self.add_flags = add_flags
        self.nan_undefined_grade = nan_undefined_grade

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        if self.add_flags and EVALS in X:
            X[C.FLAG_NO_EVAL] = (X[EVALS] == 0).astype(int)
        if self.add_flags and APPROVED in X:
            X[C.FLAG_NO_APPROVED] = (X[APPROVED] == 0).astype(int)
        if self.nan_undefined_grade and GRADE in X and APPROVED in X:
            X[GRADE] = X[GRADE].where(X[APPROVED] > 0, np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.add_flags:
            names += [n for n, src in ((C.FLAG_NO_EVAL, EVALS), (C.FLAG_NO_APPROVED, APPROVED)) if src in names]
        return np.asarray(names, dtype=object)


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Elimina columnas (para ablaciones) dentro del pipeline."""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns])
