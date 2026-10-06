"""Pipelines de preprocesamiento por familia de modelo (todo se ajusta solo con train).

- Lineal: imputación + StandardScaler (numéricas), imputación + OneHot con agrupación de
  categorías infrecuentes (nominales). El lineal necesita OHE porque el código entero de una
  nacionalidad no tiene orden.
- Árboles/boosting: imputación + OrdinalEncoder con agrupación de infrecuentes; sin escalado
  (los árboles son invariantes a transformaciones monótonas). En HistGradientBoosting las
  columnas ordinales se declaran categóricas => soporte NATIVO (particiones por grupos de
  categorías, no por orden). RandomForest de sklearn 1.4 no tiene soporte nativo: recibe los
  códigos ordinales, que el árbol puede aislar con varios cortes (limitación documentada en ADR-06).
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src import config as C
from src.features import ColumnDropper, FirstSemesterFeatures

MIN_FREQ = 0.01  # categorías con <1 % de train se agrupan en "infrecuente"


def _columns(drop=(), add_flags=True):
    num = [c for c in C.NUMERIC_BASE_COLS if c not in drop]
    cat = [c for c in C.CATEGORICAL_COLS if c not in drop]
    binary = [c for c in C.BINARY_COLS if c not in drop]
    if add_flags:
        binary += [f for f, src in ((C.FLAG_NO_EVAL, "Curricular units 1st sem (evaluations)"),
                                    (C.FLAG_NO_APPROVED, "Curricular units 1st sem (approved)")) if src not in drop]
    return num, cat, binary


def build_preprocessor(family: str, drop=(), add_flags=True) -> Pipeline:
    num, cat, binary = _columns(drop, add_flags)
    if family == "linear":
        ct = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=False)),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(min_frequency=MIN_FREQ, handle_unknown="infrequent_if_exist",
                                                    sparse_output=False))]), cat),
            ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ], remainder="drop", verbose_feature_names_out=False)
    elif family == "tree":
        ct = ColumnTransformer([
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("ord", OrdinalEncoder(min_frequency=MIN_FREQ, handle_unknown="use_encoded_value",
                                                     unknown_value=-1, dtype=np.float64))]), cat),
            ("num", SimpleImputer(strategy="median"), num),
            ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ], remainder="drop", verbose_feature_names_out=False)
    else:
        raise ValueError(family)
    return Pipeline([
        ("drop", ColumnDropper(columns=tuple(drop))),
        ("fs1", FirstSemesterFeatures(add_flags=add_flags)),
        ("ct", ct),
    ])


def n_categorical(drop=()) -> int:
    """Nº de columnas categóricas al inicio de la salida del preprocesador 'tree' (para HGB)."""
    return len([c for c in C.CATEGORICAL_COLS if c not in drop])


def build_pipeline(model, family: str, drop=(), add_flags=True) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(family, drop, add_flags)), ("model", model)])
