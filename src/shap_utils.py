"""Utilidades SHAP para interpretabilidad del modelo LogReg (TF1).

Funciones principales:
- get_feature_names(pipeline)       -> lista de nombres tras el preprocesador
- build_explainer(pipeline, X_bg)   -> shap.LinearExplainer listo para usar
- compute_shap_values(explainer, pipeline, X) -> np.ndarray de valores SHAP
- plot_summary_bar(...)             -> guarda PNG de importancia media |SHAP|
- plot_beeswarm(...)                -> guarda PNG de beeswarm global
- plot_waterfall_single(...)        -> guarda PNG de waterfall para un individuo
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap


# ---------------------------------------------------------------------------
# Extraccion de nombres de features
# ---------------------------------------------------------------------------

def get_feature_names(pipeline):
    """Devuelve la lista de nombres de features tras el paso 'prep' del pipeline.

    El preprocesador tiene la estructura:
        prep -> drop (ColumnDropper) -> fs1 (FirstSemesterFeatures) -> ct (ColumnTransformer)

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline completo (prep + model).

    Returns
    -------
    list[str]
        Nombres de columnas en el mismo orden que la salida del transformador.
    """
    prep = pipeline.named_steps["prep"]
    ct = prep.named_steps["ct"]
    return list(ct.get_feature_names_out())


# ---------------------------------------------------------------------------
# Transformacion de datos
# ---------------------------------------------------------------------------

def transform_X(pipeline, X):
    """Aplica el paso 'prep' del pipeline y devuelve un array denso.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
    X : pd.DataFrame

    Returns
    -------
    np.ndarray  shape (n_samples, n_features)
    """
    prep = pipeline.named_steps["prep"]
    X_t = prep.transform(X)
    # Puede ser una matriz dispersa (OHE) -> densificar
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    return np.asarray(X_t, dtype=float)


# ---------------------------------------------------------------------------
# Construccion del explainer
# ---------------------------------------------------------------------------

def build_explainer(pipeline, X_background):
    """Crea un shap.LinearExplainer para el modelo LogReg del pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline ajustado con pasos 'prep' y 'model'.
    X_background : pd.DataFrame
        Subconjunto de entrenamiento usado como distribucion de fondo.
        Se recomienda una muestra de ~500 filas.

    Returns
    -------
    explainer : shap.LinearExplainer
    X_bg_transformed : np.ndarray
        Datos de fondo ya transformados (util para plots posteriores).
    """
    model = pipeline.named_steps["model"]
    X_bg_t = transform_X(pipeline, X_background)
    masker = shap.maskers.Independent(X_bg_t, max_samples=X_bg_t.shape[0])
    explainer = shap.LinearExplainer(model, masker=masker)
    return explainer, X_bg_t


# ---------------------------------------------------------------------------
# Calculo de valores SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(explainer, pipeline, X):
    """Transforma X y calcula los valores SHAP.

    Parameters
    ----------
    explainer : shap.LinearExplainer
    pipeline : sklearn.pipeline.Pipeline
    X : pd.DataFrame

    Returns
    -------
    shap_vals : np.ndarray  shape (n_samples, n_features)
        Valores SHAP para la clase positiva (Dropout = 1).
    X_transformed : np.ndarray
        X ya transformado (necesario para plots de beeswarm / waterfall).
    """
    X_t = transform_X(pipeline, X)
    explanation = explainer(X_t)

    # LinearExplainer puede devolver Explanation o ndarray
    if hasattr(explanation, "values"):
        shap_vals = explanation.values
    else:
        shap_vals = np.asarray(explanation)

    # Si el modelo es multiclase o devuelve 2 columnas, tomar la clase 1
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    return shap_vals, X_t


# ---------------------------------------------------------------------------
# Plots globales
# ---------------------------------------------------------------------------

def plot_summary_bar(shap_vals, feature_names, out_path, top_n=20):
    """Grafico de barras con la importancia media |SHAP| (global).

    Parameters
    ----------
    shap_vals : np.ndarray  shape (n_samples, n_features)
    feature_names : list[str]
    out_path : str | Path
    top_n : int
        Cuantas features mostrar (las mas importantes).
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:top_n]

    top_names = [feature_names[i] for i in idx]
    top_vals = mean_abs[idx]

    # Orden ascendente para que la mas importante quede arriba
    top_names = top_names[::-1]
    top_vals = top_vals[::-1]

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    bars = ax.barh(range(len(top_names)), top_vals, color="#1f77b4")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Importancia media |valor SHAP|")
    ax.set_title(f"Top {top_n} features por importancia SHAP global (LogReg)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Guardado: {out_path}")


def plot_beeswarm(shap_vals, X_transformed, feature_names, out_path, top_n=20):
    """Beeswarm plot de SHAP (distribucion de valores por feature).

    Parameters
    ----------
    shap_vals : np.ndarray  shape (n_samples, n_features)
    X_transformed : np.ndarray  shape (n_samples, n_features)
    feature_names : list[str]
    out_path : str | Path
    top_n : int
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx_top = np.argsort(mean_abs)[::-1][:top_n]

    sv_top = shap_vals[:, idx_top]
    xv_top = X_transformed[:, idx_top]
    fn_top = [feature_names[i] for i in idx_top]

    explanation = shap.Explanation(
        values=sv_top,
        data=xv_top,
        feature_names=fn_top,
    )

    fig, ax = plt.subplots(figsize=(9, top_n * 0.38 + 1.5))
    shap.plots.beeswarm(explanation, show=False, max_display=top_n)
    plt.title(f"Beeswarm SHAP - Top {top_n} features (LogReg)", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardado: {out_path}")


# ---------------------------------------------------------------------------
# Plot local (waterfall)
# ---------------------------------------------------------------------------

def plot_waterfall_single(
    shap_vals_single, X_single, feature_names, expected_value, out_path, label=""
):
    """Waterfall plot para un unico estudiante.

    Parameters
    ----------
    shap_vals_single : np.ndarray  shape (n_features,)
    X_single : np.ndarray  shape (n_features,)
    feature_names : list[str]
    expected_value : float
        Valor base del explainer (E[f(X)]).
    out_path : str | Path
    label : str
        Titulo adicional para identificar al estudiante.
    """
    explanation = shap.Explanation(
        values=shap_vals_single,
        base_values=expected_value,
        data=X_single,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(explanation, show=False, max_display=15)
    if label:
        plt.title(f"Waterfall SHAP - {label}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardado: {out_path}")
