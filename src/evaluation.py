"""Evaluación: CV repetida estratificada (pareada entre modelos), OOF, bootstrap, umbral."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

from src import config as C

N_SPLITS, N_REPEATS = 5, 5


def get_cv():
    """Mismo objeto/semilla para todos los modelos => folds idénticos => comparación pareada."""
    return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=C.SEED)


def _fit_fold(est, X, y, tr, va):
    m = clone(est).fit(X.iloc[tr], y.iloc[tr])
    p = m.predict_proba(X.iloc[va])[:, 1]
    yv = y.iloc[va].to_numpy()
    return va, p, {
        "pr_auc": average_precision_score(yv, p),
        "roc_auc": roc_auc_score(yv, p),
        "brier": brier_score_loss(yv, p),
        "f1@0.5": f1_score(yv, (p >= 0.5).astype(int), zero_division=0),
    }


def cross_validate_oof(est, X, y, n_jobs=-1):
    """Devuelve (métricas por fold [25 filas], OOF promedio por alumno sobre las 5 repeticiones)."""
    splits = list(get_cv().split(X, y))
    out = Parallel(n_jobs=n_jobs)(delayed(_fit_fold)(est, X, y, tr, va) for tr, va in splits)
    oof = np.zeros((N_REPEATS, len(y)))
    rows = []
    for i, (va, p, met) in enumerate(out):
        oof[i // N_SPLITS, va] = p
        rows.append({"fold": i, "repeticion": i // N_SPLITS, **met})
    return pd.DataFrame(rows), oof.mean(axis=0)


def summarize_folds(fold_tables: dict) -> pd.DataFrame:
    rows = []
    for name, df in fold_tables.items():
        r = {"modelo": name}
        for m in ["pr_auc", "roc_auc", "brier", "f1@0.5"]:
            r[f"{m}_media"] = df[m].mean()
            r[f"{m}_sd"] = df[m].std(ddof=1)
        rows.append(r)
    return pd.DataFrame(rows).set_index("modelo")


def paired_differences(fold_tables: dict, ref: str, metric="pr_auc") -> pd.DataFrame:
    """Diferencia pareada por fold (modelo - ref). 'real' solo si |media| >= 1 sd de las diferencias."""
    rows = []
    for name, df in fold_tables.items():
        if name == ref:
            continue
        d = df[metric].to_numpy() - fold_tables[ref][metric].to_numpy()
        rows.append({"modelo": name, "ref": ref, "dif_media": d.mean(), "dif_sd": d.std(ddof=1),
                     "folds_modelo_gana": int((d > 0).sum()), "n_folds": len(d),
                     "diferencia_real(|media|>=1sd)": bool(abs(d.mean()) >= d.std(ddof=1))})
    return pd.DataFrame(rows).set_index("modelo")


def bootstrap_ci(y, p, metric_fn, n_boot=2000, seed=C.SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    n = len(y); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric_fn(y[idx], p[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric_fn(y, p)), float(lo), float(hi)


def paired_bootstrap_diff(y, p_a, p_b, metric_fn=average_precision_score, n_boot=2000, seed=C.SEED):
    """IC 95 % de metric(a) - metric(b) remuestreando los mismos alumnos."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); n = len(y); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(y[idx], p_a[idx]) - metric_fn(y[idx], p_b[idx]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(metric_fn(y, p_a) - metric_fn(y, p_b)), float(lo), float(hi)


def choose_threshold(y, p, min_recall=0.80):
    """Máxima precisión sujeta a recall >= min_recall (criterio declarado en ADR-09)."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]  # el último punto no tiene umbral asociado
    ok = rec >= min_recall
    i = np.argmax(np.where(ok, prec, -1))
    return float(thr[i]), float(prec[i]), float(rec[i])


def operating_point(y, p, thr):
    yhat = (np.asarray(p) >= thr).astype(int); y = np.asarray(y)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    return {"umbral": thr, "recall": recall_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "f1": f1_score(y, yhat, zero_division=0), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "alertas_por_100_alumnos": 100 * (tp + fp) / len(y)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    ph = k / n; den = 1 + z ** 2 / n
    c = (ph + z ** 2 / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
    return ph, max(0, c - h), min(1, c + h)


def segment_errors(df_seg: pd.DataFrame, y, p, thr, segment: str) -> pd.DataFrame:
    """FNR (desertores no detectados) y FPR (falsas alertas) por segmento, con IC de Wilson."""
    yhat = (np.asarray(p) >= thr).astype(int); y = np.asarray(y); rows = []
    for g, idx in df_seg.groupby(segment).indices.items():
        yy, hh = y[idx], yhat[idx]
        pos, neg = (yy == 1).sum(), (yy == 0).sum()
        fnr = wilson(int(((hh == 0) & (yy == 1)).sum()), int(pos))
        fpr = wilson(int(((hh == 1) & (yy == 0)).sum()), int(neg))
        rows.append({"segmento": segment, "grupo": g, "n": len(idx), "desertores": int(pos),
                     "prevalencia": yy.mean(), "FNR": fnr[0], "FNR_IC95": f"[{fnr[1]:.2f}, {fnr[2]:.2f}]",
                     "FPR": fpr[0], "FPR_IC95": f"[{fpr[1]:.2f}, {fpr[2]:.2f}]"})
    return pd.DataFrame(rows)
