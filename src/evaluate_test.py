"""Evaluación ÚNICA del test congelado, con modelo y umbral ya fijados.

Uso:
  python -m src.evaluate_test --model <nombre>   # primera y única vez: crea el registro
  python -m src.evaluate_test --verify           # reproducibilidad: recalcula y compara con el registro
                                                 # (no se toma ninguna decisión con este resultado)
Si el registro ya existe, el modo normal se niega a correr.
"""
import argparse
import datetime as dt
import json
import sys

import joblib
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src import config as C
from src.data import load_test_for_final_evaluation, sha256_file
from src.evaluation import bootstrap_ci, operating_point
from src.train import model_paths


def _compute(name):
    pkl, meta_path = model_paths(name)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert sha256_file(pkl) == meta["modelo_sha256"], "El modelo cambió desde que se fijó"
    pipe = joblib.load(pkl)
    X, y, _ = load_test_for_final_evaluation()
    p = pipe.predict_proba(X.drop(columns=[C.ID_COL]))[:, 1]
    thr = meta["umbral_decision"]
    ap = bootstrap_ci(y, p, average_precision_score)
    roc = bootstrap_ci(y, p, roc_auc_score)
    br = bootstrap_ci(y, p, brier_score_loss)
    res = {"pr_auc": ap, "roc_auc": roc, "brier": br, "prevalencia_test": float(y.mean()),
           "punto_operacion": operating_point(y, p, thr)}
    return res, meta, pkl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        log = json.loads(C.TEST_LOG.read_text(encoding="utf-8"))
        res, _, _ = _compute(log["modelo"])
        same = json.dumps(res["punto_operacion"], sort_keys=True) == json.dumps(log["resultados"]["punto_operacion"], sort_keys=True) \
            and abs(res["pr_auc"][0] - log["resultados"]["pr_auc"][0]) < 1e-12
        print("VERIFICACIÓN:", "IDÉNTICO al registro" if same else "DIFERENTE (revisar)")
        sys.exit(0 if same else 1)
    if C.TEST_LOG.exists():
        sys.exit(f"El test ya fue evaluado ({C.TEST_LOG}). No se reevalúa. Use --verify.")
    res, meta, pkl = _compute(a.model)
    log = {"fecha_evaluacion": dt.datetime.now().isoformat(timespec="seconds"), "modelo": a.model,
           "n_evaluaciones_del_test_para_decision": 1,
           "test_sha256": sha256_file(C.TEST_PATH), "modelo_sha256": sha256_file(pkl),
           "umbral_fijado_en_train_oof": meta["umbral_decision"], "resultados": res}
    C.TEST_LOG.parent.mkdir(exist_ok=True)
    C.TEST_LOG.write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(log, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
