import json
import numpy as np
import pandas as pd
from src.data import load_train
from src.models import BASELINES, MODELS, make_model
from src.evaluation import cross_validate_oof, summarize_folds, paired_bootstrap_diff, choose_threshold
import os

def main():
    X, y, df = load_train()
    
    # Run cross validation
    fold_tables = {}
    results_oof = {}
    
    # Models to run: baseline (dummy, rule), models
    all_models = BASELINES + MODELS
    for name in all_models:
        df_folds, oof = cross_validate_oof(make_model(name), X, y)
        fold_tables[name] = df_folds
        results_oof[name] = oof
        
    summary = summarize_folds(fold_tables)
    
    # Paired differences
    diff_hgb, lo_hgb, hi_hgb = paired_bootstrap_diff(y, results_oof['hist_gb'], results_oof['logreg'])
    diff_rule, lo_rule, hi_rule = paired_bootstrap_diff(y, results_oof['logreg'], results_oof['regla_aprobadas0'])
    diff_dummy, lo_dummy, hi_dummy = paired_bootstrap_diff(y, results_oof['logreg'], results_oof['dummy_prior'])
    
    # Ablation
    from src import config as C
    df_folds_ab1, oof_ab1 = cross_validate_oof(make_model('logreg', drop=C.FINANCIAL_COLS, add_flags=True), X, y)
    diff_ab1, lo_ab1, hi_ab1 = paired_bootstrap_diff(y, results_oof['logreg'], oof_ab1)
    
    df_folds_ab2, oof_ab2 = cross_validate_oof(make_model('logreg', drop=C.APPROVED_SIGNAL_COLS, add_flags=False), X, y)
    diff_ab2, lo_ab2, hi_ab2 = paired_bootstrap_diff(y, results_oof['logreg'], oof_ab2)
    
    with open('reports/test_evaluation_log.json', 'r', encoding='utf-8') as f:
        test_res = json.load(f)
        
    out = {
        "prevalencia_dropout_train": float(y.mean()),
        "modelos_oof": {},
        "comparaciones": {
            "logreg_vs_rule": {"diff": diff_rule, "ci_lower": lo_rule, "ci_upper": hi_rule},
            "logreg_vs_dummy": {"diff": diff_dummy, "ci_lower": lo_dummy, "ci_upper": hi_dummy},
            "hgb_vs_logreg": {"diff": diff_hgb, "ci_lower": lo_hgb, "ci_upper": hi_hgb}
        },
        "ablacion": {
            "sin_financieras": {"diff": diff_ab1, "ci_lower": lo_ab1, "ci_upper": hi_ab1},
            "sin_aprobadas": {"diff": diff_ab2, "ci_lower": lo_ab2, "ci_upper": hi_ab2}
        },
        "test": test_res
    }
    
    for name in all_models:
        row = summary.loc[name]
        out["modelos_oof"][name] = {
            "pr_auc_media": float(row["pr_auc_media"]),
            "pr_auc_sd": float(row["pr_auc_sd"]),
            "roc_auc_media": float(row["roc_auc_media"]),
            "roc_auc_sd": float(row["roc_auc_sd"])
        }
        
    os.makedirs('reports', exist_ok=True)
    with open('reports/numeros_oficiales.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
        
if __name__ == '__main__':
    main()
