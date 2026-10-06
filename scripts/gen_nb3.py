import json
import subprocess

def create_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Baseline y Modelos (Fase C.5)\n",
                "Orquestación de entrenamiento, validación cruzada, y elección de umbral."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.append('..')\n",
                "\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "import pandas as pd\n",
                "from src.data import load_train\n",
                "from src.models import BASELINES, MODELS, make_model\n",
                "from src.evaluation import cross_validate_oof, summarize_folds, paired_differences, paired_bootstrap_diff, choose_threshold, segment_errors\n",
                "from src.train import train_and_save\n",
                "\n",
                "X, y, df = load_train()\n",
                "print(f'Train shape: {X.shape}, Positives: {y.sum()} ({y.mean():.2%})')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Baselines"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "results_oof = {}\n",
                "fold_tables = {}\n",
                "\n",
                "for name in BASELINES:\n",
                "    print(f'Evaluando {name}...')\n",
                "    df_folds, oof = cross_validate_oof(make_model(name), X, y)\n",
                "    fold_tables[name] = df_folds\n",
                "    results_oof[name] = oof\n",
                "\n",
                "display(summarize_folds(fold_tables).loc[BASELINES])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Modelos (LogReg, Random Forest, HistGradientBoosting)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for name in MODELS:\n",
                "    print(f'Evaluando {name}...')\n",
                "    df_folds, oof = cross_validate_oof(make_model(name), X, y)\n",
                "    fold_tables[name] = df_folds\n",
                "    results_oof[name] = oof\n",
                "\n",
                "display(summarize_folds(fold_tables).loc[MODELS])\n",
                "\n",
                "print('\\nDiferencias pareadas vs Regla de Aprobadas = 0:')\n",
                "display(paired_differences(fold_tables, ref='regla_aprobadas0'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Diferencias Pareadas entre Modelos (OOF Bootstrap)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "diff, lo, hi = paired_bootstrap_diff(y, results_oof['hist_gb'], results_oof['logreg'])\n",
                "print(f'HGB vs LogReg (PR-AUC): Diff={diff:.4f} 95% CI [{lo:.4f}, {hi:.4f}]')\n",
                "\n",
                "diff_rf, lo_rf, hi_rf = paired_bootstrap_diff(y, results_oof['random_forest'], results_oof['logreg'])\n",
                "print(f'RF vs LogReg (PR-AUC): Diff={diff_rf:.4f} 95% CI [{lo_rf:.4f}, {hi_rf:.4f}]')\n",
                "\n",
                "best_model = 'logreg' if lo <= 0 and lo_rf <= 0 else ('hist_gb' if diff > diff_rf else 'random_forest')\n",
                "print(f'\\nModelo seleccionado: {best_model}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Ablación"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src import config as C\n",
                "for ablation_name, cols_to_drop, drop_flags in [\n",
                "    ('Sin Financieras', C.FINANCIAL_COLS, False),\n",
                "    ('Sin Aprobadas', C.APPROVED_SIGNAL_COLS, True)\n",
                "]:\n",
                "    print(f'\\nAblación: {ablation_name}')\n",
                "    df_folds_ab, oof_ab = cross_validate_oof(make_model(best_model, drop=cols_to_drop, add_flags=not drop_flags), X, y)\n",
                "    diff_ab, lo_ab, hi_ab = paired_bootstrap_diff(y, results_oof[best_model], oof_ab)\n",
                "    print(f'Impacto (Diff OOF PR-AUC original vs ablación): {diff_ab:.4f} 95% CI [{lo_ab:.4f}, {hi_ab:.4f}]')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Selección de Umbral en OOF (y Entrenamiento Final)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "thr, prec, rec = choose_threshold(y, results_oof[best_model], min_recall=0.85)\n",
                "print(f'Umbral seleccionado: {thr:.4f} (Recall OOF: {rec:.4f}, Prec OOF: {prec:.4f})')\n",
                "\n",
                "print('\\nEntrenando modelo final en todo train y guardando...')\n",
                "pipe, meta = train_and_save(best_model, thr, summarize_folds({best_model: fold_tables[best_model]}).iloc[0].to_dict())\n",
                "print(f\"Guardado exitoso. Hash: {meta['modelo_sha256']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Análisis Preliminar de Errores por Segmento (sobre OOF)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "for segment in C.SENSITIVE_COLS:\n",
                "    err = segment_errors(df, y, results_oof[best_model], thr, segment)\n",
                "    display(err[['segmento', 'grupo', 'n', 'FNR', 'FNR_IC95', 'FPR', 'FPR_IC95']])"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open("notebooks/03_baseline_y_modelos.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
        
    print("Ejecutando notebook con nbconvert...")
    subprocess.run([
        r".\.venv\Scripts\jupyter.exe", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=1200",
        "notebooks/03_baseline_y_modelos.ipynb"
    ], check=True)

if __name__ == "__main__":
    create_notebook()
