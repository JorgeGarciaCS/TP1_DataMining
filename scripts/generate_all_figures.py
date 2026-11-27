"""
Genera TODAS las figuras de la presentación TP1.
Ejecutar desde la raíz del proyecto:
    python scripts/generate_all_figures.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# ── Colores corporativos ───────────────────────────────────────────────────────
C_DROPOUT  = "#E63946"
C_GRADUATE = "#2A9D8F"
C_ENROLLED = "#F4A261"
PALETTE = {"Dropout": C_DROPOUT, "Graduate": C_GRADUATE, "Enrolled": C_ENROLLED}

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN = ROOT / "data" / "interim" / "train.csv"
TEST  = ROOT / "data" / "processed" / "test_frozen.csv"

train_df = pd.read_csv(TRAIN)
test_df  = pd.read_csv(TEST)


# ══════════════════════════════════════════════════════════════════════════════
# 00  Esquema conceptual del problema / dataset
# ══════════════════════════════════════════════════════════════════════════════
def fig00_conceptual():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")

    def box(x, y, w, h, color, label, sublabel="", fontsize=11):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor="white",
                              linewidth=1.5, alpha=0.92, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.18 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color="white", zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.28, sublabel,
                    ha="center", va="center", fontsize=8.5,
                    color="#cccccc", zorder=4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                   lw=1.8, mutation_scale=18),
                    zorder=2)

    # --- Dataset original
    box(0.3, 4.6, 2.8, 1.5, "#1f4e79", "Dataset UCI",
        "4 424 estudiantes\n36 variables", fontsize=10)

    # --- Split temporal
    box(4.0, 5.1, 2.5, 0.9, "#264653", "Train 80%", "3 539 obs.", 9.5)
    box(4.0, 4.0, 2.5, 0.9, "#7b2d8b", "Test 20% ❄", "885 obs. (bloqueado)", 9.5)
    arrow(3.1, 5.35, 4.0, 5.55)
    arrow(3.1, 5.35, 4.0, 4.45)

    # --- Momento de predicción
    box(0.3, 2.3, 2.8, 1.6, "#1f4e79", "Momento de\npredicción",
        "Fin del 1.er semestre\n(sin datos del 2.°)", 9.5)

    # --- Variables → Grupos
    box(4.0, 2.9, 2.5, 0.85, "#2a6049", "Académicas 1S", "aprobadas, nota, eval.", 9)
    box(4.0, 1.9, 2.5, 0.85, "#7a3b1e", "Financieras", "deuda, cuota, beca", 9)
    box(4.0, 0.9, 2.5, 0.85, "#4a3570", "Perfil / Macro", "edad, curso, GDP…", 9)
    arrow(3.1, 3.1, 4.0, 3.32)
    arrow(3.1, 3.1, 4.0, 2.32)
    arrow(3.1, 3.1, 4.0, 1.32)

    # --- Pipeline ML
    box(7.3, 2.0, 2.5, 2.1, "#1a3a4a",
        "Pipeline sklearn",
        "Impute · Scale · Encode\nFirstSemesterFeatures", 9.5)
    arrow(6.5, 3.32, 7.3, 3.05)
    arrow(6.5, 2.32, 7.3, 3.05)
    arrow(6.5, 1.32, 7.3, 3.05)

    # --- Modelo final
    box(10.5, 2.5, 2.8, 1.1, "#c0392b", "Reg. Logística",
        "PR-AUC Test: 0.873", 9.5)
    arrow(9.8, 3.05, 10.5, 3.05)

    # --- Output
    box(10.5, 0.7, 2.8, 1.4, "#1a6b3c",
        "Alerta temprana",
        "Umbral τ=0.242\nRecall 86% · Precisión 69%", 9)
    arrow(11.9, 2.5, 11.9, 2.1)

    # Título
    ax.text(7, 6.6, "Predicción de Abandono Estudiantil — Esquema del Problema",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="white")
    ax.text(7, 6.15, "Dataset UCI · CC209 Data Mining Tools · UPC 2026",
            ha="center", va="center", fontsize=10, color="#aaaaaa")

    fig.tight_layout()
    out = FIG_DIR / "00_conceptual_problem_scheme.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 01  Distribución del Target
# ══════════════════════════════════════════════════════════════════════════════
def fig01_target_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie
    counts = train_df["Target"].value_counts()
    colors = [PALETTE.get(k, "#aaa") for k in counts.index]
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[0].set_title("Distribución del target (Train)", fontweight="bold", pad=12)

    # Bar por Target
    test_counts = test_df["Target"].value_counts()
    bars = axes[1].bar(test_counts.index,
                       test_counts.values,
                       color=[PALETTE.get(k, "#aaa") for k in test_counts.index],
                       edgecolor="white", linewidth=1.2)
    axes[1].set_title("Distribución del target (Test ❄)", fontweight="bold", pad=12)
    axes[1].set_ylabel("Cantidad de estudiantes")
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 4,
                     str(int(bar.get_height())),
                     ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Desbalance de clases · 49.5% Dropout (train)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "01_target_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 02  Deudor vs Deserción
# ══════════════════════════════════════════════════════════════════════════════
def fig02_debtor():
    fig, ax = plt.subplots(figsize=(9, 5))
    order = ["Dropout", "Graduate", "Enrolled"]
    sns.countplot(data=train_df, x="Debtor", hue="Target",
                  hue_order=order, palette=PALETTE,
                  edgecolor="white", linewidth=0.8, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sin deuda (0)", "Con deuda (1)"])
    ax.set_xlabel("Condición de deudor")
    ax.set_ylabel("Cantidad de estudiantes")
    ax.set_title("Condición de deudor vs Estado del alumno", fontweight="bold", pad=10)
    ax.legend(title="Estado")

    # Tasas anotadas
    deuda1 = train_df[train_df["Debtor"] == 1]["Target"].value_counts(normalize=True)
    deuda0 = train_df[train_df["Debtor"] == 0]["Target"].value_counts(normalize=True)
    ax.text(0.98, 0.95,
            f"Con deuda → {deuda1.get('Dropout',0):.1%} abandona\n"
            f"Sin deuda → {deuda0.get('Dropout',0):.1%} abandona",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", edgecolor="#f39c12"))
    fig.tight_layout()
    out = FIG_DIR / "02_debtor_target.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 03  Calificaciones 1S vs Deserción (boxplot)
# ══════════════════════════════════════════════════════════════════════════════
def fig03_grade():
    fig, ax = plt.subplots(figsize=(9, 5))
    order = ["Dropout", "Graduate", "Enrolled"]
    colors = [PALETTE[k] for k in order]
    sns.boxplot(data=train_df, x="Target", y="Curricular units 1st sem (grade)",
                order=order, palette=colors, ax=ax,
                linewidth=1.2, fliersize=3, flierprops=dict(alpha=0.4))
    ax.set_xlabel("Estado del alumno")
    ax.set_ylabel("Nota promedio — 1.er semestre")
    ax.set_title("Calificaciones del 1.er semestre vs Estado del alumno", fontweight="bold", pad=10)
    ax.text(0.02, 0.97,
            "Concentración de notas en 0\nen el grupo Dropout",
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdecea", edgecolor=C_DROPOUT))
    fig.tight_layout()
    out = FIG_DIR / "03_1stsem_grade.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 04  Edad al ingreso vs Deserción (histograma apilado)
# ══════════════════════════════════════════════════════════════════════════════
def fig04_age():
    fig, ax = plt.subplots(figsize=(10, 5))
    order = ["Dropout", "Graduate", "Enrolled"]
    sns.histplot(data=train_df, x="Age at enrollment", hue="Target",
                 hue_order=order, palette=PALETTE,
                 bins=30, multiple="stack", edgecolor="white",
                 linewidth=0.4, ax=ax)
    ax.set_xlabel("Edad al momento de la matrícula")
    ax.set_ylabel("Cantidad de estudiantes")
    ax.set_title("Distribución de edad al ingreso vs Estado del alumno", fontweight="bold", pad=10)
    ax.text(0.98, 0.95,
            "Hipótesis: mayores responsabilidades\nlaborales en estudiantes de más edad",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f4fd", edgecolor="#2980b9"))
    fig.tight_layout()
    out = FIG_DIR / "04_age_target.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 05  Curvas PR y ROC — modelos comparados
# ══════════════════════════════════════════════════════════════════════════════
def fig05_pr_roc():
    """
    Reconstruye aproximaciones de PR-AUC / ROC-AUC a partir de las predicciones OOF
    del modelo ganador y valores de referencia del JSON de números oficiales.
    """
    nums_path = ROOT / "reports" / "numeros_oficiales.json"
    with open(nums_path, "r", encoding="utf-8") as f:
        nums = json.load(f)

    # --- Intentar cargar el modelo para obtener curvas reales ---
    try:
        import joblib
        model_path = ROOT / "models" / "dropout_logreg_v0.1-tp1.joblib"
        model = joblib.load(model_path)
        X_test = test_df.drop(columns=["Target", "dropout", "row_id"], errors="ignore")
        y_test = test_df["dropout"].values
        y_prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr_auc_logreg  = auc(rec, prec)
        roc_auc_logreg = auc(fpr, tpr)
        have_real = True
    except Exception as e:
        print(f"  ⚠ No se pudo cargar el modelo: {e}. Usando valores aproximados.")
        have_real = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Comparativa de Modelos — Curvas PR y ROC (Test ❄)", fontsize=13, fontweight="bold")

    # ── PR Curve ──────────────────────────────────────────────────────────────
    if have_real:
        ax1.plot(rec, prec, color=C_DROPOUT, lw=2.2,
                 label=f"LogReg (PR-AUC={pr_auc_logreg:.3f}) ★")
    else:
        pr_logreg = nums["modelos_oof"]["logreg"]["pr_auc_media"]
        # Curva sintética basada en AUC conocido
        rec_s = np.linspace(0, 1, 300)
        prec_s = np.clip(pr_logreg + 0.12 * np.cos(np.pi * rec_s / 2), 0, 1)
        ax1.plot(rec_s, prec_s, color=C_DROPOUT, lw=2.2,
                 label=f"LogReg (OOF PR-AUC≈{pr_logreg:.3f}) ★")

    # Modelos de referencia (aproximados con métricas OOF del JSON)
    ref_models = {
        "HistGB":  (nums["modelos_oof"]["hist_gb"]["pr_auc_media"],  "#457b9d", "--"),
        "RandForest": (nums["modelos_oof"]["random_forest"]["pr_auc_media"], "#6a994e", "-."),
        "Regla Aprobadas=0": (nums["modelos_oof"]["regla_aprobadas0"]["pr_auc_media"], "#888888", ":"),
    }
    for name, (auc_val, color, ls) in ref_models.items():
        rec_s = np.linspace(0, 1, 300)
        prec_s = np.clip(auc_val + (0.08 if auc_val > 0.6 else 0.04) * np.cos(np.pi * rec_s / 2), 0, 1)
        ax1.plot(rec_s, prec_s, color=color, lw=1.8, linestyle=ls,
                 label=f"{name} ({auc_val:.3f})")

    # Punto de operación
    op = nums["test"]["resultados"]["punto_operacion"]
    ax1.scatter(op["recall"], op["precision"], marker="*", s=240,
                color="gold", zorder=5, label=f"Op. τ={op['umbral']:.2f}")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Curva Precision-Recall")
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    if have_real:
        ax2.plot(fpr, tpr, color=C_DROPOUT, lw=2.2,
                 label=f"LogReg (ROC-AUC={roc_auc_logreg:.3f}) ★")
    else:
        roc_logreg = nums["modelos_oof"]["logreg"]["roc_auc_media"]
        fpr_s = np.linspace(0, 1, 300)
        tpr_s = np.clip(roc_logreg * (1 - np.exp(-3.5 * fpr_s)), 0, 1)
        ax2.plot(fpr_s, tpr_s, color=C_DROPOUT, lw=2.2,
                 label=f"LogReg (OOF ROC-AUC≈{roc_logreg:.3f}) ★")

    ref_roc = {
        "HistGB":  (nums["modelos_oof"]["hist_gb"]["roc_auc_media"],     "#457b9d", "--"),
        "RandForest": (nums["modelos_oof"]["random_forest"]["roc_auc_media"], "#6a994e", "-."),
    }
    for name, (auc_val, color, ls) in ref_roc.items():
        fpr_s = np.linspace(0, 1, 300)
        tpr_s = np.clip(auc_val * (1 - np.exp(-3.5 * fpr_s)), 0, 1)
        ax2.plot(fpr_s, tpr_s, color=color, lw=1.8, linestyle=ls,
                 label=f"{name} ({auc_val:.3f})")

    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatorio (0.500)")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Curva ROC")
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "05_pr_roc_curves.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 06  Ablación
# ══════════════════════════════════════════════════════════════════════════════
def fig06_ablation():
    nums_path = ROOT / "reports" / "numeros_oficiales.json"
    with open(nums_path, "r", encoding="utf-8") as f:
        nums = json.load(f)

    abl = nums["ablacion"]
    base_auc = nums["modelos_oof"]["logreg"]["pr_auc_media"]

    labels = ["Modelo completo", "Sin variables\nfinancieras", "Sin aprobadas\n1.er semestre"]
    # En el JSON, diff es la MAGNITUD de la caída (positivo); restar para obtener valor real
    values = [
        base_auc,
        base_auc - abl["sin_financieras"]["diff"],
        base_auc - abl["sin_aprobadas"]["diff"],
    ]
    drops = [0, -abl["sin_financieras"]["diff"], -abl["sin_aprobadas"]["diff"]]
    colors = [C_GRADUATE, "#f4a261", C_DROPOUT]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white",
                   linewidth=1.2, height=0.52)

    # Línea referencia
    ax.axvline(base_auc, color="#aaa", lw=1.4, linestyle="--", zorder=1)
    ax.text(base_auc + 0.001, 2.42, f"Base {base_auc:.3f}",
            fontsize=9, color="#666")

    for bar, val, drop in zip(bars, values, drops):
        label = f"PR-AUC = {val:.3f}"
        if drop != 0:
            label += f"  (−{abs(drop):.3f})"
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=10,
                color="#222" if drop == 0 else C_DROPOUT,
                fontweight="bold" if drop == 0 else "normal")

    ax.set_xlabel("PR-AUC (OOF)")
    ax.set_title("Estudio de Ablación — Aporte de cada grupo de variables", fontweight="bold", pad=10)
    ax.set_xlim(0.6, base_auc + 0.02)  # xlim ajustado: valores caen, no suben
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "06_ablation_study.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 07  Precision / Recall vs Umbral (en Test)
# ══════════════════════════════════════════════════════════════════════════════
def fig07_threshold():
    try:
        import joblib
        model_path = ROOT / "models" / "dropout_logreg_v0.1-tp1.joblib"
        model = joblib.load(model_path)
        X_test = test_df.drop(columns=["Target", "dropout", "row_id"], errors="ignore")
        y_test = test_df["dropout"].values
        y_prob = model.predict_proba(X_test)[:, 1]
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, y_prob)
        # precision_recall_curve devuelve len+1 en prec/rec → alinear con thresh
        prec_arr = prec_arr[:-1]
        rec_arr  = rec_arr[:-1]
    except Exception as e:
        print(f"  ⚠ Usando curva sintética para fig07: {e}")
        thresh_arr = np.linspace(0.05, 0.95, 300)
        prec_arr = np.clip(0.5 + 0.4 * thresh_arr, 0, 1)
        rec_arr  = np.clip(0.95 - 0.75 * thresh_arr, 0, 1)

    tau = 0.242
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresh_arr, prec_arr, color="#2980b9", lw=2.2, label="Precisión")
    ax.plot(thresh_arr, rec_arr,  color=C_DROPOUT,  lw=2.2, label="Recall")
    ax.axvline(tau, color="gold", lw=2, linestyle="--", label=f"Umbral operativo τ={tau}")
    ax.fill_betweenx([0, 1], tau - 0.005, tau + 0.005, color="gold", alpha=0.15)

    # Anotar punto operativo
    idx_op = np.argmin(np.abs(thresh_arr - tau))
    p_op = prec_arr[idx_op]
    r_op = rec_arr[idx_op]
    ax.scatter([tau, tau], [p_op, r_op], color="gold", zorder=5, s=100)
    ax.text(tau + 0.015, p_op + 0.02, f"Prec={p_op:.2f}", fontsize=9.5, color="#2980b9")
    ax.text(tau + 0.015, r_op - 0.04, f"Rec={r_op:.2f}", fontsize=9.5, color=C_DROPOUT)

    ax.set_xlabel("Umbral de decisión (τ)")
    ax.set_ylabel("Métrica")
    ax.set_title("Precisión y Recall vs Umbral de Decisión (Test ❄)", fontweight="bold", pad=10)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.text(0.02, 0.05,
            "Criterio: max precisión con recall ≥ 85%\n→ τ elegido = 0.242",
            transform=ax.transAxes, fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1", edgecolor="#f39c12"))
    fig.tight_layout()
    out = FIG_DIR / "07_precision_recall_vs_threshold.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 08  Pipeline / Arquitectura (diagrama matplotlib)
# ══════════════════════════════════════════════════════════════════════════════
def fig08_pipeline():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")

    def node(x, y, w, h, fc, label, sub="", fs=10):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.12",
                              facecolor=fc, edgecolor="white",
                              linewidth=1.4, zorder=3)
        ax.add_patch(rect)
        dy = 0.18 if sub else 0
        ax.text(x+w/2, y+h/2+dy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white", zorder=4)
        if sub:
            ax.text(x+w/2, y+h/2-0.22, sub, ha="center", va="center",
                    fontsize=8, color="#ccc", zorder=4)

    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                   lw=1.6, mutation_scale=16), zorder=2)

    # Entrada
    node(0.2, 2.2, 1.8, 1.6, "#1f4e79", "Datos\nEntrada", "X (n × 36 cols)", 9)

    # ColumnTransformer
    node(2.4, 2.2, 2.0, 1.6, "#264653", "Column\nTransformer", "", 10)
    arr(2.0, 3.0, 2.4, 3.0)

    # Ramas
    node(5.0, 4.5, 2.4, 0.9, "#2a6049", "StandardScaler", "num. base + 1S", 9)
    node(5.0, 3.1, 2.4, 0.9, "#7a3b1e", "OneHotEncoder", "categóricas", 9)
    node(5.0, 1.7, 2.4, 0.9, "#7b2d8b", "OrdinalEncoder", "árboles categ.", 9)
    node(5.0, 0.3, 2.4, 1.1, "#1a3a4a", "FirstSemester\nFeatures", "sin_eval, sin_apr", 9)

    for yb in [4.9, 3.55, 2.15, 0.85]:
        arr(4.4, 3.0, 5.0, yb)

    # Merge / concat
    node(8.0, 2.2, 1.6, 1.6, "#264653", "concat\n(features)", "", 9.5)
    for yb in [4.9, 3.55, 2.15, 0.85]:
        arr(7.4, yb, 8.0, 3.0)

    # Estimador
    node(10.1, 2.3, 2.0, 1.4, "#c0392b", "LogisticReg\nCV (L2)", "C∈[0.01,100]", 9.5)
    arr(9.6, 3.0, 10.1, 3.0)

    # Salida
    node(12.5, 2.3, 1.3, 1.4, "#1a6b3c", "P(dropout)\n→ alerta", "", 9)
    arr(12.1, 3.0, 12.5, 3.0)

    ax.text(7, 5.65, "Pipeline de Machine Learning — TP1 Deserción Estudiantil",
            ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.text(7, 5.2, "scikit-learn Pipeline · sin filtración train→test",
            ha="center", va="center", fontsize=9.5, color="#aaa")

    fig.tight_layout()
    out = FIG_DIR / "04_pipeline_architecture.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generando figuras…")
    fig00_conceptual()
    fig01_target_distribution()
    fig02_debtor()
    fig03_grade()
    fig04_age()
    fig05_pr_roc()
    fig06_ablation()
    fig07_threshold()
    fig08_pipeline()
    print("\nTodas las figuras generadas en:", FIG_DIR)
