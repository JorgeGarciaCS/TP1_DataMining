"""
Genera la presentación completa presentacion_tp1.pptx con todas las figuras embebidas.
Ejecutar desde la raíz del proyecto:
    python scripts/gen_pptx.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
import lxml.etree as etree

FIGS = ROOT / "reports" / "figures"
NUMS = ROOT / "reports" / "numeros_oficiales.json"
OUT  = ROOT / "reports" / "presentacion_tp1.pptx"

with open(NUMS, "r", encoding="utf-8") as f:
    nums = json.load(f)

# ── Paleta ────────────────────────────────────────────────────────────────────
BG_DARK  = RGBColor(0x0d, 0x11, 0x17)   # fondo oscuro (diapositivas figura)
BG_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT   = RGBColor(0xE6, 0x39, 0x46)   # rojo
TEAL     = RGBColor(0x2A, 0x9D, 0x8F)
GOLD     = RGBColor(0xF4, 0xA2, 0x61)
NAVY     = RGBColor(0x1F, 0x4E, 0x79)
LGRAY    = RGBColor(0xF5, 0xF5, 0xF5)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _set_bg(slide, color: RGBColor):
    """Rellena el fondo de la diapositiva con un color sólido."""
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_textbox(slide, text, left, top, width, height,
                 font_size=20, bold=False, color=RGBColor(0x22, 0x22, 0x22),
                 align=PP_ALIGN.LEFT, wrap=True):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return tb


def _add_title_bar(slide, title, subtitle=""):
    """Banda superior oscura con título blanco."""
    slide_w = Inches(13.33)
    _add_textbox(slide, title, Inches(0.4), Inches(0.12), Inches(12.5), Inches(0.6),
                 font_size=24, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), align=PP_ALIGN.LEFT)
    if subtitle:
        _add_textbox(slide, subtitle, Inches(0.4), Inches(0.72), Inches(12.5), Inches(0.35),
                     font_size=13, bold=False, color=RGBColor(0xAA, 0xAA, 0xAA))


def _add_bullet_box(slide, bullets, left, top, width, height,
                    font_size=16, bg_color=None, text_color=RGBColor(0x22,0x22,0x22)):
    """Caja con viñetas."""
    if bg_color:
        rect = slide.shapes.add_shape(1, left, top, width, height)  # MSO_SHAPE.RECTANGLE
        rect.fill.solid()
        rect.fill.fore_color.rgb = bg_color
        rect.line.color.rgb = RGBColor(0xCC,0xCC,0xCC)
    tb = slide.shapes.add_textbox(left + Inches(0.12), top + Inches(0.1),
                                  width - Inches(0.24), height - Inches(0.2))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, b in enumerate(bullets):
        p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = ("• " if not b.startswith("  ") else "   ◦ ") + b.lstrip()
        run.font.size = Pt(font_size)
        run.font.color.rgb = text_color


def _embed_figure(slide, path, left, top, width, height):
    if Path(path).exists():
        slide.shapes.add_picture(str(path), left, top, width, height)
    else:
        # Placeholder de texto si falta la imagen
        _add_textbox(slide, f"[Figura: {Path(path).name}]",
                     left, top, width, height,
                     font_size=12, color=RGBColor(0x88,0x88,0x88), align=PP_ALIGN.CENTER)


def new_blank_slide(prs):
    blank_layout = prs.slide_layouts[6]  # completamente en blanco
    return prs.slides.add_slide(blank_layout)


def add_divider(slide, y=Inches(1.0)):
    """Línea horizontal decorativa."""
    from pptx.util import Pt as _Pt
    line = slide.shapes.add_shape(1, Inches(0.4), y, Inches(12.5), Inches(0.025))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()


# ═════════════════════════════════════════════════════════════════════════════
# BUILD
# ═════════════════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

op   = nums["test"]["resultados"]["punto_operacion"]
logreg = nums["modelos_oof"]["logreg"]
hgb    = nums["modelos_oof"]["hist_gb"]
rf     = nums["modelos_oof"]["random_forest"]
rule   = nums["modelos_oof"]["regla_aprobadas0"]
cmp    = nums["comparaciones"]
abl    = nums["ablacion"]

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — Portada
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, BG_DARK)

# Banda de acento
rect = sl.shapes.add_shape(1, 0, Inches(2.8), Inches(13.33), Inches(0.06))
rect.fill.solid(); rect.fill.fore_color.rgb = ACCENT
rect.line.fill.background()

_add_textbox(sl, "PREDICCIÓN DE ABANDONO ESTUDIANTIL",
             Inches(0.6), Inches(1.0), Inches(12.0), Inches(0.9),
             font_size=34, bold=True, color=RGBColor(0xFF,0xFF,0xFF), align=PP_ALIGN.CENTER)
_add_textbox(sl, "Trabajo Práctico 1 — Data Mining Tools  |  CC209  |  UPC 2026",
             Inches(0.6), Inches(1.95), Inches(12.0), Inches(0.5),
             font_size=15, bold=False, color=RGBColor(0xAA,0xAA,0xAA), align=PP_ALIGN.CENTER)
_add_textbox(sl, "Jorge Garcia   ·   Jose Villanueva   ·   Aldair Rivas",
             Inches(0.6), Inches(3.1), Inches(12.0), Inches(0.5),
             font_size=18, bold=True, color=GOLD, align=PP_ALIGN.CENTER)
_add_textbox(sl, "Modelo: Regresión Logística  |  PR-AUC Test: 0.873  |  Recall: 86%",
             Inches(0.6), Inches(4.0), Inches(12.0), Inches(0.45),
             font_size=14, bold=False, color=RGBColor(0xCC,0xCC,0xCC), align=PP_ALIGN.CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — Esquema conceptual (figura oscura ocupa toda la slide)
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, BG_DARK)
_add_textbox(sl, "EL PROBLEMA — Dataset y flujo de trabajo",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=RGBColor(0xFF,0xFF,0xFF))
add_divider(sl, Inches(0.75))
_embed_figure(sl, FIGS / "00_conceptual_problem_scheme.png",
              Inches(0.3), Inches(0.85), Inches(12.73), Inches(6.3))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — Dataset & contexto
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, LGRAY)
_set_bg(sl, RGBColor(0xF8,0xF9,0xFA))
_add_textbox(sl, "DATASET — Origen, escala y desbalance",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=NAVY)
add_divider(sl, Inches(0.75))

_embed_figure(sl, FIGS / "01_target_distribution.png",
              Inches(5.0), Inches(0.85), Inches(8.0), Inches(4.1))

bullets_dataset = [
    "Fuente: UCI Machine Learning Repository",
    "Una institución de Portugal (sesgo geográfico)",
    "4 424 estudiantes, 36 variables de ingreso",
    "Momento de predicción: fin del 1.er semestre",
    f"Prevalencia Dropout (train): {nums['prevalencia_dropout_train']:.1%}",
    "Clases desbalanceadas → métrica PR-AUC (no Accuracy)",
    "Variables del 2.° semestre eliminadas (ADR-01)",
]
_add_bullet_box(sl, bullets_dataset, Inches(0.3), Inches(0.9), Inches(4.5), Inches(4.5),
                font_size=15, bg_color=RGBColor(0xE8,0xEF,0xF8),
                text_color=RGBColor(0x1A,0x1A,0x2E))

_add_textbox(sl, "Desbalance → se optimiza PR-AUC en lugar de Accuracy",
             Inches(0.3), Inches(5.6), Inches(12.5), Inches(0.4),
             font_size=12, bold=False, color=RGBColor(0x55,0x55,0x55))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — Gradiente socioeconómico (Debtor)
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, RGBColor(0xF8,0xF9,0xFA))
_add_textbox(sl, "EDA — Gradiente socioeconómico",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=NAVY)
add_divider(sl, Inches(0.75))

_embed_figure(sl, FIGS / "02_debtor_target.png",
              Inches(0.3), Inches(0.85), Inches(7.5), Inches(4.5))

stats = [
    "Con deuda → 63.0% abandona",
    "Sin deuda → 28.1% abandona",
    "──────────────────────────",
    "Cuota al día → 24.9% dropout",
    "Cuota en mora → 86.7% dropout",
    "──────────────────────────",
    "Con beca → 12.1% dropout",
    "Sin beca → 38.6% dropout",
]
_add_bullet_box(sl, stats, Inches(8.1), Inches(1.0), Inches(4.9), Inches(4.3),
                font_size=14.5, bg_color=RGBColor(0xFD,0xEC,0xEA),
                text_color=RGBColor(0x2D,0x2D,0x2D))
_add_textbox(sl, "Tasas empíricas — dataset de entrenamiento",
             Inches(8.1), Inches(5.3), Inches(4.9), Inches(0.3),
             font_size=10, color=RGBColor(0x88,0x88,0x88))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — Desempeño académico 1S (boxplot notas)
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, RGBColor(0xF8,0xF9,0xFA))
_add_textbox(sl, "EDA — Desempeño académico del 1.er semestre",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=NAVY)
add_divider(sl, Inches(0.75))

_embed_figure(sl, FIGS / "03_1stsem_grade.png",
              Inches(0.3), Inches(0.85), Inches(7.5), Inches(4.5))

facts = [
    "Mediana de nota = 0 en Dropout",
    "El 0 NO siempre es 'desaprobado':",
    "  ◦ Nota=0 + evals=0 → 68.5% dropout",
    "  ◦ Nota=0 + evals>0 → 86.6% dropout",
    "Paradoja del cero: ausencia ≠ bajo rendimiento",
    "Decisión: flag sin_evaluaciones_1s (ADR-03)",
]
_add_bullet_box(sl, facts, Inches(8.1), Inches(1.0), Inches(4.9), Inches(4.0),
                font_size=14.5, bg_color=RGBColor(0xE8,0xF8,0xF4),
                text_color=RGBColor(0x1A,0x3A,0x2E))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6 — Edad al ingreso
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, RGBColor(0xF8,0xF9,0xFA))
_add_textbox(sl, "EDA — Edad al ingreso",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=NAVY)
add_divider(sl, Inches(0.75))
_embed_figure(sl, FIGS / "04_age_target.png",
              Inches(0.3), Inches(0.85), Inches(8.5), Inches(5.0))

age_notes = [
    "Pico de Dropout: 18-22 años",
    "Dropout crece nuevamente >30 años",
    "Hipótesis: responsabilidades laborales",
    "  (marcado como hipótesis, no causal)",
]
_add_bullet_box(sl, age_notes, Inches(9.0), Inches(1.5), Inches(4.0), Inches(3.0),
                font_size=14, bg_color=RGBColor(0xE8,0xEF,0xF8),
                text_color=RGBColor(0x1A,0x1A,0x2E))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7 — Pipeline / Arquitectura
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, BG_DARK)
_add_textbox(sl, "INGENIERÍA — Arquitectura del Pipeline ML",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=RGBColor(0xFF,0xFF,0xFF))
add_divider(sl, Inches(0.75))
_embed_figure(sl, FIGS / "04_pipeline_architecture.png",
              Inches(0.2), Inches(0.85), Inches(12.9), Inches(5.9))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8 — Resultados comparativos (curvas PR + ROC)
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, RGBColor(0xF8,0xF9,0xFA))
_add_textbox(sl, "RESULTADOS — Comparativa de modelos",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=NAVY)
add_divider(sl, Inches(0.75))
_embed_figure(sl, FIGS / "05_pr_roc_curves.png",
              Inches(0.3), Inches(0.85), Inches(9.0), Inches(5.5))

tabla = [
    "Modelo           PR-AUC    ROC-AUC",
    "─────────────────────────────────",
    f"★ LogReg L2      {logreg['pr_auc_media']:.3f}     {logreg['roc_auc_media']:.3f}",
    f"  HistGB          {hgb['pr_auc_media']:.3f}     {hgb['roc_auc_media']:.3f}",
    f"  Rand Forest     {rf['pr_auc_media']:.3f}     {rf['roc_auc_media']:.3f}",
    f"  Regla base      {rule['pr_auc_media']:.3f}     {rule['roc_auc_media']:.3f}",
    "─────────────────────────────────",
    "LogReg elegido por parsimonia",
    "e interpretabilidad (empate",
    "estadístico con HistGB, IC95%",
    "del diff incluye 0)",
]
_add_bullet_box(sl, tabla, Inches(9.5), Inches(1.0), Inches(3.6), Inches(5.0),
                font_size=11, bg_color=RGBColor(0xE8,0xEF,0xF8),
                text_color=RGBColor(0x1A,0x1A,0x2E))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9 — Ablación
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, RGBColor(0xF8,0xF9,0xFA))
_add_textbox(sl, "ANÁLISIS DE SENSIBILIDAD — Estudio de Ablación",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=NAVY)
add_divider(sl, Inches(0.75))
_embed_figure(sl, FIGS / "06_ablation_study.png",
              Inches(0.3), Inches(0.85), Inches(8.0), Inches(5.0))

abl_notes = [
    f"Sin variables financieras:",
    f"  ◦ Caída: −{abl['sin_financieras']['diff']:.3f} PR-AUC",
    f"  ◦ IC 95%: [{abl['sin_financieras']['ci_lower']:.3f}, {abl['sin_financieras']['ci_upper']:.3f}]",
    "",
    f"Sin aprobadas 1S:",
    f"  ◦ Caída: −{abl['sin_aprobadas']['diff']:.3f} PR-AUC",
    f"  ◦ IC 95%: [{abl['sin_aprobadas']['ci_lower']:.3f}, {abl['sin_aprobadas']['ci_upper']:.3f}]",
    "",
    "Cada grupo aporta información",
    "independiente (no sinergia formal).",
    "El señor académico 1S es el más",
    "influyente.",
]
_add_bullet_box(sl, abl_notes, Inches(8.5), Inches(1.0), Inches(4.6), Inches(5.3),
                font_size=13.5, bg_color=RGBColor(0xFD,0xEC,0xEA),
                text_color=RGBColor(0x2D,0x2D,0x2D))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 10 — Precision/Recall vs Umbral
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, RGBColor(0xF8,0xF9,0xFA))
_add_textbox(sl, "CALIBRACIÓN DEL UMBRAL — Punto de operación",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=NAVY)
add_divider(sl, Inches(0.75))
_embed_figure(sl, FIGS / "07_precision_recall_vs_threshold.png",
              Inches(0.3), Inches(0.85), Inches(8.5), Inches(5.5))

thr_notes = [
    f"Umbral τ = {op['umbral']:.3f}",
    "Criterio: max precisión con",
    "  recall ≥ 85% (OOF)",
    "",
    f"Resultados en Test ❄",
    f"  Recall:    {op['recall']:.1%}",
    f"  Precision: {op['precision']:.1%}",
    f"  F1:        {op['f1']:.3f}",
    f"  TP: {op['TP']}  FP: {op['FP']}",
    f"  FN: {op['FN']}  TN: {op['TN']}",
]
_add_bullet_box(sl, thr_notes, Inches(9.1), Inches(1.0), Inches(4.0), Inches(5.5),
                font_size=13.5, bg_color=RGBColor(0xE8,0xEF,0xF8),
                text_color=RGBColor(0x1A,0x1A,0x2E))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 11 — Impacto de Negocio
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, BG_DARK)
_add_textbox(sl, "IMPACTO — ¿Qué significa en la práctica?",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=RGBColor(0xFF,0xFF,0xFF))
add_divider(sl, Inches(0.75))

cards = [
    ("86%", "de los estudiantes\nque abandonarían\nson detectados", ACCENT),
    (f"{op['alertas_por_100_alumnos']:.0f}", "alertas por cada\n100 estudiantes", GOLD),
    (f"{1-op['precision']:.0%}", "son falsas alarmas\n(costo acotado)", TEAL),
    ("1 sola vez", "evaluado en test\n(protocolo sellado)", RGBColor(0x6A,0x99,0x4E)),
]

x_start = Inches(0.5)
for i, (num_txt, label, color) in enumerate(cards):
    x = x_start + i * Inches(3.2)
    rect = sl.shapes.add_shape(1, x, Inches(1.5), Inches(2.9), Inches(3.5))
    rect.fill.solid(); rect.fill.fore_color.rgb = color
    rect.line.fill.background()
    _add_textbox(sl, num_txt, x, Inches(1.8), Inches(2.9), Inches(1.2),
                 font_size=48, bold=True, color=RGBColor(0xFF,0xFF,0xFF), align=PP_ALIGN.CENTER)
    _add_textbox(sl, label, x, Inches(2.95), Inches(2.9), Inches(1.3),
                 font_size=14, bold=False, color=RGBColor(0xFF,0xFF,0xFF), align=PP_ALIGN.CENTER)

_add_textbox(sl, "Comparado con umbral τ=0.50 (default): se recuperarían 77 estudiantes adicionales",
             Inches(0.5), Inches(5.5), Inches(12.0), Inches(0.5),
             font_size=13, bold=False, color=RGBColor(0xAA,0xAA,0xAA), align=PP_ALIGN.CENTER)

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 12 — Plan TF1
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, BG_DARK)
_add_textbox(sl, "PRÓXIMOS PASOS — Plan hacia el TF1",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=22, bold=True, color=RGBColor(0xFF,0xFF,0xFF))
add_divider(sl, Inches(0.75))

steps = [
    ("MLflow", "Tracking de experimentos (semanas 8-9)", TEAL),
    ("Optuna", "Tuning bayesiano: HistGradientBoosting / LightGBM (sem. 10-11)", GOLD),
    ("SHAP", "Interpretabilidad global + local (sem. 12)", ACCENT),
    ("FastAPI + Streamlit", "API de inferencia y UI de riesgo (sem. 13-14)", RGBColor(0x6A,0x99,0x4E)),
]
for i, (step, desc, color) in enumerate(steps):
    y = Inches(1.3) + i * Inches(1.3)
    rect = sl.shapes.add_shape(1, Inches(0.4), y, Inches(12.5), Inches(1.1))
    rect.fill.solid(); rect.fill.fore_color.rgb = RGBColor(0x1A,0x22,0x2E)
    rect.line.color.rgb = color; rect.line.width = Pt(2)
    num_box = sl.shapes.add_shape(1, Inches(0.4), y, Inches(0.5), Inches(1.1))
    num_box.fill.solid(); num_box.fill.fore_color.rgb = color
    num_box.line.fill.background()
    _add_textbox(sl, str(i+1), Inches(0.4), y + Inches(0.2), Inches(0.5), Inches(0.6),
                 font_size=20, bold=True, color=RGBColor(0xFF,0xFF,0xFF), align=PP_ALIGN.CENTER)
    _add_textbox(sl, step, Inches(1.1), y + Inches(0.08), Inches(3.0), Inches(0.45),
                 font_size=16, bold=True, color=color)
    _add_textbox(sl, desc, Inches(1.1), y + Inches(0.55), Inches(11.5), Inches(0.45),
                 font_size=13, bold=False, color=RGBColor(0xCC,0xCC,0xCC))

# ─────────────────────────────────────────────────────────────────────────────
# SLIDE BACKUP — Matriz de confusión
# ─────────────────────────────────────────────────────────────────────────────
sl = new_blank_slide(prs)
_set_bg(sl, RGBColor(0xF8,0xF9,0xFA))
_add_textbox(sl, "[Backup] Matriz de Confusión — Test ❄",
             Inches(0.4), Inches(0.1), Inches(12.5), Inches(0.6),
             font_size=20, bold=True, color=NAVY)
add_divider(sl, Inches(0.75))

matrix = [
    f"Verdaderos Positivos (TP) — Abandonos detectados:   {op['TP']}",
    f"Falsos Positivos (FP) — Alertas innecesarias:         {op['FP']}",
    f"Falsos Negativos (FN) — Abandonos no detectados:    {op['FN']}",
    f"Verdaderos Negativos (TN) — Retenidos correctos:    {op['TN']}",
    "",
    f"Umbral: {op['umbral']:.3f}   |   Precision: {op['precision']:.1%}   |   Recall: {op['recall']:.1%}   |   F1: {op['f1']:.3f}",
    f"PR-AUC Test: {nums['test']['resultados']['pr_auc'][0]:.4f}  IC95%: [{nums['test']['resultados']['pr_auc'][1]:.3f}, {nums['test']['resultados']['pr_auc'][2]:.3f}]",
    f"ROC-AUC Test: {nums['test']['resultados']['roc_auc'][0]:.4f}",
]
_add_bullet_box(sl, matrix, Inches(1.0), Inches(1.3), Inches(11.0), Inches(5.0),
                font_size=16, bg_color=RGBColor(0xE8,0xEF,0xF8),
                text_color=RGBColor(0x1A,0x1A,0x2E))

# ─────────────────────────────────────────────────────────────────────────────
prs.save(str(OUT))
print(f"✓ Presentación guardada en: {OUT}")
print(f"  Diapositivas: {len(prs.slides)}")
