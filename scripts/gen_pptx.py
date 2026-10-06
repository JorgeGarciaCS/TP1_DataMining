import collections
import collections.abc
import json
from pptx import Presentation
from pptx.util import Inches, Pt
import os

def add_slide(prs, title, content, notes=None):
    layout = prs.slide_layouts[1] # Title and Content
    slide = prs.slides.add_slide(layout)
    title_shape = slide.shapes.title
    title_shape.text = title
    
    body_shape = slide.shapes.placeholders[1]
    tf = body_shape.text_frame
    
    if isinstance(content, list):
        for i, item in enumerate(content):
            p = tf.add_paragraph() if i > 0 else tf.paragraphs[0]
            p.text = item
            p.level = 0
    else:
        tf.paragraphs[0].text = content
        
    if notes:
        notes_slide = slide.notes_slide
        notes_tf = notes_slide.notes_text_frame
        notes_tf.text = notes
        
def main():
    with open('reports/numeros_oficiales.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    prs = Presentation()
    
    # 1. Problema y pregunta
    add_slide(prs, "Predicción de Abandono Estudiantil (TP1)",
              ["¿Podemos predecir qué estudiantes abandonarán usando su estado al final del 1.er semestre?",
               "Objetivo: Intervenir tempranamente (reducción de deserción).",
               "INTEGRANTES: Jorge Garcia, Jose Villanueva, Aldair Rivas"],
              notes="Expositor: Jorge Garcia. Presenta el problema del negocio.")
              
    # 2. Dataset y limitaciones
    add_slide(prs, "Dataset y Limitaciones",
              ["Fuente: UCI Machine Learning Repository (Instituto en Portugal).",
               f"Prevalencia de abandono: {data['prevalencia_dropout_train']:.1%}",
               "Limitación: Solo una institución. Riesgo de sesgo temporal y espacial."],
              notes="Expositor: Jorge Garcia. Detallar la fuente y el desbalance de clases.")
              
    # 3. EDA
    add_slide(prs, "Análisis Exploratorio (EDA)",
              ["El 0 en calificaciones usualmente indica inasistencia total, no mal desempeño.",
               "Fuerte señal socioeconómica: Deudores y sin beca desertan más."],
              notes="Expositor: Jose Villanueva. Mostrar cómo el '0' nos llevó a replantear el preprocesamiento.")
              
    # 4. Calidad y decisiones
    add_slide(prs, "Calidad de Datos y Preprocesamiento",
              ["Transformación de '0' a NaN en modelos lineales para evitar sesgo continuo.",
               "Categorías agrupadas y truncadas para evitar sobreajuste (Alta cardinalidad).",
               "Codificación especial: OneHot para lineal, Ordinal para árboles."],
              notes="Expositor: Jose Villanueva. Explicar el manejo del 0.")
              
    # 5. Split y Anti-Leakage
    add_slide(prs, "Particionamiento y Anti-Leakage",
              ["Momento de predicción: Exclusivamente fin del 1.er semestre.",
               "Se eliminaron variables del 2.º semestre (Temporal Leakage).",
               "Test set del 20% bloqueado por hash (evaluado 1 sola vez)."],
              notes="Expositor: Aldair Rivas. Resaltar la importancia del momento de predicción.")
              
    # 6. Pipeline
    add_slide(prs, "Arquitectura del Pipeline",
              ["Pipelines de scikit-learn garantizan cero filtración de train a test.",
               "Imputación de medianas solo calculadas dentro del fold de validación.",
               "Modelos evaluados: LogReg, Random Forest, HistGradientBoosting."],
              notes="Expositor: Aldair Rivas. Explicar por qué usar sklearn Pipelines es crucial.")
              
    # 7. Baseline vs Modelos
    rule_diff = data['comparaciones']['logreg_vs_rule']
    logreg = data['modelos_oof']['logreg']
    test = data['test']['resultados']
    add_slide(prs, "Resultados: Modelo Final",
              [f"Modelo ganador: Regresión Logística (PR-AUC OOF: {logreg['pr_auc_media']:.2f})",
               f"Supera a la regla base ('aprobadas=0') por +{rule_diff['diff']:.2f} PR-AUC (IC 95%: [{rule_diff['ci_lower']:.2f}, {rule_diff['ci_upper']:.2f}]).",
               f"Resultados en TEST: PR-AUC {test['pr_auc'][0]:.2f}, Recall {test['punto_operacion']['recall']:.0%}, Precision {test['punto_operacion']['precision']:.0%}."],
              notes="Expositor: Jorge Garcia. Interpretar que el PR-AUC superior indica menos falsas alarmas.")
              
    # 8. Ablación e Interpretación
    abl_fin = data['ablacion']['sin_financieras']
    abl_apr = data['ablacion']['sin_aprobadas']
    add_slide(prs, "Ablación y Sensibilidad",
              [f"Sin variables financieras: Cae {abl_fin['diff']:.3f} en PR-AUC.",
               f"Sin datos de aprobación: Cae {abl_apr['diff']:.3f} en PR-AUC.",
               "Conclusión: El modelo no es dependiente de una sola señal trivial; fusiona demografía y academia."],
              notes="Expositor: Jose Villanueva. Hablar sobre el riesgo prospectivo de las variables financieras.")
              
    # 9. Conclusión de Negocio
    ops = test['punto_operacion']
    add_slide(prs, "Impacto de Negocio (En Test)",
              [f"Umbral seleccionado: {ops['umbral']:.2f}",
               f"De cada 100 alumnos, levantamos {ops['alertas_por_100_alumnos']:.0f} alertas.",
               f"Detectamos al {ops['recall']:.0%} de los estudiantes que abandonarían.",
               f"Sólo {1 - ops['precision']:.0%} son falsas alarmas (Precision {ops['precision']:.0%}).",
               "El modelo identifica casos complejos que la regla simple ignora."],
              notes="Expositor: Aldair Rivas. Traducir el TP, FP a estudiantes de carne y hueso.")
              
    # 10. Plan TF1
    add_slide(prs, "Plan hacia el TF1",
              ["Integrar Optuna para optimización bayesiana de ensambles.",
               "Implementar MLFlow para rastreo de experimentos.",
               "Interpretabilidad con SHAP (valores locales).",
               "Despliegue simulado con FastAPI y Streamlit."],
              notes="Expositor: Equipo. Mostrar la evolución técnica pendiente.")
              
    # Backups
    add_slide(prs, "[Backup] Matriz de Confusión",
              [f"Verdaderos Positivos (Abandonos detectados): {ops['TP']}",
               f"Falsos Positivos (Alarmas innecesarias): {ops['FP']}",
               f"Falsos Negativos (Abandonos omitidos): {ops['FN']}",
               f"Verdaderos Negativos (Retenidos omitidos): {ops['TN']}"],
              notes="Diapositiva de respaldo para preguntas.")
              
    os.makedirs('reports', exist_ok=True)
    prs.save('reports/presentacion_tp1.pptx')

if __name__ == '__main__':
    main()
