# Checklist Rúbrica TP1 (Trabajo Parcial)

A continuación, se presenta la rúbrica del TP1 convertida en un checklist verificable para garantizar que todos los requisitos sean cumplidos de manera trazable. Cada ítem debe estar enlazado a un archivo, celda o figura que evidencie el cumplimiento.

### 1. Problema y Dataset (3 puntos)
- [ ] **Definición del Problema**: Contexto, necesidad, unidad de análisis, pregunta principal y tipo de problema (clasificación/regresión/etc.) formulados. Evidencia: `docs/03_problema_y_dataset.md`.
- [ ] **Utilidad**: Se explica la utilidad esperada y los criterios de utilidad del resultado. Evidencia: `docs/03_problema_y_dataset.md`.
- [ ] **Detalles del Dataset**: Fuente, procedencia, número de observaciones/variables, período temporal, variable objetivo, diccionario de variables documentados. Evidencia: `docs/03_problema_y_dataset.md`.
- [ ] **Licencia e Integridad**: Licencia o condiciones de uso identificadas y registradas con cita a la página oficial. Limitaciones conocidas evaluadas. Evidencia: `docs/03_problema_y_dataset.md`.

### 2. EDA e Interpretación (4 puntos)
- [ ] **Preguntas del EDA**: Se formularon de 5 a 8 preguntas analíticas ligadas a la principal *antes* de graficar. Evidencia: `docs/04_preguntas_eda.md` y `notebooks/01_eda.ipynb`.
- [ ] **Análisis Visual**: Se incluyen gráficos para distribuciones, variable objetivo, relaciones entre variables, diferencias y patrones. Evidencia: `notebooks/01_eda.ipynb`.
- [ ] **Interpretación Tripartita**: Cada gráfico importante incluye: (a) hecho [lo que muestran], (b) interpretación [hipótesis] y (c) decisión para el modelado. Evidencia: `notebooks/01_eda.ipynb`.
- [ ] **Uso del Set Correcto**: El EDA se realizó solo sobre datos de entrenamiento. Evidencia: `notebooks/01_eda.ipynb`.

### 3. Calidad y Preparación de Datos (4 puntos)
- [ ] **Tabla de Decisiones**: Registro completo de detección, evidencia, decisión, justificación, alternativa y riesgo para: faltantes (MCAR/MAR/MNAR), duplicados, tipos, inconsistencias, outliers y transformaciones. Evidencia: `docs/05_decisiones_calidad.md`.
- [ ] **Implementación en Código**: Las decisiones de calidad están implementadas en el pipeline reproducible. Evidencia: `notebooks/02_calidad_preparacion.ipynb` y código en `src/`.

### 4. Train / Validation / Test y Anti-Leakage (2 puntos)
- [ ] **Estrategia de Partición**: Proporciones y estrategia justificada según la naturaleza de los datos. Evidencia: `docs/06_leakage_checklist.md` y `notebooks/02_calidad_preparacion.ipynb`.
- [ ] **Anti-Leakage por Diseño**: Auditoría contra la taxonomía de Kapoor & Narayanan (2023). El set de pruebas está congelado y no se usó para tomar decisiones. Evidencia: `docs/06_leakage_checklist.md`.
- [ ] **Test de Leakage**: Se incluye un test de código automático (`tests/test_no_leakage.py`). Evidencia: `tests/test_no_leakage.py`.

### 5. Flujo Reproducible de Preprocesamiento (2 puntos)
- [ ] **Estructura Pipeline**: Uso de `Pipeline` y `ColumnTransformer` (scikit-learn) para centralizar y aplicar el preprocesamiento de manera segura. Evidencia: `src/pipelines/` y `notebooks/03_baseline_y_modelos.ipynb`.
- [ ] **Reproducibilidad Pura**: Semillas (random_state) fijas, `requirements.txt` actualizado y estructura modular exportable. Evidencia: `requirements.txt` y código en `src/`.

### 6. Baseline, Modelos Preliminares y Evaluación (3 puntos)
- [ ] **Baseline Justificado**: Se empleó `DummyClassifier`/`DummyRegressor` o regla de negocio como base de comparación para demostrar si se supera una solución trivial. Evidencia: `notebooks/03_baseline_y_modelos.ipynb`.
- [ ] **Modelos Comparados**: Al menos dos (sugeridos tres) modelos de diferente familia, entrenados usando CV en el set de entrenamiento. Evidencia: `notebooks/03_baseline_y_modelos.ipynb`.
- [ ] **Evaluación con Incertidumbre**: Reporte de métricas acordes al problema e interpretadas en términos de negocio, con intervalos de incertidumbre para la CV. Evidencia: `notebooks/03_baseline_y_modelos.ipynb`.

### 7. Análisis Crítico y Plan hacia TF1 (2 puntos)
- [ ] **Conclusiones y Limitaciones**: Identificación explícita de hallazgos, problemas no resueltos y lo que *no se puede concluir*. Evidencia: `docs/07_estado_y_plan_tf1.md`.
- [ ] **Plan hacia TF1**: Mapeo y cronograma específico hacia la experimentación con Optuna/MLflow, explicabilidad con SHAP, app con Streamlit, y reporte. Evidencia: `docs/07_estado_y_plan_tf1.md`.
