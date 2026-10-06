# Informe Final TP1: Predicción de Abandono Estudiantil

## 1. Problema de Negocio
El abandono universitario representa una pérdida académica y financiera severa. Identificar a los estudiantes con alto riesgo de deserción permite a las instituciones asignar recursos de intervención (tutorías, becas) de forma eficiente. El objetivo es predecir si un estudiante se matriculará, graduará o abandonará (reduciendo el problema a **Abandona vs. Retiene/Gradúa**).

## 2. Dataset
Utilizamos el dataset de la UCI **Predict students' dropout and academic success** (CC BY 4.0).
- Contiene datos demográficos, socioeconómicos, y de desempeño académico del primer semestre.
- Se ha filtrado toda información correspondiente al segundo semestre para evitar fuga de información (*temporal data leakage*).

## 3. EDA y Hallazgos Principales
- **Desbalance**: El 32% de los estudiantes abandonan; el 68% restante está matriculado o graduado.
- **Relación con Calificaciones**: Estudiantes con 0 evaluaciones aprobadas tienen una tasa altísima de abandono.
- **Factores Socioeconómicos**: Estudiantes clasificados como "Deudores" o sin beca presentan tasas de abandono significativamente mayores.

## 4. Calidad y Decisiones
- Se detectó que la calificación "0" indicaba ausencia de evaluaciones. Se crearon indicadores booleanos explícitos y se imputó el 0 por `NaN` en modelos lineales.
- Cero valores nulos reales (las categorías faltantes vienen codificadas).

## 5. Particionamiento (Split) y Anti-Leakage
- Se reservó y bloqueó un test set (20%) con semilla aleatoria fija (42).
- Se elaboró un test de `pytest` que verifica la ausencia de columnas prohibidas y la independencia del test set.

## 6. Pipeline Reproducible
Todo el preprocesamiento fue embebido en objetos `Pipeline` de scikit-learn, con diferentes tratamientos de categorías raras (OneHot con truncamiento para modelos lineales; Ordinal para árboles).

## 7. Baseline y Comparación
El Baseline base es una regla de dominio: "Si no aprueba ninguna unidad, abandona". El modelo final debía mejorar estadísticamente esta regla simple.

## 8. Modelos y Experimentación
Se evaluaron Regresión Logística, Random Forest y HistGradientBoosting. Se midió el PR-AUC empleando Validación Cruzada Estratificada Repetida (5x5). Para afirmar que un modelo mejora a otro, calculamos diferencias pareadas e intervalos Bootstrap sobre OOF.

## 9. Evaluación y Umbral
- **Selección de Modelo**: La Regresión Logística logró el mejor balance entre desempeño (PR-AUC 0.86 en validación) y simplicidad.
- **Resultados en Test**: En la prueba ciega se obtuvo un Recall de ~86% y un PR-AUC de 0.87. Operativamente, el modelo alerta sobre el 40% del alumnado, y 2 de cada 3 alertas son reales.

## 10. Limitaciones y Plan a TF1
**NO podemos concluir causalidad**. El sistema predice el riesgo basándose en el perfil actual del estudiante en la Universidad particular del estudio.
Hacia el TF1 (Semana 15), integraremos Optuna (Ajuste bayesiano), MLFlow (Tracking de experimentos), SHAP (Interpretabilidad explicada) y FastAPI+Streamlit para el despliegue del simulador.
