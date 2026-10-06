# Matriz de Herramientas (Decisiones)

| Etapa | Herramienta | Decisión TP1 | Alternativa | Motivo (Evidencia) | Pendiente TF1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EDA & Visualización | `pandas`, `seaborn` | Aprobada | `plotly` | Rápido análisis estático en notebooks. | No. Se podría usar `plotly` para interactividad en Streamlit. |
| Gestión de Entornos | `pip` + `requirements.txt` | Aprobada | `poetry` / `conda` | Menos fricción inicial; reproducible con versiones exactas. | Posible paso a `uv` o `poetry` si crecen las dependencias. |
| Seguimiento de Modelos | Archivos JSON | Manual (.json, .joblib) | `mlflow` | Simplicidad en TP1. Bloqueo de reescritura implementado a mano en `evaluate_test.py`. | **Sí**. Implementación de servidor local `mlflow` para tracking. |
| Preprocesamiento | Scikit-Learn Pipelines | Aprobada | `pandas` apply / `Feature-engine` | Previene categóricamente el data leakage temporal aislando `fit` en cada iteración de `KFold`. | No. Seguir con Pipelines. |
| Selección de Modelos | CV Stratified 5x5 | Aprobada | Train-Validation-Test (holdout simple) | Dataset ruidoso; 1 hold-out tiene mucha varianza. Bootstrap pareado confirmó la significancia. | Búsqueda bayesiana de hiperparámetros (`Optuna`). |
| Algoritmo | `LogisticRegressionCV` | Aprobada (Ganador) | Ensambles (RF, HGB) | Resultó el mejor en OOF frente a los árboles gracias al tratamiento OHE de alta cardinalidad. | Probar LightGBM o XGBoost con Optuna. |
| Interpretabilidad | Coeficientes Lineales / Importancias | Básico | `SHAP` | Suficiente para baseline. | **Sí**. Explicabilidad local con SHAP values. |
| Servidor / API | N/A | No aplica en TP1 | `FastAPI` | El TP1 finaliza en la validación de test offline. | **Sí**. Exponer el modelo final en un endpoint. |
| Interfaz Gráfica | N/A | No aplica en TP1 | `Streamlit` | El TP1 es 100% código/notebooks. | **Sí**. Subir un CSV de estudiantes simulado. |
