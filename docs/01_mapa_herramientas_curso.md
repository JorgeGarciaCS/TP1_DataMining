# Mapa de Herramientas del Curso

Basado en el material de las clases y laboratorios del curso Data Mining Tools (CC209):

## Fase: Análisis Exploratorio de Datos (EDA)
- **Pandas**: Manipulación y exploración tabular.
- **Matplotlib / Seaborn**: Visualización de distribuciones, relaciones y diferencias entre grupos.
- **ydata-profiling**: Soporte para autoEDA rápido (solo como apoyo, no como entregable final).

## Fase: Preparación de Datos y Feature Engineering
- **Scikit-Learn**: 
  - `SimpleImputer` para valores faltantes.
  - `StandardScaler`, `MinMaxScaler` para escalado.
  - `OneHotEncoder`, `OrdinalEncoder` para variables categóricas.
- **ColumnTransformer**: Para aplicar diferentes transformaciones según el tipo de dato de manera paralela.

## Fase: Modelamiento y Pipelines
- **Scikit-Learn**:
  - `Pipeline`: Para encapsular el preprocesamiento y el modelo en un flujo unificado y evitar data leakage.
  - Modelos base: `DummyClassifier`, `DummyRegressor`.
  - Modelos preliminares: Regresión Logística, Ridge, Random Forest, HistGradientBoosting, XGBoost, LightGBM.

## Fase: Experimentación y Evaluación
- **Scikit-Learn**:
  - Validación cruzada (`cross_val_score`, `StratifiedKFold`, `KFold`).
  - Métricas de clasificación: Matriz de confusión, Precision, Recall, F1-score, ROC-AUC, PR-AUC.
  - Métricas de regresión: MAE, RMSE, R².
- **Optuna**: Búsqueda de hiperparámetros (para el TF1).
- **MLflow**: Tracking de experimentos y modelos (para el TF1).

## Fase: Interpretabilidad y Análisis de Errores
- **SHAP**: Importancia de variables e interpretaciones locales/globales (para el TF1).

## Otras Técnicas (Segunda mitad del curso / TF1)
- **Clustering** (ej. K-Means, DBSCAN).
- **Reducción Dimensional**: PCA, UMAP.
- **Deep Learning / Transfer Learning**.

## Fase: Despliegue (TF1)
- **Streamlit** / **FastAPI**: Creación de apps o APIs para interactuar con el modelo.
- **Docker**: Contenedorización para portabilidad.
- **Joblib**: Persistencia del pipeline (artefacto serializado).
