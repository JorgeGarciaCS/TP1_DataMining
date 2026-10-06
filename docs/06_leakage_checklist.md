# Checklist de Data Leakage (Kapoor & Narayanan, 2023)

| Tipo de Leakage | Estado | Evidencia |
| :--- | :--- | :--- |
| **Leakage temporal** (Variables futuras predecidas o usadas) | **CUMPLE** | Se descartaron todas las variables del 2.º semestre (ver `src/config.py: SECOND_SEM_COLS` y exclusión en `src/data.py`). El momento de predicción está definido al fin del 1.er semestre. |
| **Falta de independencia** (Filas solapadas entre train y test) | **CUMPLE** | `test_no_leakage.py` verifica que no hay intersección de filas entre `train.csv` y `test_frozen.csv`. |
| **Preprocesamiento sobre todo el dataset** (Imputar o escalar con test) | **CUMPLE** | Todo preprocesamiento está encapsulado en un `Pipeline` de Scikit-Learn (`src/pipelines/preprocessor.py`). Las transformaciones se ajustan solo con las estadísticas de `train` durante CV y en el modelo final. Confirmado en `test_no_leakage.py`. |
| **Variables prospectivas mal caracterizadas** (Financial/Macro) | **RIESGO** | Las variables `Debtor`, `Tuition fees up to date` e indicadores macroeconómicos no tienen un timestamp explícito en el diccionario de datos original. Se abordará con *ablación* en la Fase C.5. |
| **Ajuste de hiperparámetros con test set** | **CUMPLE** | El umbral se elige en las predicciones OOF (Out-of-Fold) de la validación cruzada sobre train. La grilla pequeña de `LogisticRegressionCV` usa CV anidada solo en train. El test se toca 1 vez. |
| **Falta de fijación de semilla** | **CUMPLE** | El particionamiento inicial y la partición K-Fold se realizan con `SEED = 42` y están versionados en `src/config.py`. |
