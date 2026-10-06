# Predicción de Abandono Estudiantil - TP1

**Curso:** CC209 Data Mining Tools (UPC)
**Integrantes:** Jorge Garcia, Jose Villanueva, Aldair Rivas

## Problema y Pregunta
La deserción universitaria en los primeros semestres es crítica. ¿Podemos predecir qué estudiantes están en riesgo de abandono al finalizar su primer semestre para intervenir a tiempo, utilizando datos demográficos, académicos y macroeconómicos, superando a la regla trivial de "detectar a los que reprueban todo"?

## Dataset
- **Fuente:** [Predict students' dropout and academic success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (UCI Machine Learning Repository).
- **Licencia:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Variable Objetivo:** `Target_Bin` (1 = Dropout, 0 = Graduate + Enrolled). Momento de predicción fijado al fin del 1.er semestre.

## Estructura del Repositorio
```text
.
├── data/               # raw (original), interim, processed (test congelado)
├── docs/               # Informes de decisiones, jueces y checklists (C.1 a C.3)
├── models/             # Modelos finales y metadatos (.joblib, .json)
├── notebooks/          # Notebooks reproducibles de EDA, Preparación y Modelos
├── reports/            # Informe oficial, JSON con números finales y test log
├── scripts/            # Utilidades de generación
├── src/                # Código fuente del pipeline (data, features, models, train)
└── tests/              # Pruebas unitarias de Data Leakage (pytest)
```

## Instrucciones de Ejecución
El proyecto está diseñado para reproducibilidad bit a bit. En un clon limpio:
1. Instalar dependencias: `pip install -r requirements.txt`
2. Correr pruebas (Opcional): `pytest tests/`
3. Reproducir Todo: Las salidas en `notebooks/` y el archivo `models/` ya están generados. Si desea regenerarlos, borre `data/processed/` y ejecute `python -m src.generate_notebooks`. (Nota: La evaluación del test está bloqueada por diseño; el script `src/evaluate_test.py` detendrá ejecuciones accidentales a menos que use `--verify`).

## Resultados Principales
El modelo seleccionado (Regresión Logística con tratamiento de alta cardinalidad) alcanza un **PR-AUC de 0.86 (CV)** y **0.87 (Test)**, capturando al **86%** de los estudiantes en riesgo con un umbral enfocado en recall. 

## Limitaciones y Lo que NO se puede concluir
- No hay relación de **causalidad**.
- El modelo solo evalúa variables al inicio de la carrera o cierre del 1.er semestre; no reemplaza el seguimiento continuo posterior.
- Los datos pertenecen a un único instituto en Portugal; no es un modelo universalmente transferible al Perú sin reentrenamiento.

## Declaración de Uso de IA
Se utilizó IA (pair-programming) para revisiones adversarias iterativas, refactorización a Pipelines de scikit-learn y redacción de tests de leakage. *Toda cifra y decisión fueron verificadas empíricamente con el código del repositorio*. (Ver `docs/declaracion_uso_ia.md` para detalles).
