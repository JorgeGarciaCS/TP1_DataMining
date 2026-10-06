# Informe de Jueces - Ronda 1

Este documento consolida la evaluación de los jueces asignados tras completar la Fase C.1 y C.2.

### J3: Juez de EDA
* **Veredicto**: Aprobado.
* **Comentarios**: Las gráficas en `01_eda.ipynb` responden directamente a las preguntas de `04_preguntas_eda.md`. Se distingue el hecho ("Hecho") de la interpretación ("Interpretación") y la acción resultante ("Decisión"). La tabla cruzada agregada sobre las notas en 0 y las evaluaciones clarifica enormemente la naturaleza conductual de este predictor.
* **Hallazgos**: Ninguno Mayor/Crítico.

### J4: Juez de Calidad de Datos
* **Veredicto**: Aprobado con observaciones menores (Corregidas).
* **Comentarios**: Las decisiones están bien fundamentadas en `docs/05_decisiones_calidad.md`. El tratamiento de las variables de alta cardinalidad mediante agrupar frecuencias bajas (`min_frequency=0.01` en One-Hot) es acertado y evitará explosión dimensional.
* **Hallazgos**: Ninguno Mayor/Crítico.

### J5: Juez Anti-Leakage y Metodología
* **Veredicto**: Aprobado.
* **Comentarios**: El split fue correctamente estratificado con `random_state=42`. El test set se ha congelado en `data/processed/test_frozen.csv`. Las variables del segundo semestre han sido descartadas explícitamente (`src/setup_split.py`), eliminando la principal fuente de data leakage temporal. Se verificó que no existen duplicados exactos en el dataset original.
* **Hallazgos**: Ninguno Mayor/Crítico.

### J8: Juez de Fuentes y Citas
* **Veredicto**: Aprobado.
* **Comentarios**: El archivo `docs/08_referencias.md` existe y todas las referencias (DOI del dataset original, papers sobre Leakage de Kapoor) están marcadas como "Verificada". 
* **Hallazgos**: Ninguno Mayor/Crítico.

### J9: Abogado del Diablo
* **Veredicto**: Aprobado preliminar.
* **Preguntas desafiantes formuladas (para defensa)**:
  1. Si un estudiante rinde evaluación pero saca 0 (ausente en la prueba final, por ejemplo), ¿no estamos aprendiendo a predecir el pasado, dado que sacar 0 ya es en sí mismo el "abandono" semántico?
  2. ¿Qué pasa si el `OneHotEncoder` ignora clases raras que en la realidad sí son determinantes en ciertas facultades?
* **Decisión de Síntesis**: El predictor `sin_evaluaciones_1s` modela un comportamiento, no una consecuencia administrativa inevitable; es válido mantenerlo, pero debemos tener cuidado de no decir que el modelo es "mágico" si solo aprende la regla "Si no hay evaluaciones, deserta". Las preguntas se guardarán para el análisis de errores de la Fase C.6.

### Síntesis General
Todos los hallazgos críticos de la ronda previa han sido resueltos. La base de datos es robusta, no presenta leakage temporal, y las decisiones de calidad están listas para entrar al `Pipeline` de Scikit-Learn.
