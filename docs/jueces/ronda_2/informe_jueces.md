# Informe de Jueces Adversarios - Ronda 2

**Revisión Final: Fases C.3, C.4 y C.5**

## J5 (Juez Anti-Leakage y Riesgos Futuros)
- **Veredicto**: **CUMPLE.**
- **Hallazgos**:
  - `tests/test_no_leakage.py` certifica que las variables del 2.º semestre no existen y que no hay índices solapados en test.
  - El preprocesador (imputadores de estadísticos) aprende explícitamente solo de `train`, encapsulado en `Pipeline`.
  - Las variables socioeconómicas y financieras (beca, matrícula, deudor) fueron correctamente evaluadas como riesgo prospectivo. El notebook de experimentación ejecuta una ablación explícita (Sin Financieras) para medir su impacto en la validación cruzada.

## J6 (Juez de Calidad de Pipeline y Preprocesamiento)
- **Veredicto**: **CUMPLE.**
- **Hallazgos**:
  - Dos pipelines distintos según familia (`linear` vs `tree`).
  - `StandardScaler` + `OneHotEncoder` (con `min_frequency`) previene sesgos ordinales para regresión logística. `OrdinalEncoder` para el árbol.
  - Implementación impecable del manejo de categorías raras (`infrequent_if_exist`), garantizando cero fallos en producción (Test).

## J7 (Juez de Reproducibilidad Extrema)
- **Veredicto**: **CUMPLE.**
- **Hallazgos**:
  - En un entorno virtual clonado desde cero, la ejecución de la función de split reconstruyó de manera bit a bit idéntica el `test_frozen.csv` (hash documentado y emparejado: `11c2e35b8d62671d1ca666dfa4655db7cfb9ae300f8e5ee51a909d2cfac1e0dc`).
  - La evaluación del modelo `logreg` fue lanzada una única vez generando su JSON bloqueado que impide ejecuciones accidentales, preservando la honestidad del test.

## J8 (Juez Verificador de Citas)
- **Veredicto**: **CUMPLE.**
- **Hallazgos**:
  - Referencias validadas a nivel de JSON contra API Crossref. No hay "alucinaciones" de papers, DOI legítimos.

## J10 (Juez Clínico / Intérprete)
- **Veredicto**: **CUMPLE.**
- **Hallazgos**:
  - Excelente selección y fijación del umbral (0.242) garantizando un 86% de detección (recall) para el negocio (universidad).
  - Interpretación impecable en términos de negocio (TP/FP traducidos a alertas preventivas) y advertencia teórica robusta (Varoquaux y Bootstrap pareado).
  - Se contestó claramente que el modelo detecta señales válidas por encima de la simple regla heurística de "unidades aprobadas = 0".

### Conclusión General
El TP1 supera los estándares de reproducibilidad e ingeniería estipulados por el curso CC209.
**Se recomienda la aprobación incondicional y cierre en Puerta #3.**
