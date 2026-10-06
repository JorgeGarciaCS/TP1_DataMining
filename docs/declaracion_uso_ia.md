# Declaración de Uso de IA

De acuerdo con las políticas del curso CC209 Data Mining Tools, declaramos el uso de herramientas de Inteligencia Artificial (IA) en el desarrollo de este Trabajo Parcial (TP1).

## Herramientas Utilizadas
- **Claude 3.5 Sonnet / Gemini 1.5 Pro / GPT-4o**: Como pair programmers integrados al entorno (Antigravity).

## Fases y Aplicación
1. **Fase A (Exploración y Rúbrica)**:
   - Resumen del material de clase y syllabus para abstraer convenciones.
   - Conversión de la rúbrica en un checklist de evidencias verificables (`docs/02_checklist_rubrica.md`).
2. **Fase B (Definición del Problema)**:
   - Búsqueda y validación de fuentes bibliográficas (DOI, referenciación cruzada).
   - Estructuración del Datasheet for Datasets (Gebru et al., 2021).
3. **Fase C.1 - C.2 (EDA y Calidad)**:
   - Análisis crítico iterativo (rol de "Abogado del Diablo" y jueces de calidad) para validar desbalance y reglas lógicas (ej. la relación entre `grade == 0` y `evaluations == 0`).
4. **Fase C.3 - C.5 (Leakage, Pipeline, Modelos)**:
   - Detección sistemática de leakage (basado en Kapoor & Narayanan, 2023).
   - Creación de código para intervalos de confianza bootstrap y métricas pareadas.
   - Redacción de pruebas unitarias (`pytest`) para certificar aislamiento del conjunto de test.

## Justificación
La IA no se utilizó para inventar datos o tomar decisiones sin supervisión; por el contrario, cada línea de código escrita o sugerida fue revisada, ejecutada y verificada contra el dataset original para garantizar la trazabilidad y la **reproducibilidad total**.
