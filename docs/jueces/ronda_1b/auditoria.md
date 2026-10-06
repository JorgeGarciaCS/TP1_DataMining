# Auditoría Independiente (Ronda 1b)

1. **Notebooks 01 y 02 y EDA (Leakage de datos)**:
   - Se verificó que los notebooks `01_eda.ipynb` y `02_calidad_preparacion.ipynb` leen exclusivamente `data/interim/train.csv` (las celdas ejecutadas muestran la lectura correcta). Ningún notebook carga el test set congelado o el dataset crudo completo.
   - Veredicto: **CUMPLE**.

2. **Split y Duplicados**:
   - `src/data.py` genera el split con `random_state=42` fijado. Los duplicados se miden *antes* del particionamiento.
   - Veredicto: **CUMPLE**.

3. **Variables Disponibles al Momento de Predicción**:
   - Variables macroeconómicas (GDP, inflación, desempleo) y financieras (Debtor, Tuition fees up to date) se consideran disponibles en el momento de inscripción o durante el semestre, según los autores. Sin embargo, dado que no hay timestamps específicos para las variables financieras, se clasificaron como riesgo en `src/config.py` y se ha dispuesto un grupo para ablación.
   - Veredicto: **CUMPLE CON RIESGO** (mitigado por ablación planeada en C.5).

4. **Regla sin_evaluaciones_1s**:
   - Construida de forma puramente funcional sobre `Curricular units 1st sem (evaluations)` en `src/features.py`. No usa la variable objetivo (Target o Dropout).
   - Veredicto: **CUMPLE**.

5. **Referencias**:
   - Verificadas mediante un script contra la API de Crossref, arXiv y JMLR. El archivo `verificacion_referencias.json` documenta los DOIs confirmados.
   - Veredicto: **CUMPLE**.
