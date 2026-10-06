# Estado Reconstruido (Ronda 1b)

## Archivos Confirmados
- `scripts/verify_references.py` existe y se ejecutó (hay un JSON `verificacion_referencias.json`).
- `src/config.py`: Definición de variables y paths.
- `src/data.py`: Carga y split, congelamiento de datos con hash en metadata.
- `src/features.py`: Creados `FirstSemesterFeatures` y `ColumnDropper` stateless.
- `src/pipelines/preprocessor.py`: Creados transformadores lineales y de árboles.
- `src/models.py`: Implementa clasificadores `DummyClassifier`, la regla de dominio `ApprovedZeroRule`, regresión logística con CV, RF y HGB.
- `src/evaluation.py`: Métricas y bootstrap (OOF, wilson, umbral).
- `src/train.py`: Creado, maneja el guardado con `joblib` y metadata JSON.
- `src/evaluate_test.py`: El script de evaluación final con guardas para el log.

## Git Status
- Todos los cambios no confirmados se pasaron a un commit en la rama `feature/fase-c3-c5`.
- El directorio de `tests/` no fue creado.

## Completitud
Los módulos escritos parecen completos. Falta orquestar la llamada a validación (notebook `03_baseline_y_modelos.ipynb` o pipeline principal). Falta revisión exhaustiva.
