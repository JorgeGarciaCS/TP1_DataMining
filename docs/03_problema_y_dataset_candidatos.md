# Propuesta de Problemas y Datasets Candidatos

Dado que el tema y dataset están "POR DEFINIR", presentamos 3 opciones que cumplen estrictamente con los criterios de la rúbrica del TP1/TF1: no son "datasets de juguete", tienen suficientes observaciones y variables, presentan problemas de calidad reales para justificar la fase de preparación, permiten modelamiento (clasificación) y aplicación de interpretabilidad (SHAP).

## Candidato 1: Predicción de Abandono y Éxito Académico Estudiantil
* **Problema y Pregunta**: ¿Podemos predecir tempranamente si un estudiante universitario abandonará sus estudios, seguirá matriculado o se graduará, basándonos en sus datos demográficos, socioeconómicos y rendimiento inicial?
* **Unidad de análisis**: El estudiante universitario al final del primer o segundo semestre.
* **Tipo de problema**: Clasificación (multiclase: Dropout, Enrolled, Graduate; o binaria si agrupamos).
* **Dataset**: "Predict students' dropout and academic success" (UCI Machine Learning Repository).
* **Características**: ~4,424 observaciones y 36 variables. Contiene un buen mix de variables categóricas (estado civil, ocupación de padres) y numéricas (calificaciones, tasas de desempleo local).
* **Licencia**: CC BY 4.0 (Uso académico y comercial permitido).
* **Por qué es adecuado**: Relevancia enorme para el contexto universitario. Permite aplicar validación cruzada, tiene desbalance natural de clases, requiere preprocesamiento (encoding, escalado), y es ideal para explicar con SHAP qué factores afectan la retención estudiantil.

## Candidato 2: Predicción de Riesgo de Siniestros de Tránsito Fatales (Perú)
* **Problema y Pregunta**: ¿Qué factores determinan la severidad de un accidente de tránsito (con heridos vs. fatal) en el Perú?
* **Unidad de análisis**: El siniestro de tránsito (accidente individual).
* **Tipo de problema**: Clasificación (Binaria o Multiclase según severidad).
* **Dataset**: Datos abiertos de Siniestros de Tránsito del Observatorio Nacional de Seguridad Vial (ONSV) del Perú.
* **Características**: Miles de registros históricos. Variables incluyen tipo de vía, clima, tipo de vehículo, edad del conductor, departamento.
* **Licencia**: Datos Abiertos del Estado Peruano.
* **Por qué es adecuado**: Cumple con el *plus* de relevancia para el contexto peruano. Requerirá un intenso trabajo de preparación de datos (valores faltantes, categorías inconsistentes escritas a mano a veces, limpieza de fechas), lo que asegura puntaje en esa sección de la rúbrica.

## Candidato 3: Predicción de Enfermedades Cardiovasculares
* **Problema y Pregunta**: ¿Se puede predecir la presencia o ausencia de enfermedad cardiovascular en un paciente a partir de exámenes médicos y hábitos de vida?
* **Unidad de análisis**: El paciente adulto.
* **Tipo de problema**: Clasificación binaria.
* **Dataset**: Cardiovascular Disease dataset (disponible en Kaggle / OpenML).
* **Características**: 70,000 observaciones y 11 variables (edad, altura, peso, presión arterial sistólica/diastólica, colesterol, glucosa, fumador, alcohol, actividad física).
* **Licencia**: CC0 / Dominio Público (verificable en OpenML/Kaggle).
* **Por qué es adecuado**: Es un problema clásico de ML interpretable en salud. Los datos tienen "errores" reales insertados de origen (ej. presiones arteriales negativas o alturas de 10 cm) que obligan a aplicar reglas de negocio para detección de outliers y limpieza, perfecto para la rúbrica.

---

### Recomendación del Equipo
**Recomendamos el Candidato 1: Predicción de Abandono Estudiantil (UCI)**.
* **Razón**: Es el más equilibrado. La documentación oficial del dataset en UCI es impecable (facilita referenciar su procedencia de forma verificable). El problema resuena directamente con nosotros como estudiantes universitarios. El número de características (36) es lo suficientemente grande para justificar Feature Engineering, Reducción Dimensional (PCA/UMAP para el TF1) y modelos basados en árboles (Random Forest, LightGBM), pero lo suficientemente manejable para que los tiempos de experimentación con Optuna no exploten los recursos. Además, está libre de riesgos éticos severos (los datos están anonimizados y desvinculados de PII).

**Riesgos evaluados**:
- Candidato 1: Riesgo bajo. Los datos ya están limpios de PII.
- Candidato 2: Riesgo medio. Los portales del estado a veces cambian las URLs o los diccionarios de datos son escuetos, dificultando el diccionario de variables.
- Candidato 3: Riesgo bajo, pero es un dataset de salud, donde una mala interpretación podría ser éticamente cuestionable si no se acota como "ejercicio académico".

Esperamos la aprobación o selección humana (Puerta de Control #1) para proceder con la FASE C.
