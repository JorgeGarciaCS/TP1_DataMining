# Definición del Problema y Dataset

## 1. Problema de Data Science
* **Contexto**: Las instituciones de educación superior enfrentan el reto constante de la deserción estudiantil (dropout). Identificar tempranamente a los estudiantes en riesgo permite asignar recursos de apoyo (tutorías, becas, seguimiento) para mejorar la retención y el éxito académico.
* **Necesidad**: Intervenir a tiempo antes de que el estudiante abandone.
* **Unidad de análisis**: Un estudiante matriculado en su primer año de grado universitario.
* **Pregunta principal**: ¿Podemos predecir si un estudiante abandonará la carrera, basándonos en su información sociodemográfica y su rendimiento al terminar el primer semestre?
* **Tipo de problema**: Clasificación (con clases: Dropout, Enrolled, Graduate).
* **Utilidad esperada**: Un sistema de alerta temprana para la oficina de bienestar estudiantil.
* **Criterio de éxito**: El modelo debe detectar a la mayoría de estudiantes en riesgo (alta sensibilidad/recall en la clase 'Dropout') para poder intervenirlos, justificando el costo de las tutorías frente a la pérdida por matrícula.

## 2. Dataset y Procedencia (Datasheet resumido)
* **Fuente**: UCI Machine Learning Repository.
* **Referencia Oficial**: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.
* **Número de observaciones**: 4,424
* **Número de variables**: 36 originales + 1 variable objetivo.
* **Variable objetivo**: `Target` (Dropout, Enrolled, Graduate).
* **Licencia**: Creative Commons Attribution 4.0 International (CC BY 4.0). Permite uso, compartición y adaptación dando el crédito apropiado.

### Datasheet (Basado en Gebru et al., 2021)
* **Motivación**: Contribuir a la reducción del abandono académico y el fracaso en la educación superior mediante la identificación temprana de estudiantes en riesgo. Financiado por el programa SATDAP (Portugal).
* **Composición**: Datos tabulares. Cada fila es un estudiante. El dataset ya fue previamente curado (sin valores faltantes técnicos reportados) y las variables categóricas vienen pre-codificadas como enteros.
* **Recolección**: Datos extraídos de diferentes bases de datos institucionales de una universidad (registros académicos, demográficos).
* **Usos recomendados**: Entrenamiento de modelos predictivos y estudios de factores de riesgo de deserción en entornos educativos similares.
* **Riesgos Éticos (Corrección)**: A diferencia de la asunción inicial, el dataset **sí contiene atributos sensibles**: Género (`Gender`), Nacionalidad (`Nacionality`), Estado Civil (`Marital status`) y nivel socioeconómico indirecto (`Scholarship holder`, `Debtor`). El riesgo es que el modelo pueda aprender sesgos históricos (ej. penalizar a ciertos grupos demográficos). *Mitigación planificada*: En el TF1 se deberá evaluar la equidad del modelo segmentando el rendimiento por género y nivel socioeconómico para asegurar que los falsos positivos/negativos estén equilibrados.
* **Limitaciones**: Los datos corresponden a una única institución en Portugal, por lo que el modelo podría no generalizar a universidades con distintos contextos de admisión o currículo.

## 3. Momento de Predicción y Riesgo de Leakage Temporal
* **Momento de predicción**: **Final del primer semestre**. La intervención se diseñará para aplicarse antes de que inicie el segundo semestre.
* **Variables no disponibles**: En ese instante, no existe información sobre el rendimiento en el segundo semestre.
* **Decisión de prevención de leakage**: Se eliminarán explícitamente todas las variables relacionadas con el segundo semestre (`Curricular units 2nd sem (credited)`, `Curricular units 2nd sem (enrolled)`, `Curricular units 2nd sem (evaluations)`, `Curricular units 2nd sem (approved)`, `Curricular units 2nd sem (grade)`, `Curricular units 2nd sem (without evaluations)`). Mantenerlas constituiría *Data Leakage* por disponibilidad temporal, inflarían artificialmente el desempeño y harían inútil el modelo en producción.
