# Estado Actual y Plan para TF1

## 1. Conclusiones del Modelo (TP1)

El modelo seleccionado fue **Regresión Logística**, que resultó empates o superó a modelos más complejos (Random Forest, HistGradientBoosting) gracias a la regularización robusta y el tratamiento adecuado de las variables categóricas de alta dimensionalidad.

### Interpretación del Desempeño (Lenguaje de Proyecto)
- **Umbral OOF Fijo**: Se ajustó el umbral de decisión a **0.242** en los datos de validación para garantizar capturar al menos al 85% de los potenciales abandonos, priorizando la retención.
- **Rendimiento en Test (Honesto)**: 
  - De cada 100 estudiantes evaluados con el modelo al fin del 1.er semestre, se alertará sobre 40 de ellos.
  - El **86% de los alumnos que efectivamente terminarían abandonando son detectados** y recibirán intervención temprana (TP = 245, FN = 39).
  - De las alarmas levantadas, el **68%** son casos reales de abandono (Precision). Un 32% (FP = 112) recibirán seguimiento preventivo pese a no estar en riesgo extremo.
- **Comparación con el Baseline (Regla Aprobadas=0)**: El modelo mejora drásticamente respecto a la regla determinista. Mientras que la regla detecta de forma trivial a los que no aprueban cursos, el modelo logra predecir con alta precisión el abandono de estudiantes que *sí* asistieron y fueron evaluados, usando información demográfica y socioeconómica combinada.
- **Error Estándar vs Bootstrap**: Como advierte Varoquaux (2018), el error estándar calculado entre los folds de la validación cruzada subestima la incertidumbre real de generalización. Por ello, hemos utilizado intervalos de confianza Bootstrap sobre las predicciones *Out-Of-Fold* completas, garantizando significancia robusta antes de decir que un modelo supera al baseline.

### Lo que NO podemos concluir
- **Causalidad**: No podemos decir que becar a un estudiante *causará* que no abandone, solo que las becas correlacionan con retención.
- **Predicción a largo plazo**: Las predicciones son válidas solo para el estado al término del 1.er semestre. Variables post-inscripción pero pre-semestre pueden tener ruido temporal.

## 2. Plan de Trabajo para TF1 (Trabajo Final)

Para el TF1 (Semana 15), se proponen las siguientes evoluciones (todo código y pipeline actual es reutilizable):

1. **Ajuste Fino (Hyperparameter Tuning)**:
   - Implementar `Optuna` con búsqueda bayesiana para afinar el HistGradientBoosting o LightGBM, dado que en el TP1 se corrieron con hiperparámetros por defecto.

2. **Tracking de Experimentos**:
   - Integrar `MLFlow` local para rastrear modelos, artefactos, umbrales y gráficos asociados a cada corrida en vez de diccionarios JSON aislados.

3. **Interpretabilidad Profunda (SHAP)**:
   - Usar valores de Shapley para explicar la predicción a nivel de individuo. **Precaución**: SHAP puede verse engañado por variables altamente correlacionadas. Se debe agrupar u omitir colineales antes de interpretar.

4. **Despliegue / Aplicación**:
   - Encapsular la inferencia en una API REST con `FastAPI`.
   - Construir una interfaz en `Streamlit` para que el personal administrativo cargue perfiles o lotes de estudiantes y visualice alertas tempranas.

## 3. Cronograma TF1 (Semanas 8 a 15)
- **Semanas 8-9**: Integración de MLFlow y refactorización de configuración.
- **Semanas 10-11**: Experimentación con Optuna (exploración de SMOTE, class weights, interacciones complejas).
- **Semana 12**: Módulo de interpretabilidad con SHAP (cálculo global y local).
- **Semanas 13-14**: Desarrollo de la API FastAPI y la aplicación Streamlit.
- **Semana 15**: Presentación y defensa del TF1, con despliegue local reproducible.
