# Preguntas de Defensa (TP1)

**1. ¿Por qué excluyeron el 2.º semestre si eso baja las métricas?**
Porque el objetivo del proyecto es predecir el abandono *a tiempo* para intervenir. Si esperamos a que termine el 2.º semestre para evaluar, el alumno ya desertó o se consolidó, haciendo que la predicción sea inútil operativamente (Data Leakage temporal).

**2. ¿Por qué eligieron un problema binario si el dataset original es multiclase (Dropout, Enrolled, Graduate)?**
Para la universidad, la prioridad operativa es evitar la deserción ("Dropout"). Agrupar "Enrolled" y "Graduate" en la clase negativa permite concentrar el costo de los falsos positivos y falsos negativos exclusivamente en los alumnos en riesgo de abandonar, simplificando la intervención.

**3. ¿Por qué conservar los ceros en las notas del 1.er semestre? ¿No ensucian el promedio?**
Un "0" en este dataset generalmente significa que el alumno no tomó o no aprobó ninguna evaluación, lo cual es una señal de comportamiento, no una medición de conocimiento. Para evitar que los modelos lineales interpreten el "0" como una nota continua baja (lo que introduciría un sesgo numérico), lo reemplazamos por `NaN` y creamos variables indicadoras booleanas (ej. `sin_evaluaciones_1s`).

**4. ¿El modelo de Machine Learning mejora de verdad frente a la regla de "aprobadas = 0"?**
Sí. De acuerdo a la validación cruzada, la regresión logística superó significativamente a la regla estática. La regla de "aprobadas = 0" detecta de forma trivial a los que nunca asisten, pero el modelo logra identificar el abandono en alumnos que *sí* asistieron y tienen notas, utilizando su contexto socioeconómico. El incremento en PR-AUC es de más de +10% con significancia (intervalo bootstrap excluye el 0).

**5. ¿Por qué el PR-AUC es la métrica principal y por qué es creíble?**
El dataset está desbalanceado (32% de prevalencia de Dropout). ROC-AUC infla el rendimiento porque premia los verdaderos negativos (alumnos que se gradúan y el modelo acierta), que son la mayoría. PR-AUC se enfoca en la clase minoritaria, penalizando fuertemente los Falsos Positivos entre las alertas levantadas. Es creíble porque se reporta la media de una validación cruzada 5x5 anidada y el intervalo Bootstrap sobre OOF, no un simple hold-out.

**6. ¿Qué pasa con las variables financieras (Debtor, Tuition up to date)? ¿No hay leakage?**
Existe un **riesgo prospectivo** (Kapoor & Narayanan, 2023) dado que el diccionario original de UCI no especifica si el estado de deudor es al inicio o al final de la carrera. Como mitigación, incluimos una ablación: al quitar las financieras, el modelo LogReg pierde ~3.8% de PR-AUC, lo que indica que, aunque ayudan, el modelo retiene capacidad predictiva sin ellas.

**7. ¿Por qué eligieron el umbral 0.242?**
El umbral no se fijó en 0.5. Se calibró utilizando las predicciones Out-of-Fold (validación) para garantizar capturar al menos al 85% de los potenciales abandonos (Recall objetivo). Un umbral de 0.242 maximiza la precisión sujeta a esa restricción operativa.

**8. ¿Qué riesgos éticos tiene usar este modelo en la vida real?**
El modelo utiliza atributos demográficos sensibles (nacionalidad, edad, becas). El análisis de segmentos en OOF mostró que la tasa de falsos positivos (FPR) varía entre grupos. Podría generar un perfilado injusto, sometiendo a tutorías obligatorias a alumnos internacionales o mayores solo por su demografía.

**9. ¿Qué hace falta para que funcione en *nuestra* universidad (UPC)?**
No es transferible directamente. Entrenamos con datos de un Instituto Politécnico en Portugal (2008-2019) con otra malla curricular, sistema de calificaciones, inflación y PBI. Para aplicarlo en la UPC se requiere un pipeline de reentrenamiento completo con datos históricos propios, utilizando este mismo código como motor (pipeline MLOps).

**10. ¿Por qué la Regresión Logística le ganó al Random Forest?**
Al tener muchas variables categóricas de alta cardinalidad aplicadas con OneHotEncoding, el espacio de features es ralo (sparse). La regresión logística (con regularización fuerte) es altamente eficiente y robusta en este régimen, mientras que el Random Forest puede sufrir sobreajuste o dilución al seleccionar cortes en variables dispersas.

**11. Al final del semestre, ¿cuántas "falsas alarmas" generará el sistema?**
Con las métricas de test, de cada 100 estudiantes evaluados, el sistema levanta aproximadamente 40 alertas. Dado que la precisión es ~68%, alrededor de 12-13 de esas 40 alertas serán estudiantes que en realidad no iban a abandonar (falsos positivos). La universidad debe definir si tiene presupuesto para asistir a estos 13 alumnos "extra".

**12. ¿Por qué afirman que los folds "subestiman la incertidumbre"?**
Basado en Varoquaux (2018), calcular la desviación estándar entre los resultados de los K folds asume independencia entre ellos, pero sus conjuntos de entrenamiento están altísimamente solapados (correlacionados). Para decisiones comparativas formales (saber si un modelo le gana a otro), usamos diferencias pareadas evaluadas mediante Bootstrap.
