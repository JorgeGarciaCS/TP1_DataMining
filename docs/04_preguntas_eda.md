# Preguntas de Análisis Exploratorio de Datos (EDA)

Antes de visualizar los datos, se formulan las siguientes preguntas analíticas orientadas por el negocio (la institución educativa) para guiar el EDA. Estas preguntas buscan descubrir perfiles de riesgo en el conjunto de entrenamiento.

1. **Análisis de la Variable Objetivo (Desbalance):** ¿Cuál es la proporción de estudiantes que abandonan (`Dropout`) frente a los que continúan o se gradúan? ¿Existe un desbalance severo que condicione la métrica a optimizar (ej. priorizar Recall o usar pesos de clase)?
2. **Impacto Económico:** ¿Cómo se relaciona el hecho de tener la colegiatura al día (`Tuition fees up to date`), tener deudas (`Debtor`) o ser becario (`Scholarship holder`) con la probabilidad de deserción?
3. **Rendimiento Académico Temprano:** ¿Existe una diferencia significativa en las calificaciones medias del primer semestre (`Curricular units 1st sem (grade)`) entre los estudiantes que abandonan y los que se gradúan?
4. **Carga Académica:** ¿Los estudiantes que aprueban un bajo porcentaje de los créditos matriculados en el primer semestre (`approved` vs `enrolled`) concentran el mayor índice de deserción?
5. **Edad y Perfil Demográfico:** ¿La edad al momento de la matrícula (`Age at enrollment`) influye en la tasa de deserción? (Por ejemplo, ¿los estudiantes de mayor edad o reingresantes abandonan más por obligaciones laborales/familiares?)
6. **Factores Macro y Contexto Externo:** ¿Se observa algún patrón que relacione la tasa de desempleo local (`Unemployment rate`) o la inflación (`Inflation rate`) con el abandono de los estudiantes?
7. **Colinealidad:** ¿Existen variables de las unidades curriculares del primer semestre que estén altamente correlacionadas entre sí (ej. créditos evaluados vs. créditos inscritos) y que puedan generar multicolinealidad en modelos lineales?
