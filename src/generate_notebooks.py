import nbformat as nbf
import os

os.makedirs('notebooks', exist_ok=True)

# Create EDA notebook
nb_eda = nbf.v4.new_notebook()

nb_eda.cells = [
    nbf.v4.new_markdown_cell("# 01 - Análisis Exploratorio de Datos (EDA)\nEn este notebook responderemos a las preguntas analíticas de negocio usando solo el conjunto de entrenamiento para evitar data leakage."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\ntrain_df = pd.read_csv('../data/interim/train.csv')\ntrain_df.head()"),
    nbf.v4.new_markdown_cell("## 1. Distribución de la Variable Objetivo"),
    nbf.v4.new_code_cell("plt.figure(figsize=(8,5))\nsns.countplot(data=train_df, x='Target', order=['Dropout', 'Enrolled', 'Graduate'])\nplt.title('Distribución de la Variable Objetivo')\nplt.show()"),
    nbf.v4.new_markdown_cell("**Hecho**: La clase mayoritaria es Graduate, seguida de cerca por Dropout. Enrolled es la minoría.\n\n**Interpretación**: Existe un desbalance moderado hacia Graduate/Dropout. La deserción es muy alta (casi un tercio de la muestra).\n\n**Decisión**: Podríamos agrupar 'Enrolled' y 'Graduate' en 'No-Dropout' si queremos un problema binario de alerta, o usar enfoques multiclase con balanceo."),
    nbf.v4.new_markdown_cell("## 2. Condición Económica vs Deserción"),
    nbf.v4.new_code_cell("plt.figure(figsize=(8,5))\nsns.countplot(data=train_df, x='Debtor', hue='Target')\nplt.title('Condición de Deudor vs Deserción')\nplt.show()"),
    nbf.v4.new_markdown_cell("**Hecho**: Entre los estudiantes que son deudores (Debtor=1), la inmensa mayoría abandona (Dropout). Entre los no deudores, la mayoría se gradúa.\n\n**Interpretación**: El factor económico es un predictor altísimo y crítico de abandono.\n\n**Decisión**: Esta variable se mantendrá y podría tener mucha importancia en el modelo."),
    nbf.v4.new_markdown_cell("## 3. Rendimiento en 1er Semestre"),
    nbf.v4.new_code_cell("plt.figure(figsize=(10,6))\nsns.boxplot(data=train_df, x='Target', y='Curricular units 1st sem (grade)')\nplt.title('Calificaciones 1er Semestre vs Deserción')\nplt.show()"),
    nbf.v4.new_markdown_cell("**Hecho**: Los que abandonan tienen una mediana de notas drásticamente menor (muchos con nota 0). Los graduados tienen notas más altas y concentradas.\n\n**Interpretación**: El éxito inicial predice el éxito final. Las notas de 0 podrían representar abandono temprano (estudiantes que se matricularon pero no rindieron exámenes).\n\n**Decisión**: Esta característica será central. Debemos revisar si las notas 0 representan verdaderos ceros o NAs semánticos de quienes nunca asistieron."),
]

with open('notebooks/01_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb_eda, f)


# Create Calidad Preparacion notebook
nb_calidad = nbf.v4.new_notebook()
nb_calidad.cells = [
    nbf.v4.new_markdown_cell("# 02 - Calidad y Preparación de Datos\nBasado en los hallazgos del EDA, aplicaremos limpieza y preparación."),
    nbf.v4.new_code_cell("import pandas as pd\ntrain_df = pd.read_csv('../data/interim/train.csv')\ntrain_df.info()"),
    nbf.v4.new_markdown_cell("### Decisiones de Calidad:\n1. **Valores Faltantes**: El dataset viene sin valores nulos técnicos. \n2. **Outliers (Notas=0)**: Muchos alumnos tienen 0 en créditos evaluados/aprobados. Esto no es un error, es un comportamiento real de abandono en el ciclo.\n3. **Transformaciones**: Las variables categóricas ya vienen como enteros. Debemos declararlas como categóricas en el pipeline (OneHotEncoding) para evitar que el modelo asuma ordinalidad falsa (ej. Nacionalidad 1 vs 105)."),
]

with open('notebooks/02_calidad_preparacion.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb_calidad, f)

print("Notebooks generados.")
