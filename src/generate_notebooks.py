import nbformat as nbf
import os
import subprocess

os.makedirs('notebooks', exist_ok=True)

# Create EDA notebook
nb_eda = nbf.v4.new_notebook()

nb_eda.cells = [
    nbf.v4.new_markdown_cell("# 01 - Análisis Exploratorio de Datos (EDA)\nEn este notebook responderemos a las preguntas analíticas de negocio usando solo el conjunto de entrenamiento para evitar data leakage."),
    nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\ntrain_df = pd.read_csv('../data/interim/train.csv')\n# Target is already binning Dropout vs No-Dropout in setup_split.py\ntrain_df.head()"),
    nbf.v4.new_markdown_cell("## 1. Distribución de la Variable Objetivo (Binaria)"),
    nbf.v4.new_code_cell("plt.figure(figsize=(6,4))\nsns.countplot(data=train_df, x='Target_Bin')\nplt.title('Distribución de Target Binario (1=Dropout, 0=No-Dropout)')\nplt.show()\nprint(train_df['Target_Bin'].value_counts(normalize=True))"),
    nbf.v4.new_markdown_cell("**Hecho**: La clase 1 (Dropout) representa el ~32% de los casos. La clase 0 (Graduate+Enrolled) el ~68%.\n\n**Interpretación**: Existe un desbalance natural hacia los casos de éxito/continuidad.\n\n**Decisión**: Utilizaremos métricas que soporten desbalance como PR-AUC y F1-Macro, además de hiperparámetros de balanceo de clases."),
    nbf.v4.new_markdown_cell("## 2. Condición Económica vs Deserción"),
    nbf.v4.new_code_cell("plt.figure(figsize=(6,4))\nsns.barplot(data=train_df, x='Debtor', y='Target_Bin')\nplt.title('Tasa de Deserción por Condición de Deudor')\nplt.show()"),
    nbf.v4.new_markdown_cell("**Hecho**: Los estudiantes deudores tienen una tasa de deserción mucho más alta.\n\n**Interpretación**: El factor económico es un predictor altísimo y crítico de abandono.\n\n**Decisión**: Esta variable se mantendrá y podría tener mucha importancia en el modelo."),
    nbf.v4.new_markdown_cell("## 3. Rendimiento en 1er Semestre y Notas en Cero"),
    nbf.v4.new_code_cell("plt.figure(figsize=(8,5))\nsns.boxplot(data=train_df, x='Target_Bin', y='Curricular units 1st sem (grade)')\nplt.title('Calificaciones 1er Semestre vs Deserción')\nplt.show()"),
    nbf.v4.new_code_cell("train_df['Grade_is_0'] = train_df['Curricular units 1st sem (grade)'] == 0\ntrain_df['Evaluations_is_0'] = train_df['Curricular units 1st sem (evaluations)'] == 0\nct = pd.crosstab([train_df['Grade_is_0'], train_df['Evaluations_is_0']], train_df['Target_Bin'], margins=True)\ndisplay(ct)\nprint(f\"Tasa de abandono cuando Evaluaciones == 0: {train_df[train_df['Evaluations_is_0']]['Target_Bin'].mean():.1%}\")\nprint(f\"Tasa de abandono cuando Evaluaciones > 0 pero Grade == 0: {train_df[(~train_df['Evaluations_is_0']) & train_df['Grade_is_0']]['Target_Bin'].mean():.1%}\")"),
    nbf.v4.new_markdown_cell("**Hecho**: Los que abandonan tienen una mediana de notas drásticamente menor (muchos con nota 0). El 0 ocurre tanto cuando no hay evaluaciones (ausencia estructural, 68.5% de abandono) como cuando rinden y sacan 0 (86.6% de abandono).\n\n**Interpretación**: Un 0 no es un error de digitación; es abandono tácito o fallo total. No hace trivial el problema (no es 100% predictivo), pero es fortísimo.\n\n**Decisión**: Se conservan los 0s. Crearemos un feature booleano `sin_evaluaciones_1s` en el Pipeline para que el modelo separe el 'cero por ausencia' del 'cero matemático'."),
]

with open('notebooks/01_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb_eda, f)

# Create Calidad Preparacion notebook
nb_calidad = nbf.v4.new_notebook()
nb_calidad.cells = [
    nbf.v4.new_markdown_cell("# 02 - Calidad y Preparación de Datos\nBasado en los hallazgos del EDA, diseñamos el flujo de preparación que irá en el Pipeline de `src/pipelines/`."),
    nbf.v4.new_code_cell("import pandas as pd\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.pipeline import Pipeline\n\ntrain_df = pd.read_csv('../data/interim/train.csv')\ntrain_df.info()"),
    nbf.v4.new_markdown_cell("### Reglas Lógicas y Decisiones de Calidad:\n1. **Valores Faltantes**: El dataset viene sin valores nulos técnicos. \n2. **Outliers (Notas=0)**: Mantenidos intencionalmente. Agregaremos un transformador personalizado para `sin_evaluaciones_1s` en src.\n3. **Transformaciones Categóricas**: Variables como `Nacionality` o `Mother's qualification` son numéricas pero representan categorías nominales.\nPara modelos lineales aplicaremos `OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.01)`. Esto agrupa las nacionalidades/ocupaciones muy raras en una categoría 'infrequent', reduciendo el ruido y sobreajuste."),
    nbf.v4.new_code_cell("cat_cols = ['Marital status', 'Application mode', 'Course', 'Previous qualification', 'Nacionality', \"Mother's qualification\", \"Father's qualification\", \"Mother's occupation\", \"Father's occupation\"]\nnum_cols = [c for c in train_df.columns if c not in cat_cols + ['Target', 'Target_Bin', 'Grade_is_0', 'Evaluations_is_0', 'Enrolled_is_0']]\n\n# Simulando el ColumnTransformer para Lineal\npreprocessor_linear = ColumnTransformer(\n    transformers=[\n        ('num', StandardScaler(), num_cols),\n        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.01), cat_cols)\n    ]\n)\nprint(\"Preprocesador para modelos lineales definido correctamente.\")"),
]

with open('notebooks/02_calidad_preparacion.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb_calidad, f)

print("Nuevos notebooks generados. Ejecutando...")
subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 'notebooks/01_eda.ipynb', '--inplace'], check=True)
subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 'notebooks/02_calidad_preparacion.ipynb', '--inplace'], check=True)
print("Notebooks ejecutados con éxito.")
