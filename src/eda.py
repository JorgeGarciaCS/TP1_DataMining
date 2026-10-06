import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# Create directories
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/interim', exist_ok=True)

# Load data
df = pd.read_csv('data/raw/data.csv', sep=';')

# Split data (80/20 stratified)
X = df.drop(columns=['Target'])
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_df = pd.concat([X_train, y_train], axis=1)

# Drop 2nd sem variables from training set (leakage prevention)
cols_to_drop = [
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)'
]
train_df = train_df.drop(columns=cols_to_drop)

# Save train set for reproducibility
train_df.to_csv('data/interim/train.csv', index=False)

# ---- EDA Plots ----

# 1. Target Distribution
plt.figure(figsize=(8,5))
sns.countplot(data=train_df, x='Target', order=['Dropout', 'Enrolled', 'Graduate'])
plt.title('Distribución de la Variable Objetivo')
plt.savefig('reports/figures/01_target_distribution.png')
plt.close()

# 2. Debtor status vs Target
plt.figure(figsize=(8,5))
sns.countplot(data=train_df, x='Debtor', hue='Target')
plt.title('Condición de Deudor vs Deserción')
plt.savefig('reports/figures/02_debtor_target.png')
plt.close()

# 3. 1st Sem Grade vs Target
plt.figure(figsize=(10,6))
sns.boxplot(data=train_df, x='Target', y='Curricular units 1st sem (grade)')
plt.title('Calificaciones 1er Semestre vs Deserción')
plt.savefig('reports/figures/03_1stsem_grade.png')
plt.close()

# 4. Age at enrollment vs Target
plt.figure(figsize=(10,6))
sns.histplot(data=train_df, x='Age at enrollment', hue='Target', multiple='stack', bins=20)
plt.title('Edad de Matrícula vs Deserción')
plt.savefig('reports/figures/04_age_target.png')
plt.close()

# Print some stats for Puerta #2
print("=== ESTADISTICAS PARA EDA ===")
print("Distribución Objetivo:\n", train_df['Target'].value_counts(normalize=True))
print("\nMatriz de correlación de variables académicas 1er sem:")
academic_cols = [c for c in train_df.columns if '1st sem' in c]
print(train_df[academic_cols].corr())

print("\nConteo de valores únicos para detectar posibles errores en variables categóricas:")
for col in ['Marital status', 'Nacionality', 'Gender', 'Debtor', 'Scholarship holder']:
    print(f"{col}: {train_df[col].unique()}")
