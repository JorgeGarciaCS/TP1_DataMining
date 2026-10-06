import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/interim', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Cargar dataset
df = pd.read_csv('data/raw/data.csv', sep=';')

# 1. Tratar Target y definir problema (Binario: Dropout vs No-Dropout)
# Target original: Dropout, Enrolled, Graduate
# Agrupamos Enrolled y Graduate como '0' (No-Dropout) y Dropout como '1'
df['Target_Bin'] = df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)

# Verificar duplicados exactos en el dataset
dups = df.duplicated().sum()
print(f"Duplicados en dataset original: {dups}")

# Eliminar variables del 2do semestre por Leakage Temporal
cols_to_drop = [
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)'
]
df = df.drop(columns=cols_to_drop)

# Separación (Split) 80/20 Estratificado
X = df.drop(columns=['Target', 'Target_Bin'])
y = df['Target_Bin']

# Usamos random_state=42 para reproducibilidad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verificar desbalance
print("\nDistribución en Train:")
print(y_train.value_counts(normalize=True))

# Verificar duplicados entre particiones
# Pandas no tiene un 'intersect' nativo directo por filas sin id, usamos merge
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

intersect = pd.merge(train_df, test_df, how='inner')
print(f"\nFilas idénticas entre Train y Test: {len(intersect)}")

# Congelar test set
test_df.to_csv('data/processed/test_frozen.csv', index=False)
train_df.to_csv('data/interim/train.csv', index=False)

# Análisis de Notas en 0 (Punto 2)
print("\n--- ANÁLISIS DE NOTAS EN 0 EN TRAIN ---")
# Tabla cruzada: Grade == 0 vs Evaluaciones vs Aprobados vs Target
train_df['Grade_is_0'] = train_df['Curricular units 1st sem (grade)'] == 0
train_df['Evaluations_is_0'] = train_df['Curricular units 1st sem (evaluations)'] == 0
train_df['Enrolled_is_0'] = train_df['Curricular units 1st sem (enrolled)'] == 0

ct = pd.crosstab(
    [train_df['Grade_is_0'], train_df['Evaluations_is_0']], 
    train_df['Target_Bin'], 
    margins=True
)
print(ct)

# Indicador: El 0 ocurre casi exclusivamente cuando no hay evaluaciones
# ¿Es un predictor trivial? 
dropout_rate_when_0_evals = train_df[train_df['Evaluations_is_0']]['Target_Bin'].mean()
print(f"\nTasa de abandono cuando Evaluaciones == 0: {dropout_rate_when_0_evals:.1%}")

# Decisión: crear un indicador 'sin_evaluaciones_1s'
