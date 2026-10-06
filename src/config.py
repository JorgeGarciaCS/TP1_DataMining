"""Configuración central del proyecto (rutas, semilla, grupos de columnas).

Todo el resto del código importa de aquí: cambiar una decisión (p. ej. qué columnas
se excluyen) se hace en un solo lugar y queda versionado.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "data.csv"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
TRAIN_PATH = DATA_INTERIM / "train.csv"
TEST_PATH = DATA_PROCESSED / "test_frozen.csv"
SPLIT_MANIFEST = DATA_PROCESSED / "split_manifest.json"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
TEST_LOG = REPORTS_DIR / "test_evaluation_log.json"

SEED = 42
TEST_SIZE = 0.20
ID_COL = "row_id"          # índice original de data.csv (permite auditar intersecciones)
TARGET_RAW = "Target"      # Dropout / Enrolled / Graduate (se conserva para análisis)
TARGET = "dropout"         # 1 = Dropout ; 0 = Graduate + Enrolled (ADR-04)

# ADR-01: no disponibles al fin del 1.er semestre (momento de predicción)
SECOND_SEM_COLS = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

# Nominales codificadas como enteros (el número no tiene orden)
CATEGORICAL_COLS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
BINARY_COLS = [
    "Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]
FIRST_SEM_COLS = [
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
]
NUMERIC_BASE_COLS = [
    "Application order", "Previous qualification (grade)", "Admission grade",
    "Age at enrollment", "Unemployment rate", "Inflation rate", "GDP",
] + FIRST_SEM_COLS

# Grupos para ablación (ADR-05): fecha de registro no documentada por la fuente
FINANCIAL_COLS = ["Debtor", "Tuition fees up to date", "Scholarship holder"]
MACRO_COLS = ["Unemployment rate", "Inflation rate", "GDP"]
# Columnas que codifican "aprobadas = 0" directa o indirectamente (nota = 0 <=> aprobadas = 0)
APPROVED_SIGNAL_COLS = ["Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)"]

# Atributos sensibles (solo para análisis de errores por segmento, ver datasheet)
SENSITIVE_COLS = ["Gender", "Nacionality", "Marital status", "Scholarship holder", "Age at enrollment"]

# Variables derivadas creadas DENTRO del pipeline (features.FirstSemesterFeatures)
FLAG_NO_EVAL = "sin_evaluaciones_1s"
FLAG_NO_APPROVED = "sin_aprobadas_1s"
