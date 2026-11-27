"""API de inferencia para el modelo de prediccion de desercion estudiantil.

Endpoints:
    GET  /health     -> estado de la API y version del modelo
    POST /predict    -> prediccion para un estudiante

Para iniciar el servidor (dejarlo correr en una terminal aparte):
    uvicorn app.api:app --reload --port 8000

Desde la raiz del proyecto.
"""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Asegurar que src este en el path cuando se ejecuta desde la raiz
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from app.schemas import StudentFeatures, PredictionResponse, HealthResponse

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

MODEL_PATH = ROOT / "models" / "dropout_logreg_v0.1-tp1.joblib"
MODEL_VERSION = "v0.1-tp1"

# Umbral optimo encontrado en TP1 (ADR-06, maximiza F1 en validacion)
THRESHOLD = 0.242

# ---------------------------------------------------------------------------
# Carga del modelo al arrancar la aplicacion (una sola vez)
# ---------------------------------------------------------------------------

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    _pipeline = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------------
# Aplicacion FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API de prediccion de desercion estudiantil",
    description=(
        "Modelo LogReg L2 entrenado sobre datos del UCI Student Performance dataset. "
        "Predice riesgo de desercion al finalizar el primer semestre. "
        "Proyecto TF1 - CC209 Data Mining Tools - UPC."
    ),
    version=MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Estado"])
def health():
    """Verifica que la API esta activa y el modelo cargado."""
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        threshold=THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediccion"])
def predict(student: StudentFeatures):
    """Predice la probabilidad de desercion para un estudiante.

    Recibe los atributos del estudiante al finalizar el primer semestre
    y devuelve la probabilidad estimada junto con la etiqueta binaria
    segun el umbral de decision calibrado.

    - **probability_dropout**: valor entre 0 y 1
    - **prediction**: 1 si P >= umbral, 0 en caso contrario
    - **threshold**: umbral aplicado (fijo en esta version)
    - **label**: descripcion legible del resultado
    """
    try:
        # Convertir el schema a DataFrame con los nombres originales de columna
        data = student.model_dump(by_alias=True)
        df = pd.DataFrame([data])

        prob = float(_pipeline.predict_proba(df)[0, 1])
        pred = int(prob >= THRESHOLD)
        label = "Riesgo de desercion" if pred == 1 else "Sin riesgo detectado"

        return PredictionResponse(
            probability_dropout=round(prob, 4),
            prediction=pred,
            threshold=THRESHOLD,
            label=label,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
