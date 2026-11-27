"""Schemas Pydantic para la API de inferencia.

Request:  StudentFeatures  — todos los campos de entrada del modelo
Response: PredictionResponse — probabilidad, etiqueta y umbral aplicado
"""

from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    """Campos de entrada necesarios para la prediccion de desercion.

    Los nombres coinciden exactamente con las columnas que espera el pipeline
    entrenado (antes del preprocesamiento interno).
    """

    # --- Demograficos ---
    Marital_status: int = Field(alias="Marital status", default=1)
    Application_mode: int = Field(alias="Application mode", default=1)
    Application_order: int = Field(alias="Application order", default=1)
    Course: int = Field(alias="Course", default=171)
    Daytime_evening_attendance: int = Field(alias="Daytime/evening attendance", default=1)
    Previous_qualification: int = Field(alias="Previous qualification", default=1)
    Previous_qualification_grade: float = Field(alias="Previous qualification (grade)", default=122.0)
    Nacionality: int = Field(alias="Nacionality", default=1)
    Mothers_qualification: int = Field(alias="Mother's qualification", default=1)
    Fathers_qualification: int = Field(alias="Father's qualification", default=1)
    Mothers_occupation: int = Field(alias="Mother's occupation", default=5)
    Fathers_occupation: int = Field(alias="Father's occupation", default=5)
    Admission_grade: float = Field(alias="Admission grade", default=127.0)
    Displaced: int = Field(alias="Displaced", default=0)
    Educational_special_needs: int = Field(alias="Educational special needs", default=0)
    Debtor: int = Field(alias="Debtor", default=0)
    Tuition_fees_up_to_date: int = Field(alias="Tuition fees up to date", default=1)
    Gender: int = Field(alias="Gender", default=1)
    Scholarship_holder: int = Field(alias="Scholarship holder", default=0)
    Age_at_enrollment: int = Field(alias="Age at enrollment", default=20)
    International: int = Field(alias="International", default=0)

    # --- 1er semestre ---
    Curricular_units_1st_sem_credited: int = Field(alias="Curricular units 1st sem (credited)", default=0)
    Curricular_units_1st_sem_enrolled: int = Field(alias="Curricular units 1st sem (enrolled)", default=6)
    Curricular_units_1st_sem_evaluations: int = Field(alias="Curricular units 1st sem (evaluations)", default=6)
    Curricular_units_1st_sem_approved: int = Field(alias="Curricular units 1st sem (approved)", default=5)
    Curricular_units_1st_sem_grade: float = Field(alias="Curricular units 1st sem (grade)", default=12.5)
    Curricular_units_1st_sem_without_evaluations: int = Field(
        alias="Curricular units 1st sem (without evaluations)", default=0
    )

    # --- Macroeconomicos ---
    Unemployment_rate: float = Field(alias="Unemployment rate", default=10.8)
    Inflation_rate: float = Field(alias="Inflation rate", default=1.4)
    GDP: float = Field(alias="GDP", default=1.74)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    """Respuesta de la API para una solicitud de prediccion."""

    probability_dropout: float = Field(
        description="Probabilidad estimada de desercion (0.0 a 1.0)"
    )
    prediction: int = Field(
        description="Etiqueta binaria: 1 = Riesgo de desercion, 0 = Sin riesgo"
    )
    threshold: float = Field(
        description="Umbral de decision aplicado (ADR-06)"
    )
    label: str = Field(
        description="Descripcion legible del resultado"
    )


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""

    status: str
    model_version: str
    threshold: float
