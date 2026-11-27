"""Interfaz Streamlit para el modelo de prediccion de desercion estudiantil.

Consume la API FastAPI corriendo en http://127.0.0.1:8000.

Para iniciar (en una terminal separada, con la API ya corriendo):
    streamlit run app/streamlit_app.py
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

# ---------------------------------------------------------------------------
# Configuracion de pagina
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Prediccion de desercion estudiantil",
    page_icon=None,
    layout="wide",
)

st.title("Prediccion de desercion estudiantil")
st.caption(
    "Modelo LogReg L2 - CC209 Data Mining Tools - UPC | "
    "Jorge Garcia, Jose Villanueva, Aldair Rivas"
)
st.markdown(
    "Completar los datos del estudiante al finalizar el primer semestre "
    "y presionar **Predecir** para obtener el resultado."
)
st.divider()

# ---------------------------------------------------------------------------
# Formulario de entrada
# ---------------------------------------------------------------------------

with st.form("formulario_prediccion"):

    col1, col2, col3 = st.columns(3)

    # -- Columna 1: Academico 1er semestre --
    with col1:
        st.subheader("Primer semestre")
        approved = st.number_input(
            "Unidades aprobadas (1S)", min_value=0, max_value=30, value=5
        )
        enrolled = st.number_input(
            "Unidades matriculadas (1S)", min_value=0, max_value=30, value=6
        )
        evaluations = st.number_input(
            "Unidades evaluadas (1S)", min_value=0, max_value=50, value=6
        )
        grade = st.number_input(
            "Nota media 1S (0-20)", min_value=0.0, max_value=20.0, value=12.5, step=0.1
        )
        credited = st.number_input(
            "Unidades acreditadas (1S)", min_value=0, max_value=20, value=0
        )
        without_eval = st.number_input(
            "Sin evaluaciones (1S)", min_value=0, max_value=20, value=0
        )

    # -- Columna 2: Datos personales e ingreso --
    with col2:
        st.subheader("Datos personales")
        age = st.number_input(
            "Edad al matricularse", min_value=15, max_value=70, value=20
        )
        gender = st.selectbox("Genero", options=[(1, "Masculino"), (0, "Femenino")], format_func=lambda x: x[1])
        marital = st.selectbox(
            "Estado civil",
            options=[(1, "Soltero"), (2, "Casado"), (3, "Viudo"), (4, "Divorciado"), (5, "Union de hecho"), (6, "Separado")],
            format_func=lambda x: x[1],
        )
        scholarship = st.selectbox("Becario", options=[(0, "No"), (1, "Si")], format_func=lambda x: x[1])
        displaced = st.selectbox("Desplazado", options=[(0, "No"), (1, "Si")], format_func=lambda x: x[1])
        international = st.selectbox("Internacional", options=[(0, "No"), (1, "Si")], format_func=lambda x: x[1])
        special_needs = st.selectbox(
            "Necesidades educativas especiales", options=[(0, "No"), (1, "Si")], format_func=lambda x: x[1]
        )

    # -- Columna 3: Financiero, academico previo y macroeconomia --
    with col3:
        st.subheader("Financiero y contexto")
        tuition = st.selectbox(
            "Cuotas al dia", options=[(1, "Si"), (0, "No")], format_func=lambda x: x[1]
        )
        debtor = st.selectbox("Deudor", options=[(0, "No"), (1, "Si")], format_func=lambda x: x[1])
        admission_grade = st.number_input(
            "Nota de admision (0-200)", min_value=0.0, max_value=200.0, value=127.0, step=0.1
        )
        prev_grade = st.number_input(
            "Nota calificacion previa (0-200)", min_value=0.0, max_value=200.0, value=122.0, step=0.1
        )
        unemployment = st.number_input(
            "Tasa de desempleo (%)", min_value=0.0, max_value=30.0, value=10.8, step=0.1
        )
        inflation = st.number_input(
            "Tasa de inflacion (%)", min_value=-5.0, max_value=20.0, value=1.4, step=0.1
        )
        gdp = st.number_input(
            "GDP (variacion %)", min_value=-10.0, max_value=10.0, value=1.74, step=0.01
        )

    # Campos con valores por defecto fijos (menos relevantes para la demo)
    app_mode = 1
    app_order = 1
    course = 171
    prev_qual = 1
    nacionality = 1
    mothers_qual = 1
    fathers_qual = 1
    mothers_occ = 5
    fathers_occ = 5
    daytime = 1

    submitted = st.form_submit_button("Predecir", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Llamada a la API y resultado
# ---------------------------------------------------------------------------

if submitted:
    payload = {
        "Marital status": marital[0],
        "Application mode": app_mode,
        "Application order": app_order,
        "Course": course,
        "Daytime/evening attendance": daytime,
        "Previous qualification": prev_qual,
        "Previous qualification (grade)": prev_grade,
        "Nacionality": nacionality,
        "Mother's qualification": mothers_qual,
        "Father's qualification": fathers_qual,
        "Mother's occupation": mothers_occ,
        "Father's occupation": fathers_occ,
        "Admission grade": admission_grade,
        "Displaced": displaced[0],
        "Educational special needs": special_needs[0],
        "Debtor": debtor[0],
        "Tuition fees up to date": tuition[0],
        "Gender": gender[0],
        "Scholarship holder": scholarship[0],
        "Age at enrollment": age,
        "International": international[0],
        "Curricular units 1st sem (credited)": credited,
        "Curricular units 1st sem (enrolled)": enrolled,
        "Curricular units 1st sem (evaluations)": evaluations,
        "Curricular units 1st sem (approved)": approved,
        "Curricular units 1st sem (grade)": grade,
        "Curricular units 1st sem (without evaluations)": without_eval,
        "Unemployment rate": unemployment,
        "Inflation rate": inflation,
        "GDP": gdp,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()

        prob = result["probability_dropout"]
        pred = result["prediction"]
        label = result["label"]
        threshold = result["threshold"]

        st.divider()
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            if pred == 1:
                st.error(f"**Resultado: {label}**")
            else:
                st.success(f"**Resultado: {label}**")

            st.metric(label="Probabilidad de desercion", value=f"{prob:.1%}")
            st.caption(f"Umbral de decision aplicado: {threshold}")

        with col_res2:
            st.markdown("**Interpretacion:**")
            if pred == 1:
                st.markdown(
                    f"El modelo asigna una probabilidad de **{prob:.1%}** de que este estudiante "
                    f"deserte. Supera el umbral de decision ({threshold}). "
                    "Se recomienda derivar el caso a seguimiento academico."
                )
            else:
                st.markdown(
                    f"El modelo asigna una probabilidad de **{prob:.1%}** de que este estudiante "
                    f"deserte. No supera el umbral de decision ({threshold})."
                )

            st.info(
                "Esta prediccion es una estimacion estadistica basada en datos historicos. "
                "No debe usarse como unica fuente de decision sobre un estudiante."
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "No se pudo conectar con la API. "
            "Verificar que uvicorn este corriendo en http://127.0.0.1:8000"
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"Error en la API: {e}")
    except Exception as e:
        st.error(f"Error inesperado: {e}")
