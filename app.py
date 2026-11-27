# Código: Dairo Alberto Herrera Escobar
# App de despliegue: predicción del puntaje de matemáticas (Saber 11, periodo 20252)

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predicción puntaje de matemáticas", page_icon="📐")

# 1. Cargar el modelo, el escalador de la edad y las columnas usadas en el entrenamiento
@st.cache_resource
def cargar():
    modelo = joblib.load("modelo_red_neuronal.pkl")
    escalador = joblib.load("escalador_edad.pkl")
    columnas = joblib.load("columnas_modelo.pkl")
    return modelo, escalador, columnas

modelo, escalador, columnas = cargar()

# 2. Preguntas del formulario (nombre de la variable -> texto que ve el usuario)
preguntas = {
    "cole_area_ubicacion": "Zona del colegio",
    "cole_depto_ubicacion": "Departamento del colegio",
    "cole_jornada": "Jornada del colegio",
    "cole_naturaleza": "Naturaleza del colegio",
    "fami_estratovivienda": "Estrato de la vivienda",
    "fami_educacionmadre": "Nivel educativo de la madre",
    "fami_educacionpadre": "Nivel educativo del padre",
    "fami_numlibros": "Libros en el hogar",
    "fami_personashogar": "Personas en el hogar",
    "fami_cuartoshogar": "Cuartos para dormir en el hogar",
    "fami_tienecomputador": "¿El hogar tiene computador?",
    "fami_tieneinternet": "¿El hogar tiene internet?",
    "estu_dedicacionlecturadiaria": "Tiempo diario de lectura por entretenimiento",
    "estu_horassemanatrabaja": "Horas de trabajo remunerado a la semana",
    "estu_horastrabnoremu": "Horas de trabajo no remunerado a la semana",
    "estu_desplazacolegio": "Medio de transporte al colegio",
}

# Orden lógico para las variables ordinales (las demás se muestran en orden alfabético)
educacion = ["Ninguno", "Primaria incompleta", "Primaria completa",
             "Secundaria (Bachillerato) incompleta", "Secundaria (Bachillerato) completa",
             "Técnica o tecnológica incompleta", "Técnica o tecnológica completa",
             "Educación profesional incompleta", "Educación profesional completa",
             "Postgrado", "No sabe", "No Aplica"]
horas = ["0", "Menos de 10 horas", "Entre 11 y 20 horas", "Entre 21 y 30 horas", "Más de 30 horas"]
orden = {
    "fami_educacionmadre": educacion,
    "fami_educacionpadre": educacion,
    "fami_cuartoshogar": ["Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis o mas"],
    "estu_dedicacionlecturadiaria": ["No leo por entretenimiento", "30 minutos o menos",
                                     "Entre 30 y 60 minutos", "Entre 1 y 2 horas", "Más de 2 horas"],
    "estu_horassemanatrabaja": horas,
    "estu_horastrabnoremu": horas,
}

# Las opciones de cada pregunta salen de las columnas dummy del entrenamiento
def opciones(variable):
    prefijo = variable + "_"
    lista = [c[len(prefijo):] for c in columnas if c.startswith(prefijo)]
    if variable in orden:
        return [o for o in orden[variable] if o in lista] + sorted(o for o in lista if o not in orden[variable])
    return sorted(lista)

# 3. Interfaz
st.title("📐 Predicción del puntaje de matemáticas")
st.write(
    "Estima el puntaje de matemáticas de las pruebas **Saber 11** a partir de "
    "características familiares, escolares y personales del estudiante. "
    "Modelo: red neuronal entrenada con los resultados del periodo 2025-2."
)

with st.form("formulario"):
    st.subheader("Datos del colegio")
    c1, c2 = st.columns(2)
    respuestas = {}
    for i, var in enumerate(["cole_area_ubicacion", "cole_depto_ubicacion", "cole_jornada", "cole_naturaleza"]):
        respuestas[var] = (c1 if i % 2 == 0 else c2).selectbox(preguntas[var], opciones(var))

    st.subheader("Datos del hogar")
    c1, c2 = st.columns(2)
    variables_hogar = ["fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre", "fami_numlibros",
                       "fami_personashogar", "fami_cuartoshogar", "fami_tienecomputador", "fami_tieneinternet"]
    for i, var in enumerate(variables_hogar):
        respuestas[var] = (c1 if i % 2 == 0 else c2).selectbox(preguntas[var], opciones(var))

    st.subheader("Datos del estudiante")
    c1, c2 = st.columns(2)
    edad = c1.number_input("Edad (años)", min_value=12, max_value=87, value=17, step=1)
    variables_estu = ["estu_dedicacionlecturadiaria", "estu_horassemanatrabaja",
                      "estu_horastrabnoremu", "estu_desplazacolegio"]
    for i, var in enumerate(variables_estu):
        respuestas[var] = (c2 if i % 2 == 0 else c1).selectbox(preguntas[var], opciones(var))

    enviar = st.form_submit_button("Calcular puntaje estimado")

# 4. Predicción: se repiten las MISMAS transformaciones del entrenamiento
if enviar:
    fila = pd.DataFrame(0, index=[0], columns=columnas)               # todas las columnas en 0
    fila["edad"] = escalador.transform(pd.DataFrame({"edad": [edad]}))[:, 0]  # misma escala 0-1
    for var, valor in respuestas.items():
        fila[f"{var}_{valor}"] = 1                                     # dummy de la opción elegida

    puntaje = float(modelo.predict(fila)[0])
    puntaje = min(max(puntaje, 0), 100)                                # escala oficial 0-100

    st.success(f"### Puntaje estimado en matemáticas: {puntaje:.0f} / 100")
    st.caption(
        "El modelo tiene un error promedio (MAE) de 8,9 puntos: el puntaje real puede estar "
        f"aproximadamente entre {max(puntaje - 9, 0):.0f} y {min(puntaje + 9, 100):.0f}. "
        "La estimación muestra tendencias asociadas a las condiciones del estudiante; "
        "no es una causa ni un resultado garantizado."
    )
