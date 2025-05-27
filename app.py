import streamlit as st
from modules import explicaciones
from modules import arbol_decision, regresion_lineal, k_means, regresion_multiple

st.set_page_config(page_title="App de Algoritmos", layout="wide")

st.sidebar.title("📚 Explicaciones")

explicacion_seleccionada = st.sidebar.radio(
    "Selecciona la explicación a mostrar:",
    (
        "Ninguna",
        "Árbol de Decisión",
        "Regresión Lineal",
        "K-means",
        "Regresión Múltiple"   # Agregado
    )
)

st.sidebar.markdown("---")
st.sidebar.title("🔧 Ejecutar Algoritmos")

algoritmo_seleccionado = st.sidebar.selectbox(
    "Selecciona el algoritmo para ejecutar:",
    (
        "Árbol de Decisión",
        "Regresión Lineal",
        "K-means",
        "Regresión Múltiple"
    )
)

if explicacion_seleccionada != "Ninguna":
    if explicacion_seleccionada == "Árbol de Decisión":
        explicaciones.mostrar_explicacion_id3()
    elif explicacion_seleccionada == "Regresión Lineal":
        explicaciones.mostrar_explicacion_regresion_lineal()
    elif explicacion_seleccionada == "K-means":
        explicaciones.mostrar_explicacion_k_means()
    elif explicacion_seleccionada == "Regresión Múltiple":
        explicaciones.mostrar_explicacion_regresion_multiple()  # Nuevo método
else:
    if algoritmo_seleccionado == "Árbol de Decisión":
        arbol_decision.run()
    elif algoritmo_seleccionado == "Regresión Lineal":
        regresion_lineal.run()
    elif algoritmo_seleccionado == "K-means":
        k_means.run()
    elif algoritmo_seleccionado == "Regresión Múltiple":
        regresion_multiple.run()
