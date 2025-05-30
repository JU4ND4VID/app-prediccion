import streamlit as st
from modules import explicaciones
from modules import arbol_decision, regresion_lineal, k_medias, regresion_multiple, k_modas  

st.set_page_config(page_title="App de Algoritmos", layout="wide")

st.sidebar.title("📚 Explicaciones 📚")

explicacion_seleccionada = st.sidebar.radio(
    "Selecciona la explicación a mostrar:",
    (
        "Algoritmos",
        "Árbol de Decisión",
        "Regresión Lineal",
        "Regresión Múltiple",
        "K-media",
        "K-modas"        
        
    )
)

st.sidebar.markdown("---")
st.sidebar.title("🔥 Ejecutar Algoritmos 🔥")

algoritmo_seleccionado = st.sidebar.selectbox(
    "Selecciona el algoritmo para ejecutar:",
    (
        "Árbol de Decisión",
        "Regresión Lineal",
        "Regresión Múltiple",
        "K-medias",
        "K-modas"
        
    )
)

if explicacion_seleccionada != "Algoritmos":
    if explicacion_seleccionada == "Árbol de Decisión":
        explicaciones.mostrar_explicacion_id3()
    elif explicacion_seleccionada == "Regresión Lineal":
        explicaciones.mostrar_explicacion_regresion_lineal()
    elif explicacion_seleccionada == "Regresión Múltiple":
        explicaciones.mostrar_explicacion_regresion_multiple()
    elif explicacion_seleccionada == "K-medias":
        explicaciones.mostrar_explicacion_k_medias()
    elif explicacion_seleccionada == "K-modas":
        explicaciones.mostrar_explicacion_k_modas()

else:
    if algoritmo_seleccionado == "Árbol de Decisión":
        arbol_decision.run()
    elif algoritmo_seleccionado == "Regresión Lineal":
        regresion_lineal.run()
    elif algoritmo_seleccionado == "Regresión Múltiple":
        regresion_multiple.run()
    elif algoritmo_seleccionado == "K-medias":
        k_medias.run()
    elif algoritmo_seleccionado == "K-modas":
        k_modas.run()

