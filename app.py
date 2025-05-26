import streamlit as st
import importlib

st.set_page_config(page_title="App de Predicción", layout="wide")

st.sidebar.title("📂 Menú de algoritmos")
opcion = st.sidebar.selectbox(
    "Selecciona el algoritmo que deseas ejecutar:",
    (
        "Árbol de Decisión",
        "Explicación ID3",
        "K-means",
        "Regresión Lineal",
        "Regresión Múltiple",
    )
)

# Diccionario que mapea opciones a módulos
modulos = {
    "Árbol de Decisión": "modules.arbol_decision",
    "Explicación ID3": "modules.explicacion_id3",
    "K-means": "modules.k_means",
    "Regresión Lineal": "modules.regresion_lineal",
    "Regresión Múltiple": "modules.regresion_multiple",
}

modulo_seleccionado = modulos.get(opcion)

if modulo_seleccionado:
    mod = importlib.import_module(modulo_seleccionado)
    # Asumimos que cada módulo tiene función 'run' o 'procesar_*'
    if hasattr(mod, "run"):
        mod.run()
    elif hasattr(mod, "procesar_arbol_decision"):
        mod.procesar_arbol_decision()
    # Agrega más condiciones según funciones en módulos
else:
    st.warning("Selecciona una opción válida.")
