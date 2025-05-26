import streamlit as st
from modules.explicacion_id3 import mostrar_explicacion_id3

import importlib

st.set_page_config(page_title="App de Predicción", layout="wide")

st.sidebar.title("📂 Menú de algoritmos")
opcion = st.sidebar.selectbox(
    "Selecciona el algoritmo que deseas ejecutar:",
    (
        "Árbol de Decisión",
        "K-means",
        "Regresión Lineal",
        "Regresión Múltiple",
    )
)

mostrar_exp = st.sidebar.button("Mostrar explicación ID3")

if mostrar_exp:
    mostrar_explicacion_id3()
else:
    modulos = {
        "Árbol de Decisión": "modules.arbol_decision",
        "K-means": "modules.k_means",
        "Regresión Lineal": "modules.regresion_lineal",
        "Regresión Múltiple": "modules.regresion_multiple",
    }
    modulo_seleccionado = modulos.get(opcion)

    if modulo_seleccionado:
        mod = importlib.import_module(modulo_seleccionado)
        if hasattr(mod, "run"):
            mod.run()
        elif hasattr(mod, "procesar_arbol_decision"):
            mod.procesar_arbol_decision()
    else:
        st.warning("Selecciona una opción válida.")
