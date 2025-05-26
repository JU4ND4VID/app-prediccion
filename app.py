import streamlit as st
import importlib

st.set_page_config(page_title="App de Predicción", layout="wide")

st.sidebar.title("📂 Menú de algoritmos")
opcion = st.sidebar.selectbox(
    "Selecciona el algoritmo que deseas ejecutar:",
    ("Árbol de Decisión", "K-means", "Regresión Lineal", "Regresión Múltiple"),
)

mostrar_explicacion = st.sidebar.button("Mostrar explicación ID3")

if mostrar_explicacion:
    from modules.explicacion_id3 import mostrar_explicacion_id3
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
        else:
            st.warning("El módulo seleccionado no tiene función 'run()'.")
    else:
        st.warning("Selecciona una opción válida.")
