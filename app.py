import streamlit as st

# Título principal
st.set_page_config(page_title="App de Predicción", layout="wide")
st.title("🧠 Aplicación de Predicción de Datos")

# Menú de selección en el sidebar
st.sidebar.title("📂 Menú de algoritmos")
opcion = st.sidebar.selectbox(
    "Selecciona el algoritmo que deseas ejecutar:",
    (
        "Árbol de Decisión",
        "Regresión Lineal",
        "Regresión Múltiple",
        "K-means"
    )
)

# Cargar los módulos según opción
if opcion == "Árbol de Decisión":
    from pages.arbol_decision import procesar_arbol_decision
    procesar_arbol_decision()

elif opcion == "Regresión Lineal":
    from pages.regresion_lineal import procesar_regresion_lineal
    procesar_regresion_lineal()

elif opcion == "Regresión Múltiple":
    from pages.regresion_multiple import procesar_regresion_multiple
    procesar_regresion_multiple()

elif opcion == "K-means":
    from pages.k_means import procesar_k_means
    procesar_k_means()
