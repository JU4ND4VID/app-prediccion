import streamlit as st

st.set_page_config(page_title="App de Predicción", layout="wide")
hide_streamlit_style = """
    <style>
    /* Oculta la barra de búsqueda en el sidebar */
    .css-1d391kg input[type="search"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



# Mantener solo título y menú en sidebar
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

# Botón para ir a la explicación
if st.sidebar.button("Mostrar explicación ID3"):
    pagina = "explicacion"
else:
    pagina = "main"

# Navegación condicional
if pagina == "explicacion":
    # Aquí cargas la explicación en la página principal
    from pages.explicacion_id3 import mostrar_explicacion
    mostrar_explicacion()

else:
    # Ejecutar módulo según algoritmo seleccionado
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
