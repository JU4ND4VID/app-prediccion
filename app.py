import streamlit as st

# Diccionario con descripciones cortas de cada algoritmo
descripciones_algoritmos = {
    "Árbol de Decisión": """
    El algoritmo ID3 construye árboles de decisión usando entropía y ganancia de información.
    Selecciona el mejor atributo para dividir recursivamente los datos hasta obtener hojas puras.
    Ideal para problemas de clasificación.
    """,
    "Regresión Lineal": """
    La regresión lineal modela la relación entre variables independientes y una variable dependiente.
    Utiliza una función lineal para predecir valores continuos.
    """,
    "Regresión Múltiple": """
    Extiende la regresión lineal considerando múltiples variables independientes simultáneamente.
    Es útil para modelar relaciones más complejas.
    """,
    "K-means": """
    Algoritmo de clustering que agrupa datos en k clusters basados en la proximidad.
    Es un método no supervisado para descubrir patrones.
    """
}

st.set_page_config(page_title="App de Predicción", layout="wide")
st.title("🧠 Aplicación de Predicción de Datos")

st.sidebar.title("📂 Menú de algoritmos")

# Mostrar descripción dinámica arriba del selectbox
opcion = st.sidebar.selectbox(
    "Selecciona el algoritmo que deseas ejecutar:",
    tuple(descripciones_algoritmos.keys())
)

st.sidebar.markdown("---")
st.sidebar.markdown(descripciones_algoritmos[opcion])

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
