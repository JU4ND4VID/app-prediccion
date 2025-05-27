import streamlit as st
from modules import explicaciones

st.sidebar.title("Menú de algoritmos")
opcion = st.sidebar.selectbox(
    "Selecciona el algoritmo que deseas ejecutar:",
    ["Árbol de Decisión", "K-means", "Regresión Lineal", "Regresión Múltiple"]
)

mostrar_explicacion = st.sidebar.button("Mostrar explicación")

if mostrar_explicacion:
    if opcion == "Árbol de Decisión":
        explicaciones.mostrar_explicacion_id3()
    elif opcion == "Regresión Lineal":
        explicaciones.mostrar_explicacion_regresion_lineal()
    elif opcion == "K-means":
        explicaciones.mostrar_explicacion_k_means()
    else:
        st.warning("Explicación no disponible para esta opción.")
else:
    if opcion == "Árbol de Decisión":
        from modules import arbol_decision
        arbol_decision.run()
    elif opcion == "Regresión Lineal":
        from modules import regresion_lineal
        regresion_lineal.run()
    elif opcion == "K-means":
        from modules import k_means
        k_means.run()
    elif opcion == "Regresión Múltiple":
        from modules import regresion_multiple
        regresion_multiple.run()
