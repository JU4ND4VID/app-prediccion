import streamlit as st
from modules import explicaciones
from modules import arbol_decision, regresion_lineal, k_means, regresion_multiple

st.title("App de Algoritmos")

# Menú lateral para seleccionar el algoritmo a ejecutar
opcion = st.sidebar.selectbox("Selecciona el algoritmo:", 
                              ["Árbol de Decisión", "Regresión Lineal", "K-means", "Regresión Múltiple"])

# En la parte principal, mostramos botones independientes para explicaciones
st.markdown("## Explicaciones")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Explicación Árbol de Decisión"):
        explicaciones.mostrar_explicacion_id3()

with col2:
    if st.button("Explicación Regresión Lineal"):
        explicaciones.mostrar_explicacion_regresion_lineal()

with col3:
    if st.button("Explicación K-means"):
        explicaciones.mostrar_explicacion_k_means()

# Ejecución del módulo seleccionado
st.markdown("---")
if opcion == "Árbol de Decisión":
    arbol_decision.run()
elif opcion == "Regresión Lineal":
    regresion_lineal.run()
elif opcion == "K-means":
    k_means.run()
elif opcion == "Regresión Múltiple":
    regresion_multiple.run()
