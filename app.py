import streamlit as st
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

mostrar_explicacion = st.sidebar.button("Mostrar explicación ID3")

def mostrar_explicacion_id3():
    st.title("Explicación del algoritmo Árbol de Decisión ID3")

    st.markdown("""
    # Proceso de construcción del Árbol de Decisión ID3

    1. **Introducción**  
    El algoritmo ID3 construye un árbol de decisión que clasifica datos usando el criterio de máxima ganancia de información, basada en la entropía.

    2. **Entropía**  
    Mide la impureza o incertidumbre de un conjunto de datos.  
    Fórmula:  
    $$
    Entropía(S) = - \sum_{i=1}^c p_i \log_2(p_i)
    $$
    donde:  
    - \(S\) es el conjunto de datos,  
    - \(c\) es el número de clases,  
    - \(p_i\) es la proporción de ejemplos en la clase \(i\).

    Si todos los datos pertenecen a una sola clase, la entropía es 0 (conjunto puro).  
    Si las clases están distribuidas uniformemente, la entropía es máxima.

    3. **Ganancia de Información**  
    Mide cuánto reduce la entropía un atributo al dividir los datos.  
    Fórmula:  
    $$
    Ganancia(S, A) = Entropía(S) - \sum_{v \in Valores(A)} \frac{|S_v|}{|S|} Entropía(S_v)
    $$
    donde:  
    - \(S_v\) es el subconjunto de \(S\) donde el atributo \(A\) toma el valor \(v\).

    Elegimos el atributo con máxima ganancia para dividir.

    4. **Proceso Recursivo**  
    Calcula la entropía y ganancia para cada atributo.  
    Escoge el atributo con mayor ganancia para crear un nodo.  
    Divide el conjunto según los valores del atributo.  
    Repite recursivamente en cada subconjunto hasta que:  
    - Todos los ejemplos son de la misma clase (entropía = 0).  
    - No quedan más atributos para dividir.

    5. **Construcción del Árbol**  
    El nodo raíz es el atributo con mayor ganancia.  
    Cada rama corresponde a un valor del atributo.  
    Las hojas contienen las clases finales.

    6. **Extracción de Reglas**  
    Cada camino desde la raíz hasta una hoja representa una regla.  
    La regla concatena las condiciones de cada nodo en el camino.  

    Ejemplo:  
    `Si Nivel académico = Magíster y Estrato socioeconómico = Medio y Área de estudio = Ingeniería, entonces Categoría = Titular`
    """)

if mostrar_explicacion:
    mostrar_explicacion_id3()
else:
    # Carga el módulo correspondiente según selección
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
