import pandas as pd
import numpy as np
import streamlit as st
from graphviz import Digraph

# Función para calcular la entropía de un conjunto
def calcular_entropia(etiquetas):
    valores, conteos = np.unique(etiquetas, return_counts=True)
    probabilidades = conteos / conteos.sum()
    entropia = -np.sum(probabilidades * np.log2(probabilidades))
    return entropia

# Función para calcular la ganancia de información
def ganancia_informacion(data, atributo, target):
    entropia_total = calcular_entropia(data[target])
    valores, conteos = np.unique(data[atributo], return_counts=True)
    entropia_condicional = 0
    for v, c in zip(valores, conteos):
        subset = data[data[atributo] == v]
        entropia_subset = calcular_entropia(subset[target])
        entropia_condicional += (c / conteos.sum()) * entropia_subset
    ganancia = entropia_total - entropia_condicional
    return ganancia

# Nodo del árbol
class NodoDecision:
    def __init__(self, atributo=None, hijos=None, es_hoja=False, clase=None):
        self.atributo = atributo
        self.hijos = hijos if hijos is not None else {}
        self.es_hoja = es_hoja
        self.clase = clase

# Construcción recursiva del árbol ID3
def construir_arbol(data, atributos, target):
    # Si todos los ejemplos tienen la misma clase, crear hoja
    if len(data[target].unique()) == 1:
        return NodoDecision(es_hoja=True, clase=data[target].iloc[0])
    
    # Si no hay atributos para dividir, crear hoja con la clase mayoritaria
    if len(atributos) == 0:
        clase_mayoritaria = data[target].mode()[0]
        return NodoDecision(es_hoja=True, clase=clase_mayoritaria)
    
    # Calcular ganancia de información para cada atributo
    ganancias = {atributo: ganancia_informacion(data, atributo, target) for atributo in atributos}
    
    # Seleccionar el atributo con mayor ganancia
    mejor_atributo = max(ganancias, key=ganancias.get)
    
    # Crear nodo raíz con el mejor atributo
    nodo = NodoDecision(atributo=mejor_atributo)
    
    # Para cada valor del mejor atributo, crear ramas recursivas
    valores_unicos = data[mejor_atributo].unique()
    for valor in valores_unicos:
        subset = data[data[mejor_atributo] == valor]
        if subset.empty:
            clase_mayoritaria = data[target].mode()[0]
            nodo.hijos[valor] = NodoDecision(es_hoja=True, clase=clase_mayoritaria)
        else:
            atributos_restantes = [a for a in atributos if a != mejor_atributo]
            nodo.hijos[valor] = construir_arbol(subset, atributos_restantes, target)
    return nodo

# Función para extraer reglas en formato legible
def extraer_reglas(nodo, camino=[]):
    if nodo.es_hoja:
        regla = " y ".join(camino) if camino else "(sin condición)"
        return [f"Si {regla}, entonces Categoría = {nodo.clase}"]
    reglas = []
    for valor, hijo in nodo.hijos.items():
        nueva_condicion = f"{nodo.atributo} = {valor}"
        reglas.extend(extraer_reglas(hijo, camino + [nueva_condicion]))
    return reglas

# Función para dibujar árbol con Graphviz
def dibujar_arbol(nodo, dot=None, padre=None, etiqueta=None, contador=[0]):
    if dot is None:
        dot = Digraph()
        dot.node(name='0', label='Inicio')
        contador[0] = 1
        padre = '0'
    
    nodo_id = str(contador[0])
    contador[0] +=1
    
    if nodo.es_hoja:
        dot.node(nodo_id, label=f"Categoría: {nodo.clase}", shape='box', style='filled', color='lightgreen')
    else:
        dot.node(nodo_id, label=f"{nodo.atributo}", shape='ellipse', style='filled', color='lightblue')
    
    dot.edge(padre, nodo_id, label=etiqueta if etiqueta else "")
    
    if not nodo.es_hoja:
        for valor, hijo in nodo.hijos.items():
            dibujar_arbol(hijo, dot, nodo_id, str(valor), contador)
    return dot

# Streamlit app
def app():
    st.title("🌳 Árbol de Decisión ID3 desde cero")

    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Por favor, sube un archivo para continuar.")
        return
    
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Vista previa de datos")
    st.dataframe(df)

    columnas = df.columns.tolist()
    target_col = st.selectbox("Selecciona la variable a predecir (target)", columnas)
    input_cols = st.multiselect("Selecciona las variables de entrada (features)", [col for col in columnas if col != target_col])

    if st.button("Generar árbol ID3"):
        if not input_cols:
            st.error("Selecciona al menos una variable de entrada.")
            return

        # Convertir variables a string para evitar problemas
        df_model = df[input_cols + [target_col]].astype(str)

        # Construir árbol
        arbol = construir_arbol(df_model, input_cols, target_col)
        st.success("Árbol construido correctamente.")

        # Extraer reglas
        reglas = extraer_reglas(arbol)
        st.subheader("📋 Reglas de Clasificación Generadas")
        for i, regla in enumerate(reglas, 1):
            st.markdown(f"**Regla {i}:** {regla}")

        # Dibujar árbol
        st.subheader("🌳 Diagrama del Árbol (estilo profesorxxxxx)")
        dot = dibujar_arbol(arbol)
        st.graphviz_chart(dot)

if __name__ == "__main__":
    app()
