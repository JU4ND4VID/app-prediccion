import pandas as pd
import numpy as np
import streamlit as st
from graphviz import Digraph

def calcular_entropia(etiquetas):
    valores, conteos = np.unique(etiquetas, return_counts=True)
    probabilidades = conteos / conteos.sum()
    entropia = -np.sum(probabilidades * np.log2(probabilidades + 1e-9))  # Evitar log(0)
    return entropia

def ganancia_informacion(data, atributo, target):
    entropia_total = calcular_entropia(data[target])
    valores, conteos = np.unique(data[atributo].dropna(), return_counts=True)
    entropia_condicional = 0
    detalles = []
    for v, c in zip(valores, conteos):
        subset = data[data[atributo] == v]
        entropia_subset = calcular_entropia(subset[target])
        peso = c / conteos.sum()
        entropia_condicional += peso * entropia_subset
        detalles.append({
            "Valor": v,
            "Cantidad": c,
            "Peso": round(peso, 3),
            "Entropía Subconjunto": round(entropia_subset, 3)
        })
    ganancia = entropia_total - entropia_condicional
    return ganancia, detalles

class NodoDecision:
    def __init__(self, atributo=None, hijos=None, es_hoja=False, clase=None):
        self.atributo = atributo
        self.hijos = hijos if hijos is not None else {}
        self.es_hoja = es_hoja
        self.clase = clase

def construir_arbol_interactivo(data, atributos, target, nivel=0):
    indent = "    " * nivel

    # Eliminar filas con target nulo o '?'
    data = data[data[target].notna() & (data[target] != '?')]
    if len(data) == 0:
        st.markdown(f"{indent}⚠️ No hay datos para construir el nodo, se asigna clase 'Desconocido'")
        return NodoDecision(es_hoja=True, clase="Desconocido")

    if len(data[target].unique()) == 1:
        st.markdown(f"{indent}📌 Nodo hoja con clase: **{data[target].iloc[0]}**")
        return NodoDecision(es_hoja=True, clase=data[target].iloc[0])

    if len(atributos) == 0:
        clase_mayoritaria = data[target].mode()[0]
        st.markdown(f"{indent}⚠️ Sin atributos restantes. Nodo hoja con clase mayoritaria: **{clase_mayoritaria}**")
        return NodoDecision(es_hoja=True, clase=clase_mayoritaria)

    entropia_total = calcular_entropia(data[target])
    st.markdown(f"{indent}🔸 Entropía total del conjunto: **{entropia_total:.4f}**")

    ganancias = {}
    for atributo in atributos:
        # Ignorar filas con NaN o '?' en atributo para cálculo
        data_filtrada = data[data[atributo].notna() & (data[atributo] != '?')]
        if len(data_filtrada) == 0:
            ganancias[atributo] = 0
            continue
        ganancia, detalles = ganancia_informacion(data_filtrada, atributo, target)
        st.markdown(f"{indent}**Atributo '{atributo}'**: Ganancia = **{ganancia:.4f}**")
        df_detalles = pd.DataFrame(detalles)
        st.dataframe(df_detalles)
        ganancias[atributo] = ganancia

    mejor_atributo = max(ganancias, key=ganancias.get)
    st.markdown(f"{indent}➡️ **Mejor atributo para dividir: '{mejor_atributo}'**\n")

    nodo = NodoDecision(atributo=mejor_atributo)
    valores_unicos = data[mejor_atributo].dropna().unique()

    for valor in valores_unicos:
        if valor == '?':
            continue
        st.markdown(f"{indent}▷ Particionando para **{mejor_atributo} = {valor}**")
        subset = data[data[mejor_atributo] == valor]
        atributos_restantes = [a for a in atributos if a != mejor_atributo]
        nodo.hijos[valor] = construir_arbol_interactivo(subset, atributos_restantes, target, nivel+1)

    # Manejar casos con valores faltantes ('?') en el atributo de división como nodo hoja con clase mayoritaria
    if any((data[mejor_atributo] == '?')):
        clase_mayoritaria = data[target].mode()[0]
        st.markdown(f"{indent}▷ Valores faltantes '?' en '{mejor_atributo}', asignando clase mayoritaria: **{clase_mayoritaria}**")
        nodo.hijos['?'] = NodoDecision(es_hoja=True, clase=clase_mayoritaria)

    return nodo

def extraer_reglas(nodo, camino=[]):
    if nodo.es_hoja:
        regla = " y ".join(camino) if camino else "(sin condición)"
        return [f"Si {regla}, entonces Categoría = {nodo.clase}"]
    reglas = []
    for valor, hijo in nodo.hijos.items():
        nueva_condicion = f"{nodo.atributo} = {valor}"
        reglas.extend(extraer_reglas(hijo, camino + [nueva_condicion]))
    return reglas

def dibujar_arbol(nodo, dot=None, padre=None, etiqueta=None, contador=[0]):
    if dot is None:
        dot = Digraph()
        dot.node(name='0', label='Inicio')
        contador[0] = 1
        padre = '0'

    nodo_id = str(contador[0])
    contador[0] += 1

    if nodo.es_hoja:
        dot.node(nodo_id, label=f"Categoría: {nodo.clase}", shape='box', style='filled', color='lightgreen')
    else:
        dot.node(nodo_id, label=f"{nodo.atributo}", shape='ellipse', style='filled', color='lightblue')

    dot.edge(padre, nodo_id, label=etiqueta if etiqueta else "")

    if not nodo.es_hoja:
        for valor, hijo in nodo.hijos.items():
            dibujar_arbol(hijo, dot, nodo_id, str(valor), contador)
    return dot

def predecir(nodo, ejemplo):
    if nodo.es_hoja:
        return nodo.clase
    valor = ejemplo.get(nodo.atributo, None)
    if valor not in nodo.hijos or valor == '?' or valor is None or pd.isna(valor):
        # Elegir la rama '?' si existe, si no la primera disponible
        if '?' in nodo.hijos:
            valor = '?'
        else:
            valor = next(iter(nodo.hijos.keys()))
    return predecir(nodo.hijos[valor], ejemplo)

def procesar_arbol_decision():
    st.title("🌳 Árbol de Decisión ID3 con explicación paso a paso")

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

    if st.button("Generar árbol ID3 con explicación"):
        if not input_cols:
            st.error("Selecciona al menos una variable de entrada.")
            return

        # Reemplazar '?' por NaN para mejor manejo
        df_model = df[input_cols + [target_col]].replace('?', np.nan).astype(str)

        st.markdown("## Construcción recursiva del árbol")
        arbol = construir_arbol_interactivo(df_model, input_cols, target_col)

        st.success("Árbol construido correctamente.")

        reglas = extraer_reglas(arbol)
        st.subheader("📋 Reglas de Clasificación Generadas")
        for i, regla in enumerate(reglas, 1):
            st.markdown(f"**Regla {i}:** {regla}")

        st.subheader("🌳 Diagrama del Árbol ")
        dot = dibujar_arbol(arbol)
        st.graphviz_chart(dot)

        # Ejemplo de predicción
        st.subheader("🧪 Prueba de predicción con valores con '?'")
        ejemplo = {}
        for col in input_cols:
            ejemplo[col] = st.text_input(f"Ingrese valor para {col} (use '?' para desconocido)", '?')

        if st.button("Predecir categoría para el ejemplo ingresado"):
            prediccion = predecir(arbol, ejemplo)
            st.success(f"La predicción para el ejemplo ingresado es: **{prediccion}**")

def run():
    procesar_arbol_decision()

if __name__ == "__main__":
    run()
