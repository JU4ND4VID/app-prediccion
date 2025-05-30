import pandas as pd
import numpy as np
import streamlit as st
from graphviz import Digraph
import unicodedata

# --- Normalización y limpieza ---
def normalizar_texto(texto):
    texto = str(texto).strip().lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto

def corregir_errores_ortograficos(valor):
    correcciones = {'ingnieria': 'ingenieria'}
    return correcciones.get(valor, valor)

def limpiar_y_normalizar_df(df, columnas):
    for col in columnas:
        df[col] = df[col].astype(str).apply(normalizar_texto)
        df[col] = df[col].apply(corregir_errores_ortograficos)
    return df

# --- Cálculo de entropía ---
def calcular_entropia(etiquetas, base=None):
    valores, conteos = np.unique(etiquetas, return_counts=True)
    probabilidades = conteos / conteos.sum()
    k = base or len(valores)
    ent = 0.0
    for p in probabilidades:
        if p > 0:
            ent -= p * np.log(p) / np.log(k)
    return ent

# --- Entropía condicional estilo profesor ---
def entropia_condicional(data, atributo, target, clases_global):
    df = data[data[atributo] != '?']
    valores, conteos = np.unique(df[atributo], return_counts=True)
    n_total = conteos.sum()
    k = len(clases_global)

    with st.expander(f"🔍 Cálculo E(S|{atributo})", expanded=True):
        st.markdown(f"### Para '{atributo}':")
        E_cond = 0.0
        for v, c in zip(valores, conteos):
            subset = df[df[atributo] == v][target]
            n_sub = len(subset)
            terminos = []
            for cls in clases_global:
                cnt = np.sum(subset == cls)
                terminos.append(f"{cnt}/{n_sub}*LOG({cnt}/{n_sub};{k})")
            formula = ' + '.join(terminos)
            contrib = (c / n_total) * calcular_entropia(subset, base=k)
            E_cond += contrib
            st.code(f"= {c}/{n_total} * (-[{formula}]) = {contrib:.9f}", language='text')
        st.markdown(f"**E(S|{atributo}) = {E_cond:.9f}**")
    return E_cond

# --- Nodo de decisión ---
class NodoDecision:
    def __init__(self, atributo=None, hijos=None, es_hoja=False, clase=None):
        self.atributo = atributo
        self.hijos = hijos or {}
        self.es_hoja = es_hoja
        self.clase = clase

# --- Construcción recursiva ---
def construir_arbol_interactivo(data, atributos, target, clases_global=None):
    if clases_global is None:
        clases_global = list(np.unique(data[target]))
    df = data[data[target] != '?']
    if df.empty:
        st.write("⚠️ No hay datos, asigno clase 'Desconocido'")
        return NodoDecision(es_hoja=True, clase='Desconocido')

    entropias = {attr: entropia_condicional(df, attr, target, clases_global) for attr in atributos}
    mejor = min(entropias, key=entropias.get)
    st.write(f"➡️ Mejor atributo = {mejor} (E(S|{mejor}) = {entropias[mejor]:.9f})\n")
    nodo = NodoDecision(atributo=mejor)

    for val in np.unique(df[mejor]):
        subset = df[df[mejor] == val]
        if len(subset[target].unique()) == 1:
            clase_leaf = subset[target].iloc[0]
            st.write(f"▷ Particionando {mejor} = {val} → Nodo hoja con clase: {clase_leaf}\n")
            nodo.hijos[val] = NodoDecision(es_hoja=True, clase=clase_leaf)
        else:
            resto = [a for a in atributos if a != mejor]
            nodo.hijos[val] = construir_arbol_interactivo(subset, resto, target, clases_global)
    return nodo

# --- Extracción de reglas ---
def extraer_reglas(nodo, camino=None):
    camino = camino or []
    if nodo.es_hoja:
        cond = ' y '.join(camino) if camino else '(sin condición)'
        return [f"Si {cond}, entonces Categoría = {nodo.clase}"]
    reglas = []
    for v, h in nodo.hijos.items():
        reglas.extend(extraer_reglas(h, camino + [f"{nodo.atributo} = {v}"]))
    return reglas

# --- Dibujo del árbol ---
def dibujar_arbol(nodo, dot=None, padre=None, etiqueta=None, contador=[0]):
    if dot is None:
        dot = Digraph(); dot.node('0','Inicio'); padre='0'; contador[0] = 1
    nid = str(contador[0]); contador[0] += 1
    if nodo.es_hoja:
        dot.node(nid, f"Categoría: {nodo.clase}", shape='box', style='filled', color='lightgreen')
    else:
        dot.node(nid, nodo.atributo, shape='ellipse', style='filled', color='lightblue')
    dot.edge(padre, nid, label=etiqueta or '')
    if not nodo.es_hoja:
        for v, h in nodo.hijos.items():
            dibujar_arbol(h, dot, nid, str(v), contador)
    return dot

# --- Predicción ---
def predecir(nodo, ejemplo):
    if nodo.es_hoja:
        return nodo.clase
    val = ejemplo.get(nodo.atributo)
    if val not in nodo.hijos:
        val = next(iter(nodo.hijos))
    return predecir(nodo.hijos[val], ejemplo)

# --- App Streamlit ---
def procesar_arbol_decision():
    st.title("🌳 Árbol ID3")

    uploaded = st.file_uploader("Sube CSV o Excel", type=["csv","xlsx"])
    if not uploaded:
        st.info("Por favor sube un archivo para continuar.")
        return
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, na_values=["?","? "], keep_default_na=False)
        else:
            df = pd.read_excel(uploaded, na_values=["?","? "], keep_default_na=False)
    except Exception as e:
        st.error(f"Error cargando el archivo: {e}")
        return

    df = df.dropna(how='any')
    df = limpiar_y_normalizar_df(df, df.columns.tolist())
    st.session_state['df'] = df

    st.subheader("Datos cargados")
    st.dataframe(df)

    cols = df.columns.tolist()
    target = st.selectbox("Selecciona la variable a predecir", cols, index=0)
    features = st.multiselect("Selecciona las variables de entrada", [c for c in cols if c != target])

    if st.button("Generar árbol ID3"):
        if not features:
            st.error("Selecciona al menos una variable de entrada.")
            return
        df_model = df[features + [target]].astype(str)
        arbol = construir_arbol_interactivo(df_model, features, target)
        st.session_state['arbol'] = arbol
        st.session_state['reglas'] = extraer_reglas(arbol)
        st.session_state['ok'] = True
        st.session_state['features'] = features
        st.session_state['target'] = target

    if st.session_state.get('ok'):
        features = st.session_state['features']
        target = st.session_state['target']
        df_model = st.session_state['df'][features + [target]].astype(str)

        st.success("Árbol construido correctamente.")
        st.subheader("Reglas de Clasificación")
        for i, regla in enumerate(st.session_state['reglas'], 1):
            st.markdown(f"**Regla {i}:** {regla}")

        st.subheader("Diagrama del Árbol")
        st.graphviz_chart(dibujar_arbol(st.session_state['arbol']))

        st.subheader("Prueba de predicción")
        with st.form('form_pred'):
            ejemplo = {}
            for c in features:
                opts = sorted(set(df_model[c].unique()) | {'?'})
                ejemplo[c] = st.selectbox(c, opts, index=opts.index('?'))
            if st.form_submit_button('Predecir'):
                pr = predecir(st.session_state['arbol'], ejemplo)
                if pr:
                    st.success(f"Predicción: {pr}")
                else:
                    st.error("No se pudo predecir.")

# Función run
def run():
    procesar_arbol_decision()

if __name__ == '__main__':
    run()
