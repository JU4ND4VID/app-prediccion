import pandas as pd
import numpy as np
import streamlit as st
from graphviz import Digraph
import unicodedata

# --- Normalización y limpieza ---
def normalizar_texto(texto):
    texto = str(texto).strip().lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto)
                    if unicodedata.category(c) != 'Mn')
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
    # entropía S
    ent = 0.0
    for p in probabilidades:
        if p > 0:
            ent -= p * np.log(p + 1e-9) / np.log(k)
    return ent

# --- Entropía condicional estilo profesor ---
def entropia_condicional(data, atributo, target, indent=""):
    """
    Muestra los cálculos como en el Excel del profesor:
    =c/n * ( -sum_i ( cnt_i/n * LOG(cnt_i/n; k) ) )
    y suma las contribuciones.
    """
    # filtrar faltantes
    df = data[data[atributo] != '?']
    valores, conteos = np.unique(df[atributo], return_counts=True)
    n_total = conteos.sum()
    k = len(np.unique(df[target]))
    # imprimir encabezado
    st.write(f"{indent}Para '{atributo}':")

    E_cond = 0.0
    # para cada valor de A
    for v, c in zip(valores, conteos):
        subset = df[df[atributo] == v][target]
        n_sub = len(subset)
        # construir términos internos
        clases, cnts = np.unique(subset, return_counts=True)
        terminos = []
        for cnt in cnts:
            if cnt == 0:
                terminos.append('0')
            else:
                terminos.append(f"{cnt}/{n_sub}*LOG({cnt}/{n_sub};{k})")
        formula = '-'.join(terminos)
        contrib = (c / n_total) * calcular_entropia(subset, base=k)
        E_cond += contrib
        # mostrar línea de cálculo
        st.write(f"{indent}= {c}/{n_total} * ( -{formula} ) = {contrib:.9f}")

    # suma final
    st.write(f"{indent}E(S|{atributo}) = {E_cond:.9f}\n")
    return E_cond

# --- Nodo de decisión ---
class NodoDecision:
    def __init__(self, atributo=None, hijos=None, es_hoja=False, clase=None):
        self.atributo = atributo
        self.hijos = hijos or {}
        self.es_hoja = es_hoja
        self.clase = clase

# --- Construcción recursiva ---
def construir_arbol_interactivo(data, atributos, target, nivel=0):
    indent = '    ' * nivel
    df = data[data[target] != '?']

    # caso sin datos
    if df.empty:
        st.write(f"{indent}⚠️ No hay datos, asigno clase 'Desconocido'")
        return NodoDecision(es_hoja=True, clase='Desconocido')
    # caso puro
    if len(df[target].unique()) == 1:
        clase_unica = df[target].iloc[0]
        st.write(f"{indent}▷ Particionando {data.columns[-1]} = {df.iloc[0, :-1].tolist()}? 📌 Nodo hoja con clase: {clase_unica}")
        return NodoDecision(es_hoja=True, clase=clase_unica)
    # caso sin atributos restantes
    if not atributos:
        clase_mayoria = df[target].mode()[0]
        st.write(f"{indent}⚠️ Sin atributos, nodo hoja clase mayoritaria: {clase_mayoria}")
        return NodoDecision(es_hoja=True, clase=clase_mayoria)

    # nivel actual
    st.write(f"{indent}=== Nivel {nivel}: Dividiendo sobre {len(df)} instancias ===")
    # cálculos de entropía condicional para cada atributo
    entropias = {}
    for attr in atributos:
        entropias[attr] = entropia_condicional(df, attr, target, indent + '    ')
    # seleccionar atributo con menor E_cond
    mejor = min(entropias, key=entropias.get)
    st.write(f"{indent}➡️ Mejor atributo = {mejor} (E(S|{mejor}) = {entropias[mejor]:.9f})\n")

    nodo = NodoDecision(atributo=mejor)
    # particionar en hijos
    for val in np.unique(df[mejor]):
        st.write(f"{indent}▷ Particionando {mejor} = {val}")
        sub = df[df[mejor] == val]
        if len(sub[target].unique()) == 1:
            clase_leaf = sub[target].iloc[0]
            st.write(f"{indent}📌 Nodo hoja con clase: {clase_leaf}\n")
            nodo.hijos[val] = NodoDecision(es_hoja=True, clase=clase_leaf)
        else:
            resto = [a for a in atributos if a != mejor]
            nodo.hijos[val] = construir_arbol_interactivo(sub, resto, target, nivel+1)
    return nodo

# --- Extracción de reglas y dibujo ---
def extraer_reglas(nodo, camino=None):
    camino = camino or []
    if nodo.es_hoja:
        cond = ' y '.join(camino) if camino else '(sin condición)'
        return [f"Si {cond}, entonces Categoría = {nodo.clase}"]
    reglas = []
    for v, h in nodo.hijos.items():
        reglas += extraer_reglas(h, camino + [f"{nodo.atributo} = {v}"])
    return reglas


def dibujar_arbol(nodo, dot=None, padre=None, etiqueta=None, contador=[0]):
    if dot is None:
        dot = Digraph(); dot.node('0','Inicio'); padre='0'; contador[0]=1
    nid = str(contador[0]); contador[0]+=1
    if nodo.es_hoja:
        dot.node(nid, f"Categoría: {nodo.clase}", shape='box', style='filled', color='lightgreen')
    else:
        dot.node(nid, nodo.atributo, shape='ellipse', style='filled', color='lightblue')
    dot.edge(padre, nid, label=etiqueta or '')
    if not nodo.es_hoja:
        for v, h in nodo.hijos.items():
            dibujar_arbol(h, dot, nid, str(v), contador)
    return dot

# --- Predicción interactiva ---
def predecir(nodo, ejemplo):
    if nodo.es_hoja:
        return nodo.clase
    val = ejemplo.get(nodo.atributo)
    if val not in nodo.hijos:
        val = next(iter(nodo.hijos))
    return predecir(nodo.hijos[val], ejemplo)

# --- App Streamlit ---
def procesar_arbol_decision():
    st.title("🌳 Árbol de Decisión ID3 - Proceso Profesor")
    uf = st.file_uploader("Sube CSV/XLSX", type=['csv','xlsx'])
    if uf:
        df = pd.read_csv(uf) if uf.name.endswith('csv') else pd.read_excel(uf)
        df = limpiar_y_normalizar_df(df, df.columns.tolist())
        st.session_state['df'] = df
    if 'df' not in st.session_state:
        st.info("Por favor, sube un archivo para continuar.")
        return
    df = st.session_state['df']
    st.subheader("Datos cargados")
    st.dataframe(df)

    cols = df.columns.tolist()
    target = st.selectbox("Seleccione target", cols, index=0)
    feats = st.multiselect("Seleccione features", [c for c in cols if c != target])

    if st.button("Construir árbol ID3"):
        if not feats:
            st.error("Debe seleccionar al menos una variable.")
            return
        df_model = df[feats + [target]].astype(str)
        st.session_state['arbol'] = construir_arbol_interactivo(df_logger, feats, target)
        st.session_state['reglas'] = extraer_reglas(st.session_state['arbol'])
        st.session_state['ok'] = True

    if st.session_state.get('ok'):
        st.success("Árbol construido.")
        st.subheader("Reglas")
        for i, r in enumerate(st.session_state['reglas'], 1):
            st.markdown(f"**Regla {i}:** {r}")
        st.subheader("Diagrama")
        st.graphviz_chart(dibujar_arbol(st.session_state['arbol']))

        st.subheader("Prueba de predicción")
        with st.form('form_pred'):
            ej = {}
            for c in feats:
                opts = sorted(set(df[c].unique()) | {'?'})
                ej[c] = st.selectbox(c, opts, index=opts.index('?'))
            if st.form_submit_button('Predecir'):
                pr = predecir(st.session_state['arbol'], ej)
                st.success(f"Predicción: {pr}")

if __name__ == '__main__':
    procesar_arbol_decision()
