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

# --- Cálculo de entropía y ganancia con detalles ---
def calcular_entropia(etiquetas, base=None):
    valores, conteos = np.unique(etiquetas, return_counts=True)
    probabilidades = conteos / conteos.sum()
    k = base or len(valores)
    # entropía base k
    return -np.sum(probabilidades * np.log(probabilidades + 1e-9) / np.log(k))


def ganancia_informacion(data, atributo, target, indent=""):
    """
    Calcula E(S), E(S|atributo) y G(S,atributo), construye una tabla con:
    Valor, Cantidad, Peso, Entropía Subconjunto, Contribución y Distribución.
    """
    # Entropía total
    E_total = calcular_entropia(data[target], base=len(data[target].unique()))
    st.write(f"{indent}🔸 Entropía total E(S) = {E_total:.4f}")

    # Detalles por valor de atributo
    filas = []
    valores, conteos = np.unique(data[atributo], return_counts=True)
    n_total = conteos.sum()
    E_cond = 0.0
    for v, c in zip(valores, conteos):
        peso = c / n_total
        subset = data[data[atributo] == v][target]
        E_sub = calcular_entropia(subset, base=len(data[target].unique()))
        contrib = peso * E_sub
        # distribución de clases dentro de S_v
        clases, cnts = np.unique(subset, return_counts=True)
        dist = "; ".join([f"{clase}:{cnts[i]}/{len(subset)}" for i, clase in enumerate(clases)])
        filas.append({
            'Valor': v,
            'Cantidad': int(c),
            'Peso': round(peso, 3),
            'Entropía Subconjunto': round(E_sub, 3),
            'Contribución': round(contrib, 3),
            'Distribución': dist
        })
        E_cond += contrib

    df_detalles = pd.DataFrame(filas)
    st.write(f"{indent}🔹 Detalles para atributo '{atributo}':")
    st.dataframe(df_detalles)
    st.write(f"{indent}🔹 E(S|{atributo}) = {E_cond:.4f}")

    G = E_total - E_cond
    st.write(f"{indent}➡️ Ganancia G(S,{atributo}) = {G:.4f}")
    st.write(f"{indent}{'-'*30}")
    return G

# --- Nodo de decisión ---
class NodoDecision:
    def __init__(self, atributo=None, hijos=None, es_hoja=False, clase=None):
        self.atributo = atributo
        self.hijos = hijos or {}
        self.es_hoja = es_hoja
        self.clase = clase

# --- Construcción recursiva con explicación ---
def construir_arbol_interactivo(data, atributos, target, nivel=0):
    indent = '    ' * nivel
    data = data[data[target] != '?']

    # caso sin datos
    if data.empty:
        st.write(f"{indent}⚠️ No hay datos, asigno clase 'Desconocido'")
        return NodoDecision(es_hoja=True, clase='Desconocido')

    # caso puro
    if len(data[target].unique()) == 1:
        clase_unica = data[target].iloc[0]
        st.write(f"{indent}📌 Nodo hoja con clase: {clase_unica}")
        return NodoDecision(es_hoja=True, clase=clase_unica)

    # caso sin atributos
    if not atributos:
        clase_mayoria = data[target].mode()[0]
        st.write(f"{indent}⚠️ Sin atributos restantes, clase mayoritaria: {clase_mayoria}")
        return NodoDecision(es_hoja=True, clase=clase_mayoria)

    # nivel actual
    st.write(f"{indent}=== Nivel {nivel}: Dividiendo sobre {len(data)} instancias ===")
    # calcular ganancias para cada atributo
    ganancias = {}
    for attr in atributos:
        st.write(f"{indent}--- Procesando atributo '{attr}' ---")
        ganancias[attr] = ganancia_informacion(data, attr, target, indent + '    ')

    # seleccionar mejor (máxima ganancia => mínima entropía condicional)
    mejor = max(ganancias, key=ganancias.get)
    st.write(f"{indent}➡️ **Mejor atributo**: {mejor} (G = {ganancias[mejor]:.4f})\n")

    nodo = NodoDecision(atributo=mejor)
    # particionar
    for val in np.unique(data[mejor]):
        sub = data[data[mejor] == val]
        # si es hoja
        if len(sub[target].unique()) == 1:
            clase_sub = sub[target].iloc[0]
            st.write(f"{indent}▷ Particionando {mejor} = {val} → Nodo hoja: {clase_sub}")
            nodo.hijos[val] = NodoDecision(es_hoja=True, clase=clase_sub)
        else:
            st.write(f"{indent}▷ Particionando {mejor} = {val}")
            resto = [a for a in atributos if a != mejor]
            nodo.hijos[val] = construir_arbol_interactivo(sub, resto, target, nivel + 1)

    return nodo

# --- Extracción de reglas, dibujo y predicción ---
def extraer_reglas(nodo, camino=None):
    camino = camino or []
    if nodo.es_hoja:
        cond = ' y '.join(camino) or '(sin condición)'
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


def predecir(nodo, ejemplo):
    if nodo.es_hoja:
        return nodo.clase
    val = ejemplo.get(nodo.atributo)
    if val not in nodo.hijos:
        val = max(nodo.hijos, key=lambda k: len(nodo.hijos[k].hijos) if not nodo.hijos[k].es_hoja else 0)
    return predecir(nodo.hijos[val], ejemplo)


def procesar_arbol_decision():
    st.title("🌳 Árbol de Decisión ID3 con explicación paso a paso")
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

    if st.button("Generar árbol ID3"):
        if not feats:
            st.error("Debe seleccionar al menos una variable de entrada.")
            return
        df_model = df[feats + [target]].astype(str)
        st.session_state['arbol'] = construir_arbol_interactivo(df_model, feats, target)
        st.session_state['reglas'] = extraer_reglas(st.session_state['arbol'])
        st.session_state['ok'] = True

    if st.session_state.get('ok'):
        st.success("Árbol construido correctamente.")
        st.subheader("📋 Reglas de Clasificación")
        for i, regla in enumerate(st.session_state['reglas'], 1):
            st.markdown(f"**Regla {i}:** {regla}")
        st.subheader("🌳 Diagrama del Árbol")
        st.graphviz_chart(dibujar_arbol(st.session_state['arbol']))

        st.subheader("🧪 Prueba de predicción")
        with st.form('form_pred'):
            ejemplo = {}
            for c in feats:
                opts = sorted(set(st.session_state['df'][c].unique()) | {'?'})
                ejemplo[c] = st.selectbox(c, opts, index=opts.index('?'))
            if st.form_submit_button('Predecir'):
                pr = predecir(st.session_state['arbol'], ejemplo)
                st.success(f"Predicción: {pr}")


def run():
    procesar_arbol_decision()

if __name__ == '__main__':
    run()
