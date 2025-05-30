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
    correcciones = { 'ingnieria': 'ingenieria' }
    return correcciones.get(valor, valor)

def limpiar_y_normalizar_df(df, columnas):
    for col in columnas:
        df[col] = df[col].astype(str).apply(normalizar_texto)
        df[col] = df[col].apply(corregir_errores_ortograficos)
    return df

# --- Cálculo de entropía y ganancia ---
def calcular_entropia(etiquetas, base=None):
    """
    Entropía de un vector de etiquetas, con logaritmo en base `base` (por defecto número de clases).
    """
    valores, conteos = np.unique(etiquetas, return_counts=True)
    probabilidades = conteos / conteos.sum()
    k = base or len(valores)
    ent = -np.sum(probabilidades * np.log(probabilidades + 1e-9) / np.log(k))
    return ent


def ganancia_informacion(data, atributo, target):
    """
    Imprime y retorna la ganancia de información G(S, atributo), mostrando solo resultados.
    """
    # Entropía total
    E_total = calcular_entropia(data[target], base=len(data[target].unique()))
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
        filas.append({
            'Valor': v,
            'Cantidad': c,
            'Peso': round(peso, 3),
            'Entropía Subconjunto': round(E_sub, 3),
            'Contribución': round(contrib, 3)
        })
        E_cond += contrib

    df_detalles = pd.DataFrame(filas)
    st.write(f"🔸 Entropía total E(S): **{E_total:.4f}**")
    st.write(f"🔹 Detalles para atributo '{atributo}':")
    st.dataframe(df_detalles)
    st.write(f"🔹 E(S|{atributo}) = **{E_cond:.4f}**")

    G = E_total - E_cond
    st.write(f"➡️ Ganancia G(S,{atributo}) = E(S) – E(S|{atributo}) = **{G:.4f}**")
    st.write('---')
    return G

# --- Clase de nodo ---
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

    if len(data) == 0:
        st.write(f"{indent}⚠️ No hay datos, asigno clase 'Desconocido'")
        return NodoDecision(es_hoja=True, clase='Desconocido')

    if len(data[target].unique()) == 1:
        unico = data[target].iloc[0]
        st.write(f"{indent}📌 Nodo hoja con clase: **{unico}**")
        return NodoDecision(es_hoja=True, clase=unico)

    if not atributos:
        mayor = data[target].mode()[0]
        st.write(f"{indent}⚠️ Sin atributos restantes, clase mayoritaria: **{mayor}**")
        return NodoDecision(es_hoja=True, clase=mayor)

    # Entropía total y ganancias
    st.write(f"{indent}=== Nivel {nivel}: Dividiendo sobre {len(data)} instancias ===")
    ganancias = {}
    for attr in atributos:
        gan = ganancia_informacion(data, attr, target)
        ganancias[attr] = gan

    mejor = max(ganancias, key=ganancias.get)
    st.write(f"{indent}➡️ Mejor atributo: **{mejor}** (G = {ganancias[mejor]:.4f})")

    nodo = NodoDecision(atributo=mejor)
    for val in np.unique(data[mejor]):
        st.write(f"{indent}▷ Particionando {mejor} = {val}")
        sub = data[data[mejor] == val]
        resto = [a for a in atributos if a != mejor]
        nodo.hijos[val] = construir_arbol_interactivo(sub, resto, target, nivel+1)

    return nodo

# --- Extracción de reglas, dibujo y predicción (sin cambios) ---
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
        for v, h in nodo.hijos.items(): dibujar_arbol(h, dot, nid, str(v), contador)
    return dot


def predecir(nodo, ejemplo):
    if nodo.es_hoja: return nodo.clase
    val = ejemplo.get(nodo.atributo)
    if val not in nodo.hijos:
        val = max(nodo.hijos, key=lambda k: len(nodo.hijos[k].hijos) if not nodo.hijos[k].es_hoja else 0)
    return predecir(nodo.hijos[val], ejemplo)


def procesar_arbol_decision():
    st.title("🌳 Árbol ID3 con resultados paso a paso")
    uf = st.file_uploader("Sube CSV/XLSX", type=['csv','xlsx'])
    if uf:
        df = pd.read_csv(uf) if uf.name.endswith('csv') else pd.read_excel(uf)
        df = limpiar_y_normalizar_df(df, df.columns.tolist())
        st.session_state['df']=df
    if 'df' not in st.session_state:
        st.info("Sube un archivo para continuar")
        return
    df = st.session_state['df']; st.subheader("Datos")
    st.dataframe(df)
    cols = df.columns.tolist()
    target = st.selectbox("Target", cols, index=0)
    feats = st.multiselect("Features", [c for c in cols if c!=target])
    if st.button("Generar árbol"):
        if not feats: st.error("Selecciona variables"); return
        dfm = df[feats+[target]].astype(str)
        st.session_state['arbol']=construir_arbol_interactivo(dfm, feats, target)
        st.session_state['reglas']=extraer_reglas(st.session_state['arbol'])
        st.session_state['ok']=True
    if st.session_state.get('ok'):
        st.success("Árbol listo")
        st.subheader("Reglas")
        for i,r in enumerate(st.session_state['reglas'],1): st.markdown(f"**Regla {i}:** {r}")
        st.subheader("Diagrama")
        st.graphviz_chart(dibujar_arbol(st.session_state['arbol']))
        st.subheader("Prueba")
        with st.form('p'):
            ej={}
            for c in feats:
                opts=sorted(set(st.session_state['df'][c].unique())|{'?'})
                ej[c]=st.selectbox(c, opts, index=opts.index('?'))
            if st.form_submit_button('Predecir'):
                res=predecir(st.session_state['arbol'], ej)
                st.success(f"Predicción: {res}")


def run(): procesar_arbol_decision()

if __name__=='__main__': run()
