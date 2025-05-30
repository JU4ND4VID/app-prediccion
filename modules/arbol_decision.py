import pandas as pd
import numpy as np
import streamlit as st
from graphviz import Digraph
import unicodedata


def normalizar_texto(texto):
    texto = str(texto).strip().lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    return texto


def corregir_errores_ortograficos(valor):
    correcciones = {
        'ingnieria': 'ingenieria',
        # Añade aquí más correcciones si las detectes
    }
    return correcciones.get(valor, valor)


def limpiar_y_normalizar_df(df, columnas):
    for col in columnas:
        df[col] = df[col].astype(str).apply(normalizar_texto)
        df[col] = df[col].apply(corregir_errores_ortograficos)
    return df


def calcular_entropia(etiquetas, base=None, indent=""):
    """
    Muestra paso a paso la entropía de un conjunto de etiquetas,
    usando logaritmo en base `base` (si no se da, base = número de clases).
    """
    valores, conteos = np.unique(etiquetas, return_counts=True)
    probabilidades = conteos / conteos.sum()
    k = base or len(valores)
    ent = 0.0
    st.markdown(f"{indent}• Entropía (base={k}) de {len(etiquetas)} instancias:")
    for v, p in zip(valores, probabilidades):
        contrib = -p * np.log(p + 1e-9) / np.log(k)
        st.markdown(f"{indent}    p(c={v})={p:.3f} → −p·log(p)/log({k}) = {contrib:.3f}")
        ent += contrib
    st.markdown(f"{indent}    **E = {ent:.4f}**\n")
    return ent


def ganancia_informacion(data, atributo, target, indent=""):
    """
    Muestra el cálculo completo de E(S), E(S|A) y G(S,A) paso a paso para un atributo.
    """
    # 1) Entropía total
    st.markdown(f"{indent}🔸 Calculando entropía total E(S) para '{target}'")
    E_total = calcular_entropia(data[target], base=len(data[target].unique()), indent=indent + "  ")

    # 2) Entropía condicional
    st.markdown(f"{indent}🔹 Entropía condicional E(S|{atributo}):")
    valores, conteos = np.unique(data[atributo], return_counts=True)
    n_total = conteos.sum()
    E_cond = 0.0
    for v, c in zip(valores, conteos):
        peso = c / n_total
        subset = data[data[atributo] == v][target]
        st.markdown(f"{indent}  • Subconjunto {atributo} = **{v}** (n={c}, peso={peso:.3f}):")
        E_sub = calcular_entropia(subset, base=len(data[target].unique()), indent=indent + "    ")
        E_cond += peso * E_sub

    st.markdown(f"{indent}    **E(S|{atributo}) = {E_cond:.4f}**\n")

    # 3) Ganancia
    G = E_total - E_cond
    st.markdown(f"{indent}➡️ **G(S,{atributo}) = E(S) – E(S|{atributo}) = {E_total:.4f} – {E_cond:.4f} = {G:.4f}**\n")
    return G


class NodoDecision:
    def __init__(self, atributo=None, hijos=None, es_hoja=False, clase=None):
        self.atributo = atributo
        self.hijos = hijos if hijos is not None else {}
        self.es_hoja = es_hoja
        self.clase = clase


def construir_arbol_interactivo(data, atributos, target, nivel=0):
    indent = "    " * nivel
    data = data[data[target] != '?']

    # Caso base: sin datos
    if len(data) == 0:
        st.markdown(f"{indent}⚠️ No hay datos para construir el nodo, se asigna clase 'Desconocido'")
        return NodoDecision(es_hoja=True, clase="Desconocido")

    # Caso base: nodo puro
    if len(data[target].unique()) == 1:
        clase_unica = data[target].iloc[0]
        st.markdown(f"{indent}📌 Nodo hoja con clase: **{clase_unica}**")
        return NodoDecision(es_hoja=True, clase=clase_unica)

    # Caso base: sin atributos restantes
    if len(atributos) == 0:
        clase_mayoritaria = data[target].mode()[0]
        st.markdown(f"{indent}⚠️ Sin atributos restantes. Nodo hoja con clase mayoritaria: **{clase_mayoritaria}**")
        return NodoDecision(es_hoja=True, clase=clase_mayoritaria)

    # Cálculo de ganancias para cada atributo
    ganancias = {}
    for atributo in atributos:
        st.markdown(f"{indent}--- Procesando atributo '{atributo}' ---")
        gan = ganancia_informacion(data, atributo, target, indent=indent + "  ")
        ganancias[atributo] = gan

    # Selección del mejor atributo
    mejor_atributo = max(ganancias, key=ganancias.get)
    st.markdown(f"{indent}➡️ **Mejor atributo para dividir en nivel {nivel}: '{mejor_atributo}' → Ganancia = {ganancias[mejor_atributo]:.4f}**\n")

    # Crear nodo y particionar
    nodo = NodoDecision(atributo=mejor_atributo)
    for valor in np.unique(data[mejor_atributo]):
        st.markdown(f"{indent}▷ Particionando para **{mejor_atributo} = {valor}**")
        subset = data[data[mejor_atributo] == valor]
        atributos_restantes = [a for a in atributos if a != mejor_atributo]
        nodo.hijos[valor] = construir_arbol_interactivo(subset, atributos_restantes, target, nivel + 1)

    return nodo


def extraer_reglas(nodo, camino=None):
    camino = camino or []
    if nodo.es_hoja:
        regla = ' y '.join(camino) if camino else '(sin condición)'
        return [f"Si {regla}, entonces Categoría = {nodo.clase}"]
    reglas = []
    for valor, hijo in nodo.hijos.items():
        nueva_cond = f"{nodo.atributo} = {valor}"
        reglas.extend(extraer_reglas(hijo, camino + [nueva_cond]))
    return reglas


def dibujar_arbol(nodo, dot=None, padre=None, etiqueta=None, contador=[0]):
    if dot is None:
        dot = Digraph()
        dot.node('0', 'Inicio')
        contador[0] = 1
        padre = '0'

    nid = str(contador[0])
    contador[0] += 1

    if nodo.es_hoja:
        dot.node(nid, f"Categoría: {nodo.clase}", shape='box', style='filled', color='lightgreen')
    else:
        dot.node(nid, nodo.atributo, shape='ellipse', style='filled', color='lightblue')
    dot.edge(padre, nid, label=etiqueta or "")

    if not nodo.es_hoja:
        for val, child in nodo.hijos.items():
            dibujar_arbol(child, dot, nid, str(val), contador)
    return dot


def predecir(nodo, ejemplo):
    if nodo.es_hoja:
        return nodo.clase
    val = ejemplo.get(nodo.atributo, None)
    if val == '?' or val not in nodo.hijos:
        # escoger niño con mayor ramificación
        if nodo.hijos:
            val = max(nodo.hijos, key=lambda k: len(nodo.hijos[k].hijos) if not nodo.hijos[k].es_hoja else 0)
        else:
            return None
    return predecir(nodo.hijos[val], ejemplo)


def procesar_arbol_decision():
    st.title("🌳 Árbol de Decisión ID3 con explicación paso a paso")

    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df = limpiar_y_normalizar_df(df, df.columns.tolist())
        st.session_state['df'] = df

    if 'df' not in st.session_state:
        st.info("Por favor, sube un archivo para continuar.")
        return

    df = st.session_state['df']
    st.subheader("Vista previa de datos")
    st.dataframe(df)

    cols = df.columns.tolist()
    target_col = st.selectbox("Selecciona la variable a predecir (target)", cols, index=0)
    input_cols = st.multiselect("Selecciona las variables de entrada (features)", [c for c in cols if c != target_col])

    if st.button("Generar árbol ID3 con explicación"):
        if not input_cols:
            st.error("Selecciona al menos una variable de entrada.")
            return
        df_model = df[input_cols + [target_col]].astype(str)
        st.session_state['arbol'] = construir_arbol_interactivo(df_model, input_cols, target_col)
        st.session_state['reglas'] = extraer_reglas(st.session_state['arbol'])
        st.session_state['arbol_creado'] = True

    if st.session_state.get('arbol_creado'):
        st.success("Árbol construido correctamente.")
        st.subheader("📋 Reglas de Clasificación Generadas")
        for i, regla in enumerate(st.session_state['reglas'], 1):
            st.markdown(f"**Regla {i}:** {regla}")
        st.subheader("🌳 Diagrama del Árbol")
        dot = dibujar_arbol(st.session_state['arbol'])
        st.graphviz_chart(dot)

        st.subheader("🧪 Prueba de predicción con valores '?' ")
        with st.form(key='form_pred'):
            ejemplo = {}
            for idx, col in enumerate(input_cols):
                opciones = sorted(set(st.session_state['df'][col].unique()).union({'?'}))
                ejemplo[col] = st.selectbox(f"Valor para {col}", opciones, index=opciones.index('?'))
            submitted = st.form_submit_button("Predecir")
        if submitted:
            pred = predecir(st.session_state['arbol'], ejemplo)
            if pred is None:
                st.error("No se pudo predecir para el ejemplo dado.")
            else:
                st.success(f"La predicción es: **{pred}**")


def run():
    procesar_arbol_decision()


if __name__ == '__main__':
    run()
