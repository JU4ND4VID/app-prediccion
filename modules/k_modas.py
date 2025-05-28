# modules/k_modas.py

import streamlit as st
import pandas as pd
import numpy as np

def contar_modas_por_cluster(df_known, cat_col, clase_col):
    clases = df_known[clase_col].unique()
    valores = df_known[cat_col].unique()
    conteo = pd.DataFrame(index=valores, columns=clases).fillna(0)

    for c in clases:
        subset = df_known.loc[df_known[clase_col] == c, cat_col]
        conteo[c] = subset.value_counts()

    conteo = conteo.fillna(0).astype(int)
    modas = {c: conteo[c].idxmax() for c in clases}
    return modas, conteo

def diferencia_binaria(a, b):
    return 0 if a == b else 1

def procesar_k_modas():
    st.title("📊 K-modes clustering paso a paso")

    uploaded = st.file_uploader("Sube CSV o Excel", type=["csv","xlsx"])
    if not uploaded:
        st.info("Por favor sube un archivo para continuar.")
        return

    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded, na_values=["?"])
        else:
            df = pd.read_excel(uploaded, na_values=["?"])
    except Exception as e:
        st.error(f"Error cargando el archivo: {e}")
        return

    st.subheader("Vista previa del dataset")
    st.dataframe(df)

    cat_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])]
    if not cat_cols:
        st.error("No se encontraron columnas categóricas para clustering.")
        return

    cat_col = st.selectbox("Selecciona la columna categórica para clusterizar", cat_cols)
    clase_col = "Clase" if "Clase" in df.columns else None
    if not clase_col:
        st.error("No se encontró columna 'Clase' para definir clusters.")
        return

    k = df[clase_col].nunique()
    st.markdown(f"Clusters detectados (según columna '{clase_col}'): **{k}**")

    mask_known = df[[cat_col, clase_col]].notna().all(axis=1)
    df_known = df.loc[mask_known].copy()
    df_missing = df.loc[~mask_known].copy()

    modas, conteo = contar_modas_por_cluster(df_known, cat_col, clase_col)

    st.markdown("### Conteo de valores por cluster")
    st.dataframe(conteo)

    st.markdown("### Modas iniciales por cluster")
    modas_df = pd.DataFrame.from_dict(modas, orient='index', columns=[cat_col])
    modas_df.index.name = "Cluster"
    st.dataframe(modas_df)

    asign_prev = None
    convergencia = False
    max_iter = 20
    clases = list(modas.keys())

    for it in range(1, max_iter+1):
        distancias = []
        for c in clases:
            modo_c = modas[c]
            dist_c = df_known.apply(lambda row:
                diferencia_binaria(row[cat_col], modo_c) + diferencia_binaria(row[clase_col], c),
                axis=1)
            distancias.append(dist_c.values)
        dist_arr = np.vstack(distancias).T

        asign_idx = np.argmin(dist_arr, axis=1)
        asign = [clases[i] for i in asign_idx]

        tabla = pd.DataFrame({cat_col: df_known[cat_col], clase_col: df_known[clase_col]})
        for j, c in enumerate(clases):
            tabla[f"Distancia Cluster {c}"] = dist_arr[:, j]
        tabla["Cluster Más Cercano"] = asign

        st.markdown(f"## Iteración {it}")
        st.dataframe(tabla)

        modas_df = pd.DataFrame.from_dict(modas, orient='index', columns=[cat_col])
        modas_df.index.name = "Cluster"
        st.markdown("### Modas (centroides) actuales")
        st.dataframe(modas_df)

        if asign_prev is not None and asign == asign_prev:
            st.success(f"Convergencia alcanzada en iteración {it}")
            convergencia = True
            break

        asign_prev = asign.copy()

        df_known["Cluster Temporal"] = asign
        nuevas_modas = {}
        for c in clases:
            vals = df_known.loc[df_known["Cluster Temporal"] == c, cat_col]
            if len(vals) > 0:
                nuevas_modas[c] = vals.mode().iloc[0]
            else:
                nuevas_modas[c] = modas[c]

        modas = nuevas_modas
        df_known.drop(columns=["Cluster Temporal"], inplace=True)

    if not convergencia:
        st.warning(f"No se alcanzó convergencia en {max_iter} iteraciones.")

    if not df_missing.empty:
        df_missing = df_missing.copy()
        df_missing["Cluster asignado"] = df_missing[clase_col]
        df_missing[cat_col] = df_missing["Cluster asignado"].map(modas)
        st.markdown("### Imputación de datos faltantes")
        st.dataframe(df_missing[[cat_col, clase_col, "Cluster asignado"]])

    df_known["Cluster asignado"] = asign
    resultado = pd.concat([df_known, df_missing], axis=0)
    st.markdown("### Resultado final")
    st.dataframe(resultado.reset_index(drop=True))

def run():
    procesar_k_modas()
