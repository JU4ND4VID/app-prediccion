import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def inicializar_centroides_por_clase(df_known, x_cols, cat_col):
    clases = df_known[cat_col].unique()
    centroides = []
    for c in clases:
        media = df_known.loc[df_known[cat_col] == c, x_cols].mean().values
        centroides.append(media)
    return np.vstack(centroides), np.array(clases)

def inicializar_centroides_aleatorios(X_known, k):
    idx = np.random.choice(len(X_known), k, replace=False)
    return X_known[idx]

def mostrar_grafica_pca(X, asign_idx, centroides, titulo):
    if X.shape[1] < 2:
        st.warning("PCA requiere al menos dos columnas numéricas para visualizar.")
        return
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    C2 = pca.transform(centroides)

    plt.figure(figsize=(7,5))
    scatter = plt.scatter(X2[:,0], X2[:,1], c=asign_idx, cmap='tab10', s=60, alpha=0.7)
    plt.scatter(C2[:,0], C2[:,1], c='black', s=200, marker='X')
    plt.title(titulo)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

def procesar_k_medias():
    st.title("📊 K-medias")

    # 1) Carga de datos
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

    # 2) Selección de columnas numéricas
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        st.error("No se encontraron columnas numéricas.")
        return

    x_cols = st.multiselect("Selecciona columnas numéricas para clustering", num_cols, default=num_cols)
    if len(x_cols) < 1:
        st.warning("Selecciona al menos una columna numérica.")
        return

    # 3) Detección automática de columna de clase
    cat_col = "Clase" if "Clase" in df.columns else None
    if cat_col:
        st.markdown(f"### Usando columna categórica para inicializar centroides: **{cat_col}**")
        k = df[cat_col].nunique()
    else:
        st.info("No se encontró columna 'Clase', inicializando aleatoriamente k=2")
        k = 2

    # 4) Separar filas completas y faltantes
    mask_known = df[x_cols].notna().all(axis=1)
    df_known   = df.loc[mask_known].copy()
    df_missing = df.loc[~mask_known].copy()

    X_known = df_known[x_cols].to_numpy()

    # 5) Inicializar centroides
    if cat_col:
        centroides, clases = inicializar_centroides_por_clase(df_known, x_cols, cat_col)
    else:
        centroides = inicializar_centroides_aleatorios(X_known, k)
        clases = np.arange(k)

    asign_prev = None
    convergencia = False
    max_iter = 20

    # 6) Iteraciones de K-medias
    for it in range(1, max_iter+1):
        dist = np.vstack([euclidean_distance(X_known, c) for c in centroides]).T  # n×k
        asign_idx = np.argmin(dist, axis=1)
        asign = clases[asign_idx]

        # Tabla de distancias y asignaciones
        tabla = df_known[x_cols].reset_index(drop=True)
        for j,c in enumerate(clases):
            tabla[f"Distancia Cluster {c}"] = dist[:,j].round(2)
        tabla["Cluster Más Cercano"] = asign
        st.markdown(f"## Iteración {it}")
        st.dataframe(tabla)

        # Tabla de centroides
        cent_df = pd.DataFrame(centroides, columns=x_cols)
        cent_df["Cluster"] = clases
        st.markdown("### Centroides")
        st.dataframe(cent_df.round(2))

        # Gráfica PCA
        if len(x_cols) >= 2:
            mostrar_grafica_pca(X_known, asign_idx, centroides, f"Clusters iteración {it}")

        # Verificar convergencia
        if asign_prev is not None and np.array_equal(asign_idx, asign_prev):
            st.success(f"Convergencia alcanzada en iteración {it}")
            convergencia = True
            break

        asign_prev = asign_idx.copy()

        # Recalcular centroides
        nuevos = []
        for j in range(len(centroides)):
            pts = X_known[asign_idx == j]
            nuevos.append(pts.mean(axis=0) if len(pts)>0 else centroides[j])
        centroides = np.vstack(nuevos)

    if not convergencia:
        st.warning(f"No se alcanzó convergencia en {max_iter} iteraciones.")

    # 7) Imputar valores faltantes (con sus clases)
    if cat_col and not df_missing.empty:
        mapa = {c: centroides[i] for i,c in enumerate(clases)}
        df_missing = df_missing.copy()
        df_missing["Cluster asignado"] = df_missing[cat_col]
        for col in x_cols:
            df_missing[col] = df_missing["Cluster asignado"].map(lambda cl: mapa[cl][x_cols.index(col)])
        st.markdown("### Imputación de datos faltantes")
        st.dataframe(df_missing[[*x_cols, cat_col, "Cluster asignado"]].round(2))

    # 8) Resultado final
    df_known["Cluster asignado"] = asign
    resultado = pd.concat([df_known, df_missing], axis=0)
    st.markdown("### Resultado final")
    st.dataframe(resultado.reset_index(drop=True).round(2))

def run():
    procesar_k_medias()
