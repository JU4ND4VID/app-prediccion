import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def imputar_valores(df, cols_num):
    for c in cols_num:
        if df[c].isnull().any():
            media = df[c].mean()
            df[c].fillna(media, inplace=True)
    return df

def inicializar_centroides_por_clase(df, cols_num, col_clase):
    clases = df[col_clase].dropna().unique()
    centroides = []
    for c in clases:
        media = df.loc[df[col_clase] == c, cols_num].mean().values
        centroides.append(media)
    return np.array(centroides), clases

def calcular_distancias(X, centroides):
    return np.linalg.norm(X[:, None] - centroides, axis=2)

def mostrar_grafica_pca(X, asignaciones, centroides, titulo):
    if X.shape[1] < 2:
        st.warning("PCA requiere al menos dos columnas numéricas para visualizar.")
        return
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centroides_pca = pca.transform(centroides)

    plt.figure(figsize=(7,5))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=asignaciones, cmap='tab10', s=60, alpha=0.7)
    plt.scatter(centroides_pca[:,0], centroides_pca[:,1], c='black', s=200, marker='X')
    plt.title(titulo)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

def procesar_k_means():
    st.title("📊 K-means clustering paso a paso")

    uploaded_file = st.file_uploader("Sube archivo CSV o Excel", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Sube un archivo para continuar.")
        return

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, na_values=["?"])
        else:
            df = pd.read_excel(uploaded_file, na_values=["?"])
    except Exception as e:
        st.error(f"Error al cargar archivo: {e}")
        return

    st.subheader("Vista previa del dataset")
    st.dataframe(df)

    columnas = df.columns.tolist()
    num_cols = [col for col in columnas if pd.api.types.is_numeric_dtype(df[col])]
    if not num_cols:
        st.error("No se encontraron columnas numéricas para clustering.")
        return

    x_cols = st.multiselect("Selecciona columnas numéricas para clustering", num_cols, default=num_cols)
    if len(x_cols) < 1:
        st.warning("Selecciona al menos una columna numérica para clustering.")
        return

    cat_cols = [col for col in columnas if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])]
    k = 2
    cat_col = None
    if cat_cols:
        cat_col = st.selectbox("Selecciona columna categórica para definir número de clusters (opcional)", [None] + cat_cols)
        if cat_col:
            k = df[cat_col].nunique()
    else:
        st.info("No se encontró columna categórica, se usará k=2 por defecto.")

    df = imputar_valores(df, x_cols)
    X = df[x_cols].values

    if not st.button("Ejecutar K-means paso a paso"):
        return

    if cat_col:
        centroides, clases = inicializar_centroides_por_clase(df, x_cols, cat_col)
    else:
        indices = np.random.choice(len(X), k, replace=False)
        centroides = X[indices]
        clases = np.arange(k)

    asign_prev = None
    convergencia = False
    max_iter = 20

    for i in range(1, max_iter + 1):
        distancias = calcular_distancias(X, centroides)
        asign_idx = np.argmin(distancias, axis=1)
        asignaciones = clases[asign_idx]

        tabla = pd.DataFrame(X, columns=x_cols)
        for idx, c in enumerate(clases):
            tabla[f"Distancia Cluster {c}"] = distancias[:, idx].round(2)
        tabla['Cluster Más Cercano'] = asignaciones

        st.markdown(f"## Iteración {i}")
        st.dataframe(tabla)

        st.markdown("### Centroides")
        centroides_df = pd.DataFrame(centroides, columns=x_cols)
        centroides_df['Cluster'] = clases
        st.dataframe(centroides_df)

        if asign_prev is not None and np.array_equal(asign_idx, asign_prev):
            st.success(f"Convergencia alcanzada en iteración {i}")
            convergencia = True
            break

        asign_prev = asign_idx.copy()

        nuevos_centroides = []
        for idx, c in enumerate(clases):
            puntos = X[asign_idx == idx]
            if len(puntos) > 0:
                nuevos_centroides.append(puntos.mean(axis=0))
            else:
                nuevos_centroides.append(centroides[idx])
        centroides = np.array(nuevos_centroides)

        if len(x_cols) >= 2:
            mostrar_grafica_pca(X, asign_idx, centroides, f"Clusters iteración {i}")

    if not convergencia:
        st.warning(f"No se alcanzó convergencia en {max_iter} iteraciones.")

    df['Cluster asignado'] = asignaciones
    st.markdown("### Resultado final")
    st.dataframe(df[[*x_cols, 'Cluster asignado']])

def run():
    procesar_k_means()
