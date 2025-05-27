import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def inicializar_centroides(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def k_means_iterativo(X, k, max_iter=100):
    centroides = inicializar_centroides(X, k)
    asignaciones = np.full(shape=len(X), fill_value=-1)
    for iter_num in range(1, max_iter+1):
        distancias = np.array([euclidean_distance(X, c) for c in centroides]).T  # shape (n_samples, k)
        nuevas_asignaciones = np.argmin(distancias, axis=1)
        if np.array_equal(asignaciones, nuevas_asignaciones):
            break
        asignaciones = nuevas_asignaciones
        for i in range(k):
            if np.any(asignaciones == i):
                centroides[i] = X[asignaciones == i].mean(axis=0)
    return centroides, asignaciones, iter_num

def mostrar_grafica_pca(X, asignaciones, centroides, titulo):
    # Validar dimensiones para PCA
    if min(X.shape) < 2:
        st.warning("No es posible realizar PCA con menos de 2 muestras o 2 características.")
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
    st.title("📊 K-means clustering con visualización paso a paso")

    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Por favor, sube un archivo para continuar.")
        return

    # Cargar DataFrame
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return

    st.subheader("Vista previa del dataset")
    st.dataframe(df)

    columnas = df.columns.tolist()

    # Selección de columnas numéricas para clustering
    num_cols = [col for col in columnas if pd.api.types.is_numeric_dtype(df[col])]
    if not num_cols:
        st.error("No se encontraron columnas numéricas para clustering.")
        return
    x_cols = st.multiselect("Selecciona columnas numéricas para clustering", num_cols, default=num_cols)

    if len(x_cols) < 2:
        st.warning("Selecciona al menos dos columnas numéricas para realizar PCA y mostrar gráfica.")
        return

    # Selección columna categórica para detectar k (opcional)
    cat_cols = [col for col in columnas if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])]
    k = 2
    cat_col = None
    if cat_cols:
        cat_col = st.selectbox("Selecciona columna categórica para definir número de clusters (opcional)", [None] + cat_cols)
        if cat_col:
            k = df[cat_col].nunique()
    else:
        st.info("No se encontró columna categórica, se usará k=2 por defecto.")

    if not x_cols:
        st.warning("Selecciona al menos una columna numérica.")
        return

    # Extraer datos numéricos y manejar valores faltantes
    X_raw = df[x_cols]
    # Guardar mask de filas con NaNs para asignar después
    mask_nan = X_raw.isnull().any(axis=1)
    # Imputar NaNs con media columna
    X = X_raw.fillna(X_raw.mean()).to_numpy()

    st.markdown(f"### Número de clusters (k): {k}")
    st.markdown(f"### Columnas usadas para clustering: {x_cols}")

    calcular = st.button("Ejecutar K-means paso a paso")

    if not calcular:
        return

    # Ejecutar K-means iterativo mostrando paso a paso
    centroides = None
    asignaciones = None

    max_iter = 20
    asignaciones_previas = None
    convergencia = False

    # Para mostrar iteraciones:
    # inicializar centroides aleatorios
    centroides = inicializar_centroides(X, k)

    for i in range(1, max_iter+1):
        distancias = np.array([euclidean_distance(X, c) for c in centroides]).T
        nuevas_asignaciones = np.argmin(distancias, axis=1)

        st.markdown(f"## Iteración {i}")
        # Tabla centroides
        st.markdown("### Centroides")
        centroides_df = pd.DataFrame(centroides, columns=x_cols)
        st.dataframe(centroides_df)

        # Tabla distancias (solo primeras 10 filas para evitar saturar)
        dist_df = pd.DataFrame(distancias, columns=[f"Cluster {j}" for j in range(k)])
        st.markdown("### Distancias de cada punto a los centroides (primeras 10 filas)")
        st.dataframe(dist_df.head(10))

        # Tabla asignaciones
        asign_df = pd.DataFrame({
            "Índice": df.index,
            "Asignación cluster actual": nuevas_asignaciones
        })
        st.markdown("### Asignaciones de clusters")
        st.dataframe(asign_df.head(10))

        # Mostrar gráfica con PCA
        mostrar_grafica_pca(X, nuevas_asignaciones, centroides, f"Clusters iteración {i}")

        if asignaciones_previas is not None and np.array_equal(nuevas_asignaciones, asignaciones_previas):
            convergencia = True
            st.success(f"Convergencia alcanzada en iteración {i}")
            break

        asignaciones_previas = nuevas_asignaciones.copy()
        # Recalcular centroides
        for cluster_idx in range(k):
            puntos_cluster = X[nuevas_asignaciones == cluster_idx]
            if len(puntos_cluster) > 0:
                centroides[cluster_idx] = puntos_cluster.mean(axis=0)

    if not convergencia:
        st.warning(f"No se alcanzó convergencia en {max_iter} iteraciones.")

    # Asignar filas originalmente con NaN
    if mask_nan.any():
        st.markdown("### Asignación de datos con valores faltantes (imputados durante cálculo)")
        X_nan = X_raw[mask_nan].fillna(X_raw.mean()).to_numpy()
        dist_nan = np.array([euclidean_distance(X_nan, c) for c in centroides]).T
        asign_nan = np.argmin(dist_nan, axis=1)
        df.loc[mask_nan, 'Cluster asignado'] = asign_nan
        st.dataframe(df.loc[mask_nan, ['Cluster asignado'] + x_cols])
    else:
        df['Cluster asignado'] = nuevas_asignaciones

    st.markdown("### Resultado final: asignación de clusters para todo el dataset")
    st.dataframe(df[[*x_cols, 'Cluster asignado']])

def run():
    procesar_k_means()
