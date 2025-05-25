import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def procesar_k_means():
    st.title("🔀 Agrupamiento con K-means")

    st.write("Carga un archivo con variables numéricas para aplicar el algoritmo de K-medias (K-means).")

    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Vista previa del archivo")
        st.dataframe(df)

        columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

        if not columnas_numericas:
            st.error("El archivo no contiene columnas numéricas.")
            return

        features = st.multiselect("Selecciona columnas para el clustering", columnas_numericas)

        if features:
            k = st.slider("Selecciona el número de clusters (K)", min_value=2, max_value=10, value=2)

            if st.button("Ejecutar K-means"):
                try:
                    X = df[features]
                    modelo = KMeans(n_clusters=k, n_init="auto", random_state=42)
                    clusters = modelo.fit_predict(X)
                    df_resultado = df.copy()
                    df_resultado["Cluster"] = clusters

                    st.success("K-means ejecutado correctamente.")
                    st.subheader("Resultado con cluster asignado")
                    st.dataframe(df_resultado)

                    # Mostrar centroides
                    st.subheader("📍 Centroides")
                    centroides_df = pd.DataFrame(modelo.cluster_centers_, columns=features)
                    st.dataframe(centroides_df)

                    # Gráfica si hay 2 dimensiones
                    if len(features) == 2:
                        st.subheader("📊 Gráfico de dispersión")
                        fig, ax = plt.subplots()
                        scatter = ax.scatter(X[features[0]], X[features[1]], c=clusters, cmap="Set1", s=100)
                        ax.scatter(modelo.cluster_centers_[:, 0], modelo.cluster_centers_[:, 1], c='black', marker='X', s=200, label="Centroides")
                        ax.set_xlabel(features[0])
                        ax.set_ylabel(features[1])
                        ax.set_title("Clustering K-means")
                        ax.legend()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error durante el clustering: {str(e)}")
