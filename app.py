import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

# Opcional: para KMedoids
try:
    from sklearn_extra.cluster import KMedoids
    kmedoids_available = True
except ImportError:
    kmedoids_available = False

st.title("Visualizador de CSV - Predicción")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista previa del archivo:")
    st.dataframe(df)

    columnas = df.columns.tolist()

    st.subheader("Selecciona un algoritmo")
    algoritmo = st.selectbox("Algoritmo", [
        "Regresión Lineal",
        "Regresión Múltiple",
        "Árbol de Decisión",
        "K-means",
        "K-medoids" if kmedoids_available else "K-medoids (no disponible)"
    ])

    if algoritmo in ["Regresión Lineal", "Regresión Múltiple", "Árbol de Decisión"]:
        col_target = st.selectbox("Selecciona la columna objetivo (variable dependiente)", columnas)

        # Determinar columnas numéricas para las features
        features = st.multiselect("Selecciona las columnas de entrada (variables independientes)", 
                                  [c for c in columnas if c != col_target])

        if st.button("Entrenar modelo"):
            try:
                X = df[features]
                y = df[col_target]

                # Conversión de variables categóricas
                if y.dtype == object:
                    y = LabelEncoder().fit_transform(y)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                if algoritmo == "Regresión Lineal" or algoritmo == "Regresión Múltiple":
                    modelo = LinearRegression()
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    st.success(f"MSE: {mse:.4f}")

                elif algoritmo == "Árbol de Decisión":
                    modelo = DecisionTreeClassifier()
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Precisión: {acc:.4f}")

            except Exception as e:
                st.error(f"Error al entrenar el modelo: {str(e)}")

    elif algoritmo == "K-means":
        num_clusters = st.slider("Número de Clusters (K)", 2, 10, 3)
        features = st.multiselect("Selecciona columnas numéricas para clustering", columnas)

        if st.button("Ejecutar K-means"):
            try:
                X = df[features]
                modelo = KMeans(n_clusters=num_clusters)
                df['Cluster'] = modelo.fit_predict(X)
                st.success("K-means ejecutado correctamente.")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif algoritmo == "K-medoids" and kmedoids_available:
        num_clusters = st.slider("Número de Clusters (K)", 2, 10, 3)
        features = st.multiselect("Selecciona columnas numéricas para clustering", columnas)

        if st.button("Ejecutar K-medoids"):
            try:
                X = df[features]
                modelo = KMedoids(n_clusters=num_clusters)
                df['Cluster'] = modelo.fit_predict(X)
                st.success("K-medoids ejecutado correctamente.")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error: {str(e)}")
