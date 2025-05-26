import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def procesar_regresion_multiple():
    st.title("🏠 Regresión Lineal Múltiple")

    st.write("Carga un archivo CSV con varias variables independientes y una dependiente para generar el modelo.")

    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Vista previa del archivo")
        st.dataframe(df)

        columnas = df.columns.tolist()

        y_col = st.selectbox("Selecciona la variable dependiente (Y)", columnas)
        x_cols = st.multiselect("Selecciona las variables independientes (X)", [col for col in columnas if col != y_col])

        if len(x_cols) > 0 and st.button("Entrenar modelo"):
            try:
                X = df[x_cols].values
                y = df[y_col].values

                modelo = LinearRegression()
                modelo.fit(X, y)
                y_pred = modelo.predict(X)

                st.success("Modelo entrenado exitosamente.")

                ecuacion = " + ".join([f"{coef:.4f} * {var}" for coef, var in zip(modelo.coef_, x_cols)])
                st.markdown(f"**Ecuación del modelo:**  \n`Y = {ecuacion} + {modelo.intercept_:.4f}`")
                st.markdown(f"**MSE:** {mean_squared_error(y, y_pred):.4f}")
                st.markdown(f"**R² (coeficiente de determinación):** {r2_score(y, y_pred):.4f}")

                # Predicción interactiva
                st.subheader("Haz una predicción")
                valores = []
                for col in x_cols:
                    valor = st.number_input(f"Ingrese un valor para {col}:", value=0.0)
                    valores.append(valor)

                if st.button("Predecir"):
                    prediccion = modelo.predict([valores])[0]
                    st.success(f"Predicción para los valores ingresados ➤ {y_col} = {prediccion:.2f}")

            except Exception as e:
                st.error(f"Error en el modelo: {str(e)}")
