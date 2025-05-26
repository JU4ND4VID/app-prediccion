import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

def procesar_regresion_lineal():
    st.title("📈 Regresión Lineal Simple")

    st.write("Carga un archivo CSV con una variable independiente y una dependiente para generar el modelo.")

    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Vista previa del archivo")
        st.dataframe(df)

        columnas = df.columns.tolist()

        x_col = st.selectbox("Selecciona la variable independiente (X)", columnas)
        y_col = st.selectbox("Selecciona la variable dependiente (Y)", [col for col in columnas if col != x_col])

        if st.button("Entrenar modelo"):
            try:
                X = df[[x_col]].values
                y = df[y_col].values

                modelo = LinearRegression()
                modelo.fit(X, y)
                y_pred = modelo.predict(X)

                st.success("Modelo entrenado exitosamente.")
                st.markdown(f"**Ecuación del modelo:**  \n`Y = {modelo.coef_[0]:.4f} * X + {modelo.intercept_:.4f}`")
                st.markdown(f"**MSE:** {mean_squared_error(y, y_pred):.4f}")
                st.markdown(f"**R² (coeficiente de determinación):** {r2_score(y, y_pred):.4f}")

                # Gráfica
                fig, ax = plt.subplots()
                ax.scatter(X, y, color='blue', label='Datos reales')
                ax.plot(X, y_pred, color='red', label='Línea de regresión')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title("Regresión Lineal")
                ax.legend()
                st.pyplot(fig)

                # Predicción con nuevo valor
                st.subheader("Haz una predicción")
                nuevo_valor = st.number_input(f"Ingrese un valor para {x_col}:", value=0.0)
                prediccion = modelo.predict(np.array([[nuevo_valor]]))[0]
                st.success(f"Predicción para {x_col} = {nuevo_valor:.2f} ➤ {y_col} = {prediccion:.2f}")

            except Exception as e:
                st.error(f"Error en el modelo: {str(e)}")
