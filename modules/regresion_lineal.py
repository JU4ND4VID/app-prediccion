import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def procesar_regresion_lineal():
    st.title("📈 Regresión Lineal Simple con explicación paso a paso")

    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Por favor, sube un archivo para continuar.")
        return

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Vista previa del dataset")
    st.dataframe(df)

    columnas = df.columns.tolist()
    x_col = st.selectbox("Selecciona la variable independiente (X)", columnas)
    y_col = st.selectbox("Selecciona la variable dependiente (Y)", [col for col in columnas if col != x_col])

    if st.button("Entrenar modelo"):
        try:
            X = df[[x_col]].values
            y = df[y_col].values

            st.markdown("### Paso 1: Crear el modelo de regresión lineal")
            modelo = LinearRegression()

            st.markdown("### Paso 2: Entrenar el modelo con los datos")
            modelo.fit(X, y)
            st.success("Modelo entrenado correctamente.")

            st.markdown("### Paso 3: Obtener predicciones y evaluar el modelo")
            y_pred = modelo.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            st.markdown(f"- **Ecuación del modelo:**  \n`Y = {modelo.coef_[0]:.4f} * X + {modelo.intercept_:.4f}`")
            st.markdown(f"- **Error cuadrático medio (MSE):** {mse:.4f}")
            st.markdown(f"- **Coeficiente de determinación (R²):** {r2:.4f}")

            st.markdown("### Paso 4: Visualización del ajuste del modelo")
            fig, ax = plt.subplots()
            ax.scatter(X, y, color='blue', label='Datos reales')
            ax.plot(X, y_pred, color='red', label='Línea de regresión')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Ajuste de Regresión Lineal")
            ax.legend()
            st.pyplot(fig)

            st.markdown("### Paso 5: Realizar una predicción con un nuevo valor")
            nuevo_valor = st.number_input(f"Ingrese un valor para {x_col}:", value=0.0)
            prediccion = modelo.predict(np.array([[nuevo_valor]]))[0]
            st.success(f"Predicción para {x_col} = {nuevo_valor:.2f} ➤ {y_col} = {prediccion:.2f}")

        except Exception as e:
            st.error(f"Error en el entrenamiento o predicción: {str(e)}")

def run():
    procesar_regresion_lineal()
