import streamlit as st
import pandas as pd
import numpy as np

def procesar_regresion_lineal():
    st.title("📈 Regresión Lineal Simple - Cálculo paso a paso")

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

    if st.button("Calcular regresión paso a paso"):
        try:
            X = df[x_col].values
            Y = df[y_col].values
            n = len(X)

            st.markdown("### Paso 1: Cálculo de medias")
            x_mean = np.mean(X)
            y_mean = np.mean(Y)
            st.markdown(f"Media de X: **{x_mean:.2f}**")
            st.markdown(f"Media de Y: **{y_mean:.2f}**")

            st.markdown("### Paso 2: Tabla de valores para cálculo")
            x_squared = X**2
            xy = X * Y
            tabla = pd.DataFrame({
                x_col: X,
                y_col: Y,
                f"{x_col}^2": x_squared,
                f"{x_col}*{y_col}": xy
            })
            st.dataframe(tabla)

            st.markdown("### Paso 3: Sumas necesarias")
            sum_x = np.sum(X)
            sum_y = np.sum(Y)
            sum_x2 = np.sum(x_squared)
            sum_xy = np.sum(xy)
            st.markdown(f"∑X = **{sum_x}**")
            st.markdown(f"∑Y = **{sum_y}**")
            st.markdown(f"∑X² = **{sum_x2}**")
            st.markdown(f"∑XY = **{sum_xy}**")
            st.markdown(f"n = **{n}**")

            st.markdown("### Paso 4: Cálculo de la pendiente (β₁)")
            numerador = n * sum_xy - sum_x * sum_y
            denominador = n * sum_x2 - sum_x**2
            beta_1 = numerador / denominador
            st.markdown(f"β₁ = (n * ∑XY - ∑X * ∑Y) / (n * ∑X² - (∑X)²)")
            st.markdown(f"β₁ = ({n} * {sum_xy} - {sum_x} * {sum_y}) / ({n} * {sum_x2} - {sum_x}²) = **{beta_1:.4f}**")

            st.markdown("### Paso 5: Cálculo del intercepto (β₀)")
            beta_0 = y_mean - beta_1 * x_mean
            st.markdown(f"β₀ = ȳ - β₁ * x̄")
            st.markdown(f"β₀ = {y_mean:.2f} - {beta_1:.4f} * {x_mean:.2f} = **{beta_0:.4f}**")

            st.markdown("### Paso 6: Ecuación de regresión")
            st.markdown(f"Y = {beta_0:.4f} + {beta_1:.4f} * X")

            st.markdown("### Paso 7: Predicción")
            nuevo_valor = st.number_input(f"Ingrese un valor para {x_col} para predecir {y_col}:", value=0.0)
            prediccion = beta_0 + beta_1 * nuevo_valor
            st.success(f"Predicción para {x_col} = {nuevo_valor:.2f} ➤ {y_col} = {prediccion:.2f}")

        except Exception as e:
            st.error(f"Error en el cálculo: {str(e)}")

def run():
    procesar_regresion_lineal()
