import streamlit as st
import pandas as pd
import numpy as np

def procesar_regresion_lineal():
    st.title("📈 Regresión Lineal Simple - Cálculo paso a paso")

    # Formulario 1: carga y cálculo modelo
    with st.form(key='form_calculo'):
        uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])
        columnas = []
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            columnas = df.columns.tolist()
            st.dataframe(df)

            x_col = st.selectbox("Selecciona la variable independiente (X)", columnas)
            y_col = st.selectbox("Selecciona la variable dependiente (Y)", [col for col in columnas if col != x_col])
        else:
            x_col = y_col = None

        submit_calculo = st.form_submit_button(label='Calcular regresión paso a paso')

    if submit_calculo and uploaded_file and x_col and y_col:
        X = df[x_col].values
        Y = df[y_col].values
        n = len(X)

        x_mean = np.mean(X)
        y_mean = np.mean(Y)

        x_squared = X**2
        xy = X * Y
        tabla = pd.DataFrame({
            x_col: X,
            y_col: Y,
            f"{x_col}^2": x_squared,
            f"{x_col}*{y_col}": xy
        })

        sum_x = np.sum(X)
        sum_y = np.sum(Y)
        sum_x2 = np.sum(x_squared)
        sum_xy = np.sum(xy)

        numerador = n * sum_xy - sum_x * sum_y
        denominador = n * sum_x2 - sum_x**2
        beta_1 = numerador / denominador
        beta_0 = y_mean - beta_1 * x_mean

        # Guardamos todo en session_state
        st.session_state['beta_0'] = beta_0
        st.session_state['beta_1'] = beta_1
        st.session_state['x_col'] = x_col
        st.session_state['y_col'] = y_col
        st.session_state['tabla'] = tabla
        st.session_state['calculo_realizado'] = True

    # Mostrar resultados calculados
    if st.session_state.get('calculo_realizado', False):
        st.markdown("### Paso 1: Cálculo de medias")
        st.markdown(f"Media de {st.session_state['x_col']}: **{np.mean(df[st.session_state['x_col']]):.2f}**")
        st.markdown(f"Media de {st.session_state['y_col']}: **{np.mean(df[st.session_state['y_col']]):.2f}**")

        st.markdown("### Paso 2: Tabla de valores para cálculo")
        st.dataframe(st.session_state['tabla'])

        st.markdown("### Paso 3: Sumas necesarias")
        st.markdown(f"∑X = **{np.sum(df[st.session_state['x_col']]):.2f}**")
        st.markdown(f"∑Y = **{np.sum(df[st.session_state['y_col']]):.2f}**")
        st.markdown(f"∑X² = **{np.sum(df[st.session_state['x_col']]**2):.2f}**")
        st.markdown(f"∑XY = **{np.sum(df[st.session_state['x_col']] * df[st.session_state['y_col']]):.2f}**")
        st.markdown(f"n = **{len(df)}**")

        st.markdown("### Paso 4: Cálculo de la pendiente (β₁)")
        st.markdown(f"β₁ = **{st.session_state['beta_1']:.4f}**")

        st.markdown("### Paso 5: Cálculo del intercepto (β₀)")
        st.markdown(f"β₀ = **{st.session_state['beta_0']:.4f}**")

        st.markdown("### Paso 6: Ecuación de regresión")
        st.markdown(f"Y = {st.session_state['beta_0']:.4f} + {st.session_state['beta_1']:.4f} * X")

        # Formulario 2: predicción
        with st.form(key='form_prediccion'):
            nuevo_valor = st.number_input(f"Ingrese un valor para {st.session_state['x_col']} para predecir {st.session_state['y_col']}:", value=0.0)
            submit_prediccion = st.form_submit_button(label='Calcular predicción')

        if submit_prediccion:
            prediccion = st.session_state['beta_0'] + st.session_state['beta_1'] * nuevo_valor
            st.success(f"Predicción para {st.session_state['x_col']} = {nuevo_valor:.2f} ➤ {st.session_state['y_col']} = {prediccion:.2f}")

def run():
    procesar_regresion_lineal()
