import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def procesar_regresion_multiple():
    st.title("📊 Regresión Lineal Múltiple - Cálculo paso a paso")

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
        st.session_state['df'] = df
    except Exception as e:
        st.error(f"Error cargando archivo: {e}")
        return

    st.subheader("Vista previa del dataset")
    st.dataframe(df)

    columnas = df.columns.tolist()
    y_col = st.selectbox("Selecciona la variable dependiente (Y)", columnas, key="y_col_multiple")
    x_cols = st.multiselect("Selecciona las variables independientes (X)", [col for col in columnas if col != y_col], key="x_cols_multiple")

    if not x_cols:
        st.warning("Selecciona al menos una variable independiente.")
        return

    calcular = st.button("Calcular regresión múltiple paso a paso")

    if calcular:
        try:
            X = df[x_cols].values
            Y = df[y_col].values.reshape(-1, 1)
            n, k = X.shape

            # Agregar columna de unos para intercepto
            X_b = np.hstack([np.ones((n, 1)), X])

            # Calcular coeficientes β usando ecuación normal
            beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y
            beta = beta.flatten()

            # Guardar en session_state para persistencia
            st.session_state['beta'] = beta
            st.session_state['x_cols'] = x_cols
            st.session_state['y_col'] = y_col
            st.session_state['X_b'] = X_b
            st.session_state['Y'] = Y
            st.session_state['calculo_realizado'] = True

        except np.linalg.LinAlgError:
            st.error("Error: La matriz X^T * X no es invertible. Puede haber multicolinealidad entre variables independientes.")
        except Exception as e:
            st.error(f"Error en el cálculo: {str(e)}")

    # Mostrar resultados calculados
    if st.session_state.get('calculo_realizado', False):
        st.markdown("### Matriz de diseño X (con columna de unos para intercepto)")
        st.dataframe(pd.DataFrame(st.session_state['X_b'], columns=["Intercepto"] + st.session_state['x_cols']))

        st.markdown("### Vector de variable dependiente Y")
        st.dataframe(pd.DataFrame(st.session_state['Y'], columns=[st.session_state['y_col']]))

        st.markdown("### Coeficientes calculados (β)")
        coef_df = pd.DataFrame({
            "Variable": ["Intercepto"] + st.session_state['x_cols'],
            "Coeficiente (β)": st.session_state['beta']
        })
        st.dataframe(coef_df)

        # Mostrar ecuación final
        ecuacion = f"{st.session_state['y_col']} = {st.session_state['beta'][0]:.4f}"
        for i, col in enumerate(st.session_state['x_cols'], start=1):
            coef = st.session_state['beta'][i]
            signo = "+" if coef >= 0 else "-"
            ecuacion += f" {signo} {abs(coef):.4f} * {col}"
        st.markdown(f"### Ecuación de regresión final:\n\n{ecuacion}")

        # Formulario para predicción
        with st.form(key='form_prediccion_multiple'):
            st.markdown("### Realiza una predicción")
            valores = {}
            for col in st.session_state['x_cols']:
                valores[col] = st.number_input(f"Ingrese valor para {col}", value=0.0)
            submit_button = st.form_submit_button(label='Calcular predicción')

        if submit_button:
            x_input = np.array([1] + [valores[col] for col in st.session_state['x_cols']])
            prediccion = x_input @ st.session_state['beta']
            st.success(f"Predicción para {st.session_state['y_col']}: {prediccion:.4f}")

def run():
    procesar_regresion_multiple()
