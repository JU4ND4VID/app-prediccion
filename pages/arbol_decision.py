import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

def procesar_arbol_decision():
    st.title("Árbol de Decisión - Clasificación Automática")
    st.write("Carga un archivo CSV para generar el árbol de decisión automáticamente.")

    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Vista previa del archivo")
        st.dataframe(df)

        columnas = df.columns.tolist()
        target_col = st.selectbox("Selecciona la variable a predecir (target)", columnas)
        input_cols = st.multiselect(
            "Selecciona las variables de entrada (features)",
            [col for col in columnas if col != target_col]
        )

        if st.button("Generar Árbol de Decisión"):
            try:
                df_model = df[input_cols + [target_col]].astype(str)

                # Codificar columnas categóricas
                df_encoded = df_model.copy()
                label_encoders = {}
                for col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    label_encoders[col] = le

                X = df_encoded[input_cols]
                y = df_encoded[target_col]

                clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
                clf.fit(X, y)

                st.success("Árbol de decisión entrenado correctamente.")

                # Mostrar reglas de clasificación
                rules = export_text(clf, feature_names=input_cols, show_weights=True)
                st.subheader("📋 Reglas generadas por el árbol")
                st.code(rules)

                # Mostrar visualización del árbol
                fig, ax = plt.subplots(figsize=(14, 6))
                plot_tree(
                    clf,
                    feature_names=input_cols,
                    class_names=label_encoders[target_col].classes_,
                    filled=True,
                    rounded=True
                )
                st.subheader("🌳 Visualización del árbol de decisión")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Ocurrió un error al generar el árbol: {str(e)}")
