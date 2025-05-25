import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz as graphviz
import re

def procesar_arbol_decision():
    st.title("🌳 Árbol de Decisión Categórico (Estilo Profesional)")

    uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Vista previa del archivo")
        st.dataframe(df)

        columnas = df.columns.tolist()
        columnas_validas = [col for col in columnas if df[col].dtype == 'object' or df[col].nunique() <= 20]

        target_col = st.selectbox("Selecciona la variable a predecir (target)", columnas_validas)
        input_cols = st.multiselect("Selecciona las variables de entrada (features)", [col for col in columnas_validas if col != target_col])

        if st.button("Generar Árbol de Decisión"):
            try:
                df_model = df[input_cols + [target_col]].astype(str)

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

                st.success("Árbol entrenado correctamente.")

                # 🌳 Visualización profesional con Graphviz
                st.subheader("🌐 Visualización elegante del árbol de decisión")
                dot_data = export_graphviz(
                    clf,
                    out_file=None,
                    feature_names=input_cols,
                    class_names=label_encoders[target_col].classes_,
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                st.graphviz_chart(dot_data)

                # 📋 Generación de reglas en tabla legible
                st.subheader("📋 Reglas de Clasificación Generadas")
                rules_raw = export_text(clf, feature_names=input_cols)
                rules_list = rules_raw.strip().split("\n")
                reglas = []
                condiciones_actuales = []

                for linea in rules_list:
                    nivel = linea.count("|   ")
                    texto = linea.strip().replace("|", "").strip()

                    if "class:" in texto:
                        clase = texto.split("class:")[-1].strip()
                        condicion = " y ".join(condiciones_actuales[:nivel])
                        reglas.append({
                            "Regla N°": len(reglas) + 1,
                            "Condición lógica": condicion if condicion else "(sin condición)",
                            "Clase resultante": clase
                        })
                    else:
                        partes = re.split(r"<=|>", texto)
                        campo = partes[0].strip()
                        valor = float(partes[1].strip())
                        valor_decodificado = label_encoders[campo].inverse_transform([int(valor)])[0]
                        operador = "<=" if "<=" in texto else ">"
                        condiciones_actuales = condiciones_actuales[:nivel]
                        condiciones_actuales.append(f"{campo} {operador} {valor_decodificado}")

                st.dataframe(pd.DataFrame(reglas))

            except Exception as e:
                st.error(f"❌ Error al generar el árbol: {str(e)}")
