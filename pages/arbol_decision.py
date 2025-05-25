import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.preprocessing import LabelEncoder
import graphviz as graphviz
import re

def procesar_arbol_decision():
    st.title("🌳 Árbol de Decisión - Clasificación Categórica")

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

        st.markdown("### Selección de variables")
        target_col = st.selectbox("Selecciona la variable a predecir (target)", columnas_validas)

        # Sugerencia automática si detecta que es el archivo del profesor
        columnas_defecto = ["Nivel_Acad", "Area_Estudio", "Estrato"]
        if all(col in columnas for col in columnas_defecto):
            input_cols = st.multiselect(
                "Selecciona las variables de entrada (features)",
                [col for col in columnas_validas if col != target_col],
                default=columnas_defecto
            )
        else:
            input_cols = st.multiselect(
                "Selecciona las variables de entrada (features)",
                [col for col in columnas_validas if col != target_col]
            )

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

                clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
                clf.fit(X, y)

                st.success("Árbol entrenado correctamente.")

                # Visualización elegante con Graphviz
                st.subheader("🌐 Visualización del árbol (categorías)")
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

                # Extraer reglas con decodificación
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
                        if len(partes) < 2:
                            continue  # evita errores si la línea está mal formada
                        campo = partes[0].strip().lstrip('-')
                        valor = float(partes[1].strip())
                        operador = "<=" if "<=" in texto else ">"
                        if campo in label_encoders:
                            valor_legible = label_encoders[campo].inverse_transform([int(valor)])[0]
                        else:
                            valor_legible = valor
                        condiciones_actuales = condiciones_actuales[:nivel]
                        condiciones_actuales.append(f"{campo.strip()} {operador} {valor_legible}")

                st.dataframe(pd.DataFrame(reglas))

            except Exception as e:
                st.error(f"❌ Error al generar el árbol: {str(e)}")
