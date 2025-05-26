import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import re

def procesar_arbol_decision():
    st.title("🌳 Árbol de Decisión - Estilo del Profesor")

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

                # Codificar variables categóricas
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

                # Extraer reglas con operadores y valores legibles
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
                        m = re.match(r"(.+?)\s*(<=|>)\s*(.+)", texto)
                        if m:
                            campo, operador, valor = m.groups()
                            campo = campo.strip()
                            valor = float(valor.strip())

                            # Mapear valor a categoría original si aplica
                            if campo in label_encoders:
                                # Para operadores <= y > en categóricas, muestra valor categórico
                                valor_cat = label_encoders[campo].inverse_transform([int(valor)])[0]
                                condicion = f"{campo} {operador} {valor_cat}"
                            else:
                                condicion = f"{campo} {operador} {valor}"

                            condiciones_actuales = condiciones_actuales[:nivel]
                            condiciones_actuales.append(condicion)

                st.subheader("📋 Reglas de Clasificación Generadas")
                df_reglas = pd.DataFrame(reglas)
                st.dataframe(df_reglas)

                # Visualización estilo profesor con condiciones legibles
                st.subheader("🌳 Diagrama del Árbol (estilo profesor)")
                dot = "digraph Tree {\nnode [shape=box, style=filled, color=lightblue];\n"
                nodo_id = 0
                nodos = {}

                for regla in reglas:
                    condiciones = regla["Condición lógica"].split(" y ") if regla["Condición lógica"] != "(sin condición)" else []
                    clase = regla["Clase resultante"]

                    prev = "root"
                    if prev not in nodos:
                        dot += f'{prev} [label="Inicio"];\n'
                        nodos[prev] = True

                    for cond in condiciones:
                        # Limpiar cond para id de nodo
                        nodo_actual = prev + "_" + cond.replace(" ", "_").replace("=", "").replace(".", "").replace("<=", "le").replace(">", "gt")
                        if nodo_actual not in nodos:
                            dot += f'{nodo_actual} [label="{cond}"];\n'
                            dot += f'{prev} -> {nodo_actual};\n'
                            nodos[nodo_actual] = True
                        prev = nodo_actual

                    hoja = f'{prev}_class_{nodo_id}'
                    dot += f'{hoja} [label="Categoría: {clase}", color=lightgreen];\n'
                    dot += f'{prev} -> {hoja};\n'
                    nodo_id += 1

                dot += "}"
                st.graphviz_chart(dot)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
