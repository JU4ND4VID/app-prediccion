import streamlit as st
import pandas as pd

st.title("Visualizador de CSV - Predicción")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Vista previa del archivo:")
    st.dataframe(df)
